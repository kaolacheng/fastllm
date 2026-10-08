//
// Independent CUDA test: verify the MXFP4 GPU dequant kernel produces the
// same values as a scalar CPU reference. This exercises the kernel ported
// from llama.cpp into fastllm (dequantize_block_mxfp4 in
// fastllm-ggml-cuda.cu).
//
// Layout: a GGUF MXFP4 row is a sequence of block_mxfp4 (32 elements each:
// 1 E8M0 scale byte + 16 packed e2m1 nibbles). The GPU kernel decodes
// QK_K/QK_MXFP4 = 8 block_mxfp4 per CUDA block. The reference matches the
// CPU dequantize_row_mxfp4 formula (E8M0 halved against the doubled
// kvalues_fp4 table), so we expect bit-identical results.
//

#define GGML_COMMON_IMPL_CUDA
#include "gguf.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <random>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while(0)

// ---------------- GPU kernel (copied from fastllm-ggml-cuda.cu) ----------------

template<typename dst_t>
struct DequantizeCast;

template<>
struct DequantizeCast<float> {
    static __device__ __forceinline__ float cast(float x) { return x; }
};

template<>
struct DequantizeCast<half> {
    static __device__ __forceinline__ half cast(float x) { return __float2half(x); }
};

static __device__ float ggml_cuda_e8m0_to_fp32_half(uint8_t x) {
    uint32_t bits;
    if (x < 2) {
        bits = 0x00200000u << x;
    } else {
        bits = (uint32_t)(x - 1) << 23;
    }
    return __uint_as_float(bits);
}

template<typename dst_t>
__global__ void dequantize_block_mxfp4(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i   = blockIdx.x;
    const block_mxfp4 * x = (const block_mxfp4 *) vx + i*(QK_K/QK_MXFP4);

    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8; // 0...3
    const int64_t ib  = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    const uint8_t  * q4 = x[ib].qs + 4*il;
    const float d = ggml_cuda_e8m0_to_fp32_half(x[ib].e);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = DequantizeCast<dst_t>::cast(d * kvalues_fp4[q4[j] & 0xf]);
        y[j+16] = DequantizeCast<dst_t>::cast(d * kvalues_fp4[q4[j] >>  4]);
    }
}

// ---------------- scalar reference (matches ggml-dequantize.cpp) ----------------

__device__ float ref_e8m0_to_fp32_half(uint8_t x) {
    uint32_t bits;
    if (x < 2) {
        bits = 0x00200000u << x;
    } else {
        bits = (uint32_t)(x - 1) << 23;
    }
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

__global__ void ref_dequant_mxfp4(const block_mxfp4 * x, float * y) {
    const int i = blockIdx.x;
    const block_mxfp4 * bx = x + i*(QK_K/QK_MXFP4);
    float * by = y + i*QK_K;
    for (int b = 0; b < QK_K/QK_MXFP4; ++b) {
        const float d = ref_e8m0_to_fp32_half(bx[b].e);
        for (int j = 0; j < QK_MXFP4/2; ++j) {
            by[b*QK_MXFP4 + j + 0       ] = d * kvalues_fp4[bx[b].qs[j] & 0x0F];
            by[b*QK_MXFP4 + j + QK_MXFP4/2] = d * kvalues_fp4[bx[b].qs[j] >>    4];
        }
    }
}

// ---------------- host glue ----------------

void run_test_mxfp4(int nblocks) {
    // nblocks = number of QK_K(256)-element groups; each has QK_K/QK_MXFP4=8 blocks
    const int nblock_mxfp4 = nblocks * (QK_K/QK_MXFP4);
    std::vector<block_mxfp4> blocks(nblock_mxfp4);
    std::mt19937 rng(2026);
    for (auto &b : blocks) {
        b.e = rng() & 0xFF;
        for (auto &q : b.qs) q = rng() & 0xFF;
    }

    block_mxfp4 * d_blocks; float * d_out; float * d_ref;
    CUDA_CHECK(cudaMalloc(&d_blocks, blocks.size()*sizeof(block_mxfp4)));
    CUDA_CHECK(cudaMalloc(&d_out, nblocks*QK_K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ref, nblocks*QK_K*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_blocks, blocks.data(), blocks.size()*sizeof(block_mxfp4), cudaMemcpyHostToDevice));

    dequantize_block_mxfp4<float><<<nblocks, 32>>>(d_blocks, d_out);
    ref_dequant_mxfp4<<<nblocks, 1>>>(d_blocks, d_ref);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu_out(nblocks*QK_K), ref_out(nblocks*QK_K);
    CUDA_CHECK(cudaMemcpy(gpu_out.data(), d_out, nblocks*QK_K*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ref_out.data(), d_ref, nblocks*QK_K*sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < nblocks*QK_K; i++) {
        max_err = std::max(max_err, std::fabs(gpu_out[i] - ref_out[i]));
    }
    printf("[MXFP4] %d blocks (8 x block_mxfp4 each), max_err = %.3e  %s\n", nblocks, max_err,
           max_err == 0.0f ? "PASS bit-identical" : "FAIL");

    CUDA_CHECK(cudaFree(d_blocks)); CUDA_CHECK(cudaFree(d_out)); CUDA_CHECK(cudaFree(d_ref));
}

int main() {
    printf("MXFP4 CUDA dequant vs CPU reference\n");
    run_test_mxfp4(1);
    run_test_mxfp4(4);
    run_test_mxfp4(256);
    run_test_mxfp4(4096);
    return 0;
}
