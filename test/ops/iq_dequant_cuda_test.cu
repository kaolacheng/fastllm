//
// Independent CUDA test: verify IQ1_S and IQ2_XXS GPU dequant kernels produce
// the same values as a scalar CPU reference. This exercises the dequant
// kernels ported from llama.cpp into fastllm (dequantize_block_iq1_s /
// dequantize_block_iq2_xxs in fastllm-ggml-cuda.cu).
//
// It is self-contained: it includes only gguf.h (block structs + codebook
// tables) and re-declares the two GPU kernels inline. The scalar reference is
// written from the CPU dequantize_row_* in third_party/gguf/ggml-dequantize.cpp.
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

// ---------------- GPU kernels (copied from fastllm-ggml-cuda.cu) ----------------

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

template<typename dst_t>
__global__ void dequantize_block_iq2_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i   = blockIdx.x;
    const block_iq2_xxs * x = (const block_iq2_xxs *) vx;
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8; // 0...3
    const int64_t ib  = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * q2 = x[i].qs + 4*ib;
    const uint8_t  * aux8 = (const uint8_t *)q2;
    const uint8_t  * grid = (const uint8_t *)(iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float)x[i].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 8; ++j) y[j] = DequantizeCast<dst_t>::cast(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}

template<typename dst_t>
__global__ void dequantize_block_iq1_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i   = blockIdx.x;
    const block_iq1_s * x = (const block_iq1_s *) vx;
    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    // fastllm stores iq1s_grid as uint64 (8 bytes = 8 int8 weights in {-1,0,1}),
    // so the delta here is plain +/-IQ1S_DELTA (no -1 offset). This matches the
    // CPU reference (ggml-dequantize.cpp dequantize_row_iq1_s) and differs from
    // llama.cpp's GPU path which uses a 0/1/2 nibble table with a -1 offset.
    const float delta = x[i].qh[ib] & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
    const float d = (float)x[i].d * (2*((x[i].qh[ib] >> 12) & 7) + 1);
    const uint64_t g = iq1s_grid[x[i].qs[4*ib+il] | (((x[i].qh[ib] >> 3*il) & 7) << 8)];
    const int8_t * q = (const int8_t *)&g;
    for (int j = 0; j < 8; ++j) y[j] = DequantizeCast<dst_t>::cast(d * (q[j] + delta));
}

// ---------------- Scalar CPU references (from ggml-dequantize.cpp) ----------------

static __device__ float fp16_to_fp32(ggml_half f) {
#if defined(GGML_COMMON_IMPL_CUDA) || defined(__CUDA_ARCH__)
    return __half2float(f);
#else
    uint32_t bits = (uint32_t)f;
    uint32_t sign = (bits >> 15) & 1;
    uint32_t exp  = (bits >> 10) & 0x1F;
    uint32_t man  = bits & 0x3FF;
    if (exp == 0) {
        return ldexpf((float)man, -24) * (sign ? -1.f : 1.f);
    }
    if (exp == 0x1F) {
        return man ? NAN : (sign ? -INFINITY : INFINITY);
    }
    return ldexpf((float)(0x400 | man), (int)exp - 25) * (sign ? -1.f : 1.f);
#endif
}

__device__ void ref_iq2_xxs_block(const block_iq2_xxs * x, float * y) {
    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;
    const float d = fp16_to_fp32(x->d);
    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
        memcpy(aux32, x->qs + 4*ib32, 2*sizeof(uint32_t));
        const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
            const uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
            for (int j = 0; j < 8; ++j) {
                y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
            }
            y += 8;
        }
    }
}

__device__ void ref_iq1_s_block(const block_iq1_s * x, float * y) {
    const float d = fp16_to_fp32(x->d);
    const uint8_t  * qs = x->qs;
    const uint16_t * qh = x->qh;
    for (int ib = 0; ib < QK_K/32; ++ib) {
        const float dl = d * (2*((qh[ib] >> 12) & 7) + 1);
        const float delta = qh[ib] & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
        for (int l = 0; l < 4; ++l) {
            const int8_t * grid = (const int8_t *)(iq1s_grid + (qs[l] | (((qh[ib] >> 3*l) & 7) << 8)));
            for (int j = 0; j < 8; ++j) {
                y[j] = dl * (grid[j] + delta);
            }
            y += 8;
        }
        qs += 4;
    }
}

// scalar reference kernels: one block per thread
__global__ void ref_dequant_iq2_xxs(const block_iq2_xxs * x, float * y) {
    const int i = blockIdx.x;
    ref_iq2_xxs_block(x + i, y + i*QK_K);
}
__global__ void ref_dequant_iq1_s(const block_iq1_s * x, float * y) {
    const int i = blockIdx.x;
    ref_iq1_s_block(x + i, y + i*QK_K);
}

// ---------------- host glue ----------------

template<typename T>
void run_test_iq2_xxs(int nblocks) {
    // random blocks
    std::vector<block_iq2_xxs> blocks(nblocks);
    std::mt19937 rng(777);
    for (auto &b : blocks) {
        uint32_t u = rng() & 0xFFFF; b.d = __ushort_as_half((unsigned short)u);
        for (auto &q : b.qs) q = rng() & 0xFFFF;
    }

    // device copy
    block_iq2_xxs * d_blocks; float * d_out; float * d_ref;
    CUDA_CHECK(cudaMalloc(&d_blocks, blocks.size()*sizeof(block_iq2_xxs)));
    CUDA_CHECK(cudaMalloc(&d_out, nblocks*QK_K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ref, nblocks*QK_K*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_blocks, blocks.data(), blocks.size()*sizeof(block_iq2_xxs), cudaMemcpyHostToDevice));

    // GPU dequant (fastllm kernel) + scalar reference kernel
    dequantize_block_iq2_xxs<float><<<nblocks, 32>>>(d_blocks, d_out);
    ref_dequant_iq2_xxs<<<nblocks, 1>>>(d_blocks, d_ref);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu_out(nblocks*QK_K), ref_out(nblocks*QK_K);
    CUDA_CHECK(cudaMemcpy(gpu_out.data(), d_out, nblocks*QK_K*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ref_out.data(), d_ref, nblocks*QK_K*sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < nblocks*QK_K; i++) {
        max_err = std::max(max_err, std::fabs(gpu_out[i] - ref_out[i]));
    }
    printf("[IQ2_XXS] %d blocks, max_err = %.3e  %s\n", nblocks, max_err,
           max_err < 1e-4f ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_blocks)); CUDA_CHECK(cudaFree(d_out)); CUDA_CHECK(cudaFree(d_ref));
}

void run_test_iq1_s(int nblocks) {
    std::vector<block_iq1_s> blocks(nblocks);
    std::mt19937 rng(888);
    for (auto &b : blocks) {
        uint32_t u = rng() & 0xFFFF; b.d = __ushort_as_half((unsigned short)u);
        for (auto &q : b.qs) q = rng() & 0xFF;
        for (auto &q : b.qh) q = rng() & 0xFFFF;
    }

    block_iq1_s * d_blocks; float * d_out; float * d_ref;
    CUDA_CHECK(cudaMalloc(&d_blocks, blocks.size()*sizeof(block_iq1_s)));
    CUDA_CHECK(cudaMalloc(&d_out, nblocks*QK_K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ref, nblocks*QK_K*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_blocks, blocks.data(), blocks.size()*sizeof(block_iq1_s), cudaMemcpyHostToDevice));

    dequantize_block_iq1_s<float><<<nblocks, 32>>>(d_blocks, d_out);
    ref_dequant_iq1_s<<<nblocks, 1>>>(d_blocks, d_ref);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu_out(nblocks*QK_K), ref_out(nblocks*QK_K);
    CUDA_CHECK(cudaMemcpy(gpu_out.data(), d_out, nblocks*QK_K*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ref_out.data(), d_ref, nblocks*QK_K*sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < nblocks*QK_K; i++) {
        max_err = std::max(max_err, std::fabs(gpu_out[i] - ref_out[i]));
    }
    printf("[IQ1_S] %d blocks, max_err = %.3e  %s\n", nblocks, max_err,
           max_err < 1e-4f ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_blocks)); CUDA_CHECK(cudaFree(d_out)); CUDA_CHECK(cudaFree(d_ref));
}

int main() {
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("=== IQ GPU dequant test (QK_K=%d) ===\n", QK_K);

    run_test_iq2_xxs<float>(64);
    run_test_iq1_s(64);
    printf("DONE\n");
    return 0;
}
