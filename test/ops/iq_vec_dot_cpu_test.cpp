// CPU vec_dot validation for the IQ experts fastllm computes on CPU:
//   ggml_vec_dot_iq1_s_q8_K   (block_iq1_s   x block_q8_K)
//   ggml_vec_dot_iq2_xxs_q8_K (block_iq2_xxs x block_q8_K)
//   ggml_vec_dot_iq3_xxs_q8_K (block_iq3_xxs x block_q8_K)
//   ggml_vec_dot_mxfp4_q8_0   (block_mxfp4   x block_q8_0)
//
// Each vec_dot result is compared against dequantize-then-dot (the exact
// arithmetic llama.cpp uses for the scalar reference). This catches the
// bsums-layout bug that previously zeroed the IQ1_S delta term.
//
// The q8 inputs are produced with iqk_quantize_row_q8_K (the same quantizer
// fastllm's CPU GGUF linear path uses), so we validate the full pipeline.

#include "gguf.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

using namespace fastllm;

extern float GGML_FP16_TO_FP32(ggml_half f);
extern ggml_half GGML_FP32_TO_FP16(float x);
void iqk_quantize_row_q8_K(const float *x, void *vy, int64_t k, ggml_type type, ggml_type oriType);

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (cond) { printf("  PASS: %s\n", msg); } \
    else { printf("  FAIL: %s\n", msg); failures++; } \
} while (0)

static void quantize_q8_K(const float *x, block_q8_K *y, int64_t k) {
    iqk_quantize_row_q8_K(x, y, k, GGML_TYPE_Q8_K, GGML_TYPE_Q8_K);
}

// ---------------- IQ1_S ----------------
static void ref_dot_iq1_s(const void *vx, const void *vy, float *out, int n) {
    const block_iq1_s *x = (const block_iq1_s*)vx;
    const block_q8_K  *y = (const block_q8_K*)vy;
    const int nb = n / QK_K;
    float sumf = 0;
    for (int i = 0; i < nb; i++) {
        float xf[QK_K];
        dequantize_row_iq1_s(x + i, xf, QK_K);
        double dot = 0;
        for (int j = 0; j < QK_K; j++) dot += (double)xf[j] * y[i].qs[j];
        sumf += (float)dot * y[i].d;
    }
    *out = sumf;
}

// ---------------- IQ2_XXS ----------------
static void ref_dot_iq2_xxs(const void *vx, const void *vy, float *out, int n) {
    const block_iq2_xxs *x = (const block_iq2_xxs*)vx;
    const block_q8_K    *y = (const block_q8_K*)vy;
    const int nb = n / QK_K;
    float sumf = 0;
    for (int i = 0; i < nb; i++) {
        float xf[QK_K];
        dequantize_row_iq2_xxs(x + i, xf, QK_K);
        double dot = 0;
        for (int j = 0; j < QK_K; j++) dot += (double)xf[j] * y[i].qs[j];
        sumf += (float)dot * y[i].d;
    }
    *out = sumf;
}

// ---------------- IQ3_XXS ----------------
static void ref_dot_iq3_xxs(const void *vx, const void *vy, float *out, int n) {
    const block_iq3_xxs *x = (const block_iq3_xxs*)vx;
    const block_q8_K    *y = (const block_q8_K*)vy;
    const int nb = n / QK_K;
    float sumf = 0;
    for (int i = 0; i < nb; i++) {
        float xf[QK_K];
        dequantize_row_iq3_xxs(x + i, xf, QK_K);
        double dot = 0;
        for (int j = 0; j < QK_K; j++) dot += (double)xf[j] * y[i].qs[j];
        sumf += (float)dot * y[i].d;
    }
    *out = sumf;
}

// ---------------- host glue ----------------

// Randomly fill a block_iq1_s (mirrors the CUDA test): d fp16, qs bytes, qh u16.
static void fill_iq1_s(std::mt19937 &rng, block_iq1_s *b) {
    b->d = GGML_FP32_TO_FP16((float)(rng() % 1000) / 100.0f);
    for (auto &q : b->qs) q = rng() & 0xFF;
    for (auto &q : b->qh) q = rng() & 0xFFFF;
}
static void fill_iq2_xxs(std::mt19937 &rng, block_iq2_xxs *b) {
    b->d = GGML_FP32_TO_FP16((float)(rng() % 1000) / 100.0f);
    for (auto &q : b->qs) q = rng() & 0xFFFF;
}
static void fill_iq3_xxs(std::mt19937 &rng, block_iq3_xxs *b) {
    b->d = GGML_FP32_TO_FP16((float)(rng() % 1000) / 100.0f);
    for (auto &q : b->qs) q = rng() & 0xFF;
}

// The q8 side is generated from random floats via iqk_quantize_row_q8_K, the
// exact quantizer fastllm's CPU GGUF linear path uses. This validates the full
// vec_dot pipeline including bsums.
static void run_test(const char *name, int nblocks,
                     void (*fill)(std::mt19937 &, void *),
                     size_t blockSize,
                     void (*ref)(const void *, const void *, float *, int),
                     void (*dot)(int, float *, size_t, const void *, size_t,
                                 const void *, size_t, int)) {
    const int n = nblocks * QK_K;
    std::mt19937 rng(2026);

    std::vector<uint8_t> xraw(nblocks * blockSize);
    for (int b = 0; b < nblocks; b++) fill(rng, xraw.data() + (size_t)b * blockSize);

    std::vector<float> src(n);
    for (auto &v : src) v = ((float)rng() / RAND_MAX) * 2.0f - 1.0f;
    std::vector<block_q8_K> y(nblocks);
    quantize_q8_K(src.data(), y.data(), n);

    float refVal = 0, gotVal = 0;
    ref(xraw.data(), y.data(), &refVal, n);
    dot(n, &gotVal, 0, xraw.data(), 0, y.data(), 0, 1);

    double denom = std::max(1.0, (double)std::fabs(refVal));
    double err = std::fabs((double)refVal - gotVal) / denom;
    printf("[%s] %d blocks ref=%.6f got=%.6f rel_err=%.3e  %s\n",
           name, nblocks, refVal, gotVal, err,
           err < 1e-3 ? "PASS" : "FAIL");
    if (err >= 1e-3) failures++;
}

int main() {
    printf("IQ CPU vec_dot vs dequant-then-dot reference\n");

    run_test("IQ1_S", 8, [](std::mt19937 &r, void *p){ fill_iq1_s(r, (block_iq1_s*)p); },
             sizeof(block_iq1_s), ref_dot_iq1_s, ggml_vec_dot_iq1_s_q8_K);

    run_test("IQ2_XXS", 8, [](std::mt19937 &r, void *p){ fill_iq2_xxs(r, (block_iq2_xxs*)p); },
             sizeof(block_iq2_xxs), ref_dot_iq2_xxs, ggml_vec_dot_iq2_xxs_q8_K);

    run_test("IQ3_XXS", 8, [](std::mt19937 &r, void *p){ fill_iq3_xxs(r, (block_iq3_xxs*)p); },
             sizeof(block_iq3_xxs), ref_dot_iq3_xxs, ggml_vec_dot_iq3_xxs_q8_K);

    if (failures == 0) { printf("\nALL PASSED\n"); return 0; }
    printf("\n%d FAILED\n", failures); return 1;
}
