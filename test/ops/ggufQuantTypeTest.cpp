//
// Independent test for the two newly-added GGML quant types:
//   1. IQ3_XXS - vec_dot_iq3_xxs_q8_K (ported from llama.cpp) + to_float
//   2. MXFP4   - block_mxfp4 + dequantize_row_mxfp4 (ported from llama.cpp)
//
// Verifies:
//   - type_traits table is wired up (ggml_type_size / ggml_blck_size / to_float / vec_dot)
//   - MXFP4 dequant produces expected values on a hand-crafted block
//   - IQ3_XXS vec_dot matches the dequantize-then-dot reference
//

#include "gguf.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

using namespace fastllm;

// declared in ggml-quant.cpp / ggml-dequantize.cpp (not in gguf.h)
extern float GGML_FP16_TO_FP32(ggml_half f);
extern ggml_half GGML_FP32_TO_FP16(float x);

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (cond) { printf("  PASS: %s\n", msg); } \
    else { printf("  FAIL: %s\n", msg); failures++; } \
} while (0)

// ---------- MXFP4 ----------
static void test_mxfp4_traits() {
    printf("[MXFP4] type_traits wiring\n");
    CHECK(ggml_type_size(GGML_TYPE_MXFP4) == sizeof(block_mxfp4),
          "ggml_type_size(MXFP4) == sizeof(block_mxfp4)");
    CHECK(ggml_blck_size(GGML_TYPE_MXFP4) == QK_MXFP4,
          "ggml_blck_size(MXFP4) == QK_MXFP4");
    CHECK(ggml_type_to_float(GGML_TYPE_MXFP4) != nullptr,
          "ggml_type_to_float(MXFP4) != nullptr");
    CHECK(ggml_is_quantized(GGML_TYPE_MXFP4),
          "ggml_is_quantized(MXFP4) == true");
    printf("    type_size=%zu blck_size=%lld type_name=%s\n",
           ggml_type_size(GGML_TYPE_MXFP4),
           (long long)ggml_blck_size(GGML_TYPE_MXFP4),
           ggml_type_name(GGML_TYPE_MXFP4));
}

static void test_mxfp4_dequant() {
    printf("[MXFP4] dequant correctness\n");
    block_mxfp4 b;
    b.e = 127;                          // e8m0 -> d = 2^(127-128) = 0.5 (via e8m0_to_fp32_half)
    for (int i = 0; i < QK_MXFP4/2; i++) {
        b.qs[i] = 0x11;                 // nibble 1 in both halves -> kvalue_mxfp4[1] = 1
    }
    std::vector<float> out(QK_MXFP4);
    dequantize_row_mxfp4(&b, out.data(), QK_MXFP4);
    bool ok = true;
    for (int i = 0; i < QK_MXFP4; i++) {
        if (std::fabs(out[i] - 0.5f) > 1e-5f) { ok = false; break; }
    }
    CHECK(ok, "all 32 elements == 0.5 (nibble=1, e=127)");

    // mixed nibbles: nibble 2 -> kvalue 2, nibble 3 -> kvalue 3, nibble 4 -> 4, nibble 12 -> -8
    b.e = 128;                          // d = 2^(128-128) = 1.0
    for (int i = 0; i < QK_MXFP4/2; i++) {
        b.qs[i] = (uint8_t)(i % 16) | (((i % 16) + 1) << 4);
    }
    dequantize_row_mxfp4(&b, out.data(), QK_MXFP4);
    ok = true;
    for (int i = 0; i < QK_MXFP4/2; i++) {
        int nib0 = i % 16;
        int nib1 = (i % 16) + 1;
        float v0 = (float)kvalues_mxfp4[nib0];   // d == 1.0
        float v1 = (float)kvalues_mxfp4[nib1 & 15];
        if (std::fabs(out[i] - v0) > 1e-5f || std::fabs(out[i + QK_MXFP4/2] - v1) > 1e-5f) {
            ok = false; break;
        }
    }
    CHECK(ok, "mixed nibbles match kvalues_mxfp4 table");
}

// ---------- IQ3_XXS ----------
static void test_iq3_xxs_traits() {
    printf("[IQ3_XXS] type_traits wiring\n");
    CHECK(ggml_type_to_float(GGML_TYPE_IQ3_XXS) != nullptr,
          "ggml_type_to_float(IQ3_XXS) != nullptr");
    CHECK(ggml_type_vec_dot(GGML_TYPE_IQ3_XXS) != nullptr,
          "ggml_type_vec_dot(IQ3_XXS) != nullptr");
    CHECK(ggml_type_vec_dot_type(GGML_TYPE_IQ3_XXS) == GGML_TYPE_Q8_K,
          "ggml_type_vec_dot_type(IQ3_XXS) == Q8_K");
    CHECK(ggml_blck_size(GGML_TYPE_IQ3_XXS) == QK_K,
          "ggml_blck_size(IQ3_XXS) == QK_K");
    CHECK(ggml_type_size(GGML_TYPE_IQ3_XXS) == sizeof(block_iq3_xxs),
          "ggml_type_size(IQ3_XXS) == sizeof(block_iq3_xxs)");
}

static void test_iq3_xxs_vec_dot() {
    printf("[IQ3_XXS] vec_dot vs dequant consistency\n");
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist_byte(0, 255);

    // QK_K is 256 by default
    static_assert(QK_K == 256, "test assumes QK_K == 256");

    // craft random iq3_xxs blocks
    const int nb = 3;
    std::vector<block_iq3_xxs> bx(nb);
    for (int b = 0; b < nb; b++) {
        bx[b].d = GGML_FP32_TO_FP16(0.75f + 0.01f * b);
        for (int i = 0; i < (int)sizeof(bx[b].qs); i++) {
            bx[b].qs[i] = (uint8_t)dist_byte(rng);
        }
    }

    // dequantize to float reference
    std::vector<float> xf(nb * QK_K);
    for (int b = 0; b < nb; b++) {
        dequantize_row_iq3_xxs(&bx[b], xf.data() + b * QK_K, QK_K);
    }

    // Build a standard block_q8_K by hand (d = 1.0, qs = small int8 values).
    // NOTE: fastllm's block_q8_K differs from llama.cpp's: d is float, qs is
    // int8_t, and there is no dmin field. This decouples the vec_dot test from
    // iqk_quantize_row_q8_K, so we verify pure vec_dot math against
    // dequantize-then-dot.
    std::vector<block_q8_K> q8(nb);
    for (int b = 0; b < nb; b++) {
        q8[b].d = 1.0f;
        q8[b].sum = 0.0f;
        for (int i = 0; i < QK_K; i++) {
            q8[b].qs[i] = (int8_t)((i % 200) - 100);  // -100..99
        }
    }

    // reference dot: sum xf[i] * yf[i] with yf = d * qs = qs (d == 1.0)
    float ref = 0.0f;
    for (int b = 0; b < nb; b++) {
        const float d = q8[b].d;
        for (int i = 0; i < QK_K; i++) {
            ref += xf[b * QK_K + i] * (d * q8[b].qs[i]);
        }
    }

    // vec_dot
    float got = 0.0f;
    ggml_vec_dot_iq3_xxs_q8_K(nb * QK_K, &got, 0, bx.data(), 0, q8.data(), 0, 1);

    // note: vec_dot sums per-block; since we passed n = nb*QK_K and contiguous
    // blocks, it iterates all nb blocks internally. Validate against dequant dot.
    const float abs_err = std::fabs(got - ref);
    const float rel = abs_err / (std::fabs(ref) + 1e-30f);
    printf("    got=%.9f ref=%.9f abs_err=%.3g rel=%.3g\n", got, ref, abs_err, rel);
    CHECK(rel < 1e-4f, "vec_dot_iq3_xxs matches dequantize-then-dot");
}

int main() {
    printf("=== GGUF quant type extension test ===\n");
    test_mxfp4_traits();
    test_mxfp4_dequant();
    test_iq3_xxs_traits();
    test_iq3_xxs_vec_dot();

    if (failures == 0) {
        printf("\nALL TESTS PASSED\n");
        return 0;
    } else {
        printf("\n%d TEST(S) FAILED\n", failures);
        return 1;
    }
}
