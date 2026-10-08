#pragma once

namespace fastllm {
class Data;
}

// Prepare an already materialized FP8 Linear weight for repeated small-row
// SM70 GEMM calls. This uses the existing in-place layout/FP16-scale conversion.
// The caller must select the weight's CUDA device and ensure serving is idle.
// Returns false for ineligible weights/row counts or a deferred conversion,
// without changing the weight storage.
bool FastllmCudaWarmupFp8E4M3Sm70(fastllm::Data &weight, int rows);

// Marlin-packed FP8 -> FP16 with Marlin's rounded scales; cuBLAS alpha=1.
// Keeps the packed source intact for decode. Uses cudaStreamPerThread.
bool FastllmCudaDequantFp8MarlinForCublas(fastllm::Data &weight, void *destination);

// ── DFlash2 逐行激活缩放 ────────────────────────────────────────────────
// 背景：这台机器上唯一正确的 NVFP4 GEMM 是 FP16 进 / FP16 出的，而 DFlash2
// 草稿的 GEMM 输出峰值会超过 65504（实测 L0 attn.o_proj amax=59904 且已有
// inf），存成 half 就是 inf；inf 进入后面的 RMSNorm 会立刻摊满整个张量，
// 最终让 selector 选出越界候选 id 并中止进程。
// 实测越界值是稀疏离群点（40960 个元素里只有 11~17 个），所以按行（按 token）
// 缩放：只让出现离群点的那一行承担大 scale，其余行按自然量级进入 FP16。
// s_i 取 2 的幂 ⇒ FP32 下乘除精确，正常范围内不丢尾数。
// 全部为新增内核，既有内核一行未改；scale 留在设备上，不做主机 readback，
// 因此不破坏 CUDA graph 捕获。
//
// 逐行 absmax → 2 的幂 scale，并把缩放后的输入转成 FP16 写入 halfOut。
// 只接受 FLOAT32 输入；返回 false（类型不符 / 分配失败）时调用方应退回
// 原来的直接 cast 到 FP16 的路径。
bool FastllmCudaDFlashScaleToHalf(const fastllm::Data &input,
                                  fastllm::Data &halfOut, float headroom);

// 把 GEMM 的 FP16 输出按行乘回 scale，并原地转成 outLike.dataType。
// outLike 只用于取目标 dtype，仅支持 FLOAT16 / BFLOAT16（元素宽度相同，
// 可复用同一块缓冲）。返回 false 时调用方应退回 ToDataType。
bool FastllmCudaDFlashScaleBack(fastllm::Data &halfOutput,
                                const fastllm::Data &outLike);
