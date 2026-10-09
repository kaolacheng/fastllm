#!/usr/bin/env bash
#
# Qwen3.8 27B + DFlash2 草稿 (DFlash2-FP8) FastLLM OpenAI 兼容服务 —— 生产启动脚本
#   路径: /home/kaolachen/workspace/script/qwen27b/start_dflash2.sh
#   停止: ./stop.sh   或  kill $(cat /tmp/ftllm_dflash2.pid)
#
# 定位: 替换原 cuda2/3 上的 merkyor-w4a16 生产服务, 沿用同一端口 8092,
#       对外服务名统一为 "Qwen3.8 27b"。
#
# 硬件: Tesla T10 x4 (sm75, 16GB)。本实例占用 cuda2/3; cuda0/1 留给调试实例。
#
# 实测 (tp2, cuda2/3, max_context 262144, chunked_prefill 2048, 页长 16, DFlash2 草稿):
#   KV 池           336176 token (21011 页 x 16) @4并发+关快照+ratio1.02 —— 够 262144, 余量 28%
#   并发            4 并发真的合批(日志 alive=4), 但只赚 ~1.4x:
#                   N=1 138 tok/s | N=4 合计 186 tok/s (单请求掉到 46 tok/s)
#   利用率          15195 / 14841 MiB (93.4% / 91.2%), 压力后余 736 / 1090 MiB
#   稳定性          4 并发 x 8k 预填充 4/4 全 200, 无 OOM(这是准入测试, 单请求压不出来)
#   runtime cache   858.06 MB (rank0) / 317.72 MB (rank1) @4并发+关快照
#   decode          181 tok/s; 接受长度 87.5/71.9/68.8/62.5/59.4/59.4/56.3%
#   显存差          rank0-rank1 ~326 MiB (fc 改行并行前是 622 MiB)
#   fc 权重         行并行 (axis0, 输出维 5120), output-gather 收齐;
#                   不可按输入维列并行 —— 框架要求 input.IsTensorParallelSharded(),
#                   而本处输入是每卡复制的完整 [tokens,25600], 会退回兜底路径把整块
#                   250MB 搬回 root (实测 ~2.5 份 = 618.8MB 显存不对齐, 且易 OOM)。
#
set -euo pipefail

MODEL=${MODEL:-/home/kaolachen/workspace/models/merkyor-w4a16/NVFP4/W4A16}
DRAFT=${DRAFT:-$MODEL/DFlash2-FP8}
SERVED_NAME=${SERVED_NAME:-Qwen3.8 27b}   # API 里 model 字段用的正式名称
PORT=${PORT:-8092}
MAX_BATCH=${MAX_BATCH:-2}
CHUNK=${CHUNK:-2048}
MAX_CTX=${MAX_CTX:-262144}
TP=${TP:-2}
PAGE_SIZE=${PAGE_SIZE:-16}
GPU_MEM_RATIO=${GPU_MEM_RATIO:-1.01}  # 最终档: 1.01 是 batch2+视觉下 targetFree 两卡都非负的最后档(1.02 是 -0.01 擦边, 1.04 起必崩)
LOG=${LOG:-/tmp/ftllm_dflash2.log}
PIDFILE=${PIDFILE:-/tmp/ftllm_dflash2.pid}
PREFIX_CACHE=${PREFIX_CACHE:-true}   # 默认开: 长前缀复用实测 298.7K -> 8.4s (35,711 t/s) 且答案一致; 短请求上只有开销, 无前缀可复用时略慢
# 草稿数(drafts_per_step): -1=模型默认(实测 7)。开头的 enabled 日志会打印实际值。
DRAFT_TOKENS=${DRAFT_TOKENS:--1}
# 视觉: 原生产脚本 start.sh 的写法就是 --multimodal + --image_embedding_cache 4g,
#   没有 --vision_device(默认 auto, 视觉跟 TP 设备走), 也没有 --mmproj(GGUF 专用)。
#   注意 --multimodal 会"先加载视觉权重并预分配工作区, 再分配 KV cache",
#   所以 KV 池会自动缩水, 需要看日志里的 KV Cache Token limit 确认还够 262144。
MULTIMODAL=${MULTIMODAL:-1}
IMAGE_EMBED_CACHE=${IMAGE_EMBED_CACHE:-4g}
FTLLM_BIN=${FTLLM_BIN:-/home/kaolachen/fastllm-env/bin/ftllm}

# 只用 cuda2/3 (原生产服务的位置)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
# ftllm 运行期需要的 CUDA 运行库
export LD_LIBRARY_PATH=/home/kaolachen/vllm-env/lib/python3.10/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# CUDA Graph: 关 (DFlash 草稿 + 数值探针路径在 graph 捕获期会被跳过, 优先稳定性)
export FASTLLM_CUDA_GRAPH=${FASTLLM_CUDA_GRAPH:-0}
# 数值探针: 默认关 (开启会拖慢 3~5 倍, 仅数值排查时用)
export FASTLLM_DFLASH_PROBE=${FASTLLM_DFLASH_PROBE:-0}
# 批量前缀快照: 默认关。开着时每个并发请求各留一份(草稿位点数)状态快照,
#   batch=4 即 4x8=32 份 = 1270.87 MB/卡; 关掉只留 1 份共享快照 = 317.72 MB/卡,
#   代价是被拒的草稿要走一次 target replay。
#   实测 (ratio 1.05, 每请求 512 token, 4 并发真合批 alive=4):
#     开: KV 312752, N=1 151.2 tok/s, N=4 合计 217.3 tok/s
#     关: KV 365072, N=1 137.6 tok/s, N=4 合计 185.6 tok/s
#   → 开着更快(并发 +17%), 但多吃 953 MB/卡 reserve; 本机选"关", 因为稳定
#     余量比那 17% 更值钱: 开着时 1.00 档 KV 只有 262016 < 262144, 根本起不来,
#     要用 1.05+ 才够, 而 1.05 在 4 并发 x 8k 预填充下只剩 ~300 MB 余量(危险)。
export FASTLLM_DFLASH_BATCH_PREFIX_SNAPSHOTS=${FASTLLM_DFLASH_BATCH_PREFIX_SNAPSHOTS:-0}
# KV 显存校准: 收紧保守余量, 把被重复计入的 runtime reserve 让给 KV 池
# gpu_mem_ratio (每 +0.01 约 +9000 token, 且多吃 ~145 MB/卡)
# !! 测法很关键 !! KV 池是"用多少分配多少", 刚启动时 nvidia-smi 明显偏低
#    (1.08 刚起看着还有 814 MB 余量, 真跑起来只剩 170 MB) —— 必须先用真实负载
#    (4 并发 x 8k 预填充) 压过再看稳态余量, 否则会被假数据骗着往上推。
#    1.12 和 1.08 就是这样在压力下 OOM 死的, 报错都是同一个:
#      "CUDA error when allocating 20 MB ... gpuFree: 9 MB"([1,2048,5120] fp16 激活)
# 已用 4并发x8k预填充 验证过的档位 (关快照):
#   1.00 -> KV 318192, 压力后余 902 / 1262 MiB   稳
#   1.02 -> KV 336176, 压力后余 736 / 1090 MiB   默认, 实测最激进的安全档
#   1.05 -> 推算只剩 ~300 MiB, 未验证, 不要用
export FASTLLM_QWEN35_MM_WORKSPACE_MARGIN_MB=${FASTLLM_QWEN35_MM_WORKSPACE_MARGIN_MB:-128}
export FASTLLM_KV_RUNTIME_HEADROOM_MB=${FASTLLM_KV_RUNTIME_HEADROOM_MB:-8}   # 原生产 start_2gpu_300k.sh 的取值
export FASTLLM_KV_FINAL_SAFETY_MB=${FASTLLM_KV_FINAL_SAFETY_MB:-8}   # 同上; 判定看日志 targetFree: 正数且>=0.5GB 才安全

# ALLOW_MULTI=1 用于同机并存多个实例(各自独立端口/显卡; 如 cuda0/1 的调试实例)
if [ "${ALLOW_MULTI:-0}" != "1" ] && pgrep -x ftllm >/dev/null; then
    echo "已有 ftllm 在运行 (pid: $(pgrep -x ftllm | tr '\n' ' ')), 先 ./stop.sh (或多实例: ALLOW_MULTI=1)"
    exit 1
fi

echo "启动 ftllm DFlash2  model=$MODEL"
echo "  草稿      $DRAFT"
echo "  服务名    $SERVED_NAME"
echo "  port=$PORT  tp=$TP  ctx=$MAX_CTX  chunk=$CHUNK  page=$PAGE_SIZE  gpu=$CUDA_VISIBLE_DEVICES  gpu_mem_ratio=$GPU_MEM_RATIO"
echo "  日志      $LOG"
ln -sf "$LOG" /tmp/ftllm_current.log
cd /tmp
# --draft_tokens 只接受正整数; 默认 -1 时必须整个省略, 否则 argparse 直接报错退出。
DRAFT_ARGS=()
if [ "${DRAFT_TOKENS:--1}" -gt 0 ] 2>/dev/null; then
    DRAFT_ARGS+=(--draft_tokens "$DRAFT_TOKENS")
fi

VISION_ARGS=()
if [ "$MULTIMODAL" = "1" ]; then
    VISION_ARGS+=(--multimodal --image_embedding_cache "$IMAGE_EMBED_CACHE")
fi

nohup $FTLLM_BIN server "$MODEL" \
    --model_name "$SERVED_NAME" \
    --tp "$TP" \
    --max_batch "$MAX_BATCH" \
    --max_context_length "$MAX_CTX" \
    --chunked_prefill_size "$CHUNK" \
    --page_size "$PAGE_SIZE" \
    --prefix_cache "$PREFIX_CACHE" \
    ${DRAFT_ARGS[@]+"${DRAFT_ARGS[@]}"} \
    ${VISION_ARGS[@]+"${VISION_ARGS[@]}"} \
    --kv_cache_dtype fp8 \
    --gpu_mem_ratio "$GPU_MEM_RATIO" \
    --speculative_algorithm dflash \
    --speculative_draft_model_path "$DRAFT" \
    --enable_thinking true \
    --host 0.0.0.0 \
    --port "$PORT" \
    > "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "pid=$(cat $PIDFILE)  (端口 $PORT; 日志有缓冲, 判断就绪请直接发请求)"
