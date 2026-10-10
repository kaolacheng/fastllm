# Qwen3.8-27B W4A4 DFlash2：2 × Tesla T10 (sm75) 部署与调参

[Benchmark 索引](../benchmark.md) · [DFlash2 Benchmark](qwen38_27b_dflash2.md) · [启动脚本](../../scripts/qwen27b/README.md)

本文是 [Qwen3.8-27B DFlash2 Benchmark](qwen38_27b_dflash2.md) 在 sm75 设备布局上的补充：NVFP4 W4A4 目标模型配 DFlash2-FP8 草稿，双卡 TP2，覆盖 KV 池预算方法、`targetFree` 判据、草稿数扫描、前缀缓存与视觉的实测结果。测试日期 2026-10-08 至 2026-10-09。

## 结论速览

- 2 × Tesla T10 上，Qwen3.8-27B W4A4 + DFlash2-FP8 以 TP2 可稳定服务 262144 上下文。最终档为 `max_batch 2`、`gpu_mem_ratio 1.01`、页长 16、FP8 KV，KV 池 313872 token。
- 加视觉（`--multimodal`）会缩池，且与 batch 4 不兼容：batch 4 + ratio 1.02 + 视觉实测 KV 池 252400，低于 262144。要同时保住上下文与视觉，用 batch 2 + ratio 1.01。
- 显存能否站住看日志里的 `targetFree`，不要看启动后的 nvidia-smi 空闲。KV 池按用量增长，启动时的空闲明显偏大。
- 草稿数用 checkpoint 原生值 7。扫描 3/5/7/9/11 五档，7 在可预测内容上最好，11 回落，散文类基本不受草稿数影响。
- 前缀缓存默认开：同前缀 1855 token 的请求时延从 2.79 s 降到 0.23 s（11.9×），且草稿模型 KV 一并恢复。
- 带图请求的投机接受长度明显低于纯文本（accept_len 2.17~2.27 对 4.3~6.5），因为草稿模型是纯文本路径。

## 环境与模型

| 项目 | 值 |
| --- | --- |
| GPU | 4 × NVIDIA Tesla T10，每卡 16384 MiB，sm75 |
| 卡间互联 | PHB（经 CPU PCIe root complex），Gen3 x8，无 NVLink |
| 可用显存 | 驱动另占约 454 MiB，单进程上限约 15930 MiB/卡 |
| 本次占用 | cuda2/3 跑生产实例，cuda0/1 跑参数扫描 |
| 目标模型 | Qwen3.8-27B-Coder390-W4A4（NVFP4 / W4A4），hidden 5120，24 Q 头 / 4 KV 头，head_dim 256，vocab 248320 |
| 草稿模型 | 同目录 `DFlash2-FP8`，日志报告 `layers=5` |
| 服务名 / 端口 | `Qwen3.8 27b` / 8092（沿用被替换服务的端口与名称） |
| 并行与量化 | TP2，FP8 KV Cache，页长 16，chunked prefill 2048 |
| 日期 | 2026-10-08 至 2026-10-09 |

## 推荐启动配置

完整参数如下（脚本已收录在 `scripts/qwen27b/`，其中模型路径、ftllm 安装路径需按机器修改）：

~~~bash
CUDA_VISIBLE_DEVICES=2,3 \
FASTLLM_KV_RUNTIME_HEADROOM_MB=8 \
FASTLLM_KV_FINAL_SAFETY_MB=8 \
ftllm server /mnt/models/Qwen3.8-27B-Coder390-W4A4/NVFP4/W4A4 \
  --model_name "Qwen3.8 27b" \
  --tp 2 \
  --max_batch 2 \
  --max_context_length 262144 \
  --chunked_prefill_size 2048 \
  --page_size 16 \
  --prefix_cache true \
  --multimodal \
  --image_embedding_cache 4g \
  --kv_cache_dtype fp8 \
  --gpu_mem_ratio 1.01 \
  --speculative_algorithm dflash \
  --speculative_draft_model_path /mnt/models/Qwen3.8-27B-Coder390-W4A4/NVFP4/W4A4/DFlash2-FP8 \
  --enable_thinking true \
  --host 0.0.0.0 --port 8092
~~~

启动与停止：

~~~bash
bash scripts/qwen27b/start_dflash2.sh     # 同机已有其它 ftllm 实例时加 ALLOW_MULTI=1
bash scripts/qwen27b/stop_dflash2.sh      # 只按 --port 8092 精确匹配, 不影响其它实例
~~~

草稿数默认不传参数，使用 checkpoint 原生值 7；需要覆盖时用 `--draft_tokens N`（只接受正整数，见"踩坑与限制"）。`--multimodal` 会先加载视觉权重并预热工作区，再分配 KV cache，所以 KV 池会自动缩水，必须看日志确认仍够用。

## KV 池预算与 targetFree 判据

`gpu_mem_ratio` 的作用是把显存校准中重复计入的 runtime reserve 让给 KV 池（每 +0.01 约 +9000 token，同时多吃约 145 MB/卡）。校准结束时会打印每卡的 `freeAfterWarmup` 与 `targetFree`：

~~~text
GPU 0: freeAfterWarmup=1.54 GB, targetFree=0.50 GB, localKVPerPage=0.26 MB, delayedPagedReserve=0.02 MB/page, pageLimit=19617.
~~~

判据：**`targetFree` 两卡都必须为正，且按经验不小于 0.5 GB**。它是"扣掉按最大 batch 预留的 runtime cache 后还剩多少"，为负即代表这次校准已经超额分配，实际负载下会崩。启动后的 nvidia-smi 空闲不能替代它——KV 池是按用量逐步分配的，刚启动时看着空很多。

### 开视觉、batch 2（最终档，cuda0/1）

| gpu_mem_ratio | KV 池 | targetFree rank0 / rank1 | 2 并发 × 8k 预填充 | 压后余量 rank0 / rank1 |
| ---: | ---: | --- | --- | --- |
| 1.00 | 302944 | +0.67 / +0.33 GB | 2/2 全 200 | 1018 / 1252 MiB |
| **1.01** | **313872** | **+0.50 / +0.16 GB** | **2/2 全 200** | **322 / 666 MiB** |
| 1.02 | 320912 | +0.34 / −0.01 GB | 2/2 全 200 | 214 / 538 MiB |
| 1.04 | 338912 | 0.00 / −0.34 GB | 0/2，OOM | 12 / 334 MiB |
| 1.06 | 360736 | −0.33 / −0.68 GB | 0/2，OOM | 22 / 50 MiB |

1.02 是擦边档（rank1 的 `targetFree` 已是 −0.01），压后余量也降到 214 MiB；1.01 是两卡 `targetFree` 仍非负的最后一档，因此作为最终配置。

### 关视觉、batch 4（cuda2/3）

| gpu_mem_ratio | KV 池 | targetFree rank0 / rank1 | 4 并发 × 8k 预填充 | 压后余量 rank0 / rank1 |
| ---: | ---: | --- | --- | --- |
| 1.00 | 318192 | 未记录 | 4/4 全 200 | 902 / 1262 MiB |
| 1.02 | 336176 | +0.59 / +0.05 GB | 4/4 全 200 | 736 / 1090 MiB |

### 准入测试协议

验证显存是否站得住，必须用**并发填满 max_batch 的长预填充**压过一遍，再看稳态余量。单条长请求压不出峰值：早期在 1.08 与 1.12 上就是用单请求验的，启动后看着还有 814 MB 空闲，真实负载下只剩 170 MB，最终 OOM。

OOM 时的签名（都发生在 prefill 激活分配上）：

~~~text
Error: CUDA error when allocating 20 MB memory on device 0! gpuFree: 9 MB / 15930 MB.
FastLLM fatal CUDA allocation error: Error: cuda malloc failed in Data::MallocSpace.
  requestBytes = 20971520, dataType = float16, dims = [1, 2048, 5120].
~~~

1.06 档拿到的是更大的 `requestBytes = 35651584`（`dims = [1, 2048, 8704]`），早期更高档位还见过 `cudaErrorMemoryAllocation at fastllm-cuda.cu:5404`。

### 两个余量环境变量的默认值

脚本里设的 8 MB 是对校准默认值的显式覆盖：

| 环境变量 | 校准默认值 |
| --- | --- |
| `FASTLLM_KV_RUNTIME_HEADROOM_MB` | `min(max(512 MB, total/100), 2 GB)`，再被 `available/4` 兜底；设了覆盖值时取 `min(覆盖值, available/4)` |
| `FASTLLM_KV_FINAL_SAFETY_MB` | `min(max(128 MB, total/200), 512 MB)` |

多卡部署里这两项是不随卡数扩展的固定预留，收紧它们可以把显存让给 KV 池。

## 接受率与步率的正确口径

早期版本的 `pos_accept_rate` 是进程累计计数，永不归零，因此从日志反算出的接受长度是假的：同一服务打印 3.30~4.19，而按窗口统计的真值是 6.19~6.45。现改为"窗口增量 + 指数移动平均"（α = 0.3，窗口 64 次验证）：

~~~text
[Qwen3.5 DFlash2] pos_accept_rate(EMA)=[64.80%, 28.84%, 16.30%, 7.52%, 4.83%, 3.48%, 1.07%] accept_len=2.27 tokens/step (window=64 validations, total=256 validations).
~~~

口径换算：

- `accept_len = 1 + Σ pos_accept_rate`，即每步平均产出 token 数。
- `steps/s = 客户端可见吞吐 ÷ accept_len`。

注意它是**混合窗口值**：一个进程里混跑多类任务时，会被接受率最低的那类拉低。上面那条 2.27 来自带图请求，不能与纯文本长生成（4.3~6.5）直接比较；要得到每类任务各自的口径，需要每档只跑一类并跑够窗口长度。

## 草稿数扫描

条件：`temperature 0.5`、`max_context_length 5000`、`max_tokens 512`、每类任务 3 遍取中位；三类任务分别为"从 0 数到 1000"（高可预测）、写 Triton GEMV 内核（中）、写散文（低）。

| 草稿数 | 数数（高） | GEMV（中） | 散文（低） | accept_len |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 99.6 | 82.4 | 40.0 | 1.55~1.62 |
| 5 | 84.9 | 71.2 | 43.3 | 1.58~1.60 |
| **7（原生）** | **112.2** | **78.4** | 39.2 | **1.71** |
| 9 | 112.2（三次原始值 112 / 112 / 141） | 待实测 | 待实测 | 待实测 |
| 11 | 94.4 | 77.6 | 39.4 | 1.69 |

单位均为客户端可见 token/s。结论：**默认的 7 最优**；11 明显回落，因为位置 8 之后的接受率只剩约 1%，多出的草稿只增加延迟；3 与 5 都不如 7；9 在数数上打平但未超过。散文类在 39~43 之间基本不随草稿数变化——低可预测内容里大部分草稿被拒，多猜的收益被延迟抵消。

第一轮扫描中，草稿数 7 的实例启动成功后静默退出（已确认校准正常、KV 池 313872、`targetFree` +0.50/+0.16，无 OOM 报错），原因未定位；第二轮同配置运行正常，因此不作为结论。

## 前缀缓存

`--prefix_cache true` 为默认。同前缀 1855 token 的对照实测：

| 请求 | 时延 | prompt token |
| --- | ---: | ---: |
| 冷启动 | 2.79 s | 1855 |
| 同前缀复用 | 0.23 s | 1855 |

即 11.9× 提升。日志三行对应播种与复用：

~~~text
[Qwen3.5 DFlash2] long prefill cache seeded: tokens=1855, chunk=256.
[Qwen3.5 DFlash2] prefix cache restored: tokens=1792, draft_kv_tokens=1792.
[Qwen3.5 MTP] prefix cache hit: tokens=1792.
~~~

`draft_kv_tokens=1792` 表示草稿模型 KV 一并恢复，复用请求不需要重新为草稿热身。

代价在冷启动路径：开启后冷 prefill 会按 `chunk=256` 播种缓存（上面第一行），本次 1855 token 冷启动约 665 token/s，低于关闭时的水平。前缀缓存偏向"多轮或多请求共享长前缀"的用法；若流量全是互不相同的短请求，它会净亏一点预填充速度。

同机另一生产档（merkyor-w4a16 模型，非本篇配置）记录过 298.7K 前缀复用 8.4 秒（35,711 token/s），仅作量级参考。

## 视觉

视觉用 `--multimodal` 开启，配合 `--image_embedding_cache 4g`（CPU 侧 embedding 缓存，按需分配）。`--mmproj` 只支持 Qwen3.5 家族的 GGUF 模型，safetensors 部署不用它；`--vision_device` 不指定时视觉塔跟随 TP 设备。

工作区在 KV cache 之前预热，因此池子会缩水：

~~~text
[Vision] Multimodal warmup before KV cache: cuda:0, heads=8, max patches=2048, fixed workspace=192.00 MiB.
[Vision] Multimodal workspace ready: cuda:0, peak=45.16 MiB, live=0.00 MiB; remaining memory is available for KV cache.
~~~

固定工作区 192.00 MiB/卡，实测峰值 45.16 / 45.10 MiB——**预留量远大于实际峰值**。但同档位下池子仍明显变小：关视觉时 batch 4 / ratio 1.02 为 336176（cuda2/3），开视觉后 batch 4 / ratio 1.02 实测 252400（cuda0/1），已低于 262144，该组合不可用。两组不在同一卡对，作方向性参考；实际落点是 batch 2 + ratio 1.01（见上文档位表）。

单张 PNG 图片的请求实测：

| 项目 | 值 |
| --- | --- |
| 视觉 feature token | 400 |
| HTTP / 时延 | 200 / 14.5 s |
| prompt / completion | 461 / 639 token |
| embedding 缓存 | 未命中后写入 8192000 / 4294967296 字节 |
| accept_len（EMA） | 2.27、2.17 |

带图请求的接受长度显著低于纯文本（4.3~6.5），草稿模型为纯文本路径，预测不了对图像内容的描述。视觉 + 投机解码在 sm75 上的其他组合（更大 batch、更大 ratio）未验证。

## 代码改动清单

按五组列出本分支相对上游的改动，commit 号为短哈希。

**一、DFlash2 草稿路径**

- `bcca7786`：`dflash.fc` 与 attention q/k/v 加入张量并行（行并行 + output-gather，关闭两级融合保证头边界对齐）；`RunDFlashTpLinear` 统一分片 GEMM 入口；TP 预留修正 fake 视图重复计费；草稿残差改 FP32 累加（BF16 会吞掉 1e-5 级注意力贡献）；`lm_head` 逐行缩放覆盖 NVFP4 各布局。`fc` 权重必须按输出维行并行——按输入维列并行会被框架的 `input.IsTensorParallelSharded()` 要求挡住，退回兜底路径把整块约 250 MB 搬回 root，实测约 2.5 份（618.8 MB）显存不对齐且易 OOM。
- `dd900d07`：逐行激活缩放保护 FP16 GEMM 溢出。NVFP4 GEMM 为 FP16 进出，草稿 GEMM 输出峰值可超 65504 产生 inf，污染 RMSNorm 并让 selector 越界；按行取 absmax 推出 2 的幂 scale，scale 全程留在设备上不破坏 CUDA graph 捕获。

**二、接受率统计口径**

- `bcca7786`：MTP / DFlash2 接受率日志改为 EMA 窗口增量，替换原先的进程累计计数（即上一节讨论的假值问题）。

**三、KV 显存预算**

- `80005d44`：新增 `include/utils/cuda_cache_budget.h`，`FASTLLM_KV_RUNTIME_HEADROOM_MB` 与 `FASTLLM_KV_FINAL_SAFETY_MB` 可覆盖校准默认值，供多卡部署回收不随卡数扩展的固定预留。

**四、多模态前缀缓存与视觉显存**

- `bcca7786`：按图片内容哈希键比对快照，无键输入（如视频）拒绝记录与复用；恢复边界必须越过全部视觉 token，文本尾巴走续写路径并按全序列 M-RoPE 切位置；`thread_local` 钩子在分块边界（页对齐 + 状态一致）安全记录快照。视觉侧：工作区在 KV 之前预热、激活 arena 按 chunk 尺寸收缩、`FASTLLM_QWEN35_MM_WORKSPACE_MARGIN_MB` 可调余量、`FASTLLM_QWEN35_SKIP_VISION` 供纯文本部署跳过视觉塔权重。新增 `include/utils/qwen35_mm_record_hook.h`。

**五、诊断探针清理**

- `dfe6e661`：移除 MoE 专家缓存与 NVFP4 规划路径的一次性 `[DBG-*]` / `[EP-DBG]` 打印、Qwen3.5 前缀缓存与结构探针及其环境开关、basellm 的探针打印与计数变量、`model.cpp` 的 deviceMap 打印，共删除 409 行；保留 staged fill 信息、错误上报与内存检查堆栈诊断，以及 `pos_accept_rate` 等运行信息日志。

分支还包含与本篇设备布局无关的合并与算子工作：`13c9b9ce`（合并上游 82 个提交）、`89b36ca9`（NVFP4 packed E4M3 scale 布局）、`dcc65774`（GGUF/IQ/MXFP4 算子验证测试）、`a0e51574`、`ebd29791`、`291974d8`、`6135be93`、`77067c59`。

## 踩坑与限制

- **不要用启动后的空闲显存判断 ratio**。KV 池按用量分配，1.08 档刚启动看着还有 814 MB 余量，真实负载下只剩 170 MB。必须压过一遍再看稳态余量。
- **以 `targetFree` 为准**：两卡都要为正、经验上不小于 0.5 GB。负数代表本次校准已超额分配，实测 1.04 与 1.06 在 2 并发 × 8k 下 0/2 全挂。
- **准入测试必须并发填满 `max_batch`**。单条长请求压不出 prefill 激活峰值。
- **`--draft_tokens` 只接受正整数**。默认 `-1` 表示用 checkpoint 原生值，此时必须整个省略该参数；传 `-1` 会让 argparse 直接报错退出，服务根本起不来。
- **不要在服务运行时覆盖被映射的 `.so`**，会让进程静默退出（日志停在一条正常请求之后，无任何报错）。要更新引擎文件，先写临时文件再 `mv` 原子替换，运行中的进程继续持有旧 inode。
- **`nohup` 不防 SIGTERM**。启动包装脚本被中断时，同进程组的服务会一起收到 SIGTERM，日志表现为正常的 graceful shutdown。需要真正的后台常驻用 `setsid nohup`。
- **`--enable_thinking true` 会吃掉整个输出预算**。`max_tokens` 偏小时返回 HTTP 200 但 `content` 为空（内容全在思考里），容易误判成服务故障。
- **视觉与 batch 4 在 2 × T10 上不可行**，需要降到 batch 2；同时视觉会缩小 KV 池，开启后必须重新确认 `KV Cache Token limit` 仍不小于 `max_context_length`。
- **边界**：全部数据来自单机 4 × T10（sm75 / 16 GB / PHB、无 NVLink），`temperature 0.5` 与 `1.0` 混用，单次结果噪声较大（同一提示词在不同轮次出现过明显散布）。换硬件、换模型或换量化格式都需要重新校准，不要直接外推。

## 复现与回归检查清单

1. 用上文推荐命令启动，确认进程起来且 `/v1/models` 返回 200。
2. 检查日志四行：
   - `[Qwen3.5 DFlash2] enabled: layers=5, drafts_per_step=7, ...`
   - `[Vision] Multimodal workspace ready: ...`
   - `KV Cache Token limit: 313872 tokens (pageLen=16).`（同次启动的 `pageLimit` 为 19617，二者一致）
   - 两卡 `freeAfterWarmup=... targetFree=...`
3. 通过标准：KV 池 ≥ 262144；两卡 `targetFree` 为正（期望约 +0.50 / +0.16 GB）；台账余量按 0.26 MB/页对照。
4. 准入测试：并发 `max_batch` 条 8k 预填充，全部 HTTP 200，压后余量 rank0 不低于约 300 MiB；任何一条 000 或日志出现 `cuda malloc failed` 即未通过。
5. 视觉回归：发一张图片，确认 `[Vision] ... after encode: ... feature_tokens=` 与 `Image embedding cache` 行出现，并返回 200。
6. 前缀缓存回归：同一长前缀连发两次，第二次应出现 `prefix cache restored` 与 `prefix cache hit`，时延数量级下降。
7. 接受率口径：确认日志为 `pos_accept_rate(EMA)=[...] accept_len=... (window=... total=...)` 形式；按任务类型单独跑才做横向比较。

---

# 附录：生产切到 mky W4A16、prefill 归因与 INT8 证伪（2026-10-09 / 10-10 实测）

以下内容为 10-09 至 10-10 的追加实测，硬件与上文本篇相同（4 × T10，生产跑 cuda2/3，实验跑 cuda0/1）。

## 生产模型切回 mky W4A16

生产默认模型从 `Qwen3.8-27B-Coder390-W4A4/NVFP4/W4A4` 换成 `merkyor-w4a16/NVFP4/W4A16`（脚本 `MODEL` 默认值，草稿仍取 `$MODEL/DFlash2-FP8`，mky 目录内自带该草稿）。切换后按同一配置（batch 2、ratio 1.01、视觉开、前缀缓存开、余量 8 MB）复测：

| 项 | 旧 W4A4（NVFP4） | mky W4A16 |
|---|---:|---:|
| KV 池 | 313872 | **301136** |
| 2 并发 × 8k 准入 | 2/2 通过 | 2/2 通过 |
| 压后余量（rank0/rank1） | 322 / 666 MiB | **496 / 944 MiB** |

池子略小但仍比 `max_context_length` 262144 富余约 15%，且压后余量更宽。注意 `targetFree` 在 GPU1 上读到 +0.16 GB（低于 0.5 GB 的严格线），但准入测试通过且压后留有近 500 MiB——这与上文"batch 2 + 视觉时该判据偏保守"的结论一致。

两个模型的采样默认值**逐字相同**（日志原话 `default generation config: {'repetition_penalty': 1.0, 'top_p': 0.95, 'top_k': 20, 'temperature': 1.0}`），因为它来自模型目录里的 `generation_config.json`，两份文件内容一致。所以要排查"循环/重复"类退化时，不要先怀疑这层——两模型的差异在量化路径（W4A16 只压权重，W4A4/NVFP4 连激活一起压到 4 bit）。另外 `repetition_penalty` **不在**模型配置里，因此保持 `llm.py` 内置的 1.0，等价于完全不做重复惩罚。

## prefill 实测与拟合（mky W4A16）

条件：8093 实例、`PREFIX_CACHE=false` 保证冷 prefill、提示词唯一随机内容、`max_tokens` 极小使时间几乎全为 prefill。

| 前缀 L | prompt_tokens | 耗时 | t/s |
|---:|---:|---:|---:|
| 8K | 8254 | 7.740 / 7.679 s | 1066.4 / 1074.8 |
| 64K | 65758 | 75.035 / 75.161 s | 876.4 / 874.9 |
| 250K | 250144 | 458.545 s | 545.5 |

拟合（比上文的旧式 `1000/(0.94ms + 8.9e-6·L)` 更准，旧式基于 W4A4/旧配置）：

```
累计平均   T/L  = 0.903 ms + 3.72e-6 ms · L
瞬时速率   1/s(x) = 0.903 ms + 7.44e-6 ms · x      (恰为累计式的两倍)
```

逐块交叉验证误差 < 1%：x=1024 模型 1098 vs 实测 1089（+0.8%），x=64512 模型 723 vs 实测 729（−0.9%）；用逐块速度重建总时长 74.6 s vs 实测 75.04 s（−0.5%）。

上下文长度相关项占总时间：8K **3.2%** / 64K **24.5%** / 250K **50.8%**。

## prefill 的 FLOPs 归因：FP16 侧已无余量

双卡 FP16 纸面峰值约 182 TFLOPS（56 SM × 1.59 GHz × 2），但**实测 cuBLAS fp16 在同形状只有 73.6~78.8 TFLOPS/卡**——这才是可比的基准。把 250K 那次 458.5 s 按 FLOPs 拆开：

| 项 | FLOPs | 耗时 | 实测算力 |
|---|---:|---:|---:|
| 线性（权重 GEMM） | 1.35e16 | 225.9 s | 59.8 TFLOPS |
| 上下文（16 层全注意力，∝ L²） | 1.23e16 | 232.5 s | 52.9 TFLOPS |
| 合计 | — | **458.4 s** | 实测 458.5 s |

两项都贴在 cuBLAS 的实际能力上（生产线性项 59.8 vs 纯 cuBLAS 73.6~78.8，约 80%，差额是 W4A16 解包成本），合计与实测差 0.1 s。⇒ **FP16 侧几乎没有余量**；注意力项是 O(L²) 的真实计算，这正是"decode 从 1k 到 200k 只慢约 20%，而 prefill 慢 70~80%"的根本原因：decode 每步只多读一遍 KV，prefill 则是实打实的二次增长。

### 为什么自带 profiler 量不到 prefill 分段

`FASTLLM_PRINT_PROFILE` 只统计经过 `Executor::Run` 的算子（`src/fastllm.cpp:370`，累加在 `src/executor.cpp:497-499`；且必须同时开 `FASTLLM_CUDA_SYNC=1`，否则只量到 kernel launch 时间）。但 `src/models/qwen3_5.cpp` 里 `executor->Run(...)` 出现 **0 次**——22 处算子全走 `runner.Run`，`Qwen3CudaDirectRunner::Run` 直接调 `device->Run`，**绕过了 Executor**（`include/models/qwen3_cuda_common.h:62-95`）。`FASTLLM_PROFILE*` 则只覆盖 CPU/NUMA MoE（`cpudevice.cpp:3347`、`numasdevice.cpp:8420`、`model.cpp:773`）。因此模型侧只有循环级墙钟，拿不到算子分解；本机也未安装 nsys/ncu/nvprof。

另：`FASTLLM_PAGED_CUBLAS_LINEAR_KV` 那条"直读 KV、免 gather"的快路径对 prefill 不适用，它被三重门槛挡住（`native.cu:495-509 / 646-656`：需 `qoLen<=pageLen`、需 fp16 KV、4-KV-head 时硬编码只在 arch 70 开）；`--fast_prefill` 是 DeepSeek-V4.1 专用。

## INT8 / INT4 张量核调研与证伪

结论先行：**Turing 的 INT8/INT4 指令能力是真的（2× / 4×），但在本卡上不足以替换现有 W4A16/fp16 生产线**，除非先解决"让一个操作数常驻 L2"。以下是完整证据链，记录在此以免重复尝试。

### 指令级能力（`-arch=sm_75` 实测编译 + 运行通过）

| 指令 | 实测 | 相对 FP16 |
|---|---:|---:|
| FP16 `mma.m16n8k8` | 91.1 TFLOPS | 1.00×（恰等于 56 SM × 1024 FLOP/clk × 1.59 GHz 纸面峰值） |
| INT8 `mma.m8n8k16` | **182.2 TOPS** | **2.00×** |
| INT4 `mma.m8n8k32` | **364.5 TOPS** | **4.00×** |

裸 mma 是寄存器常驻、零访存的理想值（`m16n8k16` / `m16n8k32` 的 s8 变体需要 sm_80+，本卡不可用；`m8n8k16.s8` 与 `m8n8k32.s4` 在 sm75 上均可编译并正确执行，INT4 传说不假）。

### cuBLAS 的 INT8 路径在 sm75 不可用

同形状（m=2048，生产真实值）实测：

| 形状 (m,n,k) | cuBLAS FP16 | cuBLAS INT8 | 倍率 |
|---|---:|---:|---:|
| 2048, 34816, 5120 | 76.0 TFLOPS | 17.9 TOPS | 0.24× |
| 2048, 14336, 17408 | 77.8 | 17.2 | 0.22× |
| 2048, 17408, 5120 (tp2 每卡) | 73.6 | 18.5 | 0.25× |
| 2048, 7168, 17408 (tp2 每卡) | 76.8 | 17.5 | 0.23× |

`CUBLAS_GEMM_DEFAULT_TENSOR_OP` 被忽略（与 `CUBLAS_GEMM_DEFAULT` 同数），实际走 SIMT/dp4a（约 200 ops/clk/SM）。⇒ **"直接调 cuBLAS / CUTLASS int8"这条路在本卡上是死的，必须手写 IMMA。**

### 手写 IMMA 真实 GEMM：从 0.40× 到 0.88×，仍未过线

同一进程内与 cuBLAS fp16 比 wall time（TOPS 与 TFLOPS 不能直接比大小，只能比时间）：

| 形状 (m,n,k) | v1 朴素分块 | v3（转置布局+16B 装载+双缓冲） | cuBLAS fp16 |
|---|---:|---:|---:|
| 2048, 17408, 5120 | 32.4 TOPS / 11.28 ms (0.46×) | 最好 **0.845~0.88×** | 71.0 / 5.14 ms |
| 2048, 7168, 17408 | 36.2 / 14.10 ms (0.46×) | 同上区间 | 78.8 / 6.49 ms |
| 2048, 34816, 5120 | 32.1 / 22.78 ms (0.43×) | 同上区间 | 75.0 / 9.74 ms |
| 2048, 14336, 17408 | 31.3 / 32.64 ms (0.40×) | 同上区间 | 77.9 / 13.12 ms |

v1 只吃到裸 mma 峰值（182.2 TOPS）的 20%，且 `BM=128 BN=64` 落在朴素流量下限的 0.98×（即它对自己的分块已接近最优，问题是**复用不足**）。v3 换用 k-quad-major 转置布局、16 字节向量装载与手工双缓冲后提升到 0.845~0.88×，**仍未过 1.0×**。

根因已定量定位到 **DRAM 访问的连续字节数**，而不是 mma 发射或 shared 带宽：

- `mma / LDS / barrier` 结构本身无损失：模式测试打到 157~160 TOPS，是裸峰值的 87~92%。
- 顺序读实测 376~386 GB/s；而按行面板步进读时，每行 32~64 B 只有 **195~212 GB/s**（每行 256 B 才回到 365.9 GB/s）。
- 把权重转置成 `XT[k/4][r]` 后达到 **383 GB/s = 实测 DRAM 峰值**，即贴死在"无复用流量模型 `mnk(1/BM + 1/BN)`"的地板上。
- 对照：**cuBLAS 有约 2.4× 的 L2 复用**（同形状 5.14 ms 要求其 DRAM 流量 ≲1.82 GB），而手写 kernel 是 1.0×；15 个配置（BK 32/64、六种 tile、1~7 CTA/SM、三种光栅化）**全部等于零复用模型**。

⇒ 结论：不建议用从零手写的 int8 路线替换现有生产线。INT8 的 2× 指令优势真实存在，但**只有在操作数能按容量常驻 L2 时才兑现**（k 维切分 + 廉价归约，或持久 kernel 固定 m-row 走 n）。若将来重开此方向，验收标准仍是"与同形状 cuBLAS fp16 的 wall time 之比 > 1.0×"，而不是裸 TOPS。

顺带澄清一个易踩的坑：打包字节序的两种候选（`4*gq+j` 连续 / `2*gq+(j&1)+8*(j>>1)` 分两组）在 A、B 使用**同一置换**时都正确（点积对求和顺序不变，14/14 误差为 0）。所以 W4A8 类实现跑得慢**不是 fragment 布局 bug**；但若权重是离线预打包、kernel 又在运行时按另一种约定装载，就会算错。

## 运维级坑（本轮新增）

- **客户端断开不会中止正在跑的 prefill**。日志出现 `is_disconnected!!!` / `Abort request` 之后仍有 `[Decode] pending=1`，长请求会继续算完并占着引擎，后续请求排队。实测：一个 250K 请求被中断后仍跑到 43%，导致紧随其后的 128K 实测 493 s（≈ 该 250K 剩余 57% 的 ~261 s + 干净 128K 的 ~183 s）——这条 128K 数据被判为污染并作废。**取消长生成并不省算力**，长请求建议加"断开即中止"。
- **同机基准必须每 rep 交错测量**。cuBLAS fp16 自身跨轮漂移达 4.65 ↔ 6.78 ms（74 ↔ 54 TFLOPS），受 cuda2/3 上生产服务的负载影响；单次比值是噪声。
- **第一个测点是冷启动 boost 伪影**：同形状首测 53 TFLOPS，热机后 76。任何吞吐测量都要丢弃首个点。
- **`/tmp` 是 tmpfs**，机器重启即清空（本次意外断电导致全部实验日志丢失）。长实验的产物要写到持久目录（如 `~/workspace/reports/`）。

## INT8 探索的最终结论（v4 / v5 / fp16 对照 / Triton）

### 硬件事实更正（重要）
- **T10 实测 `l2CacheSize = 4 MB`**（此前口算推演按 6 MB，作废），`sharedMemPerBlockOptin = 64 KB`。任何"几百 KB 的 slab 常驻 shared、或 6 MB 活跃窗口"的设计在 sm75 上都不成立——这直接解释了下面 v4 为什么失败。
- `m16n8k16.f16` 在 sm_75 被 ptxas **硬拒**（需 sm_80+），fp16 只能用 `m16n8k8`（0.5 条/周期/SM，而 IMMA 是 1 条/周期/SM）⇒ fp16 的计算地板本身也是 int8 的 2×。

### v4：B 驻留读取顺序重设计 —— 0.853~0.885×，复用一次都没发生
设计：CTA 认领 `(BN × BK)` 的 B slab 并在内部遍历全部 m，目标是让大操作数 B 从 DRAM 只读一次、其余靠 L2 取用。

| 形状 (m,n,k) | v4 TOPS | v4 ms | v3 ms | cuBLAS fp16 ms | v4/cuBLAS |
|---|---:|---:|---:|---:|---:|
| 2048,17408,5120 | 65.3 | 5.590 | 5.59 | 4.769 | 0.853 |
| 2048,7168,17408 | 69.0 | 7.409 | 7.29 | 6.553 | 0.885 |
| 2048,34816,5120 | 66.6 | 10.967 | 11.05 | 9.611 | 0.876 |
| 2048,14336,17408 | 68.9 | 14.843 | 14.85 | 13.129 | 0.885 |

- **反算等效流量 2.14 GB，与零复用模型 `mnk(1/BM+1/BN) = 2.139 GB` 逐位吻合**；"B 常驻"模型应为 0.80 GB（差 2.7×）。扫 m（256/512/1024/2048）反算 0.267/0.535/1.070/2.139 GB，逐点落在零复用模型上，**等效带宽恒为 384~396 GB/s = DRAM 峰值**。
- 结构天花板：把全部全局读取换成常量后同 kernel 只需 3.100 ms（117.8 TOPS）⇒ mma/LDS/barrier 不是瓶颈，耗时完全由 DRAM 决定。
- 两种光栅化都零复用；m 最快的原因是波内 B 工作集 3.5 × 1.31 MB = **4.6 MB > 4 MB L2**（只超 15%，但就是装不下）。
- **k 切分陷阱（实测）**：S=1 5.59 ms → S=2 最好 23.33 ms（0.199×，反算 8.93 GB）→ S=4 24.26 ms（9.29 GB）。每多一路切分就多一整轮 `m·n·4 B` 的读改写。最终版改为累加器常驻寄存器、整段 k 一次算完、C 只写一次、**完全不切 k**。
- 12 个配置 × 3 种 S 在 256³ 上全部 0 mismatch；寄存器 96~172，0 spill。

### v5：持久 CTA + `grid.sync()` —— 反而更慢
56 个持久 CTA + cooperative groups，每个 k-slab 前强制整波对齐（理论上让该 slab 的 B 被整波共享）。正确性全过，但 **6.506 ms，比 v4 更慢**：每 CTA 20 tile × 40 slab × 2 = 1600 次 grid barrier，开销大于复用收益。

### 分离变量的 fp16 对照：换 mma 类型精确复现"流量翻倍"
同一份代码只换 mma 数据类型（全局 `[k/2][r]` uint32、smem 行距、16B 装载、双缓冲、tile 与两种光栅化全部对齐）：

| 形状 (m,n,k) | 手写 F16 ms | F16 TFLOPS | cuBLAS ms | cuBLAS TF | 手写/cuBLAS | 手写 INT8 ms | F16/INT8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2048,17408,5120 | 11.29 | 32.2 | 5.25 | 69.5 | 2.17× | 5.487 | 1.94× |
| 2048,7168,17408 | 14.81 | 34.5 | 7.17 | 71.3 | 2.09× | 7.266 | 1.92× |
| 2048,34816,5120 | 22.37 | 32.6 | 10.68 | 68.4 | 2.12× | 10.837 | 1.96× |
| 2048,14336,17408 | 31.31 | 32.6 | 14.67 | 69.7 | 2.12× | 14.710 | 2.01× |

- 手写 fp16 = **32.0~34.5 TFLOPS**；流量模型预测 `4.278 GB / 386.7 GB/s = 11.06 ms = 32.2 TFLOPS`，实测差 <3%。
- **FP16/INT8 耗时比 = 1.82~2.01（均值 1.90）**，就是"每元素字节翻倍 ⇒ 流量翻倍"的精确复现。
- 等字节交叉验证：int8 用 64×128 tile 的零复用流量（4.278 GB）与 fp16 128×256 完全相同，实测 int8 18.018 ms vs fp16 11.29 ms ⇒ **同字节下 fp16 反而更快**，fp16 没有为 dtype 付额外代价。
- ⇒ **剩余 2.1× 差距 100% 来自 L2 复用，与 mma 类型无关。**

### 两个消融：复用是"时序脆弱"的
- **noload**（全局读取换常量）：59.4~62.2 TFLOPS = 裸 fp16 峰值的 65~68%。结构本身够快，是 DRAM 把它压到 32。
- **nomma**（只留 global→smem→LDS，去掉 mma）：8.96 / 11.72 / 17.55 / 23.49 ms ⇒ 等效 **477~511 GB/s，比 386.7 GB/s 硬件上限高 24~32%**，只能由约 20~28% 的 L2 复用解释（m-fastest 光栅化下相邻 n-CTA 共享同一 A 面板）；而 w2x4 布局是 378~401 GB/s ≈ 贴死 DRAM、零复用。
- ⇒ **纯装载调度确实能拿到跨 CTA 复用，一旦加入 mma 就失去时间对齐、复用消失**（整核反算带宽 357~378 GB/s < 386.7）。这与 v3/v4 的结论一致，是这条路线真正的死因。

### Triton 对照：fp16 能拿到 2.06× 复用，int8 不行
环境 torch 2.13.0+cu130 / triton 3.7.1（`/home/kaolachen/vllm-env`），`get_device_capability() = (7,5)`；PTX 确认走真张量核（fp16→`mma.m16n8k8`，int8→`mma.m8n8k16`，无 FMA/dp4a 退化）。

| | Triton | 手写 | cuBLAS |
|---|---:|---:|---:|
| fp16（四形状） | 46~50 TFLOPS，**0.70~0.73×** | 32~34.5，0.46× | 68~71（基准） |
| int8（四形状） | 66.6~74.3 TOPS（健康时钟峰 80.6），**0.89~0.90×** | 65~70，0.85~0.89× | — |

- **Triton fp16 拿到了 L2 复用**：反算等效带宽 **723~778 GB/s = 1.87~2.01× DRAM 峰值**（最好一轮 795 GB/s = **2.06×**），物理上不可能无复用。开关是 `GROUP_M`：形状 (17408,5120) 由 G1 的 689 GB/s(1.78×) 到 G4 的 795 GB/s(2.06×)，**快 15.4%**；形状 (14336,17408) G1→G4 **快 19.0%**。⇒ "我们拿不到 L2 复用"**不是**这张卡的普遍现象。
- **但 int8 上手写与 Triton 都没拿到**：Triton int8 反算 eqBW 只有 **391~435 GB/s = 1.01~1.13×**，仍钉在零复用地板（256×128 与手写 128×256 的 `(1/BM+1/BN)` 同为 0.01172，零复用流量同为 2.14 GB）。换 128×128 tile 能把 eqBW 推到 587 GB/s（1.52×），但绝对时间更慢 ⇒ 复用不是 int8 的唯一瓶颈。
- **根因是硬限制：sm75 没有 `cp.async`**。TTGIR 实测 `async_copy=0`、`local_alloc` 单缓冲、smem 不随 `num_stages` 变化；`num_stages=1/2/3`、`tl.range(num_stages=)`、手写 prefetch **三者全部无效** ⇒ K 循环无法 load/mma 重叠。这是 fp16 只到 0.72× 的原因，int8 受同一限制。
- 融合 dequant（将来做 W4A8/W4A16 可直接用）：**per-channel scale 放 epilogue 近免费**（+0.07~2.8%）；per-K-group128 in-loop **+5.1~13.3%**，但在 256×128 tile 上因 fp32 累加器寄存器翻倍爆到 **+421% / 88 spills**。
- satfinite 安全：最坏 `|acc| = K·128·128 = 2.85e8 < 2^31`，实测 max|acc| 1.0e4~2.1e4。

### 最终判定
手写 int8（0.85~0.89×）、手写 fp16（0.46×）、Triton int8（0.89~0.90×）、Triton fp16（0.70~0.73×）——**int8 的三个方向落在同一档，没有一个越过 1.0×**。

**INT8 的 2× 指令优势真实存在（裸 mma 182.2 TOPS），但兑现条件是"操作数驻留 L2 + load/mma 重叠"，而这张卡被三件事同时卡死**：4 MB 的 L2（装不下 int8 波内 B 工作集所需的 4.6 MB）、32K 累加器寄存器上限（tile 不能再大）、以及 sm75 缺失的 `cp.async`（K 循环无法重叠）。cuBLAS fp16 那 2.1~2.35× 的复用不是"神秘优化"，而是把这三件事都吃透了的结果。

⇒ **维持结论：生产继续用 W4A16 + Marlin**（实测 59.8 TFLOPS ≈ 同形状纯 cuBLAS fp16 的 80%）。若将来重开 INT8，验收标准不变：与同形状 cuBLAS fp16 的 wall time 之比 > 1.0×，而不是裸 TOPS。

### 原始产物
`~/workspace/reports/imma_bench/`（`i8gemm.cu`、`i8gemm_v3.cu`、`i8gemm_v4.cu`、`i8gemm_v5.cu`、`f16gemm_v4.cu`、`ldpat.cu` 装载模式带宽标定，以及各自的 `*_notes.md` 与运行日志）、`~/workspace/reports/triton_bench/REPORT.md`。
