# FastLLM（本地双卡适配分支）

[English](README_EN.md) · [本机完整实测](docs/benchmarks/qwen38_27b_dflash2_t10_tp2.md) · [启动脚本](scripts/qwen27b/README.md) · [上游仓库](https://github.com/ztxz16/fastllm)

本分支在上游 [FastLLM](https://github.com/ztxz16/fastllm) 基础上，补完**本地双卡（2 × Tesla T10，sm75，16 GB）**的部署与调参适配。主线是在这块硬件上跑通 **Qwen3.8-27B W4A4/NVFP4 目标模型 + DFlash2-FP8 投机解码**，并给出一套可复现的显存预算与验收方法。

通用引擎介绍、模型清单、完整安装与参数手册不在本文重复：英文版完整保留在 [README_EN.md](README_EN.md)，上游完整中文文档见 git 历史中 `dfe6e661` 之前的 `README.md`，或直接访问[上游仓库](https://github.com/ztxz16/fastllm)。本文只讲本地双卡这条线。

相对上游，本分支共 10 个提交、28 个文件、+2864/−195，其中与本线直接相关的改动：

- DFlash2 草稿路径支持 TP2 分片，并以逐行激活缩放避免 FP16 GEMM 溢出；
- 接受率统计改为 EMA 窗口增量，替换原先会打印假值的进程累计口径；
- KV 显存预算可通过环境变量覆盖，多卡部署可回收不随卡数扩展的固定预留；
- 多模态前缀缓存与视觉显存优化；
- 2 × T10 的完整实测数据与本机启停脚本。

## 硬件与目标

| 项目 | 值 |
| --- | --- |
| GPU | 4 × NVIDIA Tesla T10，每卡 16384 MiB，sm75 |
| 卡间互联 | PHB（经 CPU PCIe root complex），Gen3 x8，无 NVLink |
| 可用显存 | 驱动另占约 454 MiB，单进程上限约 15930 MiB/卡 |
| 本例布局 | `cuda2/3` 跑生产实例，`cuda0/1` 可跑参数扫描 |
| 目标模型 | Qwen3.8-27B-Coder390-W4A4（NVFP4 / W4A4），hidden 5120，24 Q 头 / 4 KV 头，head_dim 256，vocab 248320 |
| 草稿模型 | 同目录 `DFlash2-FP8`，日志报告 `layers=5` |
| 服务名 / 端口 | `Qwen3.8 27b` / 8092 |
| 并行与量化 | TP2，FP8 KV Cache，页长 16，chunked prefill 2048 |

四卡两两之间只有 PHB 链路，4 卡 TP 的聚合收益有限，因此本例固定用 2/3 卡做 TP2，把另外两卡留作并行验证。

## 一键启动

~~~bash
bash scripts/qwen27b/start_dflash2.sh     # 同机已有其它 ftllm 实例时加 ALLOW_MULTI=1
bash scripts/qwen27b/stop_dflash2.sh      # 只按 --port 8092 精确匹配，不影响其它实例
~~~

脚本默认值（完整命令见 [启动脚本说明](scripts/qwen27b/README.md) 与[实测篇](docs/benchmarks/qwen38_27b_dflash2_t10_tp2.md)）：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--tp` | 2 | 固定 2/3 卡 |
| `--max_batch` | 2 | 视觉开启时 batch 4 在 16 GB 卡上不成立 |
| `--max_context_length` | 262144 | KV 池实测 313872 token，满足该窗口 |
| `--chunked_prefill_size` | 2048 | |
| `--page_size` | 16 | 每页 0.26 MB |
| `--kv_cache_dtype` | fp8 | |
| `--gpu_mem_ratio` | 1.01 | 见下一节，1.01 是视觉 + batch 2 下 `targetFree` 仍非负的最后一档 |
| `--prefix_cache` | true | 长前缀复用收益远大于冷路径代价 |
| `--multimodal` | 开 | 配合 `--image_embedding_cache 4g`（CPU 侧缓存） |
| 草稿数 | 不传参 | 使用 checkpoint 原生值 7，覆盖时用 `--draft_tokens N` |
| `FASTLLM_KV_RUNTIME_HEADROOM_MB` | 8 | 校准默认值的显式覆盖 |
| `FASTLLM_KV_FINAL_SAFETY_MB` | 8 | 同上 |

启动后就绪判据是日志中的四行：`[Qwen3.5 DFlash2] enabled: layers=5, drafts_per_step=7, ...`、`[Vision] Multimodal workspace ready: ...`、`KV Cache Token limit: 313872 tokens (pageLen=16).`、以及每卡的 `freeAfterWarmup=... targetFree=...`。

## KV 池预算与 targetFree 判据

`gpu_mem_ratio` 的作用是把显存校准中重复计入的 runtime reserve 让给 KV 池：每 +0.01 约 +9000 token，同时多吃约 145 MB/卡。校准结束时每卡会打印 `freeAfterWarmup` 与 `targetFree`：

~~~text
GPU 0: freeAfterWarmup=1.54 GB, targetFree=0.50 GB, localKVPerPage=0.26 MB, delayedPagedReserve=0.02 MB/page, pageLimit=19617.
~~~

**判据：两卡 `targetFree` 都必须为正，且经验上不小于 0.5 GB。** 它表示"扣掉按最大 batch 预留的 runtime cache 之后还剩多少"，为负即说明本次校准已经超额分配，真实负载下会崩。

开视觉、batch 2（最终档）实测：

| gpu_mem_ratio | KV 池 | targetFree rank0 / rank1 | 2 并发 × 8k 预填充 | 压后余量 rank0 / rank1 |
| ---: | ---: | --- | --- | --- |
| 1.00 | 302944 | +0.67 / +0.33 GB | 2/2 全 200 | 1018 / 1252 MiB |
| **1.01** | **313872** | **+0.50 / +0.16 GB** | **2/2 全 200** | **322 / 666 MiB** |
| 1.02 | 320912 | +0.34 / −0.01 GB | 2/2 全 200 | 214 / 538 MiB |
| 1.04 | 338912 | 0.00 / −0.34 GB | 0/2，OOM | 12 / 334 MiB |
| 1.06 | 360736 | −0.33 / −0.68 GB | 0/2，OOM | 22 / 50 MiB |

1.02 已是擦边档（rank1 的 `targetFree` 为 −0.01，压后余量只剩 214 MiB），1.04 起必崩，因此 1.01 作为最终配置。关视觉、batch 4 的对照档为 1.00 → 318192 token、1.02 → 336176 token，均通过 4 并发 × 8k。

脚本里设的 8 MB 是对校准默认值的覆盖：

| 环境变量 | 校准默认值 |
| --- | --- |
| `FASTLLM_KV_RUNTIME_HEADROOM_MB` | `min(max(512 MB, total/100), 2 GB)`，再被 `available/4` 兜底；设了覆盖值时取 `min(覆盖值, available/4)` |
| `FASTLLM_KV_FINAL_SAFETY_MB` | `min(max(128 MB, total/200), 512 MB)` |

这两项是不随卡数扩展的固定预留，多卡部署里收紧它们可以把显存让给 KV 池。

## 准入测试协议

显存是否站得住，必须用**并发填满 `max_batch` 的长预填充**压过一遍，再看稳态余量。单条长请求压不出峰值：早期在 1.08 与 1.12 档就是用单请求验的，启动后看着还有 814 MB 空闲，真实负载下只剩 170 MB，最终 OOM。本线的两次 OOM 也都源于此。

OOM 签名（都发生在 prefill 激活分配上）：

~~~text
Error: CUDA error when allocating 20 MB memory on device 0! gpuFree: 9 MB / 15930 MB.
FastLLM fatal CUDA allocation error: Error: cuda malloc failed in Data::MallocSpace.
  requestBytes = 20971520, dataType = float16, dims = [1, 2048, 5120].
~~~

更高档位还见过 `requestBytes = 35651584`（`dims = [1, 2048, 8704]`）与 `cudaErrorMemoryAllocation at fastllm-cuda.cu:5404`。

## 接受率与步率的正确口径

早期版本的 `pos_accept_rate` 是进程累计计数、永不归零，从日志反算出的接受长度是假的：同一服务打印 3.30~4.19，而按窗口统计的真值为 6.19~6.45。现改为"窗口增量 + 指数移动平均"（α = 0.3，窗口 64 次验证）：

~~~text
[Qwen3.5 DFlash2] pos_accept_rate(EMA)=[64.80%, 28.84%, 16.30%, 7.52%, 4.83%, 3.48%, 1.07%] accept_len=2.27 tokens/step (window=64 validations, total=256 validations).
~~~

- `accept_len = 1 + Σ pos_accept_rate`，即每步平均产出 token 数；
- `steps/s = 客户端可见吞吐 ÷ accept_len`。

它是**混合窗口值**：一个进程里混跑多类任务时会被接受率最低的那类拉低。上面那条 2.27 来自带图请求，不能与纯文本长生成（4.3~6.5）直接比较；要得到每类任务各自的口径，需每档只跑一类并跑够窗口长度。此外单次 `temperature 1.0` 的重复测量散布明显（同一提示词出现过 3.47 / 5.67 / 6.19 / 6.45），单次结果不足以作结论。

## 草稿数扫描

条件：`temperature 0.5`、`max_context_length 5000`、`max_tokens 512`，每类任务 3 遍取中位。三类任务为"从 0 数到 1000"（高可预测）、写 Triton GEMV 内核（中）、写散文（低）。

| 草稿数 | 数数（高） | GEMV（中） | 散文（低） | accept_len |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 99.6 | 82.4 | 40.0 | 1.55~1.62 |
| 5 | 84.9 | 71.2 | 43.3 | 1.58~1.60 |
| **7（原生）** | **112.2** | **78.4** | 39.2 | **1.71** |
| 9 | 112.2（原始值 112 / 112 / 141） | 待实测 | 待实测 | 待实测 |
| 11 | 94.4 | 77.6 | 39.4 | 1.69 |

单位为客户端可见 token/s。结论：**默认的 7 最优**；11 明显回落（位置 8 之后接受率只剩约 1%，多出的草稿只增加延迟）；3 与 5 都不如 7；9 在数数上打平但未超过。散文在 39~43 之间基本不随草稿数变化——低可预测内容里大部分草稿被拒，多猜的收益被延迟抵消。

## 前缀缓存

`--prefix_cache true` 为默认。同前缀 1855 token 的对照实测：

| 请求 | 时延 | prompt token |
| --- | ---: | ---: |
| 冷启动 | 2.79 s | 1855 |
| 同前缀复用 | 0.23 s | 1855 |

即 11.9× 提升，日志三行对应播种与复用：

~~~text
[Qwen3.5 DFlash2] long prefill cache seeded: tokens=1855, chunk=256.
[Qwen3.5 DFlash2] prefix cache restored: tokens=1792, draft_kv_tokens=1792.
[Qwen3.5 MTP] prefix cache hit: tokens=1792.
~~~

`draft_kv_tokens=1792` 表示草稿模型 KV 一并恢复，复用请求无需重新为草稿热身。代价在冷路径：开启后冷 prefill 按 `chunk=256` 播种缓存，本次 1855 token 冷启动约 665 token/s，低于关闭时的水平。它偏向"多轮或多请求共享长前缀"的用法；若流量全是互不相同的短请求，会净亏一点预填充速度。

## 视觉

视觉用 `--multimodal` 开启，配合 `--image_embedding_cache 4g`（CPU 侧 embedding 缓存，按需分配）。`--mmproj` 只支持 Qwen3.5 家族的 GGUF 模型，safetensors 部署不用它；`--vision_device` 不指定时视觉塔跟随 TP 设备。工作区在 KV cache **之前**预热，因此池子会自动缩水：

~~~text
[Vision] Multimodal warmup before KV cache: cuda:0, heads=8, max patches=2048, fixed workspace=192.00 MiB.
[Vision] Multimodal workspace ready: cuda:0, peak=45.16 MiB, live=0.00 MiB; remaining memory is available for KV cache.
~~~

固定工作区 192.00 MiB/卡，实测峰值仅 45.16 / 45.10 MiB——预留量远大于实际峰值。但同档位下池子仍明显变小：关视觉 batch 4 / ratio 1.02 为 336176（cuda2/3），开视觉 batch 4 / ratio 1.02 实测 252400（cuda0/1），已低于 262144，该组合不可用（两组不在同一卡对，作方向性参考）。

单张 PNG 图片请求实测：

| 项目 | 值 |
| --- | --- |
| 视觉 feature token | 400 |
| HTTP / 时延 | 200 / 14.5 s |
| prompt / completion | 461 / 639 token |
| embedding 缓存 | 未命中后写入 8192000 / 4294967296 字节 |
| accept_len（EMA） | 2.27、2.17 |

带图请求的接受长度显著低于纯文本（4.3~6.5），因为草稿模型是纯文本路径，预测不了对图像内容的描述。视觉 + 投机解码在 sm75 上的其它组合（更大 batch、更大 ratio）未验证。

## 踩坑与限制

- **不要用启动后的空闲显存判断 ratio**。KV 池按用量逐步分配，1.08 档刚启动看着还有 814 MB 余量，真实负载下只剩 170 MB。必须压过一遍再看稳态余量。
- **以 `targetFree` 为准**。两卡都要为正、经验上不小于 0.5 GB；负数代表本次校准已超额分配，实测 1.04 与 1.06 在 2 并发 × 8k 下 0/2 全挂。
- **准入测试必须并发填满 `max_batch`**。单条长请求压不出 prefill 激活峰值。
- **`--draft_tokens` 只接受正整数**。默认 `-1` 表示用 checkpoint 原生值，此时必须整个省略该参数；传 `-1` 会让 argparse 直接报错退出，服务根本起不来。
- **不要在服务运行时覆盖被映射的 `.so`**，会让进程静默退出（日志停在一条正常请求之后，无任何报错）。更新引擎文件要先写临时文件再 `mv` 原子替换，运行中的进程继续持有旧 inode。
- **`nohup` 不防 SIGTERM**。启动包装脚本被中断时，同进程组的服务会一起收到 SIGTERM，日志表现为正常的 graceful shutdown。需要真正后台常驻用 `setsid nohup`。
- **`--enable_thinking true` 会吃掉整个输出预算**。`max_tokens` 偏小时返回 HTTP 200 但 `content` 为空（内容全在思考里），容易误判成服务故障。
- **视觉与 batch 4 在 2 × T10 上不可行**，需降到 batch 2；开视觉后必须重新确认 `KV Cache Token limit` 仍不小于 `max_context_length`。
- **残留实例会互相干扰**：同卡上并存两个实例时，显存校准会互相抢显存并污染 `targetFree`，参数扫描务必确认卡已释放。
- **边界**：全部数据来自单机 4 × T10（sm75 / 16 GB / PHB、无 NVLink），`temperature 0.5` 与 `1.0` 混用，单次结果噪声较大。换硬件、换模型或换量化格式都需重新校准，不要直接外推。

## 本分支改动清单

**一、DFlash2 草稿路径**

- `bcca7786`：`dflash.fc` 与 attention q/k/v 加入张量并行（行并行 + output-gather，关闭两级融合保证头边界对齐）；新增 `RunDFlashTpLinear` 统一分片 GEMM 入口；修正 TP 预留中 fake 视图重复计费；草稿残差改 FP32 累加（BF16 会吞掉 1e-5 级注意力贡献）；`lm_head` 逐行缩放覆盖 NVFP4 各布局。`fc` 必须按输出维行并行——按输入维列并行会被框架的 `input.IsTensorParallelSharded()` 挡住并退回兜底路径，把整块约 250 MB 搬回 root（实测约 2.5 份、618.8 MB），显存不对齐且易 OOM。
- `dd900d07`：逐行激活缩放保护 FP16 GEMM 溢出。NVFP4 GEMM 为 FP16 进出，草稿 GEMM 输出峰值可超 65504 产生 inf，污染 RMSNorm 并让 selector 越界；按行取 absmax 推出 2 的幂 scale，scale 全程留在设备上，不破坏 CUDA graph 捕获。

**二、接受率统计口径**

- `bcca7786`：MTP / DFlash2 接受率日志改为 EMA 窗口增量，替换原先的进程累计计数。

**三、KV 显存预算**

- `80005d44`：新增 `include/utils/cuda_cache_budget.h`，`FASTLLM_KV_RUNTIME_HEADROOM_MB` 与 `FASTLLM_KV_FINAL_SAFETY_MB` 可覆盖校准默认值。

**四、多模态前缀缓存与视觉显存**

- `bcca7786`：按图片内容哈希键比对快照，无键输入（如视频）拒绝记录与复用；恢复边界必须越过全部视觉 token，文本尾巴走续写路径并按全序列 M-RoPE 切位置；`thread_local` 钩子在分块边界（页对齐 + 状态一致）安全记录快照。视觉侧：工作区在 KV 之前预热、激活 arena 按 chunk 尺寸收缩、`FASTLLM_QWEN35_MM_WORKSPACE_MARGIN_MB` 可调余量、`FASTLLM_QWEN35_SKIP_VISION` 供纯文本部署跳过视觉塔权重。新增 `include/utils/qwen35_mm_record_hook.h`。

**五、诊断探针清理**

- `dfe6e661`：移除 MoE 专家缓存与 NVFP4 规划路径的一次性 `[DBG-*]` / `[EP-DBG]` 打印、Qwen3.5 前缀缓存与结构探针及其环境开关、basellm 的探针打印与计数变量、`model.cpp` 的 deviceMap 打印，共删除 409 行；保留 staged fill 信息、错误上报、内存检查堆栈诊断与 `pos_accept_rate` 等运行信息日志。

分支还包含与本设备布局无关的合并与算子工作：`13c9b9ce`（合并上游 82 个提交）、`89b36ca9`（NVFP4 packed E4M3 scale 布局）、`dcc65774`（GGUF/IQ/MXFP4 算子验证测试）、`a0e51574`、`ebd29791`、`291974d8`、`6135be93`、`77067c59`。

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [docs/benchmarks/qwen38_27b_dflash2_t10_tp2.md](docs/benchmarks/qwen38_27b_dflash2_t10_tp2.md) | 本机完整实测：KV 池档位、准入测试、草稿数扫描、前缀缓存、视觉、回归清单 |
| [docs/benchmarks/qwen38_27b_dflash2.md](docs/benchmarks/qwen38_27b_dflash2.md) | 上游 DFlash2 Benchmark（含其它设备布局） |
| [docs/benchmark.md](docs/benchmark.md) | Benchmark 索引 |
| [scripts/qwen27b/README.md](scripts/qwen27b/README.md) | 启动脚本用法、默认参数表、仓库外依赖 |
| [docs/](docs/) | 上游通用文档目录（参数、模型、后端、算子等） |
| [README_EN.md](README_EN.md) | 上游英文 README |

## 复现与回归检查清单

1. 用上文命令启动，确认进程起来且 `/v1/models` 返回 200。
2. 检查日志：`[Qwen3.5 DFlash2] enabled: layers=5, drafts_per_step=7`、`[Vision] Multimodal workspace ready`、`KV Cache Token limit: 313872 tokens`、两卡 `freeAfterWarmup / targetFree`。
3. 通过标准：KV 池 ≥ `max_context_length`；两卡 `targetFree` 为正（期望约 +0.50 / +0.16 GB）。
4. 准入测试：并发 `max_batch` 条 8k 预填充全部 HTTP 200，压后余量 rank0 不低于约 300 MiB；任一条 000 或日志出现 `cuda malloc failed` 即未通过。
5. 视觉回归：发一张图片，确认出现 `[Vision] ... after encode: ... feature_tokens=` 与 `Image embedding cache` 行，且返回 200。
6. 前缀缓存回归：同一长前缀连发两次，第二次应出现 `prefix cache restored` 与 `prefix cache hit`，时延数量级下降。
7. 接受率口径：确认日志为 `pos_accept_rate(EMA)=[...] accept_len=... (window=... total=...)` 形式。
