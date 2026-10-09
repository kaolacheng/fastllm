# Qwen3.8-27B W4A4 + DFlash2 生产启动脚本（2 × Tesla T10）

本目录收录 2 × Tesla T10（sm75，16 GB）上 Qwen3.8-27B W4A4 + DFlash2-FP8 的生产启动器。硬件布局、参数依据与全部实测数据见 [`docs/benchmarks/qwen38_27b_dflash2_t10_tp2.md`](../../docs/benchmarks/qwen38_27b_dflash2_t10_tp2.md)。

| 文件 | 作用 |
| --- | --- |
| `start_dflash2.sh` | 在 `cuda2/3` 上以 TP2 启动 OpenAI 兼容服务（端口 8092，服务名 `Qwen3.8 27b`），日志与 pid 写入 `/tmp` |
| `stop_dflash2.sh` | 只停止命中该端口的 ftllm 进程：在 `/proc/<pid>/cmdline` 中精确匹配 `--port` 的取值，不使用 `pkill -f`，因此不影响同机其它实例 |

## 用法

~~~bash
bash scripts/qwen27b/start_dflash2.sh
bash scripts/qwen27b/stop_dflash2.sh
~~~

同机已有其它 ftllm 实例时，启动脚本会拒绝启动，需显式允许多实例：

~~~bash
ALLOW_MULTI=1 bash scripts/qwen27b/start_dflash2.sh
~~~

脚本内部用 `nohup` 拉起服务。若启动命令本身运行在会被中断的包装脚本里，建议在外层用 `setsid nohup bash scripts/qwen27b/start_dflash2.sh`，否则包装脚本收到的 SIGTERM 会连带结束同进程组的服务。

## 默认参数

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MODEL` | `/mnt/models/Qwen3.8-27B-Coder390-W4A4/NVFP4/W4A4` | 目标模型（NVFP4 / W4A4） |
| `DRAFT` | `$MODEL/DFlash2-FP8` | DFlash2 草稿模型 |
| `SERVED_NAME` | `Qwen3.8 27b` | API 中 `model` 字段的取值 |
| `PORT` | `8092` | 服务端口 |
| `TP` | `2` | 张量并行度 |
| `MAX_BATCH` | `2` | 并发序列数；batch 4 与视觉不兼容，见文档 |
| `MAX_CTX` | `262144` | 单会话上下文窗口 |
| `CHUNK` | `2048` | `chunked_prefill_size` |
| `PAGE_SIZE` | `16` | Paged KV 页长 |
| `GPU_MEM_RATIO` | `1.01` | 让给 KV 池的显存比例；1.01 是 batch 2 + 视觉下两卡 `targetFree` 均为正的最后档 |
| `PREFIX_CACHE` | `true` | 前缀缓存；同前缀 1855 token 复用实测 11.9× |
| `MULTIMODAL` | `1` | `1` 时追加 `--multimodal --image_embedding_cache $IMAGE_EMBED_CACHE` |
| `IMAGE_EMBED_CACHE` | `4g` | 视觉 embedding 缓存（CPU 侧，按需分配） |
| `DRAFT_TOKENS` | `-1` | `-1` 表示省略 `--draft_tokens`，使用 checkpoint 原生值 7；覆盖时必须是正整数 |
| `FTLLM_BIN` | `/home/kaolachen/fastllm-env/bin/ftllm` | ftllm 可执行文件 |
| `LOG` | `/tmp/ftllm_dflash2.log` | 服务日志（同时软链到 `/tmp/ftllm_current.log`） |
| `PIDFILE` | `/tmp/ftllm_dflash2.pid` | pid 文件 |
| `CUDA_VISIBLE_DEVICES` | `2,3` | 本实例占用的卡 |
| `FASTLLM_CUDA_GRAPH` | `0` | 关闭 CUDA Graph |
| `FASTLLM_DFLASH_BATCH_PREFIX_SNAPSHOTS` | `0` | 关闭批量前缀快照，只保留一份共享快照 |
| `FASTLLM_QWEN35_MM_WORKSPACE_MARGIN_MB` | `128` | 视觉工作区余量 |
| `FASTLLM_KV_RUNTIME_HEADROOM_MB` | `8` | 显式覆盖校准默认值（默认是 `min(max(512 MB, total/100), 2 GB)` 再被 `available/4` 兜底） |
| `FASTLLM_KV_FINAL_SAFETY_MB` | `8` | 显式覆盖校准默认值（默认是 `min(max(128 MB, total/200), 512 MB)`） |

所有变量都可以用环境变量覆盖，例如：

~~~bash
CUDA_VISIBLE_DEVICES=0,1 PORT=8093 LOG=/tmp/ftllm_dbg.log PIDFILE=/tmp/ftllm_dbg.pid \
  ALLOW_MULTI=1 bash scripts/qwen27b/start_dflash2.sh
~~~

## 仓库外依赖（换机器必须修改）

| 依赖 | 脚本中的取值 | 说明 |
| --- | --- | --- |
| 模型与草稿路径 | `/mnt/models/Qwen3.8-27B-Coder390-W4A4/NVFP4/W4A4` | 换成实际部署路径；草稿必须是该目录下的 `DFlash2-FP8` 或另用 `DRAFT` 指定 |
| ftllm 可执行文件 | `/home/kaolachen/fastllm-env/bin/ftllm` | 换成目标环境的安装路径（或设 `FTLLM_BIN`） |
| CUDA 运行库 | `LD_LIBRARY_PATH=/home/kaolachen/vllm-env/lib/python3.10/site-packages/nvidia/cu13/lib` | 该机器上 ftllm 运行期依赖的 CUDA 库目录 |
| 日志 / pid | `/tmp/ftllm_dflash2.log`、`/tmp/ftllm_dflash2.pid` | 多实例并存时必须分别指定，否则会互相覆盖 |
| 显卡编号 | `CUDA_VISIBLE_DEVICES=2,3` | 按机器可用卡调整 |

其余参数（batch、上下文、页长、`gpu_mem_ratio`、两个 KV 余量）与显存强相关，改动前请先按文档中的准入测试协议重新校准：并发填满 `max_batch` 的 8k 预填充，再看日志 `targetFree` 与压后余量。
