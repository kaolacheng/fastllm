//
// 多模态前缀快照记录钩子
//
// 带图请求的 token 数（文本 + 视觉 token）几乎不会正好落在 KV 页边界
// (pageLen=128) 上，而 Qwen3.5 的线性注意力层状态无法回退裁剪，因此只有在
// 状态恰好页对齐的那个瞬间才能安全记录前缀快照。
//
// 多模态 prefill 内部按 chunked_prefill_size 分块推进（每块结束都会同步
// MTP draft 缓存），块边界天然满足"页对齐 + 状态一致"两个条件。basellm 在
// 调用多模态 forward 前把当前请求的 ResponseContext 放到这个 thread_local
// 里，模型侧在每块结束时取出来记录快照。
//
#pragma once

namespace fastllm {

// 仅在本请求所在的执行线程内可见；指向 ResponseContext（模型侧自行转换）。
extern thread_local void *gQwen35MultimodalRecordContext;

// 作用域守卫：进入多模态 forward 时设置，退出时恢复原值。
struct Qwen35MultimodalRecordScope {
    explicit Qwen35MultimodalRecordScope(void *context) {
        previous = gQwen35MultimodalRecordContext;
        gQwen35MultimodalRecordContext = context;
    }
    ~Qwen35MultimodalRecordScope() {
        gQwen35MultimodalRecordContext = previous;
    }
    Qwen35MultimodalRecordScope(const Qwen35MultimodalRecordScope &) = delete;
    Qwen35MultimodalRecordScope &operator=(const Qwen35MultimodalRecordScope &) = delete;

    void *previous = nullptr;
};

}  // namespace fastllm
