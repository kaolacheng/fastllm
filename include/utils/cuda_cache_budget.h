#ifndef FASTLLM_CUDA_CACHE_BUDGET_H
#define FASTLLM_CUDA_CACHE_BUDGET_H

#include <algorithm>
#include <cstdlib>

namespace fastllm {
    // Optional operator overrides for the per-device KV safety margins.  Unset
    // keeps the calibrated defaults; multi-GPU deployments can reclaim the
    // fixed per-device reservations that do not scale with device count.
    inline long long CudaCacheEnvMb(const char *name, long long fallbackBytes) {
        const char *value = std::getenv(name);
        if (value == nullptr || *value == '\0') {
            return fallbackBytes;
        }
        char *end = nullptr;
        long double mb = std::strtold(value, &end);
        if (end == value || mb < 0.0L) {
            return fallbackBytes;
        }
        return (long long)(mb * 1024.0L * 1024.0L);
    }

    // Shared by paged AutoWarmup and models with continuous KV storage.
    // Measure available memory after warming the actual serving workspaces.
    inline long long CudaCacheRuntimeHeadroom(long long total, long long available) {
        const char *overrideMb = std::getenv("FASTLLM_KV_RUNTIME_HEADROOM_MB");
        if (overrideMb != nullptr && *overrideMb != '\0') {
            const long long requested = CudaCacheEnvMb("FASTLLM_KV_RUNTIME_HEADROOM_MB", -1);
            if (requested >= 0) {
                return std::max(0LL, std::min(requested, available / 4));
            }
        }
        const long long headroom = std::min(
            std::max(512LL * 1024 * 1024, total / 100), 2LL * 1024 * 1024 * 1024);
        return std::max(0LL, std::min(headroom, available / 4));
    }

    // Absolute safety margin kept free on every device.
    inline long long CudaCacheFinalSafety(long long total) {
        const long long fallback = std::min(
            std::max(128LL * 1024 * 1024, total / 200), 512LL * 1024 * 1024);
        return CudaCacheEnvMb("FASTLLM_KV_FINAL_SAFETY_MB", fallback);
    }
}

#endif
