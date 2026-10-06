#pragma once
#include <stdint.h>
// Configure only between models, while no compute worker is active.
// The budget covers cached FP64 payloads; map metadata and scratch are separate.
extern "C" {
struct QwenDenseCacheStats {
    uint64_t budget_bytes, retained_bytes, reserved_bytes, entries, hits, misses, rejected;
#ifdef QWEN_TIMING
    double fill_thread_seconds; // Sum of parallel allocation/conversion intervals, not wall time.
#endif
};
void qwen_dense_cache_configure(uint64_t bytes);
QwenDenseCacheStats qwen_dense_cache_stats();
}
