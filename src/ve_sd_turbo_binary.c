#include <stdint.h>

/* Caller proves disjoint buffers or exact in-place ownership. No reduction. */
void sd_ve_binary_plane_f32(const float * x, float * y, float value,
                            int64_t n, int multiply) {
    if (x == y) {
        float * restrict output = y;
        if (multiply) {
            for (int64_t i = 0; i < n; ++i) output[i] = output[i] * value;
        } else {
            for (int64_t i = 0; i < n; ++i) output[i] = output[i] + value;
        }
    } else {
        const float * restrict input = x;
        float * restrict output = y;
        if (multiply) {
            for (int64_t i = 0; i < n; ++i) output[i] = input[i] * value;
        } else {
            for (int64_t i = 0; i < n; ++i) output[i] = input[i] + value;
        }
    }
}
