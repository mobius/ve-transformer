// Isolated FP32 elementwise candidate; same formula, no fast-math.
#include <math.h>
void sd_ve_silu_f32(int n, float * y, const float * x) {
    for (int i=0;i<n;++i) y[i]=x[i]/(1.0f+expf(-x[i]));
}
