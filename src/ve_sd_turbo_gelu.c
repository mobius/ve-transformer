// Isolated FP32 erf GELU; preserve formula and order, no fast-math.
#include <math.h>
void sd_ve_gelu_f32(int n, float *y, const float *x) {
    for (int i=0; i<n; ++i)
        y[i]=0.5f*x[i]*(1.0f+erff(x[i]*0.7071067811865475244f));
}
