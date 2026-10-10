// Isolated checked vector exp; subtraction and results remain FP32.
#include <math.h>
void sd_ve_softmax_exp_f32(int n,float *y,const float *x,float maximum) {
    for(int i=0;i<n;++i)y[i]=expf(x[i]-maximum);
}
