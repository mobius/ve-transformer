// Sequential FP64 accumulation; compiled without vectorization/reassociation.
double sd_ve_softmax_sum_f32(int n,const float *x) {
    double sum=0.0;
    for(int i=0;i<n;++i)sum+=(double)x[i];
    return sum;
}
