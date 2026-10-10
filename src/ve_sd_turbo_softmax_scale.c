/* Internal row primitives: copy source/destination must not overlap. */
void sd_ve_softmax_scale_f32(int n,float *x,float scale) {
    for(int i=0;i<n;++i)x[i]*=scale;
}
void sd_ve_softmax_copy_scale_f32(int n,float *restrict dst,const float *restrict src,float scale) {
#ifdef SD_SCALE_BASELINE
    for(int i=0;i<n;++i)dst[i]=src[i];
    for(int i=0;i<n;++i)dst[i]*=scale;
#else
    for(int i=0;i<n;++i)dst[i]=src[i]*scale;
#endif
}
