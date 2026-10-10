#define _POSIX_C_SOURCE 200809L
#include <cblas.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static int number(const char *text) {
    char *end=NULL; errno=0; long value=strtol(text,&end,10);
    if(errno || !text[0] || *end || value<1 || value>1048576) return 0;
    return (int)value;
}
static float input_value(size_t index,uint32_t salt) {
    uint32_t x=(uint32_t)index^salt;
    x^=x>>16; x*=0x7feb352dU; x^=x>>15; x*=0x846ca68bU; x^=x>>16;
    return ((int)(x&1023U)-512)*(1.0f/512.0f);
}
static double seconds(void) {
    struct timespec t;
    if(clock_gettime(CLOCK_MONOTONIC,&t)) { fputs("probe clock unavailable\n",stderr); exit(3); }
    return (double)t.tv_sec+(double)t.tv_nsec/1e9;
}
int main(int argc,char **argv) {
    if(argc!=9 || sizeof(size_t)!=8) return 2;
    int m=number(argv[1]),n=number(argv[2]),k=number(argv[3]);
    int lda=number(argv[4]),ldb=number(argv[5]),ldc=number(argv[6]);
    int warmup=number(argv[7]),repeats=number(argv[8]);
    if(!m || !n || !k || lda<k || ldb<k || ldc<n || warmup!=1 || repeats!=3) return 2;
    uint64_t na=(uint64_t)m*lda,nb=(uint64_t)n*ldb,nc=(uint64_t)m*ldc;
    if(4*(na+nb+nc)>4ULL*1024*1024*1024) return 2;
    float *a=malloc((size_t)na*sizeof(float)),*b=malloc((size_t)nb*sizeof(float)),*c=malloc((size_t)nc*sizeof(float));
    if(!a || !b || !c) { free(a);free(b);free(c);return 4; }
    for(size_t i=0;i<(size_t)na;i++) a[i]=input_value(i,0x12345678U);
    for(size_t i=0;i<(size_t)nb;i++) b[i]=input_value(i,0x9abcdef0U);
    for(size_t i=0;i<(size_t)nc;i++) c[i]=NAN;
    int actual=1,openmp=0;
#ifdef _OPENMP
    openmp=1;
#pragma omp parallel shared(actual)
    {
#pragma omp single
        actual=omp_get_num_threads();
    }
#endif
    printf("PROBE_CONFIG openmp=%d actual_threads=%d m=%d n=%d k=%d lda=%d ldb=%d ldc=%d bytes=%llu\n",openmp,actual,m,n,k,lda,ldb,ldc,(unsigned long long)(4*(na+nb+nc)));
    fflush(stdout);
    for(int run=0;run<warmup+repeats;run++) {
        double started=seconds();
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,n,k,1.0f,a,lda,b,ldb,0.0f,c,ldc);
        double elapsed=seconds()-started;
        printf("PROBE_TIME iteration=%d phase=%s seconds=%.9f\n",run,run<warmup?"warmup":"measure",elapsed);
        for(int sample=0;sample<20;sample++) {
            int row,column;
            if(sample==0) {row=0;column=0;}
            else if(sample==1) {row=m-1;column=n-1;}
            else if(sample==2) {row=0;column=n-1;}
            else if(sample==3) {row=m-1;column=0;}
            else {row=(sample*7919+17)%m;column=(sample*104729+23)%n;}
            printf("PROBE_SAMPLE iteration=%d sample=%d row=%d column=%d value=%.9g\n",run,sample,row,column,c[(size_t)row*ldc+column]);
        }
        fflush(stdout);
    }
    for(int row=0;row<m;row++) for(int column=0;column<n;column++) {
        if(!isfinite(c[(size_t)row*ldc+column])) {free(a);free(b);free(c);return 5;}
    }
    printf("PROBE_FINITE elements=%llu PASS\n",(unsigned long long)m*n);
    free(a);free(b);free(c);return 0;
}
