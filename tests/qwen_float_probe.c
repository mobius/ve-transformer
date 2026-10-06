#define _POSIX_C_SOURCE 200809L
// Measure the alternative of expanding only dense projections within VE capacity.
#include <cblas.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void) {struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec/1e9;}
int main(void) {
    const int m=8192,n=2048;
    float *w=malloc((size_t)m*n*4),*x=malloc(n*4),*y=malloc(m*4);
    if(!w||!x||!y) return 1;
    for(int i=0;i<m*n;++i) w[i]=((i*13+7)%101-50)*0.03125f;
    for(int i=0;i<n;++i)x[i]=((i*17+3)%97-48)*0.03125f;
    for(int rep=-1;rep<5;++rep) {
        double start=now();cblas_sgemv(CblasRowMajor,CblasNoTrans,m,n,1,w,n,x,1,0,y,1);
        double dt=now()-start,checksum=0;
        for(int r=0;r<m;++r) checksum+=y[r];
        if(rep>=0) printf("m=%d n=%d repeat=%d seconds=%.9f effective_weight_GBps=%.6f checksum=%.9g\n",m,n,rep,dt,m*(double)n*4/dt/1e9,checksum);
    }
    for(int r=0;r<m;++r) {
        double ref=0;for(int k=0;k<n;++k)ref+=(double)w[(size_t)r*n+k]*x[k];
        if(!isfinite(y[r])||fabs(y[r]-ref)>0.01+0.0001*fabs(ref)) {fprintf(stderr,"float probe FAIL\n");return 1;}
    }
    puts("float probe independent accumulation PASS");free(w);free(x);free(y);return 0;
}
