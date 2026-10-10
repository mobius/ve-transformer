#include <omp.h>
#include <cblas.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

static double now(void) {
    struct timespec t;
    if(clock_gettime(CLOCK_MONOTONIC,&t)) abort();
    return (double)t.tv_sec+(double)t.tv_nsec*1e-9;
}

int main(void) {
    const int m=512,n=16384,k=4608;
    float * a=malloc((size_t)m*k*4),*b=malloc((size_t)n*k*4),*c=malloc((size_t)m*n*4);
    if(!a||!b||!c)return 2;
    for(int i=0;i<m;++i)for(int q=0;q<k;++q)a[(size_t)i*k+q]=(float)(i%5-2)*0.25f;
    for(int j=0;j<n;++j)for(int q=0;q<k;++q)b[(size_t)j*k+q]=(float)(j%7-3)*0.125f;
    if(omp_get_max_threads()!=8)return 3;
    for(int phase=0;phase<4;++phase) {
        const int switching=phase==1||phase==2;
        for(int cycle=0;cycle<5;++cycle)for(int step=0;step<3;++step) {
            const int wanted=switching && step==1 ? 4 : 8;
            const double start=now();
            if(omp_get_max_threads()!=wanted)omp_set_num_threads(wanted);
            const double configured=now();
            int actual=0;
            uint64_t sum=0;
#pragma omp parallel shared(actual) reduction(+:sum)
            {
#pragma omp single
                actual=omp_get_num_threads();
#pragma omp for
                for(int i=0;i<65536;++i)sum+=(uint64_t)i;
            }
            const double region=now();
            if(actual!=wanted||sum!=UINT64_C(2147450880))return 4;
            cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,n,k,1.0f,a,k,b,k,0.0f,c,n);
            const double gemm=now();
            for(int i=0;i<m;++i)for(int j=0;j<n;++j) {
                const float expected=(float)(144*(i%5-2)*(j%7-3));
                if(!isfinite(c[(size_t)i*n+j])||c[(size_t)i*n+j]!=expected)return 5;
            }
            if(omp_get_max_threads()!=wanted)return 6;
            printf("OMP_SWITCH phase=%d mode=%s cycle=%d step=%d measured=%d threads=%d actual=%d after=%d config_seconds=%.9f region_seconds=%.9f gemm_seconds=%.9f integer_sum=%" PRIu64 " checked=%d\n",
                   phase,switching?"switch":"fixed",cycle,step,cycle!=0,wanted,actual,omp_get_max_threads(),configured-start,region-configured,gemm-region,sum,m*n);
            fflush(stdout);
        }
    }
    free(a);free(b);free(c);
    printf("OMP_SWITCH_PASS calls=60 checked_elements=%" PRIu64 " final_threads=%d\n",UINT64_C(60)*512*16384,omp_get_max_threads());
    return 0;
}
