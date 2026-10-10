#include <stddef.h>
#include <stdint.h>
#include <omp.h>

/* Disjoint buffers; copy only, including NaN payloads and signed zero. */
int sd_ve_gemm_pack_f32(float * restrict dst,const float * restrict src,int n,int k) {
    if(!dst || !src || n<1 || n>4096 || k<1 || k>4608 ||
       omp_in_parallel() || omp_get_max_threads()!=4) return 0;
    const size_t bytes=(size_t)n*(size_t)k*sizeof(float);
    const uintptr_t d=(uintptr_t)dst,s=(uintptr_t)src;
    if(d>UINTPTR_MAX-bytes || s>UINTPTR_MAX-bytes ||
       (d<=s ? s-d<bytes : d-s<bytes)) return 0;
    int actual=0;
#pragma omp parallel shared(actual)
    {
#pragma omp single
        actual=omp_get_num_threads();
#pragma omp for
        for(int q=0;q<k;q++) {
            if(actual==4)
                for(int i=0;i<n;i++) dst[(size_t)q*n+i]=src[(size_t)i*k+q];
        }
    }
    return actual==4;
}
