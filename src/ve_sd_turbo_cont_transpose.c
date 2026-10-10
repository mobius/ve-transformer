#include <stddef.h>
#include <stdint.h>

// Internal pointers are disjoint, as established by the public entry.
static void transpose_rows(float * restrict dst, const float * restrict src,
                           size_t m, size_t n, size_t first, size_t last) {
    for (size_t j=first;j<last;++j) {
        float * row=dst+j*m;
        const float * column=src+j;
        for (size_t i=0;i<m;++i) row[i]=column[i*n];
    }
}

int sd_ve_cont_transpose_f32(float * dst, const float * src,
                           size_t m, size_t n, int ith, int nth) {
    if (!dst || !src || m<1 || n<1 || m>4096 || n>4096 ||
        m*n>2097152 || (nth!=1 && nth!=2 && nth!=4 && nth!=8) ||
        ith<0 || ith>=nth) return 0;
    const size_t bytes=m*n*sizeof(float);
    const uintptr_t d=(uintptr_t)dst,s=(uintptr_t)src;
    if (d<=s ? s-d<bytes : d-s<bytes) return 0;
    const size_t rows=(n+(size_t)nth-1)/(size_t)nth;
    const size_t first=rows*(size_t)ith;
    const size_t last=first+rows<n ? first+rows : n;
    transpose_rows(dst,src,m,n,first,last);
    return 1;
}
