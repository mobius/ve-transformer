#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>
extern int sd_ve_gemm_pack_f32(float*,const float*,int,int);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static uint32_t bits(size_t i,int mode){static const uint32_t special[]={0,0x80000000,0x7f800000,0xff800000,0x7fc01234,0x7f801234,1,0x80000001};if(mode)return special[i%8];uint32_t x=(uint32_t)i*0x9e3779b9U;x^=x>>13;return x;}
int main(void){
 if(!getenv("VE_TRANSFORMER_TEMPERATURE_SUPERVISED"))return 8;
 omp_set_dynamic(0);omp_set_num_threads(4);
 const int sizes[3][2]={{2048,4608},{4096,2304},{4096,4608}};
 for(int shape=0;shape<3;shape++)for(int mode=0;mode<2;mode++){
 int n=sizes[shape][0],k=sizes[shape][1];size_t count=(size_t)n*k;
 float *src=malloc(count*4),*storage=malloc((count+128)*4);if(!src||!storage)return 2;float*dst=storage+64;
#pragma omp parallel for
 for(size_t i=0;i<count;i++){uint32_t v=bits(i,mode);memcpy(src+i,&v,4);}
 for(size_t i=0;i<count+128;i++){uint32_t v=0xdeadbeefU;memcpy(storage+i,&v,4);}
 double started=now();if(!sd_ve_gemm_pack_f32(dst,src,n,k))return 3;double elapsed=now()-started;
 size_t errors=0;
#pragma omp parallel for reduction(+:errors)
 for(size_t i=0;i<count;i++){uint32_t a,b;memcpy(&a,src+i,4);size_t row=i/k,col=i%k;memcpy(&b,dst+col*n+row,4);if(a!=bits(i,mode)||a!=b)errors++;}
 for(int i=0;i<64;i++){uint32_t a,b;memcpy(&a,storage+i,4);memcpy(&b,dst+count+i,4);if(a!=0xdeadbeefU||b!=0xdeadbeefU)errors++;}
 if(errors)return 4;
 printf("PACK_CHECK shape=%d n=%d k=%d mode=%d seconds=%.9f checked=%zu input_unchanged=1 guards=1 PASS\n",shape,n,k,mode,elapsed,count);fflush(stdout);free(src);free(storage);
 }
 float a[16],b[16];memset(a,0x5a,sizeof(a));memset(b,0x6b,sizeof(b));float saved[16];memcpy(saved,b,sizeof(b));
 int rejected=0;rejected+=!sd_ve_gemm_pack_f32(NULL,a,1,1);rejected+=!sd_ve_gemm_pack_f32(b,NULL,1,1);rejected+=!sd_ve_gemm_pack_f32(b,a,0,1);rejected+=!sd_ve_gemm_pack_f32(b,a,4097,1);rejected+=!sd_ve_gemm_pack_f32(b,a,1,0);rejected+=!sd_ve_gemm_pack_f32(b,a,1,4609);rejected+=!sd_ve_gemm_pack_f32(b,b,1,1);rejected+=!sd_ve_gemm_pack_f32(b+1,b,2,2);
 omp_set_num_threads(8);rejected+=!sd_ve_gemm_pack_f32(b,a,1,1);omp_set_num_threads(4);
 int nested=1;
#pragma omp parallel shared(nested)
 {
#pragma omp single
 nested=sd_ve_gemm_pack_f32(b,a,1,1);
 }
 rejected+=!nested;if(rejected!=10||memcmp(saved,b,sizeof(b)))return 5;
 puts("PACK_COMPLETE shapes=3 modes=2 actual_threads=4 rejected=10");return 0;
}
