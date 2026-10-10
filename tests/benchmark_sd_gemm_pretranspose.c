#define _POSIX_C_SOURCE 200809L
#include <cblas.h>
#include <omp.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static float value(size_t i,uint32_t salt){uint32_t x=(uint32_t)i^salt;x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;x^=x>>16;return ((int)(x&1023)-512)*(1.0f/512.0f);}
static double now(void){struct timespec t;if(clock_gettime(CLOCK_MONOTONIC,&t))abort();return t.tv_sec+t.tv_nsec*1e-9;}
static int spatial;
static double setup_s,pack_s,multiply_s,cleanup_s;
static void pack(float * restrict dst,const float * restrict src,int n,int k){
#pragma omp parallel for
 for(int q=0;q<k;q++)for(int i=0;i<n;i++)dst[(size_t)q*n+i]=src[(size_t)i*k+q];
}
static void gemm(int m,int n,int k,int mode,const float*a,const float*b,float*c){
 setup_s=pack_s=multiply_s=cleanup_s=0;
 double start=now();size_t capacity=(size_t)k*spatial;
 float *storage=mode?malloc((capacity+128)*sizeof(float)):NULL;
 if(mode&&!storage)exit(4);
 float *bt=mode?storage+64:NULL;
 if(mode)for(int q=0;q<64;q++)storage[q]=bt[capacity+q]=-12345.25f;
 setup_s=now()-start;
 for(int j=0;j<n;j+=spatial){int width=n-j<spatial?n-j:spatial;
  if(mode){start=now();pack(bt,b+(size_t)j*k,width,k);pack_s+=now()-start;}
  start=now();cblas_sgemm(CblasRowMajor,CblasNoTrans,mode?CblasNoTrans:CblasTrans,
   m,width,k,1,a,k,mode?bt:b+(size_t)j*k,mode?width:k,0,c+j,n);multiply_s+=now()-start;
 }
 start=now();
 if(mode){for(int q=0;q<64;q++)if(storage[q]!=-12345.25f||bt[capacity+q]!=-12345.25f)exit(5);free(storage);}
 cleanup_s=now()-start;
}
int main(void){
 if(!getenv("VE_TRANSFORMER_TEMPERATURE_SUPERVISED"))return 8;
 if(omp_get_max_threads()!=4)return 2;
 int actual=0;
#pragma omp parallel shared(actual)
 {
#pragma omp single
 actual=omp_get_num_threads();
 }
 if(actual!=4)return 3;
 const int tiles[3][1]={{1},{1},{1}};
 const int baselines[3]={2048,4096,4096};
 const int shapes[3][3]={{512,16384,4608},{256,65536,2304},{256,65536,4608}};
 for(int shape=0;shape<3;shape++){
 int m=shapes[shape][0],n=shapes[shape][1],k=shapes[shape][2];
 int baseline_tile=0;spatial=baselines[shape];
 printf("TILE_SPATIAL shape=%d tile=%d\n",shape,spatial);
 size_t na=(size_t)m*k,nb=(size_t)n*k,nc=(size_t)m*n;
 float*a=malloc(na*4),*b=malloc(nb*4),*storage=malloc((nc+128)*4),*ref=malloc(nc*4);
 if(!a||!b||!storage||!ref)return 4;
 float*c=storage+64;
#pragma omp parallel for
 for(size_t i=0;i<na;i++)a[i]=value(i,0x12345678U);
#pragma omp parallel for
 for(size_t i=0;i<nb;i++)b[i]=value(i,0x9abcdef0U);
 for(int q=0;q<64;q++)storage[q]=c[nc+q]=-12345.25f;
 printf("TILE_CONFIG shape=%d m=%d n=%d k=%d lda=%d ldb=%d ldc=%d actual_threads=%d baseline_tile=%d\n",shape,m,n,k,k,k,n,actual,baseline_tile);fflush(stdout);
 gemm(m,n,k,0,a,b,ref);
 for(int ti=0;ti<1;ti++)for(int arm=0;arm<4;arm++){
 int tile=(arm==1||arm==2)?tiles[shape][ti]:baseline_tile;
 for(int rep=0;rep<4;rep++){
#pragma omp parallel for
 for(size_t i=0;i<nc;i++)c[i]=NAN;
 double start=now();gemm(m,n,k,tile,a,b,c);double elapsed=now()-start;
 printf("PACK_PARTS shape=%d candidate=1 arm=%d rep=%d setup_seconds=%.9f pack_seconds=%.9f multiply_seconds=%.9f cleanup_seconds=%.9f buffer_bytes=%zu\n",shape,arm,rep,setup_s,pack_s,multiply_s,cleanup_s,tile?((size_t)k*spatial+128)*sizeof(float):0);
 size_t errors=0;double max_error=0;
#pragma omp parallel for reduction(+:errors) reduction(max:max_error)
 for(size_t i=0;i<nc;i++){double err=fabs((double)c[i]-ref[i]);if(!isfinite(c[i])||!isfinite(ref[i])||err>0.0001+0.0001*fabs((double)ref[i]))errors++;if(err>max_error)max_error=err;}
 for(int q=0;q<64;q++)if(storage[q]!=-12345.25f||c[nc+q]!=-12345.25f)return 5;
 if(errors||omp_get_max_threads()!=4)return 6;
 printf("TILE_TIME shape=%d candidate=%d arm=%d tile=%d rep=%d seconds=%.9f checked=%zu max_error=%.9g\n",shape,tiles[shape][ti],arm,tile,rep,elapsed,nc,max_error);
 for(int sample=0;sample<20;sample++){int row,column;if(sample==0){row=0;column=0;}else if(sample==1){row=m-1;column=n-1;}else if(sample==2){row=0;column=n-1;}else if(sample==3){row=m-1;column=0;}else{row=(sample*7919+17)%m;column=(sample*104729+23)%n;}printf("TILE_SAMPLE shape=%d candidate=%d arm=%d rep=%d sample=%d row=%d column=%d value=%.9g\n",shape,tiles[shape][ti],arm,rep,sample,row,column,c[(size_t)row*n+column]);}
 fflush(stdout);
 }
 }
 size_t input_errors=0;
#pragma omp parallel for reduction(+:input_errors)
 for(size_t i=0;i<na;i++)if(a[i]!=value(i,0x12345678U))input_errors++;
#pragma omp parallel for reduction(+:input_errors)
 for(size_t i=0;i<nb;i++)if(b[i]!=value(i,0x9abcdef0U))input_errors++;
 if(input_errors)return 7;
 printf("TILE_INPUT shape=%d elements=%zu PASS\n",shape,na+nb);fflush(stdout);
 free(a);free(b);free(storage);free(ref);
 }
 puts("TILE_COMPLETE shapes=3 candidates=1 arms=4 runs=4 threads=4");return 0;
}
