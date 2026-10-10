#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
typedef void (*kernel)(const float*,float*,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int,int,int,int,int,int,int,int);
extern void sd_ve_im2col_f32(const float*,float*,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int,int,int,int,int,int,int,int);
extern void sd_ve_im2col_rows_f32(const float*,float*,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int,int,int,int,int,int,int,int);
int main(void) {
 if(!getenv("VE_TRANSFORMER_TEMPERATURE_SUPERVISED"))return 8;
 omp_set_dynamic(0);omp_set_num_threads(4);
 const int shapes[4][2]={{512,128},{512,256},{256,256},{256,512}};
 for(int shape=0;shape<4;++shape) {
  const int width=shapes[shape][0],ic=shapes[shape][1];
  const size_t input=(size_t)ic*width*width,count=input*9;
  uint32_t *src=malloc(input*4),*saved=malloc(input*4),*allocation=malloc((count+4)*4),*out=allocation?allocation+2:NULL;
  if(!src||!saved||!out)return 2;
  for(size_t i=0;i<input;++i)src[i]=(i%317==0)?0x7fc01234:((i%401==0)?0x7f800000:((i%29==0)?0x80000000:(0x3f000000+(uint32_t)(i%65536))));
  memcpy(saved,src,input*4);
  for(int arm=0;arm<2;++arm) {
   const int candidate=arm==1;
   kernel fn=candidate?sd_ve_im2col_rows_f32:sd_ve_im2col_f32;
   for(int repeat=0;repeat<1;++repeat) {
    memset(allocation,0xa5,(count+4)*4);int bad_team=0;
    double start=omp_get_wtime();
    #pragma omp parallel reduction(|:bad_team)
    {
     int nth=omp_get_num_threads();bad_team|=nth!=4;
     fn((float*)src,(float*)out,1,ic,width,width,3,3,width,width,1,1,1,1,1,1,omp_get_thread_num(),nth);
    }
    double elapsed=omp_get_wtime()-start;
    if(bad_team||memcmp(src,saved,input*4)||out[count]!=0xa5a5a5a5||out[count+1]!=0xa5a5a5a5||allocation[0]!=0xa5a5a5a5||allocation[1]!=0xa5a5a5a5)return 3;
    // Scalar coordinate oracle, independent of either kernel's loop ordering.
    int bad_output=0;
    #pragma omp parallel for reduction(|:bad_output)
    for(int y=0;y<width;++y)for(int x=0;x<width;++x)for(int c=0;c<ic;++c)for(int ky=0;ky<3;++ky)for(int kx=0;kx<3;++kx) {
     int iy=y+ky-1,ix=x+kx-1;
     uint32_t expected=iy<0||iy>=width||ix<0||ix>=width?0:src[((size_t)c*width+iy)*width+ix];
     size_t index=(((size_t)y*width+x)*ic+c)*9+ky*3+kx;
     if(out[index]!=expected)bad_output=1;
    }
    if(bad_output){fprintf(stderr,"oracle mismatch\n");return 4;}
    printf("IM2COL_CURRENT4 width=%d ic=%d arm=%d mode=%s repeat=%d threads=4 seconds=%.9f checked_elements=%zu input_unchanged=1 guards=1 PASS\n",width,ic,arm,candidate?"rows":"channels",repeat,elapsed,count);fflush(stdout);
   }
  }
  free(src);free(saved);free(allocation);
 }
 puts("IM2COL_CURRENT4_COMPLETE shapes=4 modes=2 actual_threads=4");return 0;
}
