#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern void sd_ve_im2col_f32(const float*,float*,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int,int,int,int,int,int,int,int);
int main(void) {
 const int cases[][12]={
 {37,29,7,2,3,3,1,1,1,1,1,1},{257,3,3,1,1,1,1,1,0,0,1,1},
 {65,17,13,1,3,3,2,2,1,1,1,1},{33,21,5,2,3,3,1,1,2,2,2,2},
 {17,13,3,1,2,3,2,1,3,2,2,1},{1,1,1,1,3,3,1,1,4,4,1,1},
 {512,2,4,1,3,1,1,1,1,0,1,1},{11,9,3,3,3,3,3,2,1,2,1,2}};
 for(int c=0;c<8;++c)for(int t=0;t<2;++t) {
  const int *q=cases[c];int IW=q[0],IH=q[1],IC=q[2],N=q[3],KW=q[4],KH=q[5],s0=q[6],s1=q[7],p0=q[8],p1=q[9],d0=q[10],d1=q[11],nth=t?8:1;
  int OW=(IW+2*p0-d0*(KW-1)-1)/s0+1,OH=(IH+2*p1-d1*(KH-1)-1)/s1+1;
  size_t input=(size_t)N*IC*IH*IW,count=(size_t)N*OH*OW*IC*KH*KW;
  float *src=malloc(input*sizeof(float)),*dst=malloc((count+2)*sizeof(float));uint32_t *ref=calloc(count,sizeof(uint32_t));
  if(!src||!dst||!ref)return 2;
  for(size_t i=0;i<input;++i) {float v=(float)((int)(i%97)-48)/8.0f;uint32_t bits;memcpy(&bits,&v,4);if(i%317==0)bits=0x7fc01234;if(i%401==0)bits=0x7f800000;if(i%29==0)bits=0x80000000;memcpy(src+i,&bits,4);}
  memset(dst,0xa5,(count+2)*sizeof(float));
  for(int n=0;n<N;++n)for(int y=0;y<OH;++y)for(int x=0;x<OW;++x)for(int ic=0;ic<IC;++ic)for(int ky=0;ky<KH;++ky)for(int kx=0;kx<KW;++kx) {
   int iy=y*s1+ky*d1-p1,ix=x*s0+kx*d0-p0;size_t j=(((size_t)n*OH+y)*OW+x)*(IC*KH*KW)+(ic*KH+ky)*KW+kx;
   if(iy>=0&&iy<IH&&ix>=0&&ix<IW)memcpy(ref+j,src+(((size_t)n*IC+ic)*IH+iy)*IW+ix,4);
  }
  for(int ith=0;ith<nth;++ith)sd_ve_im2col_f32(src,dst,N,IC,IH,IW,KH,KW,OH,OW,s0,s1,p0,p1,d0,d1,ith,nth);
  uint32_t guards[2];memcpy(guards,dst+count,8);
  if(memcmp(dst,ref,count*4)||guards[0]!=0xa5a5a5a5||guards[1]!=0xa5a5a5a5) {printf("im2col case=%d threads=%d FAIL\n",c,nth);return 3;}
  printf("im2col case=%d threads=%d elements=%zu independent_bitwise_tail PASS\n",c,nth,count);
  free(src);free(dst);free(ref);
 }
 return 0;
}
