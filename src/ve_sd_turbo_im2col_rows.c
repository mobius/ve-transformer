// Experimental output-row ownership; original output layout is preserved.
#include <stdint.h>
void sd_ve_im2col_rows_f32(const float *src, float *dst,
 int64_t N,int64_t IC,int64_t IH,int64_t IW,int64_t KH,int64_t KW,int64_t OH,int64_t OW,
 int s0,int s1,int p0,int p1,int d0,int d1,int ith,int nth) {
 const int64_t pitch=IC*KH*KW;
 const int64_t rows=N*OH;
 const int64_t begin=rows*ith/nth,end=rows*(ith+1)/nth;
 for(int64_t r=begin;r<end;++r) {
  const int64_t n=r/OH,oh=r%OH;
  for(int64_t ic=0;ic<IC;++ic)
  for(int64_t kh=0;kh<KH;++kh)for(int64_t kw=0;kw<KW;++kw) {
   float *out=dst+r*OW*pitch+ic*KH*KW+kh*KW+kw;
   const int64_t iy=oh*s1+kh*d1-p1,offset=kw*d0-p0;
   if(iy<0||iy>=IH) {for(int64_t x=0;x<OW;++x)out[x*pitch]=0.0f;continue;}
   int64_t lo=offset<0?(-offset+s0-1)/s0:0;
   int64_t hi=IW-1-offset<0?0:(IW-1-offset)/s0+1;
   if(lo>OW)lo=OW;if(hi>OW)hi=OW;
   if(hi<lo) {for(int64_t x=0;x<OW;++x)out[x*pitch]=0.0f;continue;}
   const float *row=src+(n*IC+ic)*IH*IW+iy*IW;
   for(int64_t x=0;x<lo;++x)out[x*pitch]=0.0f;
   for(int64_t x=lo;x<hi;++x)out[x*pitch]=row[x*s0+offset];
   for(int64_t x=hi;x<OW;++x)out[x*pitch]=0.0f;
  }
 }
}
