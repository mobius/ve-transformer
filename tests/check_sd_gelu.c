#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern void sd_ve_gelu_f32(int,float*,const float*);
int main(void) {
 const int lengths[]={0,1,3,7,63,64,127,128,255,256,257,511,512,513,1025,4096,65537};
 const float edges[]={-INFINITY,INFINITY,NAN,-FLT_MAX,FLT_MAX,-1000,1000,-10,10,-0.0f,0.0f,
                      -FLT_MIN,FLT_MIN,-0x1p-149f,0x1p-149f,-4,4,-8,8};
 int cases=0; double max_error=0;
 for(int c=0;c<17;++c) for(int mode=0;mode<3;++mode) for(int inplace=0;inplace<2;++inplace) {
  int n=lengths[c];float *a=malloc((n+2)*sizeof(float)),*b=malloc((n+2)*sizeof(float)),*saved=malloc((n+2)*sizeof(float));
  if(!a||!b||!saved)return 2;
  a[0]=b[0]=12345;a[n+1]=b[n+1]=-12345;
  for(int i=0;i<n;++i) {
   uint32_t v=(uint32_t)i*1664525u+1013904223u;
   a[i+1]=mode==0 ? (float)((i%3201)-1600)/100.0f : mode==1 ? -9.0f+(float)(i%5001)/1000.0f : (float)(v%320001u)/10000.0f-16;
   if(i<(int)(sizeof(edges)/sizeof(edges[0])))a[i+1]=edges[i];
  }
  memcpy(saved,a,(n+2)*sizeof(float));float *y=inplace?a+1:b+1;
  fprintf(stderr,"fixture n=%d mode=%d inplace=%d before_kernel\n",n,mode,inplace);
  sd_ve_gelu_f32(n,y,a+1);
  fprintf(stderr,"fixture n=%d mode=%d inplace=%d after_kernel\n",n,mode,inplace);
  if(a[0]!=12345||a[n+1]!=-12345||b[0]!=12345||b[n+1]!=-12345)return 3;
  if(!inplace&&memcmp(a,saved,(n+2)*sizeof(float)))return 4;
  for(int i=0;i<n;++i) {
   float input=saved[i+1],actual=y[i];
   if(!isfinite(input)) {
    if((isnan(input)||input<0) ? !isnan(actual) : !(isinf(actual)&&actual>0))return 5;
    continue;
   }
   double x=input, expected=0.5*x*(1.0+erf(x/sqrt(2.0)));
   double error=fabs((double)actual-expected);
   if(!isfinite(actual)||error>2e-6+2e-6*fabs(expected)) {
    printf("FAIL n=%d mode=%d inplace=%d index=%d input=%g actual=%g expected=%.17g error=%g\n",n,mode,inplace,i,input,actual,expected,error);return 6;
   }
   if(input==0&&((actual!=0)||!!signbit(actual)!=!!signbit(input)))return 7;
   if(error>max_error)max_error=error;
  }
  free(a);free(b);free(saved);++cases;
 }
 printf("GELU fixtures=%d FP64_reference_special_values_signed_zero_guards_inplace PASS max_abs_error=%.9g\n",cases,max_error);
 return 0;
}
