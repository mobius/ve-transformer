#include <math.h>
#include <stdio.h>
#include <stdlib.h>
extern void sd_ve_silu_f32(int,float*,const float*);
int main(void) {
 const int lengths[]={1,37,255,256,257,1025,65537};
 for(int c=0;c<7;++c) {
  int n=lengths[c];float *x=malloc((n+2)*sizeof(float)),*y=malloc((n+2)*sizeof(float));
  if(!x||!y)return 2;
  for(int i=0;i<n;++i)x[i]=(float)((i%1001)-500)/10.0f;
  const float edges[]={-1000.0f,-100.0f,-89.0f,-88.0f,0.0f,88.0f,89.0f,100.0f,1000.0f};
  for(int i=0;i<n && i<9;++i)x[i]=edges[i];
  y[n]=12345.0f;y[n+1]=-12345.0f;
  sd_ve_silu_f32(n,y,x);double maximum=0;
  for(int i=0;i<n;++i) {double v=x[i],expected=v/(1.0+exp(-v));double error=fabs(y[i]-expected);
   if(!isfinite(y[i])||error>2e-5+2e-6*fabs(expected)) {printf("FAIL n=%d index=%d input=%g actual=%g expected=%g\n",n,i,x[i],y[i],expected);return 3;}
   if(error>maximum)maximum=error;
  }
  if(y[n]!=12345.0f||y[n+1]!=-12345.0f)return 4;
  sd_ve_silu_f32(n,x,x);
  for(int i=0;i<n;++i)if(x[i]!=y[i])return 5;
  printf("SiLU n=%d max_abs_error=%.9g independent_FP64_and_inplace PASS\n",n,maximum);
  free(x);free(y);
 }
 return 0;
}
