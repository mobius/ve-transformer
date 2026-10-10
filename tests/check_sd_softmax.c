#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern void sd_ve_softmax_exp_f32(int,float*,const float*,float);
extern double sd_ve_softmax_sum_f32(int,const float*);
int main(void) {
 const int lengths[]={1,37,77,255,256,257,4096,65537};
 for(int c=0;c<8;++c)for(int pattern=0;pattern<3;++pattern) {
  int n=lengths[c];float *x=malloc(n*sizeof(float)),*copy=malloc(n*sizeof(float)),*y=malloc((n+2)*sizeof(float));
  double *expected=malloc(n*sizeof(double));if(!x||!copy||!y||!expected)return 2;
  float maximum=-INFINITY;
  for(int i=0;i<n;++i) {
   x[i]=pattern==0?(float)((i%101)-50)/8.0f:pattern==1?(i==0?1000.0f:(float)(-1000+i%23)):(i%3==0?(float)(i%13):-INFINITY);
   if(x[i]>maximum)maximum=x[i];
  }
  memcpy(copy,x,n*sizeof(float));y[n]=12345.0f;y[n+1]=-12345.0f;
  sd_ve_softmax_exp_f32(n,y,x,maximum);double sum=sd_ve_softmax_sum_f32(n,y),ordered=0.0,reference_sum=0.0;
  for(int i=0;i<n;++i) {ordered+=(double)y[i];expected[i]=exp((double)(x[i]-maximum));reference_sum+=expected[i];}
  if(sum!=ordered||!isfinite(sum)||sum<=0.0||y[n]!=12345.0f||y[n+1]!=-12345.0f)return 3;
  double maximum_error=0.0,probability_sum=0.0;
  const float scale=(float)(1.0/sum);
  for(int i=0;i<n;++i) {
   double actual=(double)(y[i]*scale),ref=expected[i]/reference_sum,error=fabs(actual-ref);
   if(!isfinite(y[i])||y[i]<0.0f||error>2e-7+2e-6*fabs(ref))return 4;
   if(isinf(x[i])&&y[i]!=0.0f)return 5;
   if(error>maximum_error)maximum_error=error;probability_sum+=actual;
  }
  if(fabs(probability_sum-1.0)>2e-6)return 6;
  sd_ve_softmax_exp_f32(n,copy,copy,maximum);
  for(int i=0;i<n;++i)if(copy[i]!=y[i])return 7;
  printf("Softmax n=%d pattern=%d max_abs_error=%.9g independent_FP64_ordered_sum_inplace PASS\n",n,pattern,maximum_error);
  free(x);free(copy);free(y);free(expected);
 }
 return 0;
}
