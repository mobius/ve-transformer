#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <chrono>
#include <cstdlib>
extern "C" double sd_ve_softmax_sum_f32(int,const float*);
extern "C" double sd_ve_softmax_sum_o2_f32(int,const float*);
static uint64_t bits(double x){uint64_t b;std::memcpy(&b,&x,8);return b;}
static volatile double sink;
int main(){
 for(int n:{1,7,64,77,255,256,257,1024,4096}){
  std::vector<float>x(n);uint32_t state=0x91ab73;
  for(int fixture=0;fixture<12;++fixture){
   for(int i=0;i<n;++i){state=state*1664525u+1013904223u;uint32_t b=0x3f000000u+(state&0x7fffffu);if(i%31==0)b=0;if(i%31==1)b=1;if(i%31==2)b=0x00800000;if(i%31==3)b=0x3f800000;std::memcpy(&x[i],&b,4);}
   double a=sd_ve_softmax_sum_f32(n,x.data()),b=sd_ve_softmax_sum_o2_f32(n,x.data());
   if(bits(a)!=bits(b))return 2;
   std::printf("CHECK n=%d fixture=%d baseline_bits=%llu candidate_bits=%llu PASS\n",n,fixture,(unsigned long long)bits(a),(unsigned long long)bits(b));
  }
  if(n!=64&&n!=77&&n!=256&&n!=1024&&n!=4096)continue;
  for(int rep=0;rep<6;++rep)for(int arm=0;arm<4;++arm){
   const bool candidate=arm==1||arm==2;auto fn=candidate?sd_ve_softmax_sum_o2_f32:sd_ve_softmax_sum_f32;
   for(int i=0;i<100;++i)sink=fn(n,x.data());
   auto start=std::chrono::steady_clock::now();for(int i=0;i<5000;++i)sink=fn(n,x.data());
   double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
   std::printf("TIME n=%d rep=%d arm=%d mode=%s calls=5000 seconds=%.12f\n",n,rep,arm,candidate?"candidate":"baseline",seconds);
  }
 }
 std::puts("SOFTMAX_SUM_BATCH_PASS checks=108 timings=120");return 0;
}
