#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <chrono>
#include <cstdlib>
extern "C" void sd_ve_softmax_scale_f32(int,float*,float);
extern "C" void sd_ve_softmax_copy_scale_f32(int,float*,const float*,float);
extern "C" void sd_o1_softmax_scale_f32(int,float*,float);
extern "C" void sd_o1_softmax_copy_scale_f32(int,float*,const float*,float);
static void save(const char*folder,int id,const char*kind,const float*x,int n){char name[1024];std::snprintf(name,sizeof name,"%s/case%d-%s.f32",folder,id,kind);FILE*f=std::fopen(name,"wb");if(!f||std::fwrite(x,4,n,f)!=(size_t)n)std::exit(3);std::fclose(f);}
int main(int argc,char**argv){if(argc!=2)return 2;int id=0;
 for(int n:{1,7,64,77,255,256,257,1024,4096}){
  std::vector<float>src(n+2),b(n+2),c(n+2);uint32_t state=0x7391ab;
  const uint32_t edges[]={0,0x80000000,1,0x80000001,0x00800000,0x80800000,0x7f800000,0xff800000,0x7f7fffff,0xff7fffff};
  for(int i=0;i<n;++i){state=state*1664525u+1013904223u;uint32_t bits=i%31<10?edges[i%31]:(0x3f000000u+(state&0x7fffffu));std::memcpy(src.data()+1+i,&bits,4);}
  for(float scale:{0.000244140625f,0.00031234567f,0.001953125f,0.1f,0.33333334f,0.5f,1.0f,2.0f}){
   b.front()=b.back()=c.front()=c.back()=12345.0f;
   sd_o1_softmax_copy_scale_f32(n,b.data()+1,src.data()+1,scale);sd_ve_softmax_copy_scale_f32(n,c.data()+1,src.data()+1,scale);
   if(std::memcmp(b.data(),c.data(),4*(n+2)))return 4;
   save(argv[1],id,"input",src.data()+1,n);save(argv[1],id,"copy-baseline",b.data()+1,n);save(argv[1],id,"copy-candidate",c.data()+1,n);
   std::memcpy(b.data()+1,src.data()+1,n*4);std::memcpy(c.data()+1,src.data()+1,n*4);
   sd_o1_softmax_scale_f32(n,b.data()+1,scale);sd_ve_softmax_scale_f32(n,c.data()+1,scale);
   if(std::memcmp(b.data(),c.data(),4*(n+2))||b.front()!=12345.0f||b.back()!=12345.0f)return 5;
   save(argv[1],id,"scale-baseline",b.data()+1,n);save(argv[1],id,"scale-candidate",c.data()+1,n);
   std::printf("CHECK id=%d n=%d scale=%.9g both_ops_bitwise_canary=PASS\n",id++,n,scale);
  }
  if(n!=64&&n!=77&&n!=256&&n!=1024&&n!=4096)continue;
  for(int op=0;op<2;++op)for(int rep=0;rep<6;++rep)for(int arm=0;arm<4;++arm){bool candidate=arm==1||arm==2;
   auto copy=candidate?sd_ve_softmax_copy_scale_f32:sd_o1_softmax_copy_scale_f32;auto norm=candidate?sd_ve_softmax_scale_f32:sd_o1_softmax_scale_f32;
   std::memcpy(c.data()+1,src.data()+1,n*4);
   for(int i=0;i<100;++i){if(op==0)copy(n,c.data()+1,src.data()+1,.5f);else norm(n,c.data()+1,1.0f);}
   auto start=std::chrono::steady_clock::now();for(int i=0;i<5000;++i){if(op==0)copy(n,c.data()+1,src.data()+1,.5f);else norm(n,c.data()+1,1.0f);}
   double t=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();std::printf("TIME n=%d op=%s rep=%d arm=%d mode=%s calls=5000 seconds=%.12f\n",n,op==0?"copy":"scale",rep,arm,candidate?"candidate":"baseline",t);
  }
 }
 std::puts("SOFTMAX_SCALE_BATCH_PASS fixtures=72 output_pairs=144 timings=240");return 0;}
