#include "ggml.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <fstream>
static void fail(const char*s){std::fprintf(stderr,"GROUP_NORM_CENTER_FAIL %s\n",s);std::exit(2);}
static void save(const char*folder,int id,const char*name,const void*p,size_t count){char f[1024];std::snprintf(f,sizeof f,"%s/case%d-%s.f32",folder,id,name);FILE*out=std::fopen(f,"wb");if(!out||std::fwrite(p,4,count,out)!=count)fail("save output");std::fclose(out);}
static void fill(float*p,size_t n,int fixture){uint32_t state=0x81ab31;const uint32_t edges[]={0,0x80000000,1,0x80000001,0x00800000,0x80800000};for(size_t i=0;i<n;++i){state=state*1664525u+1013904223u;float v=(float)((int)((state>>8)%4001)-2000)/100.0f;if(i%31<6)std::memcpy(&v,&edges[i%31],4);if(fixture==1)v=.75f;if(fixture==2)v*=1.0e10f;p[i]=v;}}
static double compute(ggml_cgraph*g,ggml_cplan&plan,bool candidate){setenv("SD_GROUP_NORM_SCALE","1",1);setenv("SD_GROUP_NORM_CENTER_SQUARE",candidate?"1":"0",1);auto start=std::chrono::steady_clock::now();if(ggml_graph_compute(g,&plan)!=GGML_STATUS_SUCCESS)fail("graph compute");return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();}
static void run(const char*folder,int id,int64_t a,int64_t b,int64_t c,int64_t d,int groups,float eps){if(a<=0||b<=0||c<=0||d!=1||groups<=0||c%groups||!std::isfinite(eps)||eps<=0)fail("fixture geometry");size_t count=(size_t)a*b*c*d;auto*ctx=ggml_init({count*12+16*1024*1024,nullptr,false});if(!ctx)fail("context");auto*src=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,a,b,c,d);auto*out=ggml_group_norm(ctx,src,groups,eps);auto*g=ggml_new_graph(ctx);ggml_build_forward_expand(g,out);std::vector<float>original(count),baseline(count);
 for(int fixture=0;fixture<3;++fixture){fill((float*)src->data,count,fixture);std::memcpy(original.data(),src->data,count*4);
  for(int nth:{1,2,4,8}){auto params=ggml_threadpool_params_default(nth);params.poll=0;auto*pool=ggml_threadpool_new(&params);if(!pool)fail("pool");auto plan=ggml_graph_plan(g,nth,pool);if(plan.n_threads!=nth)fail("plan threads");std::vector<uint8_t>work(plan.work_size);plan.work_data=work.data();compute(g,plan,false);std::memcpy(baseline.data(),out->data,count*4);compute(g,plan,true);
   if(std::memcmp(baseline.data(),out->data,count*4)||std::memcmp(original.data(),src->data,count*4))fail("bitwise output/source differs");for(size_t i=0;i<count;++i)if(!std::isfinite(((float*)out->data)[i])||(fixture==1&&((float*)out->data)[i]!=0))fail("nonfinite or constant output");
   std::printf("CHECK id=%d fixture=%d threads=%d ne=%lld,%lld,%lld,%lld groups=%d eps=%.9g bitwise_source=PASS\n",id,fixture,nth,(long long)a,(long long)b,(long long)c,(long long)d,groups,eps);
   if(fixture==0&&nth==8){for(int rep=0;rep<6;++rep)for(int arm=0;arm<4;++arm){bool candidate=arm==1||arm==2;setenv("SD_GGML_THREADS_QUIET","1",1);setenv("SD_GROUP_NORM_SCALE","1",1);setenv("SD_GROUP_NORM_CENTER_SQUARE",candidate?"1":"0",1);auto batch_start=std::chrono::steady_clock::now();for(int call=0;call<16;++call)if(ggml_graph_compute(g,&plan)!=GGML_STATUS_SUCCESS)fail("batch compute");double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-batch_start).count()/16;setenv("SD_GGML_THREADS_QUIET","0",1);if(std::memcmp(baseline.data(),out->data,count*4)||std::memcmp(original.data(),src->data,count*4))fail("timed output/source differs");std::printf("TIME id=%d rep=%d arm=%d mode=%s threads=8 seconds=%.12f\n",id,rep,arm,candidate?"candidate":"baseline",seconds);}
    compute(g,plan,true);if(std::memcmp(baseline.data(),out->data,count*4)||std::memcmp(original.data(),src->data,count*4))fail("captured candidate output/source differs");std::printf("CAPTURE id=%d fixture=0 threads=8 mode=candidate source_unchanged=PASS\n",id);
    save(folder,id,"input",src->data,count);save(folder,id,"baseline",baseline.data(),count);save(folder,id,"candidate",out->data,count);
   }ggml_threadpool_free(pool);
  }
 }ggml_free(ctx);}
int main(int argc,char**argv){if(argc!=3)return 2;std::setvbuf(stdout,nullptr,_IOLBF,0);setenv("SD_GGML_THREADS_QUIET","0",1);setenv("SD_GROUP_NORM_SHAPE_PROFILE","0",1);setenv("SD_SOFTMAX_SHAPE_PROFILE","0",1);setenv("SD_OP_PROFILE","0",1);std::ifstream f(argv[1]);int id,groups,actual=0;int64_t a,b,c,d;float eps;while(f>>id>>a>>b>>c>>d>>groups>>eps){if(id!=actual)fail("fixture order");run(argv[2],id,a,b,c,d,groups,eps);++actual;}if(actual!=24)fail("24 fixture combinations required");std::puts("GROUP_NORM_CENTER_BATCH_PASS shapes=24 fixtures=72 checks=288 timings=576 calls_per_timing=16 captures=24");return 0;}
