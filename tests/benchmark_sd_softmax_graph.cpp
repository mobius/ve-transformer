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
static void fail(const char*s){std::fprintf(stderr,"SOFTMAX_GRAPH_FAIL %s\n",s);std::exit(2);}
static void save(const char*folder,int id,const char*name,const void*p,size_t count){char f[1024];std::snprintf(f,sizeof f,"%s/case%d-%s.f32",folder,id,name);FILE*out=std::fopen(f,"wb");if(!out||std::fwrite(p,4,count,out)!=count)fail("save output");std::fclose(out);}
static void fill(float*p,size_t n){uint32_t state=0x81ab31;for(size_t i=0;i<n;++i){state=state*1664525u+1013904223u;p[i]=(float)((int)((state>>8)%4001)-2000)/100.0f;if(i%31==0)p[i]=-100.0f;if(i%37==0)p[i]=-1.0e-38f;}}
static double compute(ggml_cgraph*g,ggml_cplan&plan,bool candidate){setenv("SD_VE_SOFTMAX_SCALE",candidate?"1":"0",1);auto start=std::chrono::steady_clock::now();if(ggml_graph_compute(g,&plan)!=GGML_STATUS_SUCCESS)fail("graph compute");return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();}
static void run(const char*folder,int id,int64_t a,int64_t b,int64_t c,int64_t d,float scale,int mask_type,bool timed){size_t count=(size_t)a*b*c*d;auto*ctx=ggml_init({count*12+16*1024*1024,nullptr,false});if(!ctx)fail("context");auto*src=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,a,b,c,d);fill((float*)src->data,count);std::vector<float>original(count);std::memcpy(original.data(),src->data,count*4);
 ggml_tensor*mask=nullptr;if(mask_type){mask=ggml_new_tensor_2d(ctx,mask_type==1?GGML_TYPE_F32:GGML_TYPE_F16,a,b);for(int64_t i=0;i<a*b;++i){float v=i%a%13==12?-INFINITY:(float)((i%a)%7)*-.125f;if(mask_type==1)((float*)mask->data)[i]=v;else ((ggml_fp16_t*)mask->data)[i]=ggml_fp32_to_fp16(v);}}
 auto*out=ggml_soft_max_ext(ctx,src,mask,scale,0.0f);auto*g=ggml_new_graph(ctx);ggml_build_forward_expand(g,out);std::vector<float>baseline(count);
 for(int nth:{1,2,4,8}){if(timed&&nth!=8)continue;auto params=ggml_threadpool_params_default(nth);params.poll=0;auto*pool=ggml_threadpool_new(&params);if(!pool)fail("pool");auto plan=ggml_graph_plan(g,nth,pool);if(plan.n_threads!=nth)fail("plan threads");std::vector<uint8_t>work(plan.work_size);plan.work_data=work.data();compute(g,plan,false);std::memcpy(baseline.data(),out->data,count*4);compute(g,plan,true);
 if(std::memcmp(baseline.data(),out->data,count*4)||std::memcmp(original.data(),src->data,count*4))fail("bitwise output/source differs");for(size_t i=0;i<count;++i)if(!std::isfinite(((float*)out->data)[i])||((float*)out->data)[i]<0)fail("nonfinite or negative probability");
 std::printf("CHECK id=%d ne=%lld,%lld,%lld,%lld scale=%.9g mask=%d threads=%d mode=%s bitwise=PASS\n",id,(long long)a,(long long)b,(long long)c,(long long)d,scale,mask_type,nth,timed?"actual":"validation");
 if(!timed&&nth==8){save(folder,id,"input",src->data,count);save(folder,id,"baseline",baseline.data(),count);save(folder,id,"candidate",out->data,count);}
 if(timed){for(int rep=0;rep<6;++rep)for(int arm=0;arm<4;++arm){bool candidate=arm==1||arm==2;double seconds=compute(g,plan,candidate);if(std::memcmp(baseline.data(),out->data,count*4)||std::memcmp(original.data(),src->data,count*4))fail("timed output/source differs");std::printf("TIME id=%d rep=%d arm=%d mode=%s threads=8 seconds=%.12f\n",id,rep,arm,candidate?"candidate":"baseline",seconds);}}
 ggml_threadpool_free(pool);}ggml_free(ctx);}
int main(int argc,char**argv){if(argc!=3)return 2;setenv("SD_VE_SOFTMAX","1",1);setenv("SD_SOFTMAX_SHAPE_PROFILE","0",1);setenv("SD_OP_PROFILE","0",1);int id=0;for(int n:{64,77,256,1024,4096})for(float scale:{1.0f,.1f,.000244140625f})for(int mask:{0,1,2})run(argv[2],id++,n,17,1,1,scale,mask,false);std::ifstream f(argv[1]);int shape;int64_t a,b,c,d;int actual=0;while(f>>shape>>a>>b>>c>>d){if(shape!=actual)fail("shape order");run(argv[2],100+actual++,a,b,c,d,1,0,true);}if(id!=45||actual!=10)fail("fixture count");std::puts("SOFTMAX_GRAPH_BATCH_PASS validation_cases=45 validation_checks=180 actual_shapes=10 timings=240");return 0;}
