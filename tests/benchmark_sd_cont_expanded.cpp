#include "ggml.h"
#include "ggml-cpu.h"
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <utility>
extern "C" int sd_ve_cont_transpose_f32(float*,const float*,size_t,size_t,int,int);
static const uint32_t sentinel=0x5a6b7c8d;
static uint32_t bits(const float *p) {uint32_t b;std::memcpy(&b,p,4);return b;}
static void put(float *p,uint32_t b) {std::memcpy(p,&b,4);}
static void fail(const char *why) {std::fprintf(stderr,"CONT_TEST_FAIL %s\n",why);std::exit(2);}
struct Shape {int id;int64_t ne[4];size_t nb[4];size_t m,n;};
static std::vector<float> input(size_t count) {
 std::vector<float> x(count+2);put(x.data(),sentinel);put(x.data()+count+1,sentinel);
 const uint32_t edges[]={0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x3f800000,0x7f800000,0xff800000,0x7f800001,0xff800001,0x7fc00001,0xffc12345,0x7fffffff,0xffffffff};
 uint32_t state=0x725312ab;
 for(size_t i=0;i<count;++i){state=state*1664525u+1013904223u;put(x.data()+1+i,i%31<15?edges[i%31]:state);}
 return x;
}
static std::vector<uint32_t> golden(const float *src,size_t m,size_t n) {
 std::vector<uint32_t> y(m*n);
 for(size_t j=0;j<n;++j)for(size_t i=0;i<m;++i)y[j*m+i]=bits(src+i*n+j);
 return y;
}
static void unchanged(const std::vector<float>&src,const std::vector<unsigned char>&before) {
 if(std::memcmp(src.data(),before.data(),before.size()))fail("source bytes/canaries modified");
}
static void direct_checks(FILE *checks,const Shape&s) {
 const size_t count=s.m*s.n;auto src=input(count);auto ref=golden(src.data()+1,s.m,s.n);
 std::vector<unsigned char> before(src.size()*4);std::memcpy(before.data(),src.data(),before.size());
 std::vector<float> dst(count+2);
 for(int nth:{1,2,4,8}) {
  const size_t dr=(s.n+nth-1)/nth;
  for(int ith=0;ith<nth;++ith) {
   for(size_t i=0;i<dst.size();++i)put(dst.data()+i,sentinel);
   if(!sd_ve_cont_transpose_f32(dst.data()+1,src.data()+1,s.m,s.n,ith,nth))fail("valid partition refused");
   const size_t first=dr*ith,last=std::min(first+dr,s.n);
   for(size_t i=0;i<count;++i) {
    const bool owned=i/s.m>=first && i/s.m<last;
    if(bits(dst.data()+1+i)!=(owned?ref[i]:sentinel))fail("partition ownership/bits differ");
   }
   if(bits(dst.data())!=sentinel||bits(dst.data()+count+1)!=sentinel)fail("destination canary modified");
   unchanged(src,before);
   std::fprintf(checks,"ownership\t%d\t%d\t%d\t%zu\t%zu\tPASS\n",s.id,nth,ith,first,last);
  }
  for(size_t i=0;i<dst.size();++i)put(dst.data()+i,sentinel);
  int seen[8]={0};int success[8]={0};
#pragma omp parallel num_threads(nth)
  {
   const int ith=omp_get_thread_num(),actual=omp_get_num_threads();
   if(ith>=8)fail("unexpected actual team");
   seen[ith]=actual;success[ith]=sd_ve_cont_transpose_f32(dst.data()+1,src.data()+1,s.m,s.n,ith,actual);
  }
  for(int i=0;i<nth;++i)if(seen[i]!=nth||success[i]!=1)fail("actual direct team/dispatch differs");
  if(std::memcmp(dst.data()+1,ref.data(),count*4)||bits(dst.data())!=sentinel||bits(dst.data()+count+1)!=sentinel)fail("concurrent copy/canaries differ");
  unchanged(src,before);std::fprintf(checks,"concurrent\t%d\t%d\tPASS\n",s.id,nth);
 }
}
struct Callback {size_t m,n;int seen[8];};
static void candidate(struct ggml_tensor*d,const struct ggml_tensor*a,int ith,int nth,void*opaque) {
 auto *c=static_cast<Callback*>(opaque);
 if(ith<0||ith>=8||nth>8)fail("unexpected graph callback team");
 c->seen[ith]=nth;
 if(!sd_ve_cont_transpose_f32(static_cast<float*>(d->data),static_cast<const float*>(a->data),c->m,c->n,ith,nth))fail("graph candidate refused");
}
static double compute(ggml_cgraph*g,ggml_cplan&plan,Callback*callback) {
 if(callback)std::fill(callback->seen,callback->seen+8,0);
 const auto start=std::chrono::steady_clock::now();
 if(ggml_graph_compute(g,&plan)!=GGML_STATUS_SUCCESS)fail("graph compute failed");
 const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 if(callback)for(int i=0;i<plan.n_threads;++i)if(callback->seen[i]!=plan.n_threads)fail("actual graph team differs");
 return elapsed;
}
static void graphs(FILE*checks,FILE*timings,const Shape&s,const char*folder) {
 const size_t count=s.m*s.n;auto src=input(count);auto ref=golden(src.data()+1,s.m,s.n);
 auto *ctx=ggml_init({count*16+8*1024*1024,nullptr,false});if(!ctx)fail("graph context allocation");
 auto *base=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,count+2);
 std::memcpy(base->data,src.data(),src.size()*4);
 auto *view=ggml_view_4d(ctx,base,s.ne[0],s.ne[1],s.ne[2],s.ne[3],s.nb[1],s.nb[2],s.nb[3],4);
 // ggml_view_4d fixes nb[0] to element size; transpose metadata must carry the actual first stride.
 view->nb[0]=s.nb[0];
 for(int i=0;i<4;++i)if(view->ne[i]!=s.ne[i]||view->nb[i]!=s.nb[i])fail("actual view geometry differs");
 auto *out_b=ggml_cont(ctx,view);Callback callback={s.m,s.n,{0}};
 auto *out_c=ggml_map_custom1(ctx,view,candidate,GGML_N_TASKS_MAX,&callback);
 auto *gb=ggml_new_graph(ctx),*gc=ggml_new_graph(ctx);ggml_build_forward_expand(gb,out_b);ggml_build_forward_expand(gc,out_c);
 for(int nth:{1,2,4,8}) {
  auto params=ggml_threadpool_params_default(nth);params.poll=0;auto pool=ggml_threadpool_new(&params);if(!pool)fail("graph pool allocation");
  auto pb=ggml_graph_plan(gb,nth,pool),pc=ggml_graph_plan(gc,nth,pool);
  std::vector<uint8_t> wb(pb.work_size),wc(pc.work_size);pb.work_data=wb.data();pc.work_data=wc.data();
  if(pb.n_threads!=nth||pc.n_threads!=nth)fail("graph planned team differs");
  auto check=[&](ggml_tensor*out,const char*mode) {
   if(std::memcmp(out->data,ref.data(),count*4)||std::memcmp(base->data,src.data(),src.size()*4))fail("actual graph output/source bytes differ");
   std::fprintf(checks,"graph\t%d\t%d\t%s\tPASS\n",s.id,nth,mode);
  };
  compute(gb,pb,nullptr);check(out_b,"baseline");compute(gc,pc,&callback);check(out_c,"candidate");
  if(nth==8) {
   for(int rep=0;rep<6;++rep)for(int arm=0;arm<4;++arm) {
    const bool c=arm==1||arm==2;const double seconds=compute(c?gc:gb,c?pc:pb,c?&callback:nullptr);
    if(std::memcmp((c?out_c:out_b)->data,ref.data(),count*4)||std::memcmp(base->data,src.data(),src.size()*4))fail("timed graph output/source differs");
    std::fprintf(timings,"%d\t%d\t%d\t%s\t8\t%.12f\n",s.id,rep,arm,c?"candidate":"baseline",seconds);
   }
   char path[1024];std::snprintf(path,sizeof path,"%s/shape%d-input.f32",folder,s.id);FILE*f=std::fopen(path,"wb");if(!f||std::fwrite(src.data()+1,4,count,f)!=count)fail("input output file");std::fclose(f);
   for(auto item:{std::make_pair(out_b,"baseline"),std::make_pair(out_c,"candidate")}) {
    std::snprintf(path,sizeof path,"%s/shape%d-%s.f32",folder,s.id,item.second);f=std::fopen(path,"wb");if(!f||std::fwrite(item.first->data,4,count,f)!=count)fail("graph output file");std::fclose(f);
   }
  }
  ggml_threadpool_free(pool);
 }
 ggml_free(ctx);
}
static void invalid_checks(FILE*checks) {
 auto src=input(64),dst=input(64);std::vector<unsigned char> before(src.size()*4),out_before(dst.size()*4);std::memcpy(before.data(),src.data(),before.size());std::memcpy(out_before.data(),dst.data(),out_before.size());
 struct Args {float*d;const float*s;size_t m,n;int ith,nth;};
 Args cases[]={{nullptr,src.data()+1,8,8,0,1},{dst.data()+1,nullptr,8,8,0,1},{dst.data()+1,src.data()+1,0,8,0,1},{dst.data()+1,src.data()+1,8,0,0,1},{dst.data()+1,src.data()+1,4097,8,0,1},{dst.data()+1,src.data()+1,8,4097,0,1},{dst.data()+1,src.data()+1,4096,4096,0,1},{dst.data()+1,src.data()+1,8,8,-1,1},{dst.data()+1,src.data()+1,8,8,1,1},{dst.data()+1,src.data()+1,8,8,0,0},{dst.data()+1,src.data()+1,8,8,0,3},{dst.data()+1,src.data()+1,8,8,0,9},{src.data()+1,src.data()+1,8,8,0,1},{src.data()+2,src.data()+1,8,8,0,1}};
 for(size_t i=0;i<sizeof(cases)/sizeof(cases[0]);++i) {
  auto a=cases[i];if(sd_ve_cont_transpose_f32(a.d,a.s,a.m,a.n,a.ith,a.nth)!=0)fail("invalid call accepted");
  unchanged(src,before);if(std::memcmp(dst.data(),out_before.data(),out_before.size()))fail("invalid call modified destination");
  std::fprintf(checks,"invalid\t%zu\tPASS\n",i);
 }
}
int main(int argc,char**argv) {
 if(argc!=3)return 2;std::ifstream in(argv[1]);std::vector<Shape> shapes;Shape s;
 while(in>>s.id>>s.ne[0]>>s.ne[1]>>s.ne[2]>>s.ne[3]>>s.nb[0]>>s.nb[1]>>s.nb[2]>>s.nb[3]){s.m=s.ne[0];s.n=s.ne[1]*s.ne[2]*s.ne[3];shapes.push_back(s);}
 if(shapes.size()!=9)fail("nine bound model shapes required");
 char path[1024];std::snprintf(path,sizeof path,"%s/checks.tsv",argv[2]);FILE*checks=std::fopen(path,"w");std::snprintf(path,sizeof path,"%s/timings.tsv",argv[2]);FILE*timings=std::fopen(path,"w");if(!checks||!timings)fail("report allocation");
 omp_set_dynamic(0);setenv("SD_CONT_SHAPE_PROFILE","0",1);setenv("SD_OP_PROFILE","0",1);
 for(const auto &r:shapes){direct_checks(checks,r);graphs(checks,timings,r,argv[2]);std::printf("CONT shape=%d m=%zu n=%zu graph/direct bitwise PASS\n",r.id,r.m,r.n);}
 int id=9;for(size_t n:{size_t(1),size_t(2),size_t(7),size_t(255),size_t(256),size_t(257),size_t(4095),size_t(4096)}) {Shape edge={id++,{17,int64_t(n),1,1},{n*4,4,n*17*4,n*17*4},17,n};direct_checks(checks,edge);}
 Shape tiny={id,{1,1,1,1},{4,4,4,4},1,1};direct_checks(checks,tiny);invalid_checks(checks);
 std::fclose(checks);std::fclose(timings);std::printf("CONT_TRANSPOSE_PASS real_shapes=9 direct_shapes=18 invalid=14 timing_rows=216\n");return 0;
}
