#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
 int cases=0;
 for(int kind=0;kind<4;++kind) for(int multiply=0;multiply<2;++multiply) for(int inplace=0;inplace<2;++inplace) {
  const int W=7,H=3,C=6,N=2,BC=kind==1?8:C;
  auto ctx=ggml_init({1024*1024,nullptr,true});auto backend=ggml_backend_cpu_init();
  if(!ctx||!backend)return 2;ggml_backend_cpu_set_n_threads(backend,kind==3 && inplace?1:8);
  ggml_backend_t backends[]={backend};auto sched=ggml_backend_sched_new(backends,nullptr,1,GGML_DEFAULT_GRAPH_SIZE,false);
  auto base=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,W,H,BC,N);
  auto input=kind==1?ggml_view_4d(ctx,base,W,H,C,N,base->nb[1],base->nb[2],base->nb[3],0):base;
  auto bias=kind==3?ggml_view_4d(ctx,base,1,1,3,1,base->nb[1],base->nb[2],base->nb[3],0):
                   ggml_new_tensor_4d(ctx,GGML_TYPE_F32,kind==2?W:1,kind==2?H:1,3,1);
  auto out=multiply ? (inplace?ggml_mul_inplace(ctx,input,bias):ggml_mul(ctx,input,bias)) :
                       (inplace?ggml_add_inplace(ctx,input,bias):ggml_add(ctx,input,bias));
  ggml_set_input(base);ggml_set_input(bias);ggml_set_output(out);
  if(!inplace) ggml_set_output(base); // Require allocator to preserve this input.
  ggml_set_output(bias);
  auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,out);
  if(!ggml_backend_sched_alloc_graph(sched,graph))return 3;
  std::vector<float> x(ggml_nelements(base)),b(ggml_nelements(bias)),actual(ggml_nelements(out));
  for(size_t i=0;i<x.size();++i)x[i]=kind==3?0.0f:float(int(i%67)-33)*0.125f;
  for(size_t i=0;i<b.size();++i)b[i]=kind==3?0.0f:float(int(i%13)-6)*0.25f;
  ggml_backend_tensor_set(base,x.data(),0,x.size()*4);
  if(kind!=3) ggml_backend_tensor_set(bias,b.data(),0,b.size()*4);
  if(ggml_backend_sched_graph_compute(sched,graph)!=GGML_STATUS_SUCCESS)return 4;
  // Get each row separately: the in-place fallback intentionally has gaps.
  for(int n=0;n<N;++n)for(int c=0;c<C;++c)for(int y=0;y<H;++y)
   ggml_backend_tensor_get(out,actual.data()+((n*C+c)*H+y)*W,n*out->nb[3]+c*out->nb[2]+y*out->nb[1],W*4);
  for(int n=0;n<N;++n)for(int c=0;c<C;++c)for(int y=0;y<H;++y)for(int xx=0;xx<W;++xx) {
   const float value=b[kind==2?(c%3)*W*H+y*W+xx:c%3];
   const float source=x[((n*BC+c)*H+y)*W+xx];
   const float expected=multiply?source*value:source+value;
   if(std::memcmp(&expected,&actual[((n*C+c)*H+y)*W+xx],4))return 5;
  }
  std::vector<float> bias_after(b.size());ggml_backend_tensor_get(bias,bias_after.data(),0,b.size()*4);
  if(std::memcmp(b.data(),bias_after.data(),b.size()*4))return 6;
  if(!inplace) {
   std::vector<float> unchanged(x.size());ggml_backend_tensor_get(base,unchanged.data(),0,x.size()*4);
   if(std::memcmp(x.data(),unchanged.data(),x.size()*4))return 7;
  }
  std::printf("BINARY_GRAPH_CASE kind=%d multiply=%d inplace=%d independent_bitwise PASS\n",kind,multiply,inplace);
  ggml_backend_sched_free(sched);ggml_backend_free(backend);ggml_free(ctx);++cases;
 }
 std::printf("BINARY_GRAPH_PASS cases=%d\n",cases);return 0;
}
