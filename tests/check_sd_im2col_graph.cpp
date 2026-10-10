#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstring>
#include <vector>
int main() {
 for(int c=0;c<3;++c) {
  const bool two=c!=1;const int IW=two?5:11,IH=two?7:1,IC=3,N=2,BC=c==2?4:IC,KW=3,KH=two?3:1;
  const int padding=two?2:1,dilation=two?2:1;
  auto *ctx=ggml_init({32*1024*1024,nullptr,true});auto backend=ggml_backend_cpu_init();
  if(!ctx||!backend)return 2;ggml_backend_cpu_set_n_threads(backend,8);
  ggml_backend_t backends[]={backend};auto sched=ggml_backend_sched_new(backends,nullptr,1,GGML_DEFAULT_GRAPH_SIZE,false);
  if(!sched)return 3;
  auto *base=two?ggml_new_tensor_4d(ctx,GGML_TYPE_F32,IW,IH,BC,N):ggml_new_tensor_3d(ctx,GGML_TYPE_F32,IW,IC,N);
  auto *src=c==2?ggml_view_4d(ctx,base,IW,IH,IC,N,base->nb[1],base->nb[2],base->nb[3],0):base;
  auto *kernel=two?ggml_new_tensor_4d(ctx,GGML_TYPE_F32,KW,KH,IC,2):ggml_new_tensor_3d(ctx,GGML_TYPE_F32,KW,IC,2);
  auto *out=ggml_im2col(ctx,kernel,src,1,1,padding,two?padding:0,dilation,1==c?1:dilation,two,GGML_TYPE_F32);
  ggml_set_input(base);ggml_set_input(kernel);ggml_set_output(out);
  auto *graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,out);
  if(!ggml_backend_sched_alloc_graph(sched,graph))return 4;
  std::vector<float> input(ggml_nelements(base)),weights(ggml_nelements(kernel),0.0f);
  for(size_t i=0;i<input.size();++i)input[i]=float(int(i%73)-36)/8.0f;
  ggml_backend_tensor_set(base,input.data(),0,input.size()*4);ggml_backend_tensor_set(kernel,weights.data(),0,weights.size()*4);
  if(ggml_backend_sched_graph_compute(sched,graph)!=GGML_STATUS_SUCCESS)return 5;
  std::vector<float> result(ggml_nelements(out)),ref(result.size(),0.0f),unchanged(input.size());
  ggml_backend_tensor_get(out,result.data(),0,result.size()*4);ggml_backend_tensor_get(base,unchanged.data(),0,unchanged.size()*4);
  const int OW=int(out->ne[1]),OH=two?int(out->ne[2]):1;
  for(int n=0;n<N;++n)for(int y=0;y<OH;++y)for(int x=0;x<OW;++x)for(int ic=0;ic<IC;++ic)for(int ky=0;ky<KH;++ky)for(int kx=0;kx<KW;++kx) {
   const int ix=x+kx*dilation-padding,iy=two?y+ky*dilation-padding:0;
   if(ix>=0&&ix<IW&&iy>=0&&iy<IH)ref[(((size_t)n*OH+y)*OW+x)*(IC*KH*KW)+(ic*KH+ky)*KW+kx]=input[(((size_t)n*BC+ic)*IH+iy)*IW+ix];
  }
  if(std::memcmp(ref.data(),result.data(),ref.size()*4)||std::memcmp(input.data(),unchanged.data(),input.size()*4))return 6;
  std::printf("im2col graph case=%d independent_bitwise_input_unchanged PASS\n",c);
  ggml_backend_sched_free(sched);ggml_backend_free(backend);ggml_free(ctx);
 }
 std::printf("IM2COL_GRAPH_PASS cases=3\n");return 0;
}
