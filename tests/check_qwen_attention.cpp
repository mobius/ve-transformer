// Independent long-double oracle for the F16 attention path and GGML head mapping.
#include "ggml.h"
#include "ggml-cpu-impl.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
extern "C" bool __wrap_ggml_cpu_extra_compute_forward(ggml_compute_params *,ggml_tensor *);
static int check(int width,int rows,int cols,int ah,int bh,int ab,int bb,int layout,int threads) {
    auto *ctx=ggml_init({1024*1024,nullptr,true});if(!ctx)return 1;
    auto *a=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,width,rows,ah,ab);
    auto *b=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,width,cols,bh,bb);
    auto *d=ggml_mul_mat(ctx,a,b);
    if(layout) {
        int stride=layout==2?2:1;
        a->nb[0]=2*stride;a->nb[1]=(width*stride+5)*2;a->nb[2]=a->nb[1]*rows+16;a->nb[3]=a->nb[2]*ah+16;
        b->nb[0]=4*stride;b->nb[1]=(width*stride+3)*4;b->nb[2]=b->nb[1]*cols+16;b->nb[3]=b->nb[2]*bh+32;
        d->nb[1]=(rows+3)*4;d->nb[2]=d->nb[1]*cols+20;d->nb[3]=d->nb[2]*bh+32;
    }
    std::vector<unsigned char> av(a->nb[3]*ab,0),bv(b->nb[3]*bb,0);
    std::vector<float> out(d->nb[3]*bb/4,12345);
    a->data=av.data();b->data=bv.data();d->data=out.data();
    for(int n=0;n<ab;++n)for(int h=0;h<ah;++h)for(int r=0;r<rows;++r)for(int k=0;k<width;++k) {
        float f=((k*13+r*17+h*19+n*23)%103-51)*0.00713f;
        ggml_fp16_t v=ggml_fp32_to_fp16(f);
        std::memcpy(av.data()+n*a->nb[3]+h*a->nb[2]+r*a->nb[1]+k*a->nb[0],&v,2);
    }
    for(int n=0;n<bb;++n)for(int h=0;h<bh;++h)for(int c=0;c<cols;++c)for(int k=0;k<width;++k) {
        float f=((k*7+c*19+h*29+n*31)%97-48)*0.00917f;
        std::memcpy(bv.data()+n*b->nb[3]+h*b->nb[2]+c*b->nb[1]+k*b->nb[0],&f,4);
    }
#ifdef QWEN_ATTENTION_PREFILL_ONLY
    if(cols==1) {
        for(int i=0;i<threads;++i) {
            ggml_compute_params p={};p.ith=i;p.nth=threads;
            if(__wrap_ggml_cpu_extra_compute_forward(&p,d))return 1;
        }
        for(float value:out)if(value!=12345)return 1;
        std::printf("attention single-column fallback threads=%d PASS\n",threads);
        ggml_free(ctx);return 0;
    }
#endif
    int failed=0;
#pragma omp parallel for num_threads(threads) reduction(+:failed)
    for(int i=0;i<threads;++i) {
        ggml_compute_params p={};p.ith=i;p.nth=threads;
        if(!__wrap_ggml_cpu_extra_compute_forward(&p,d))++failed;
    }
    double maximum=0;std::vector<bool> written(out.size(),false);
    for(int n=0;n<bb;++n)for(int h=0;h<bh;++h)for(int c=0;c<cols;++c)for(int r=0;r<rows;++r) {
        long double sum=0;
        for(int k=0;k<width;++k) {
            ggml_fp16_t x;float y;
            std::memcpy(&x,av.data()+(n/(bb/ab))*a->nb[3]+(h/(bh/ah))*a->nb[2]+r*a->nb[1]+k*a->nb[0],2);
            std::memcpy(&y,bv.data()+n*b->nb[3]+h*b->nb[2]+c*b->nb[1]+k*b->nb[0],4);
            sum+=(long double)ggml_fp16_to_fp32(x)*ggml_fp16_to_fp32(ggml_fp32_to_fp16(y));
        }
        size_t index=(n*d->nb[3]+h*d->nb[2]+c*d->nb[1]+r*d->nb[0])/4;written[index]=true;
        double error=(double)std::abs((long double)out[index]-sum);maximum=std::max(maximum,error);
        if(!std::isfinite(out[index]) || error>3e-6+3e-7*std::abs(sum))++failed;
    }
    for(size_t i=0;i<out.size();++i)if(!written[i] && out[i]!=12345)++failed;
    std::printf("attention width=%d rows=%d cols=%d heads=%d/%d batch=%d/%d layout=%d threads=%d error=%.9g %s\n",width,rows,cols,ah,bh,ab,bb,layout,threads,maximum,failed?"FAIL":"PASS");
    ggml_free(ctx);return failed?1:0;
}
int main() {
    int cases=0;
    for(int shape=0;shape<3;++shape)for(int cols:{1,17})for(int layout:{0,1,2})for(int threads:{1,3,4,8})for(int group=0;group<3;++group) {
        int width=shape==0?31:shape==1?128:256,rows=shape==0?19:shape==1?256:128;
        int ah=group==0?1:group==1?2:3,bh=group==0?1:12,ab=group==2?2:1,bb=group==0?1:2;
        if(check(width,rows,cols,ah,bh,ab,bb,layout,threads))return 1;
        ++cases;
    }
    std::printf("ATTENTION_ORACLE_PASS cases=%d\n",cases);return 0;
}
