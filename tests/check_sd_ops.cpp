// Image operators on the same VE BLAS scheduler used by the image pipeline.
// Independent scalar FP64 formulae cover odd tails, padding and channel axes.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-blas.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

struct Graph {
    ggml_context *ctx;
    ggml_backend_t backends[2];
    ggml_backend_sched_t sched;
    Graph() {
        ctx=ggml_init({16*1024*1024,nullptr,true});
        backends[0]=ggml_backend_blas_init();
        backends[1]=ggml_backend_cpu_init();
        if (!ctx || !backends[0] || !backends[1]) throw std::runtime_error("backend init");
        ggml_backend_cpu_set_n_threads(backends[1],8);
        sched=ggml_backend_sched_new(backends,nullptr,2,GGML_DEFAULT_GRAPH_SIZE,false,true);
        if (!sched) throw std::runtime_error("scheduler init");
    }
    ~Graph() {
        ggml_backend_sched_free(sched);
        ggml_backend_free(backends[0]);ggml_backend_free(backends[1]);ggml_free(ctx);
    }
    std::vector<float> run(ggml_tensor *out,
                          const std::vector<std::pair<ggml_tensor*,std::vector<float>>>& inputs,
                          bool require_blas=false) {
        for (const auto &item:inputs) ggml_set_input(item.first);
        ggml_set_output(out);
        auto *graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,out);
        if (!ggml_backend_sched_alloc_graph(sched,graph)) throw std::runtime_error("allocation");
        if (require_blas && ggml_backend_sched_get_tensor_backend(sched,out)!=backends[0])
            throw std::runtime_error("matrix operation did not select BLAS");
        for (const auto &item:inputs) {
            if (ggml_nbytes(item.first)!=item.second.size()*sizeof(float)) throw std::runtime_error("input shape");
            ggml_backend_tensor_set(item.first,item.second.data(),0,ggml_nbytes(item.first));
        }
        if (ggml_backend_sched_graph_compute(sched,graph)!=GGML_STATUS_SUCCESS)
            throw std::runtime_error("compute");
        std::vector<float> output(ggml_nelements(out));
        ggml_backend_tensor_get(out,output.data(),0,output.size()*sizeof(float));
        return output;
    }
};

static std::vector<float> values(size_t count,int factor) {
    std::vector<float> result(count);
    for (size_t i=0;i<count;++i) result[i]=(int((i*factor+3)%97)-48)*.03125f;
    return result;
}
static void check(const char *name,const std::vector<float>& actual,const std::vector<double>& expected) {
    if (actual.size()!=expected.size()) throw std::runtime_error("output shape");
    double squared=0,denominator=0,maximum=0;
    for (size_t i=0;i<actual.size();++i) {
        if (!std::isfinite(actual[i])) throw std::runtime_error("nonfinite output");
        double delta=actual[i]-expected[i];squared+=delta*delta;denominator+=expected[i]*expected[i];
        maximum=std::max(maximum,std::abs(delta));
    }
    double relative=std::sqrt(squared/std::max(denominator,1e-24));
    std::printf("%s relative_l2=%.9g max_abs=%.9g %s\n",name,relative,maximum,relative<=1e-4?"PASS":"FAIL");
    if (relative>1e-4) throw std::runtime_error("operator tolerance");
}
static void matmul(int k,int m,int n) {
    Graph g;auto *a=ggml_new_tensor_2d(g.ctx,GGML_TYPE_F32,k,m);
    auto *b=ggml_new_tensor_2d(g.ctx,GGML_TYPE_F32,k,n);
    auto av=values(size_t(k)*m,13),bv=values(size_t(k)*n,17);
    auto *out=ggml_mul_mat(g.ctx,a,b);
    auto actual=g.run(out,{{a,av},{b,bv}},true);
    std::vector<double> expected(size_t(m)*n);
    for (int j=0;j<n;++j) for (int i=0;i<m;++i) {
        double sum=0;for(int p=0;p<k;++p) sum+=double(av[size_t(i)*k+p])*bv[size_t(j)*k+p];
        expected[size_t(j)*m+i]=sum;
    }
    check("NLC SGEMM",actual,expected);
}
static void conv(int width,int height,int cin,int cout,int stride) {
    Graph g;auto *a=ggml_new_tensor_4d(g.ctx,GGML_TYPE_F32,3,3,cin,cout);
    auto *b=ggml_new_tensor_4d(g.ctx,GGML_TYPE_F32,width,height,cin,1);
    auto av=values(9*cin*cout,13),bv=values(width*height*cin,17);
    auto *out=ggml_conv_2d(g.ctx,a,b,stride,stride,1,1,1,1);
    auto actual=g.run(out,{{a,av},{b,bv}});
    int ow=(width-1)/stride+1,oh=(height-1)/stride+1;
    std::vector<double> expected(ow*oh*cout);
    for (int c=0;c<cout;++c) for (int y=0;y<oh;++y) for (int x=0;x<ow;++x) {
        double sum=0;
        for(int ic=0;ic<cin;++ic) for(int ky=0;ky<3;++ky) for(int kx=0;kx<3;++kx) {
            int iy=y*stride+ky-1,ix=x*stride+kx-1;
            if (iy>=0 && iy<height && ix>=0 && ix<width)
                sum+=double(av[((c*cin+ic)*3+ky)*3+kx])*bv[(ic*height+iy)*width+ix];
        }
        expected[(c*oh+y)*ow+x]=sum;
    }
    check("FP32 convolution",actual,expected);
}
static void unary(int op) {
    Graph g;const int width=37,rows=13;
    auto *in=ggml_new_tensor_2d(g.ctx,GGML_TYPE_F32,width,rows);
    auto input=values(width*rows,17);ggml_tensor *out=nullptr;
    if(op==0) out=ggml_norm(g.ctx,in,1e-5f);
    if(op==1) out=ggml_silu(g.ctx,in);
    if(op==2) out=ggml_gelu_erf(g.ctx,in);
    if(op==3) out=ggml_soft_max(g.ctx,in);
    auto actual=g.run(out,{{in,input}});std::vector<double> expected(input.size());
    for (int row=0;row<rows;++row) {
        double mean=0,var=0,expsum=0,maximum=-1e30;
        for(int x=0;x<width;++x) {double v=input[row*width+x];mean+=v;maximum=std::max(maximum,v);}
        mean/=width;
        for(int x=0;x<width;++x) {double v=input[row*width+x];var+=(v-mean)*(v-mean);expsum+=std::exp(v-maximum);}
        var/=width;
        for(int x=0;x<width;++x) {
            double v=input[row*width+x],result=0;
            if(op==0) result=(v-mean)/std::sqrt(var+1e-5);
            if(op==1) result=v/(1+std::exp(-v));
            if(op==2) result=.5*v*(1+std::erf(v/std::sqrt(2.0)));
            if(op==3) result=std::exp(v-maximum)/expsum;
            expected[row*width+x]=result;
        }
    }
    const char *names[]={"layer normalization","SiLU","GELU","softmax"};check(names[op],actual,expected);
}
static void group_norm() {
    Graph g;const int w=7,h=5,c=12,groups=3,pergroup=w*h*c/groups;
    auto *in=ggml_new_tensor_4d(g.ctx,GGML_TYPE_F32,w,h,c,1);
    auto input=values(w*h*c,17);auto *out=ggml_group_norm(g.ctx,in,groups,1e-5f);
    auto actual=g.run(out,{{in,input}});std::vector<double> expected(input.size());
    for(int group=0;group<groups;++group) {
        double mean=0,var=0;
        for(int i=0;i<pergroup;++i)mean+=input[group*pergroup+i];mean/=pergroup;
        for(int i=0;i<pergroup;++i){double delta=input[group*pergroup+i]-mean;var+=delta*delta;}var/=pergroup;
        for(int i=0;i<pergroup;++i)expected[group*pergroup+i]=(input[group*pergroup+i]-mean)/std::sqrt(var+1e-5);
    }
    check("group normalization",actual,expected);
}
static void upscale() {
    Graph g;const int w=7,h=5,c=3;
    auto *in=ggml_new_tensor_4d(g.ctx,GGML_TYPE_F32,w,h,c,1);
    auto input=values(w*h*c,17);auto *out=ggml_upscale(g.ctx,in,2,GGML_SCALE_MODE_NEAREST);
    auto actual=g.run(out,{{in,input}});std::vector<double> expected(w*h*c*4);
    for(int channel=0;channel<c;++channel)for(int y=0;y<h*2;++y)for(int x=0;x<w*2;++x)
        expected[(channel*h*2+y)*w*2+x]=input[(channel*h+y/2)*w+x/2];
    check("nearest upscale",actual,expected);
}
int main() {
    try {
        matmul(257,65,77);matmul(1024,1024,77);
        conv(37,29,7,13,1);conv(37,29,7,13,2);conv(64,64,4,320,1);
        for(int op=0;op<4;++op)unary(op);
        group_norm();upscale();
        std::puts("IMAGE_OPERATOR_PASS cases=11; BLAS assignment asserted for matrix tests");
    } catch(const std::exception &e) {std::fprintf(stderr,"operator failure: %s\n",e.what());return 1;}
    return 0;
}
