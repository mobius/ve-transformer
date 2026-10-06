// A bounded float32 execution path for ggml dense and selected-expert projections.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#ifdef QWEN_ACCUM_FP64
#include "qwen_dense_cache.h"
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#endif
#ifdef QWEN_NLC
#include <cblas.h>
#endif
extern "C" {
void ve_dequant_rows(ggml_type,const void *,float *,int,int,uint32_t *);
bool __real_ggml_cpu_extra_compute_forward(ggml_compute_params *,ggml_tensor *);
}
namespace {
#ifdef QWEN_ACCUM_FP64
using DenseKey=std::tuple<uintptr_t,int,int,int>;
using DenseTile=std::shared_ptr<const std::vector<double>>;
struct DenseCache {
    std::mutex mutex;
    std::map<DenseKey,DenseTile> tiles;
    QwenDenseCacheStats stats{};
};
DenseCache dense_cache;
DenseTile find_dense(const DenseKey &key) {
    std::lock_guard<std::mutex> lock(dense_cache.mutex);
    if(!dense_cache.stats.budget_bytes)return {};
    auto it=dense_cache.tiles.find(key);
    if(it==dense_cache.tiles.end()) {++dense_cache.stats.misses;return {};}
    ++dense_cache.stats.hits;return it->second;
}
DenseTile retain_dense(const DenseKey &key,const float *matrix,size_t count) {
    std::unique_lock<std::mutex> lock(dense_cache.mutex);
    if(!dense_cache.stats.budget_bytes)return {};
    auto found=dense_cache.tiles.find(key);
    if(found!=dense_cache.tiles.end())return found->second;
    size_t bytes=count*sizeof(double);
    if(bytes>dense_cache.stats.budget_bytes-dense_cache.stats.retained_bytes-dense_cache.stats.reserved_bytes) {
        ++dense_cache.stats.rejected;return {};
    }
    // Reserve before unlocking so concurrent fills cannot exceed the payload cap.
    dense_cache.stats.reserved_bytes+=bytes;
    lock.unlock();
    DenseTile tile;
    try {
        auto filled=std::make_shared<std::vector<double>>(count);
        for(size_t k=0;k<count;++k)(*filled)[k]=matrix[k];
        tile=std::move(filled);
    } catch(const std::bad_alloc &) {
        lock.lock();dense_cache.stats.reserved_bytes-=bytes;
        ++dense_cache.stats.rejected;return {};
    }
    lock.lock();
    found=dense_cache.tiles.find(key);
    if(found!=dense_cache.tiles.end()) {
        tile.reset();dense_cache.stats.reserved_bytes-=bytes;return found->second;
    }
    try {
        dense_cache.tiles.emplace(key,tile);
        dense_cache.stats.reserved_bytes-=bytes;
        dense_cache.stats.retained_bytes+=bytes;
        dense_cache.stats.entries=dense_cache.tiles.size();return tile;
    } catch(const std::bad_alloc &) {
        tile.reset();dense_cache.stats.reserved_bytes-=bytes;
        ++dense_cache.stats.rejected;return {};
    }
}
#endif
struct Workspace {std::vector<float> a,b,c;std::vector<uint32_t> packed;
#ifdef QWEN_ACCUM_FP64
    std::vector<double> da,db,dc;
#endif
};
thread_local Workspace workspace;
struct Column {int route,token;};
bool supported(const ggml_tensor *dst) {
    if(dst->op!=GGML_OP_MUL_MAT && dst->op!=GGML_OP_MUL_MAT_ID) return false;
    const auto *a=dst->src[0],*b=dst->src[1];
    if(!a||!b||b->type!=GGML_TYPE_F32||dst->type!=GGML_TYPE_F32) return false;
    if(a->type!=GGML_TYPE_F32 && a->type!=GGML_TYPE_Q4_K && a->type!=GGML_TYPE_Q5_K &&
       a->type!=GGML_TYPE_Q6_K && a->type!=GGML_TYPE_Q8_0) return false;
    if(a->ne[0]<256 || a->ne[0]>16384 || a->ne[0]%256 || a->ne[1]<32 || a->ne[1]>300000 ||
       a->ne[3]!=1 || b->ne[3]!=1 || b->nb[0]!=sizeof(float) || dst->nb[0]!=sizeof(float) ||
       a->nb[1]!=ggml_row_size(a->type,a->ne[0]) || b->ne[0]!=a->ne[0]) return false;
    if(dst->op==GGML_OP_MUL_MAT) return a->ne[2]==1 && b->ne[2]==1 && b->ne[1]<=512;
    const auto *ids=dst->src[2];
    return ids && ids->type==GGML_TYPE_I32 && ids->ne[0]>0 && ids->ne[0]<=256 &&
        ids->ne[1]>0 && ids->ne[1]<=512 && a->ne[2]<=256 && b->ne[2]==ids->ne[1] &&
        (b->ne[1]==1 || b->ne[1]==ids->ne[0]) && ids->nb[0]==sizeof(int32_t);
}
void multiply(int rows,int cols,int width,const float *a,const float *b,float *c,const double *cached=nullptr) {
#ifdef QWEN_ACCUM_FP64
#ifdef QWEN_NLC
    auto &w=workspace;
    if(!cached) {
        w.da.resize((size_t)rows*width);
        for(size_t k=0;k<w.da.size();++k)w.da[k]=a[k];
    }
    w.db.resize((size_t)cols*width);w.dc.resize((size_t)rows*cols);
    const double *matrix=cached?cached:w.da.data();
    for(size_t k=0;k<w.db.size();++k)w.db[k]=b[k];
    if(cols==1)cblas_dgemv(CblasRowMajor,CblasNoTrans,rows,width,1,matrix,width,w.db.data(),1,0,w.dc.data(),1);
    else cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,rows,cols,width,1,matrix,width,w.db.data(),width,0,w.dc.data(),cols);
    for(size_t k=0;k<w.dc.size();++k)c[k]=(float)w.dc[k];
#else
    for(int r=0;r<rows;++r)for(int col=0;col<cols;++col) {
        double sum=0;
        for(int k=0;k<width;++k)sum+=(cached?cached[(size_t)r*width+k]:(double)a[(size_t)r*width+k])*(double)b[(size_t)col*width+k];
        c[r*cols+col]=(float)sum;
    }
#endif
#elif defined(QWEN_NLC)
    if(cols==1) cblas_sgemv(CblasRowMajor,CblasNoTrans,rows,width,1,a,width,b,1,0,c,1);
    else cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,rows,cols,width,1,a,width,b,width,0,c,cols);
#else
    auto dot=ggml_get_type_traits_cpu(GGML_TYPE_F32)->vec_dot;
    for(int r=0;r<rows;++r) for(int col=0;col<cols;++col)
        dot(width,c+r*cols+col,0,a+r*width,0,b+col*width,0,1);
#endif
}

#ifdef QWEN_ACCUM_FP64
bool precise_unmasked_softmax(ggml_compute_params *params,ggml_tensor *dst) {
    if(dst->op!=GGML_OP_SOFT_MAX || dst->src[1] || dst->src[2])return false;
    auto *src=dst->src[0];
    if(!src || src->type!=GGML_TYPE_F32 || dst->type!=GGML_TYPE_F32 ||
       src->nb[0]!=sizeof(float) || dst->nb[0]!=sizeof(float) || !ggml_are_same_shape(src,dst))return false;
    float scale=0,bias=0;
    std::memcpy(&scale,dst->op_params,sizeof(scale));
    std::memcpy(&bias,(const char*)dst->op_params+sizeof(float),sizeof(bias));
    if(bias!=0)return false;
    auto &w=workspace;w.da.resize(dst->ne[0]);
    for(int64_t row=params->ith;row<ggml_nrows(dst);row+=params->nth) {
        int64_t i1=row%dst->ne[1],i2=(row/dst->ne[1])%dst->ne[2],i3=row/(dst->ne[1]*dst->ne[2]);
        const float *x=(const float*)((const char*)src->data+i1*src->nb[1]+i2*src->nb[2]+i3*src->nb[3]);
        float *y=(float*)((char*)dst->data+i1*dst->nb[1]+i2*dst->nb[2]+i3*dst->nb[3]);
        double maximum=-INFINITY;
        for(int64_t k=0;k<dst->ne[0];++k)maximum=std::max(maximum,(double)x[k]*scale);
        double sum=0;
        for(int64_t k=0;k<dst->ne[0];++k) {w.da[k]=std::exp((double)x[k]*scale-maximum);sum+=w.da[k];}
        for(int64_t k=0;k<dst->ne[0];++k)y[k]=(float)(w.da[k]/sum);
    }
    return true;
}
bool precise_delta_net(ggml_compute_params *params,ggml_tensor *dst) {
    if(dst->op!=GGML_OP_GATED_DELTA_NET || dst->type!=GGML_TYPE_F32 || !ggml_is_contiguous(dst))return false;
    for(int n=0;n<6;++n)if(!dst->src[n] || dst->src[n]->type!=GGML_TYPE_F32)return false;
    auto *q=dst->src[0],*k=dst->src[1],*v=dst->src[2],*g=dst->src[3],*beta=dst->src[4],*initial=dst->src[5];
    int64_t S=v->ne[0],H=v->ne[1],N=v->ne[2],B=v->ne[3];
    int32_t K=0;std::memcpy(&K,dst->op_params,sizeof(K));
    if(S<1 || S>256 || H<1 || N<1 || K<1 ||
       !ggml_is_contiguous_rows(q) || !ggml_is_contiguous_rows(k) || !ggml_is_contiguous_rows(v) ||
       !ggml_is_contiguous(g) || !ggml_is_contiguous(beta) || !ggml_is_contiguous(initial) ||
       q->ne[0]!=S || k->ne[0]!=S || q->ne[2]!=N || k->ne[2]!=N ||
       H%q->ne[1] || H%k->ne[1] || B%q->ne[3] || B%k->ne[3] ||
       (g->ne[0]!=1 && g->ne[0]!=S) || beta->ne[0]!=1 ||
       initial->ne[0]!=S || initial->ne[1]!=S || initial->ne[2]!=H || initial->ne[3]!=B)return false;
    for(int d=1;d<4;++d)if(g->ne[d]!=v->ne[d] || beta->ne[d]!=v->ne[d])return false;
    if(ggml_nelements(dst)!=S*H*N*B+(int64_t)K*S*S*H*B)return false;
    auto &w=workspace;w.a.resize(S*S);w.b.resize(S);w.da.resize(S);
    float *out=(float*)dst->data;const int64_t attn=S*H*N*B,snapshot=S*S*H*B;
    const double scale=1/std::sqrt((double)S);
    for(int64_t head=params->ith;head<H*B;head+=params->nth) {
        int64_t h=head%H,b=head/H;
        std::memcpy(w.a.data(),(const float*)initial->data+head*S*S,S*S*sizeof(float));
        for(int64_t t=0;t<N;++t) {
            const float *qd=(const float*)((const char*)q->data+(b/(B/q->ne[3]))*q->nb[3]+t*q->nb[2]+(h%q->ne[1])*q->nb[1]);
            const float *kd=(const float*)((const char*)k->data+(b/(B/k->ne[3]))*k->nb[3]+t*k->nb[2]+(h%k->ne[1])*k->nb[1]);
            const float *vd=(const float*)((const char*)v->data+b*v->nb[3]+t*v->nb[2]+h*v->nb[1]);
            const float *gd=(const float*)((const char*)g->data+b*g->nb[3]+t*g->nb[2]+h*g->nb[1]);
            float bv=*(const float*)((const char*)beta->data+b*beta->nb[3]+t*beta->nb[2]+h*beta->nb[1]);
            for(int64_t i=0;i<S;++i)w.da[i]=std::exp((double)gd[g->ne[0]==1?0:i]);
            for(int64_t j=0;j<S;++j) {
                double sum=0;
                for(int64_t i=0;i<S;++i) {
                    float &state=w.a[j*S+i];state=(float)((double)state*w.da[i]);
                    sum+=(double)state*kd[i];
                }
                w.b[j]=(float)(((double)vd[j]-sum)*bv);
            }
            for(int64_t j=0;j<S;++j) {
                double sum=0;
                for(int64_t i=0;i<S;++i) {
                    float &state=w.a[j*S+i];
                    // Double product and sum round once on storage to F32.
                    state=(float)((double)state+(double)kd[i]*w.b[j]);
                    sum+=(double)state*qd[i];
                }
                out[((b*N+t)*H+h)*S+j]=(float)(sum*scale);
            }
            int64_t slot=N-1-t;
            if(slot<K)std::memcpy(out+attn+slot*snapshot+head*S*S,w.a.data(),S*S*sizeof(float));
        }
    }
    return true;
}
bool precise_convolution(ggml_compute_params *params,ggml_tensor *dst) {
    if(dst->op!=GGML_OP_SSM_CONV)return false;
    auto *x=dst->src[0],*c=dst->src[1];
    if(!x || !c || x->type!=GGML_TYPE_F32 || c->type!=GGML_TYPE_F32 || dst->type!=GGML_TYPE_F32 ||
       x->nb[0]!=sizeof(float) || c->nb[0]!=sizeof(float) || dst->nb[0]!=sizeof(float) ||
       x->ne[3]!=1 || c->ne[2]!=1 || c->ne[3]!=1 || dst->ne[3]!=1 ||
       c->ne[0]<1 || c->ne[0]>128 || x->ne[0]!=c->ne[0]-1+dst->ne[1] ||
       x->ne[1]!=dst->ne[0] || c->ne[1]!=dst->ne[0] || x->ne[2]!=dst->ne[2])return false;
    for(int64_t sequence=0;sequence<dst->ne[2];++sequence)
        for(int64_t channel=params->ith;channel<dst->ne[0];channel+=params->nth) {
            const float *weights=(const float*)((const char*)c->data+channel*c->nb[1]);
            const float *input=(const float*)((const char*)x->data+sequence*x->nb[2]+channel*x->nb[1]);
            for(int64_t token=0;token<dst->ne[1];++token) {
                double sum=0;
                for(int64_t k=0;k<c->ne[0];++k)sum+=(double)input[token+k]*(double)weights[k];
                float *out=(float*)((char*)dst->data+sequence*dst->nb[2]+token*dst->nb[1]);
                out[channel]=(float)sum;
            }
        }
    return true;
}
bool precise_elementwise(ggml_compute_params *params,ggml_tensor *dst) {
    auto *src=dst->src[0];
    if(!src || src->type!=GGML_TYPE_F32 || dst->type!=GGML_TYPE_F32 ||
       src->nb[0]!=sizeof(float) || dst->nb[0]!=sizeof(float) || !ggml_are_same_shape(src,dst))return false;
    if(dst->op!=GGML_OP_RMS_NORM && dst->op!=GGML_OP_UNARY)return false;
    auto unary=dst->op==GGML_OP_UNARY?ggml_get_unary_op(dst):GGML_UNARY_OP_ABS;
    if(dst->op==GGML_OP_UNARY && unary!=GGML_UNARY_OP_SILU &&
       unary!=GGML_UNARY_OP_SIGMOID && unary!=GGML_UNARY_OP_SOFTPLUS)return false;
    float epsilon=0;
    if(dst->op==GGML_OP_RMS_NORM)std::memcpy(&epsilon,dst->op_params,sizeof(epsilon));
    for(int64_t row=params->ith;row<ggml_nrows(dst);row+=params->nth) {
        int64_t i1=row%dst->ne[1],i2=(row/dst->ne[1])%dst->ne[2],i3=row/(dst->ne[1]*dst->ne[2]);
        const float *x=(const float*)((const char*)src->data+i1*src->nb[1]+i2*src->nb[2]+i3*src->nb[3]);
        float *y=(float*)((char*)dst->data+i1*dst->nb[1]+i2*dst->nb[2]+i3*dst->nb[3]);
        if(dst->op==GGML_OP_RMS_NORM) {
            double sum=0;for(int64_t k=0;k<dst->ne[0];++k)sum+=(double)x[k]*(double)x[k];
            float scale=(float)(1/std::sqrt(sum/dst->ne[0]+epsilon));
            for(int64_t k=0;k<dst->ne[0];++k)y[k]=x[k]*scale;
        } else {
            for(int64_t k=0;k<dst->ne[0];++k) {
                double v=x[k];
                if(unary==GGML_UNARY_OP_SOFTPLUS)y[k]=(float)(v>30?v:std::log1p(std::exp(v)));
                else {
                    double sigmoid=v>=0?1/(1+std::exp(-v)):std::exp(v)/(1+std::exp(v));
                    y[k]=(float)(unary==GGML_UNARY_OP_SILU?v*sigmoid:sigmoid);
                }
            }
        }
    }
    return true;
}
#endif
void execute(ggml_compute_params *params,ggml_tensor *dst) {
    const auto *a=dst->src[0],*b=dst->src[1];
    int width=a->ne[0],m=a->ne[1];
    int begin=m*params->ith/params->nth,end=m*(params->ith+1)/params->nth;
    if(begin==end)return;
    const bool moe=dst->op==GGML_OP_MUL_MAT_ID;
    const auto *ids=moe?dst->src[2]:nullptr;
    int experts=moe?a->ne[2]:1,used=moe?ids->ne[0]:1,tokens=moe?ids->ne[1]:b->ne[1];
    std::vector<Column> columns;columns.reserve(tokens*used);
    auto &w=workspace;
    for(int expert=0;expert<experts;++expert) {
        columns.clear();
        for(int t=0;t<tokens;++t) for(int route=0;route<used;++route) {
            int chosen=0;
            if(moe) std::memcpy(&chosen,(char*)ids->data+t*ids->nb[1]+route*ids->nb[0],sizeof(chosen));
            GGML_ASSERT(chosen>=0 && chosen<experts);
            if(chosen==expert) columns.push_back({route,t});
        }
        if(columns.empty())continue;
        for(int row=begin;row<end;row+=128) {
            int rows=std::min(128,end-row);
            const char *raw=(const char*)a->data+expert*a->nb[2]+row*a->nb[1];
            const float *matrix=nullptr;
            const double *cached_matrix=nullptr;
#ifdef QWEN_ACCUM_FP64
            DenseTile tile;
            const bool cacheable=!moe && a->op==GGML_OP_NONE;
            DenseKey key((uintptr_t)raw,(int)a->type,width,rows);
            if(cacheable)tile=find_dense(key);
            if(tile)cached_matrix=tile->data();
#endif
            if(!cached_matrix) {
            if(a->type==GGML_TYPE_F32)matrix=(const float*)raw;
            else {
                w.a.resize(rows*width);w.packed.resize((rows*a->nb[1]+3)/4);
                ve_dequant_rows(a->type,raw,w.a.data(),rows,width,w.packed.data());matrix=w.a.data();
            }
#ifdef QWEN_ACCUM_FP64
            if(cacheable) {
                tile=retain_dense(key,matrix,(size_t)rows*width);
                if(tile)cached_matrix=tile->data();
            }
#endif
            }
            for(size_t first=0;first<columns.size();first+=128) {
                int nc=std::min<size_t>(128,columns.size()-first);
                w.b.resize(width*nc);w.c.resize(rows*nc);
                for(int col=0;col<nc;++col) {
                    auto position=columns[first+col];
                    size_t offset=moe?(position.route%b->ne[1])*b->nb[1]+position.token*b->nb[2]:position.token*b->nb[1];
                    std::memcpy(w.b.data()+col*width,(const char*)b->data+offset,width*sizeof(float));
                }
                multiply(rows,nc,width,matrix,w.b.data(),w.c.data(),cached_matrix);
                for(int col=0;col<nc;++col) {
                    auto position=columns[first+col];
                    size_t offset=moe?position.route*dst->nb[1]+position.token*dst->nb[2]:position.token*dst->nb[1];
                    float *output=(float*)((char*)dst->data+offset)+row;
                    for(int r=0;r<rows;++r)output[r]=w.c[r*nc+col];
                }
            }
        }
    }
}
}
extern "C" bool __wrap_ggml_cpu_extra_compute_forward(ggml_compute_params *params,ggml_tensor *dst) {
#ifdef QWEN_ACCUM_FP64
    if(precise_unmasked_softmax(params,dst))return true;
    if(precise_delta_net(params,dst))return true;
    if(precise_convolution(params,dst))return true;
    if(precise_elementwise(params,dst))return true;
#endif
    if(!supported(dst))return __real_ggml_cpu_extra_compute_forward(params,dst);
    execute(params,dst);return true;
}
#ifdef QWEN_ACCUM_FP64
extern "C" void qwen_dense_cache_configure(uint64_t bytes) {
    std::lock_guard<std::mutex> lock(dense_cache.mutex);
    GGML_ASSERT(dense_cache.stats.reserved_bytes==0);
    dense_cache.tiles.clear();dense_cache.stats={};dense_cache.stats.budget_bytes=bytes;
}
extern "C" QwenDenseCacheStats qwen_dense_cache_stats() {
    std::lock_guard<std::mutex> lock(dense_cache.mutex);return dense_cache.stats;
}
extern "C" void __wrap_ggml_vec_dot_f32(int n,float *s,size_t bs,const float *x,size_t bx,const float *y,size_t by,int nrc) {
    GGML_ASSERT(nrc==1);(void)bs;(void)bx;(void)by;
    double sum=0;for(int k=0;k<n;++k)sum+=(double)x[k]*(double)y[k];
    *s=(float)sum;
}
#endif
