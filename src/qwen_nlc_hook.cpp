#if defined(QWEN_REUSE_INPUT) && (!defined(QWEN_ACCUM_FP64) || !defined(QWEN_NLC))
#error "Input reuse requires the FP64 NLC path"
#endif
// A bounded float32 execution path for ggml dense and selected-expert projections.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#ifdef QWEN_PROFILE
#include <cstdio>
#endif
#if defined(QWEN_TIMING) || defined(QWEN_PROFILE)
#include <chrono>
#endif
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
#ifdef QWEN_PROFILE
// Diagnostic-only: each GGML worker owns its slot. Snapshots require quiescence.
struct ProfileCounts { uint64_t lock_wait=0,lookup=0,pack=0,blas=0,cast=0,scatter=0,weight=0; };
struct Profile {
    ProfileCounts timers;
    double lock_wait=0,lookup=0,pack=0,blas=0,cast=0,scatter=0,weight=0;
    uint64_t lookups=0,multiplies=0,sampled_lookups=0,sampled_multiplies=0,cursor=0;
};
struct alignas(128) ProfileSlot { Profile bins[2]; };
ProfileSlot profile_slots[256];
thread_local Profile *active_profile=nullptr;
thread_local bool profile_sample=false;
#ifndef QWEN_PROFILE_SAMPLE_BITS
#define QWEN_PROFILE_SAMPLE_BITS 6
#endif
static_assert(QWEN_PROFILE_SAMPLE_BITS>=0 && QWEN_PROFILE_SAMPLE_BITS<=12,"invalid sample rate");
void profile_draw() {
    if(!active_profile) {profile_sample=false;return;}
    // Deterministic dispersed sampling; separate counters per worker and column bin.
    auto hash=++active_profile->cursor*UINT64_C(11400714819323198485);
    profile_sample=QWEN_PROFILE_SAMPLE_BITS==0 || (hash>>(64-(QWEN_PROFILE_SAMPLE_BITS?QWEN_PROFILE_SAMPLE_BITS:1)))==0;
}
using ProfileClock=std::chrono::steady_clock;
struct ProfileTimer {
    double *counter;
    ProfileClock::time_point start;
    explicit ProfileTimer(double *value,uint64_t *count):counter(value),start(value?ProfileClock::now():ProfileClock::time_point{}) {if(count)++*count;}
    ~ProfileTimer() {if(counter)*counter+=std::chrono::duration<double>(ProfileClock::now()-start).count();}
};
void profile_worker(const ggml_compute_params *params,int cols) {
    GGML_ASSERT(params->ith>=0 && params->ith<256 && params->nth<=256);
    active_profile=&profile_slots[params->ith].bins[cols==1?0:1];
    if(!active_profile->cursor)active_profile->cursor=UINT64_C(0xd1b54a32d192ed03)*(params->ith+1);
}
#define PROFILE_SCOPE(field) ProfileTimer profile_timer(active_profile && profile_sample?&active_profile->field:nullptr,active_profile && profile_sample?&active_profile->timers.field:nullptr)
#else
#define PROFILE_SCOPE(field) ((void)0)
#endif
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
#ifdef QWEN_PROFILE
    auto wait_start=profile_sample?ProfileClock::now():ProfileClock::time_point{};
#endif
    std::lock_guard<std::mutex> lock(dense_cache.mutex);
#ifdef QWEN_PROFILE
    if(active_profile) {
        ++active_profile->lookups;
        if(profile_sample) {
            active_profile->lock_wait+=std::chrono::duration<double>(ProfileClock::now()-wait_start).count();
            ++active_profile->sampled_lookups;
            ++active_profile->timers.lock_wait;
        }
    }
    PROFILE_SCOPE(lookup);
#endif
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
#ifdef QWEN_TIMING
    auto fill_start=std::chrono::steady_clock::now();
#endif
    try {
        auto filled=std::make_shared<std::vector<double>>(count);
        for(size_t k=0;k<count;++k)(*filled)[k]=matrix[k];
        tile=std::move(filled);
    } catch(const std::bad_alloc &) {
        lock.lock();dense_cache.stats.reserved_bytes-=bytes;
        ++dense_cache.stats.rejected;return {};
    }
    lock.lock();
#ifdef QWEN_TIMING
    dense_cache.stats.fill_thread_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-fill_start).count();
#endif
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
#ifdef QWEN_REUSE_INPUT
    std::vector<double> input_double;
#endif
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
void multiply(int rows,int cols,int width,const float *a,const float *b,float *c,const double *cached=nullptr
#ifdef QWEN_REUSE_INPUT
,const double *prepared_b=nullptr
#endif
) {
#ifdef QWEN_ACCUM_FP64
#ifdef QWEN_NLC
    auto &w=workspace;
    {
    PROFILE_SCOPE(pack);
    if(!cached) {
        w.da.resize((size_t)rows*width);
        for(size_t k=0;k<w.da.size();++k)w.da[k]=a[k];
    }
#ifdef QWEN_REUSE_INPUT
    w.dc.resize((size_t)rows*cols);
    if(!prepared_b) {
        w.db.resize((size_t)cols*width);
        for(size_t k=0;k<w.db.size();++k)w.db[k]=b[k];
    }
#else
    w.db.resize((size_t)cols*width);w.dc.resize((size_t)rows*cols);
    for(size_t k=0;k<w.db.size();++k)w.db[k]=b[k];
#endif
    }
    const double *matrix=cached?cached:w.da.data();
    {
    PROFILE_SCOPE(blas);
#ifdef QWEN_PROFILE
    if(active_profile) {++active_profile->multiplies;if(profile_sample)++active_profile->sampled_multiplies;}
#endif
#ifdef QWEN_REUSE_INPUT
    const double *input=prepared_b?prepared_b:w.db.data();
#else
    const double *input=w.db.data();
#endif
    if(cols==1)cblas_dgemv(CblasRowMajor,CblasNoTrans,rows,width,1,matrix,width,input,1,0,w.dc.data(),1);
    else cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,rows,cols,width,1,matrix,width,input,width,0,w.dc.data(),cols);
    }
    {
    PROFILE_SCOPE(cast);
    for(size_t k=0;k<w.dc.size();++k)c[k]=(float)w.dc[k];
    }
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

#ifdef QWEN_ATTENTION_NLC
// Match GGML's F16 dot input rounding, including grouped head/batch broadcasting.
bool attention_multiply(ggml_compute_params *params,ggml_tensor *dst) {
    if(dst->op!=GGML_OP_MUL_MAT)return false;
    const auto *a=dst->src[0],*b=dst->src[1];
    if(!a || !b || a->type!=GGML_TYPE_F16 || b->type!=GGML_TYPE_F32 || dst->type!=GGML_TYPE_F32 ||
       a->ne[0]!=b->ne[0] || a->ne[2]<1 || a->ne[3]<1 || b->ne[2]%a->ne[2] || b->ne[3]%a->ne[3] ||
       dst->ne[0]!=a->ne[1] || dst->ne[1]!=b->ne[1] || dst->ne[2]!=b->ne[2] || dst->ne[3]!=b->ne[3] ||
       dst->nb[0]!=sizeof(float))return false;
#ifdef QWEN_ATTENTION_PREFILL_ONLY
    if(b->ne[1]==1)return false;
#endif
    const int64_t width=a->ne[0],rows=a->ne[1],cols=b->ne[1];
#ifdef QWEN_PROFILE
    profile_worker(params,cols);
#endif
    if(width<1 || width>32768 || rows<1 || rows>32768 || cols<1 || cols>512 ||
       rows*width>8*1024*1024 || rows*cols>8*1024*1024)return false;
    auto &w=workspace;
    const int64_t r2=b->ne[2]/a->ne[2],r3=b->ne[3]/a->ne[3];
    for(int64_t batch=params->ith;batch<b->ne[2]*b->ne[3];batch+=params->nth) {
#ifdef QWEN_PROFILE
        profile_draw();
#endif
        int64_t h=batch%b->ne[2],n=batch/b->ne[2];
        const char *ap=(const char*)a->data+(h/r2)*a->nb[2]+(n/r3)*a->nb[3];
        const char *bp=(const char*)b->data+h*b->nb[2]+n*b->nb[3];
        w.a.resize(rows*width);w.b.resize(cols*width);w.c.resize(rows*cols);
        for(int64_t row=0;row<rows;++row) {
            if(a->nb[0]==sizeof(ggml_fp16_t))ggml_fp16_to_fp32_row((const ggml_fp16_t*)(ap+row*a->nb[1]),w.a.data()+row*width,width);
            else for(int64_t k=0;k<width;++k) {
                ggml_fp16_t value;std::memcpy(&value,ap+row*a->nb[1]+k*a->nb[0],sizeof(value));
                w.a[row*width+k]=ggml_fp16_to_fp32(value);
            }
        }
        for(int64_t col=0;col<cols;++col)for(int64_t k=0;k<width;++k) {
            float value;std::memcpy(&value,bp+col*b->nb[1]+k*b->nb[0],sizeof(value));
            w.b[col*width+k]=ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
        }
        multiply(rows,cols,width,w.a.data(),w.b.data(),w.c.data());
        char *out=(char*)dst->data+h*dst->nb[2]+n*dst->nb[3];
        for(int64_t col=0;col<cols;++col)for(int64_t row=0;row<rows;++row) {
            float value=w.c[row*cols+col];
            std::memcpy(out+col*dst->nb[1]+row*dst->nb[0],&value,sizeof(value));
        }
    }
    return true;
}
#endif

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
#ifdef QWEN_PROJECTION_TILE_ROWS
    constexpr int tile_rows=QWEN_PROJECTION_TILE_ROWS;
    static_assert(tile_rows>0 && tile_rows<=4096,"invalid projection tile rows");
#else
    constexpr int tile_rows=128;
#endif
    int width=a->ne[0],m=a->ne[1];
    int begin=m*params->ith/params->nth,end=m*(params->ith+1)/params->nth;
    if(begin==end)return;
    const bool moe=dst->op==GGML_OP_MUL_MAT_ID;
    const auto *ids=moe?dst->src[2]:nullptr;
    int experts=moe?a->ne[2]:1,used=moe?ids->ne[0]:1,tokens=moe?ids->ne[1]:b->ne[1];
    std::vector<Column> columns;columns.reserve(tokens*used);
    auto &w=workspace;
#ifdef QWEN_PROFILE
    profile_worker(params,tokens);
#endif
    for(int expert=0;expert<experts;++expert) {
        columns.clear();
        for(int t=0;t<tokens;++t) for(int route=0;route<used;++route) {
            int chosen=0;
            if(moe) std::memcpy(&chosen,(char*)ids->data+t*ids->nb[1]+route*ids->nb[0],sizeof(chosen));
            GGML_ASSERT(chosen>=0 && chosen<experts);
            if(chosen==expert) columns.push_back({route,t});
        }
        if(columns.empty())continue;
#ifdef QWEN_REUSE_INPUT
        const double *prepared_input=nullptr;
        if(!moe) {
            // Operator-local values, overwritten on every invocation; no activation cache.
            // Supported dense shapes cap this workspace at 64 MiB per worker.
            GGML_ASSERT(columns.size()<=512 && width<=16384);
#ifdef QWEN_PROFILE
            profile_draw();
#endif
            PROFILE_SCOPE(pack);
            w.input_double.resize((size_t)width*columns.size());
            for(size_t col=0;col<columns.size();++col) {
                const float *input=(const float*)((const char*)b->data+columns[col].token*b->nb[1]);
                for(int k=0;k<width;++k)w.input_double[col*width+k]=(double)input[k];
            }
            prepared_input=w.input_double.data();
        }
#endif
        for(int row=begin;row<end;row+=tile_rows) {
#ifdef QWEN_PROFILE
            profile_draw();
#endif
            int rows=std::min(tile_rows,end-row);
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
            PROFILE_SCOPE(weight);
            if(a->type==GGML_TYPE_F32)matrix=(const float*)raw;
            else {
                w.a.resize(rows*width);
                const int decode_rows=std::min(rows,128);
                w.packed.resize((decode_rows*a->nb[1]+3)/4);
                for(int decoded=0;decoded<rows;decoded+=128) {
                    const int chunk=std::min(128,rows-decoded);
                    ve_dequant_rows(a->type,raw+decoded*a->nb[1],w.a.data()+decoded*width,chunk,width,w.packed.data());
                }
                matrix=w.a.data();
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
                {
                PROFILE_SCOPE(pack);
                w.c.resize(rows*nc);
#ifdef QWEN_REUSE_INPUT
                if(!prepared_input) {
#endif
                w.b.resize(width*nc);
                for(int col=0;col<nc;++col) {
                    auto position=columns[first+col];
                    size_t offset=moe?(position.route%b->ne[1])*b->nb[1]+position.token*b->nb[2]:position.token*b->nb[1];
                    std::memcpy(w.b.data()+col*width,(const char*)b->data+offset,width*sizeof(float));
                }
#ifdef QWEN_REUSE_INPUT
                }
#endif
                }
                multiply(rows,nc,width,matrix,w.b.data(),w.c.data(),cached_matrix
#ifdef QWEN_REUSE_INPUT
                    ,prepared_input?prepared_input+first*width:nullptr
#endif
                );
                {
                PROFILE_SCOPE(scatter);
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
}
extern "C" bool __wrap_ggml_cpu_extra_compute_forward(ggml_compute_params *params,ggml_tensor *dst) {
#ifdef QWEN_ATTENTION_NLC
    if(attention_multiply(params,dst))return true;
#endif
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
#ifdef QWEN_PROFILE
    for(auto &slot:profile_slots)for(auto &bin:slot.bins)bin=Profile{};
#endif
}
extern "C" QwenDenseCacheStats qwen_dense_cache_stats() {
    std::lock_guard<std::mutex> lock(dense_cache.mutex);
#ifdef QWEN_PROFILE
    // Called by the main thread only after processing has synchronized workers.
    double clock_samples[64];
    for(auto &sample:clock_samples) {
        auto a=ProfileClock::now();auto b=ProfileClock::now();
        sample=std::chrono::duration<double>(b-a).count();
    }
    std::sort(clock_samples,clock_samples+64);
    std::fprintf(stderr,"QWEN_CLOCK {\"min_seconds\":%.9f,\"median_seconds\":%.9f,\"max_seconds\":%.9f}\n",clock_samples[0],(clock_samples[31]+clock_samples[32])/2,clock_samples[63]);
    for(int bin=0;bin<2;++bin) {
        Profile sum;
        for(const auto &slot:profile_slots) {
            const auto &p=slot.bins[bin];
            sum.lock_wait+=p.lock_wait;sum.lookup+=p.lookup;sum.pack+=p.pack;
            sum.blas+=p.blas;sum.cast+=p.cast;sum.scatter+=p.scatter;sum.weight+=p.weight;
            sum.lookups+=p.lookups;sum.multiplies+=p.multiplies;
            sum.timers.lock_wait+=p.timers.lock_wait;sum.timers.lookup+=p.timers.lookup;
            sum.timers.pack+=p.timers.pack;sum.timers.blas+=p.timers.blas;
            sum.timers.cast+=p.timers.cast;sum.timers.scatter+=p.timers.scatter;sum.timers.weight+=p.timers.weight;
            sum.sampled_lookups+=p.sampled_lookups;sum.sampled_multiplies+=p.sampled_multiplies;
        }
        std::fprintf(stderr,"QWEN_PROFILE {\"columns\":%d,\"lock_wait\":%.9f,\"lookup\":%.9f,\"pack\":%.9f,\"blas\":%.9f,\"cast\":%.9f,\"scatter\":%.9f,\"weight\":%.9f,\"lookups\":%llu,\"multiplies\":%llu,\"sample_bits\":%d,\"sampled_lookups\":%llu,\"sampled_multiplies\":%llu,\"timer_counts\":{\"lock_wait\":%llu,\"lookup\":%llu,\"pack\":%llu,\"blas\":%llu,\"cast\":%llu,\"scatter\":%llu,\"weight\":%llu}}\n",bin==0?1:2,sum.lock_wait,sum.lookup,sum.pack,sum.blas,sum.cast,sum.scatter,sum.weight,(unsigned long long)sum.lookups,(unsigned long long)sum.multiplies,QWEN_PROFILE_SAMPLE_BITS,(unsigned long long)sum.sampled_lookups,(unsigned long long)sum.sampled_multiplies,(unsigned long long)sum.timers.lock_wait,(unsigned long long)sum.timers.lookup,(unsigned long long)sum.timers.pack,(unsigned long long)sum.timers.blas,(unsigned long long)sum.timers.cast,(unsigned long long)sum.timers.scatter,(unsigned long long)sum.timers.weight);
    }
#endif
    return dense_cache.stats;
}
extern "C" void __wrap_ggml_vec_dot_f32(int n,float *s,size_t bs,const float *x,size_t bx,const float *y,size_t by,int nrc) {
    GGML_ASSERT(nrc==1);(void)bs;(void)bx;(void)by;
    double sum=0;for(int k=0;k<n;++k)sum+=(double)x[k]*(double)y[k];
    *s=(float)sum;
}
#endif
