#define _POSIX_C_SOURCE 200809L
#include "transformer.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(VT_NLC_ATTENTION) && !defined(VT_USE_NLC)
#error VT_NLC_ATTENTION requires VT_USE_NLC
#endif
#ifdef VT_USE_NLC
#include <cblas.h>
#include <limits.h>
#endif

static void linear(const float *a, const float *b, float *c,
                   size_t m, size_t k, size_t n) {
#ifdef VT_USE_NLC
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                (cblas_int_t)m,(cblas_int_t)n,(cblas_int_t)k,
                1.0f,a,(cblas_int_t)k,b,(cblas_int_t)n,0.0f,c,(cblas_int_t)n);
#else
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) c[i*n+j] = 0;
        for (size_t p = 0; p < k; ++p) {
            const float x = a[i*k+p];
            /* Contiguous output axis permits vectorization on VE. */
            for (size_t j = 0; j < n; ++j) c[i*n+j] += x*b[p*n+j];
        }
    }
#endif
}
static void norm(const float *x, float *y, size_t t, size_t d,
                 const float *gain, const float *bias, float epsilon) {
    for (size_t i = 0; i < t; ++i) {
        float mean = 0, var = 0;
        for (size_t j = 0; j < d; ++j) mean += x[i*d+j];
        mean /= (float)d;
        for (size_t j = 0; j < d; ++j) {
            float z = x[i*d+j]-mean; var += z*z;
        }
        const float inv = 1.0f/sqrtf(var/(float)d+epsilon);
        for (size_t j = 0; j < d; ++j)
            y[i*d+j] = (x[i*d+j]-mean)*inv*gain[j]+bias[j];
    }
}
static void add_bias(float *x, const float *bias, size_t t, size_t d) {
    if (!bias) return;
    for (size_t i=0; i<t; ++i)
        for (size_t j=0; j<d; ++j) x[i*d+j]+=bias[j];
}
static double tick(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC,&ts)) return NAN;
    return (double)ts.tv_sec+ts.tv_nsec*1e-9;
}
/* Opaque fixed-shape scratch owner; no shared global state. */
struct vt_workspace {
    size_t tokens, width, heads, hidden;
    float *memory;
    int variable_tokens;
};

static int scratch_count(const vt_config *c, size_t *count) {
    if (!c || !c->tokens || !c->width || !c->heads || !c->hidden ||
        c->width%c->heads || (c->causal!=0 && c->causal!=1)) return -1;
    size_t t=c->tokens, d=c->width, f=c->hidden;
#ifdef VT_USE_NLC
    if (t>INT_MAX || d>INT_MAX || f>INT_MAX) return -1;
#endif
    const size_t cap=SIZE_MAX/sizeof(float);
    if (t>cap/d || t>cap/f || d>cap/d || d>cap/f) return -1;
    size_t td=t*d, tf=t*f, scores=t;
#ifdef VT_NLC_ATTENTION
    if (t>cap/t) return -1;
    scores=t*t;
#endif
    if (td>cap/7 || tf>cap-7*td || scores>cap-7*td-tf) return -1;
    *count=7*td+tf+scores;
    return 0;
}

int vt_workspace_create(const vt_config *c, vt_workspace **result) {
    if (!result) return -1;
    *result=NULL;
    size_t count;
    if (scratch_count(c,&count)) return -1;
    vt_workspace *ws=malloc(sizeof(*ws));
    if (!ws) return -2;
    ws->memory=malloc(count*sizeof(float));
    if (!ws->memory) { free(ws); return -2; }
    ws->variable_tokens=0;
    ws->tokens=c->tokens; ws->width=c->width;
    ws->heads=c->heads; ws->hidden=c->hidden;
    *result=ws;
    return 0;
}

int vt_workspace_create_capacity(const vt_config *c, size_t max_tokens,
                                 vt_workspace **result) {
    if (!result) return -1;
    *result=NULL;
    size_t count;
    if (scratch_count(c,&count) || max_tokens<c->tokens) return -1;
    vt_config capacity=*c;
    capacity.tokens=max_tokens;
    int rc=vt_workspace_create(&capacity,result);
    if (!rc) (*result)->variable_tokens=1;
    return rc;
}

void vt_workspace_destroy(vt_workspace *ws) {
    if (ws) { free(ws->memory); free(ws); }
}

static int forward(const vt_config *c, const vt_weights *w,
                   const float *x, float *y, vt_workspace *ws, vt_profile *profile,
                   const vt_layer_options *options) {
    double previous=0;
    if (profile) { memset(profile,0,sizeof(*profile)); previous=tick(); }
#define MARK(field) do { if (profile) { double current=tick(); \
    profile->field=(current-previous)*1000; previous=current; } } while (0)
    size_t count;
    if (scratch_count(c,&count) || !w || !x || !y || x==y ||
        !w->q || !w->k || !w->v || !w->out || !w->up || !w->down ||
        !w->gain1 || !w->bias1 || !w->gain2 || !w->bias2) return -1;
    if (ws && ((ws->variable_tokens ? c->tokens>ws->tokens : ws->tokens!=c->tokens) ||
               ws->width!=c->width ||
               ws->heads!=c->heads || ws->hidden!=c->hidden)) return -1;
    if (options && (options->activation>1 || options->unscaled_attention>1 ||
        (options->window && !c->causal) || !isfinite(options->norm_epsilon) ||
        options->norm_epsilon<=0 || options->norm_epsilon>=1)) return -1;
    float epsilon=options ? options->norm_epsilon : 1e-5f;
    size_t window=options ? options->window : 0;
    size_t t=c->tokens, d=c->width, f=c->hidden, h=c->heads, hd=d/h;
    size_t td=t*d, tf=t*f;
    float *mem=ws ? ws->memory : malloc(count*sizeof(float));
    if (!mem) return -2;
    float *n=mem, *q=n+td, *k=q+td, *v=k+td, *a=v+td;
    float *r=a+td, *z=r+td, *ff=z+td, *scores=ff+tf;
#ifndef VT_NLC_ATTENTION
    /* Only the baseline attention accumulates into existing scratch values.
       Every other buffer is completely overwritten before it is read. */
    memset(a,0,td*sizeof(float));
#endif
    MARK(allocation_ms);
    norm(x,n,t,d,w->gain1,w->bias1,epsilon);
    MARK(norm1_ms);
    linear(n,w->q,q,t,d,d); linear(n,w->k,k,t,d,d);
    linear(n,w->v,v,t,d,d);
    if (options) {
        add_bias(q,options->q_bias,t,d); add_bias(k,options->k_bias,t,d);
        add_bias(v,options->v_bias,t,d);
    }
    MARK(qkv_ms);
    const float scale=options && options->unscaled_attention ? 1.0f : 1.0f/sqrtf((float)hd);
#ifdef VT_NLC_ATTENTION
    for (size_t head=0; head<h; ++head) {
        /* Leading dimensions preserve interleaved heads without packing. */
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                    (cblas_int_t)t,(cblas_int_t)t,(cblas_int_t)hd,scale,
                    q+head*hd,(cblas_int_t)d,k+head*hd,(cblas_int_t)d,
                    0.0f,scores,(cblas_int_t)t);
        for (size_t i=0; i<t; ++i) {
            float *row=scores+i*t;
            size_t limit=c->causal ? i+1 : t;
            size_t first=window && i+1>window ? i+1-window : 0;
            float maxscore=-INFINITY,sum=0;
            for (size_t j=first; j<limit; ++j)
                if (row[j]>maxscore) maxscore=row[j];
            for (size_t j=first; j<limit; ++j) {
                row[j]=expf(row[j]-maxscore); sum+=row[j];
            }
            for (size_t j=first; j<limit; ++j) row[j]/=sum;
            for (size_t j=0; j<first; ++j) row[j]=0;
            /* Masked columns must be zero before probability x V. */
            for (size_t j=limit; j<t; ++j) row[j]=0;
        }
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                    (cblas_int_t)t,(cblas_int_t)hd,(cblas_int_t)t,1.0f,
                    scores,(cblas_int_t)t,v+head*hd,(cblas_int_t)d,
                    0.0f,a+head*hd,(cblas_int_t)d);
    }
#else
    for (size_t head=0; head<h; ++head) for (size_t i=0; i<t; ++i) {
        size_t limit=c->causal ? i+1 : t;
        size_t first=window && i+1>window ? i+1-window : 0;
        float maxscore=-INFINITY, sum=0;
        for (size_t j=first; j<limit; ++j) {
            float s=0;
            for (size_t p=0; p<hd; ++p)
                s+=q[i*d+head*hd+p]*k[j*d+head*hd+p];
            scores[j]=s*scale;
            if (scores[j]>maxscore) maxscore=scores[j];
        }
        for (size_t j=first; j<limit; ++j) {
            scores[j]=expf(scores[j]-maxscore); sum+=scores[j];
        }
        for (size_t j=first; j<limit; ++j) {
            float prob=scores[j]/sum;
            for (size_t p=0; p<hd; ++p)
                a[i*d+head*hd+p]+=prob*v[j*d+head*hd+p];
        }
    }
#endif
    MARK(attention_ms);
    linear(a,w->out,r,t,d,d);
    if (options) add_bias(r,options->out_bias,t,d);
    for (size_t i=0; i<td; ++i) r[i]+=x[i];
    MARK(projection_ms);
    norm(r,z,t,d,w->gain2,w->bias2,epsilon);
    MARK(norm2_ms);
    linear(z,w->up,ff,t,d,f);
    if (options) add_bias(ff,options->up_bias,t,f);
    if (options && options->activation==1) {
        for (size_t i=0; i<tf; ++i) {
            float value=ff[i];
            ff[i]=0.5f*value*(1.0f+tanhf(0.7978845608028654f*
                                      (value+0.044715f*value*value*value)));
        }
    } else for (size_t i=0; i<tf; ++i) if (ff[i]<0) ff[i]=0;
    linear(ff,w->down,y,t,f,d);
    if (options) add_bias(y,options->down_bias,t,d);
    for (size_t i=0; i<td; ++i) y[i]+=r[i];
    MARK(ffn_ms);
    if (!ws) free(mem);
    MARK(release_ms);
#undef MARK
    return 0;
}
int vt_forward_profile(const vt_config *c, const vt_weights *w,
                       const float *x, float *y, vt_profile *profile) {
    return forward(c,w,x,y,NULL,profile,NULL);
}
int vt_forward(const vt_config *c, const vt_weights *w, const float *x, float *y) {
    return forward(c,w,x,y,NULL,NULL,NULL);
}
int vt_forward_workspace(const vt_config *c, const vt_weights *w,
                         const float *x, float *y, vt_workspace *ws,
                         vt_profile *profile) {
    if (!ws) return -1;
    return forward(c,w,x,y,ws,profile,NULL);
}
int vt_forward_extended(const vt_config *c, const vt_weights *w,
                        const vt_layer_options *options,
                        const float *x, float *y, vt_workspace *ws, vt_profile *profile) {
    return forward(c,w,x,y,ws,profile,options);
}
