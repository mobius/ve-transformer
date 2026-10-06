#define _POSIX_C_SOURCE 200809L
#include "transformer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>
#include <string.h>

static size_t number(const char *s, size_t limit) {
    char *end;
    errno=0;
    if (!s[0] || s[0]<'0' || s[0]>'9') return 0;
    unsigned long value=strtoul(s,&end,10);
    if (errno || *end || !value || value>limit) return 0;
    return (size_t)value;
}
static int compare(const void *a, const void *b) {
    double x=*(const double *)a, y=*(const double *)b;
    return (x>y)-(x<y);
}

static double now(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC,&ts)) exit(2);
    return (double)ts.tv_sec+ts.tv_nsec*1e-9;
}
int main(int argc, char **argv) {
    /* Bounded configurable workload; single sequence, single layer. */
    vt_config c={32,64,4,128,1};
    size_t repeats=50;
    int profiling=0, reuse=0;
    while (argc>1) {
        if (!strcmp(argv[argc-1],"--profile")) profiling=1;
        else if (!strcmp(argv[argc-1],"--workspace")) reuse=1;
        else break;
        --argc;
    }
    if (argc!=1 && argc!=6) {
        fprintf(stderr,"usage: %s [tokens width heads hidden repeats] [--profile] [--workspace]\n",argv[0]);
        return 2;
    }
    if (argc==6) {
        c.tokens=number(argv[1],1024); c.width=number(argv[2],2048);
        c.heads=number(argv[3],2048); c.hidden=number(argv[4],8192);
        repeats=number(argv[5],1000);
        if (!c.tokens || !c.width || !c.heads || !c.hidden || !repeats ||
            c.width%c.heads) {
            fprintf(stderr,"invalid shape or repetitions\n"); return 2;
        }
    }
    size_t dd=c.width*c.width, df=c.width*c.hidden, td=c.tokens*c.width;
    float *mem=calloc(4*dd+2*df+2*c.width+2*td,sizeof(float));
    if (!mem) return 1;
    float *q=mem,*k=q+dd,*v=k+dd,*o=v+dd,*up=o+dd,*down=up+df;
    float *gain=down+df,*bias=gain+c.width,*x=bias+c.width,*y=x+td;
    for (size_t i=0;i<4*dd+2*df;++i) mem[i]=(float)((int)(i%17)-8)/128;
    for (size_t i=0;i<c.width;++i) gain[i]=1;
    for (size_t i=0;i<td;++i) x[i]=(float)((int)(i%19)-9)/16;
    vt_weights w={q,k,v,o,up,down,gain,bias,gain,bias};
    vt_workspace *ws=NULL;
    double setup=now();
    if (reuse && vt_workspace_create(&c,&ws)) { free(mem); return 1; }
    double setup_ms=(now()-setup)*1000;
    for (int i=0;i<5;++i) {
        int rc=reuse ? vt_forward_workspace(&c,&w,x,y,ws,NULL) : vt_forward(&c,&w,x,y);
        if (rc) { vt_workspace_destroy(ws); free(mem); return 1; }
    }
    double *samples=calloc(repeats,sizeof(double));
    if (!samples) { free(mem); return 1; }
    double start=now();
    vt_profile sums={0};
    for (size_t i=0;i<repeats;++i) {
        double one=now();
        vt_profile p;
        if (profiling) {
            int rc=reuse ? vt_forward_workspace(&c,&w,x,y,ws,&p) :
                           vt_forward_profile(&c,&w,x,y,&p);
            if (rc) return 1;
            sums.allocation_ms+=p.allocation_ms; sums.norm1_ms+=p.norm1_ms;
            sums.qkv_ms+=p.qkv_ms; sums.attention_ms+=p.attention_ms;
            sums.projection_ms+=p.projection_ms; sums.norm2_ms+=p.norm2_ms;
            sums.ffn_ms+=p.ffn_ms; sums.release_ms+=p.release_ms;
        } else {
            int rc=reuse ? vt_forward_workspace(&c,&w,x,y,ws,NULL) : vt_forward(&c,&w,x,y);
            if (rc) return 1;
        }
        samples[i]=(now()-one)*1000;
    }
    double elapsed=now()-start;
    double checksum=0;
    for (size_t i=0;i<td;++i) {
        if (!isfinite(y[i])) return 1;
        checksum+=y[i];
    }
    if (elapsed<=0) return 1;
    qsort(samples,repeats,sizeof(double),compare);
    size_t p50=(repeats+1)/2-1, p95=(95*repeats+99)/100-1;
    printf("tokens=%zu width=%zu heads=%zu hidden=%zu causal=1 warmup=5 repeats=%zu\n"
           "mean_latency_ms=%.6f p50_latency_ms=%.6f p95_latency_ms=%.6f "
           "sequence_tokens_per_second=%.3f checksum=%.9g\n",
           c.tokens,c.width,c.heads,c.hidden,repeats,elapsed*1000/repeats,
           samples[p50],samples[p95],(double)c.tokens*repeats/elapsed,checksum);
    printf("workspace_reuse=%d workspace_create_ms=%.6f\n",reuse,reuse ? setup_ms : 0.0);
    if (profiling) printf("profile_mean_ms allocation=%.6f norm1=%.6f qkv=%.6f "
                         "attention=%.6f projection=%.6f norm2=%.6f ffn=%.6f release=%.6f\n",
                         sums.allocation_ms/repeats,sums.norm1_ms/repeats,sums.qkv_ms/repeats,
                         sums.attention_ms/repeats,sums.projection_ms/repeats,sums.norm2_ms/repeats,
                         sums.ffn_ms/repeats,sums.release_ms/repeats);
    vt_workspace_destroy(ws); free(samples); free(mem); return 0;
}
