/* Single-sequence resident block executor. Weights load once; no network or
   external path is accepted from the request stream. All shapes are bounded. */
#define _POSIX_C_SOURCE 200809L
#include "transformer.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <signal.h>

#define MAX_LAYERS 32
#define MAX_PARAMETERS ((size_t)64*1024*1024)
typedef struct {
    vt_config config;
    vt_weights weights;
    vt_layer_options options;
    float *memory;
} layer;
static double now(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC,&t)) return NAN;
    return (double)t.tv_sec+t.tv_nsec*1e-9;
}
static int read_layer(FILE *file, layer *l, size_t width, size_t *total) {
    char magic[8]; uint32_t fields[8]; float epsilon;
    if (fread(magic,1,8,file)!=8 || memcmp(magic,"VTF32V2\0",8) ||
        fread(fields,sizeof(uint32_t),8,file)!=8 || fread(&epsilon,4,1,file)!=1) return -1;
    size_t t=fields[0], d=fields[1], h=fields[2], f=fields[3];
    if (t!=1 || d!=width || !h || h>d || d%h || !f || f>8192 ||
        fields[4]!=1 || fields[5]>1 || fields[6]>1 || fields[7]>1024 ||
        !isfinite(epsilon) || epsilon<=0 || epsilon>=1) return -1;
    size_t count=4*d*d+2*d*f+9*d+f+d;
    if (count>MAX_PARAMETERS-*total) return -1;
    *total+=count;
    l->memory=malloc(count*sizeof(float));
    if (!l->memory || fread(l->memory,sizeof(float),count,file)!=count) return -1;
    for (size_t i=0;i<count;++i) if (!isfinite(l->memory[i])) return -1;
    float *q=l->memory,*k=q+d*d,*v=k+d*d,*out=v+d*d;
    float *up=out+d*d,*down=up+d*f,*g1=down+d*f,*b1=g1+d,*g2=b1+d,*b2=g2+d;
    float *bias=b2+d;
    vt_weights weights={q,k,v,out,up,down,g1,b1,g2,b2};
    vt_layer_options options={bias,bias+d,bias+2*d,bias+3*d,bias+4*d,bias+4*d+f,
                              fields[5],fields[6],fields[7],epsilon};
    vt_config config={1,d,h,f,1};
    l->weights=weights; l->options=options; l->config=config;
    return 0;
}
int main(int argc, char **argv) {
    if (argc!=2) { fputs("usage: resident model.bundle\n",stderr); return 2; }
    uint32_t endian=1;
    if (*(unsigned char *)&endian!=1 || sizeof(float)!=4 || sizeof(double)!=8) return 2;
    signal(SIGPIPE,SIG_IGN);
    int rc=1;
    FILE *file=NULL;
    layer layers[MAX_LAYERS]={0};
    vt_workspace *workspace=NULL;
    float *buffers=NULL;
    char magic[8]; uint32_t fields[3];
    size_t count=0, width=0, capacity=0, parameters=0;
    double load_ms=0, load_start=now();
    file=fopen(argv[1],"rb");
    if (!file) goto cleanup;
    if (fread(magic,1,8,file)!=8 || memcmp(magic,"VTGPTN1\0",8) ||
        fread(fields,4,3,file)!=3) goto cleanup;
    count=fields[0]; width=fields[1]; capacity=fields[2];
    if (!count || count>MAX_LAYERS || !width || width>2048 || !capacity || capacity>1024)
        goto cleanup;
    for (size_t i=0;i<count;++i) {
        if (read_layer(file,&layers[i],width,&parameters)) goto cleanup;
        if (i && (layers[i].config.heads!=layers[0].config.heads ||
                  layers[i].config.hidden!=layers[0].config.hidden)) goto cleanup;
    }
    if (fgetc(file)!=EOF || ferror(file)) goto cleanup;
    fclose(file); file=NULL;
    if (vt_workspace_create_capacity(&layers[0].config,capacity,&workspace)) goto cleanup;
    buffers=malloc(2*capacity*width*sizeof(float));
    if (!buffers) goto cleanup;
    load_ms=(now()-load_start)*1000;
    if (!isfinite(load_ms) || fwrite("VTREADY1",1,8,stdout)!=8 ||
        fwrite(&load_ms,sizeof(double),1,stdout)!=1 || fflush(stdout)) goto cleanup;
    while (1) {
        uint32_t request[2];
        size_t bytes=fread(request,1,sizeof(request),stdin);
        if (!bytes && feof(stdin)) { rc=0; break; }
        if (bytes!=sizeof(request)) goto cleanup;
        size_t tokens=request[0]; unsigned last_only=request[1];
        if (!tokens) { if (last_only) goto cleanup; rc=0; break; }
        if (tokens>capacity || last_only>1) goto cleanup;
        size_t td=tokens*width;
        float *x=buffers,*y=buffers+capacity*width;
        if (fread(x,sizeof(float),td,stdin)!=td) goto cleanup;
        for (size_t i=0;i<td;++i) if (!isfinite(x[i])) goto cleanup;
        double start=now();
        for (size_t i=0;i<count;++i) {
            layers[i].config.tokens=tokens;
            if (vt_forward_extended(&layers[i].config,&layers[i].weights,&layers[i].options,
                                    x,y,workspace,NULL)) goto cleanup;
            float *swap=x; x=y; y=swap;
        }
        double compute_ms=(now()-start)*1000;
        if (!isfinite(compute_ms)) goto cleanup;
        for (size_t i=0;i<td;++i) if (!isfinite(x[i])) goto cleanup;
        uint32_t result[2]={0,last_only ? 1 : (uint32_t)tokens};
        size_t output_count=(size_t)result[1]*width;
        float *output=last_only ? x+(tokens-1)*width : x;
        if (fwrite("VTRES01\0",1,8,stdout)!=8 || fwrite(result,4,2,stdout)!=2 ||
            fwrite(&compute_ms,8,1,stdout)!=1 ||
            fwrite(output,4,output_count,stdout)!=output_count || fflush(stdout)) goto cleanup;
    }
cleanup:
    if (rc) fputs("resident: invalid model/request or execution failure\n",stderr);
    if (file) fclose(file);
    for (size_t i=0;i<MAX_LAYERS;++i) free(layers[i].memory);
    vt_workspace_destroy(workspace); free(buffers);
    return rc;
}
