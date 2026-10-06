/* Portable bounded v1/v2 interchange: v1 has 8-byte magic and 5 LE uint32s.
   v2 adds activation/unscaled/window uint32s and float32 norm epsilon, then
   six bias arrays after the usual weights and norms, before the input.
   This binary supports little-endian CPU/VE hosts only. */
#define _POSIX_C_SOURCE 200809L
#include "transformer.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char **argv) {
    int profiling=argc==4 && !strcmp(argv[3],"--profile");
    if (argc!=3 && !profiling) {
        fprintf(stderr,"usage: %s input.vtf output.f32 [--profile]\n",argv[0]); return 2;
    }
    uint32_t endian=1;
    if (*(unsigned char *)&endian!=1 || sizeof(float)!=4) return 2;
    FILE *in=fopen(argv[1],"rb");
    if (!in) { fprintf(stderr,"cannot open input\n"); return 1; }
    char magic[8]; uint32_t fields[5];
    vt_layer_options options={0};
    int extended=0;
    if (fread(magic,1,8,in)!=8 ||
        (memcmp(magic,"VTF32V1\0",8) && memcmp(magic,"VTF32V2\0",8)) ||
        fread(fields,sizeof(uint32_t),5,in)!=5) {
        fprintf(stderr,"invalid input header\n"); fclose(in); return 2;
    }
    extended=!memcmp(magic,"VTF32V2\0",8);
    if (extended) {
        uint32_t extra[3];
        if (fread(extra,sizeof(uint32_t),3,in)!=3 ||
            fread(&options.norm_epsilon,sizeof(float),1,in)!=1 ||
            extra[0]>1 || extra[1]>1 || extra[2]>1024 ||
            (extra[2] && fields[4]!=1) || !isfinite(options.norm_epsilon) ||
            options.norm_epsilon<=0 || options.norm_epsilon>=1) {
            fprintf(stderr,"invalid extended options\n"); fclose(in); return 2;
        }
        options.activation=extra[0]; options.unscaled_attention=extra[1]; options.window=extra[2];
    }
    size_t t=fields[0],d=fields[1],h=fields[2],f=fields[3];
    if (!t || t>1024 || !d || d>2048 || !h || h>d || d%h ||
        !f || f>8192 || fields[4]>1) {
        fprintf(stderr,"unsupported shape\n"); fclose(in); return 2;
    }
    size_t dd=d*d,df=d*f,td=t*d, count=4*dd+2*df+4*d+td+(extended ? 5*d+f : 0);
    float *mem=malloc((count+td)*sizeof(float));
    if (!mem) { fclose(in); return 1; }
    if (fread(mem,sizeof(float),count,in)!=count || fgetc(in)!=EOF || ferror(in)) {
        fprintf(stderr,"invalid tensor payload length\n");
        free(mem); fclose(in); return 2;
    }
    fclose(in);
    for (size_t i=0;i<count;++i) if (!isfinite(mem[i])) {
        fprintf(stderr,"nonfinite input or weight\n"); free(mem); return 2;
    }
    float *q=mem,*k=q+dd,*v=k+dd,*o=v+dd,*up=o+dd,*down=up+df;
    float *g1=down+df,*b1=g1+d,*g2=b1+d,*b2=g2+d,*x=b2+d;
    if (extended) {
        options.q_bias=x; options.k_bias=x+d; options.v_bias=x+2*d;
        options.out_bias=x+3*d; options.up_bias=x+4*d; options.down_bias=x+4*d+f;
        x+=5*d+f;
    }
    float *y=x+td;
    vt_weights w={q,k,v,o,up,down,g1,b1,g2,b2};
    vt_config cfg={t,d,h,f,(int)fields[4]};
    struct timespec before={0}, after={0};
    if (profiling && clock_gettime(CLOCK_MONOTONIC,&before)) { free(mem); return 1; }
    if (vt_forward_extended(&cfg,&w,extended ? &options : NULL,x,y,NULL,NULL)) { free(mem); return 1; }
    if (profiling && clock_gettime(CLOCK_MONOTONIC,&after)) { free(mem); return 1; }
    for (size_t i=0;i<td;++i) if (!isfinite(y[i])) {
        fprintf(stderr,"nonfinite output\n"); free(mem); return 1;
    }
    FILE *out=fopen(argv[2],"wb");
    if (!out) { fprintf(stderr,"cannot open output\n"); free(mem); return 1; }
    int ok=fwrite(y,sizeof(float),td,out)==td;
    if (fclose(out)) ok=0;
    if (profiling && ok) printf("native_forward_ms=%.6f\n",
        (after.tv_sec-before.tv_sec)*1000.0+(after.tv_nsec-before.tv_nsec)/1e6);
    free(mem); return ok ? 0 : 1;
}
