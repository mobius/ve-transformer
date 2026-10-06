/* Deterministic fixtures for elementwise cross-device/backend validation. */
#include "transformer.h"
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
static size_t number(const char *s, size_t limit) {
    char *end;
    errno=0;
    if (!*s || *s<'0' || *s>'9') return 0;
    unsigned long n=strtoul(s,&end,10);
    return errno || *end || !n || n>limit ? 0 : (size_t)n;
}
int main(int argc, char **argv) {
    int reuse=argc==7 && !strcmp(argv[6],"--workspace");
    if ((argc!=6 && !reuse) || (strcmp(argv[5],"0") && strcmp(argv[5],"1"))) return 2;
    vt_config c={number(argv[1],1024),number(argv[2],2048),
                 number(argv[3],2048),number(argv[4],8192),atoi(argv[5])};
    if (!c.tokens || !c.width || !c.heads || !c.hidden || c.width%c.heads ||
        (c.causal!=0 && c.causal!=1)) return 2;
    size_t dd=c.width*c.width, df=c.width*c.hidden, td=c.tokens*c.width;
    float *mem=calloc(4*dd+2*df+4*c.width+2*td,sizeof(float));
    if (!mem) return 1;
    float *q=mem,*k=q+dd,*v=k+dd,*o=v+dd,*up=o+dd,*down=up+df;
    float *g1=down+df,*b1=g1+c.width,*g2=b1+c.width,*b2=g2+c.width;
    float *x=b2+c.width,*y=x+td;
    for (size_t i=0;i<4*dd+2*df;++i)
        mem[i]=(float)((int)((i*7+3)%31)-15)/256;
    for (size_t i=0;i<c.width;++i) {
        g1[i]=1+(float)(i%3)/16; g2[i]=1-(float)(i%5)/32;
        b1[i]=(float)((int)(i%7)-3)/32; b2[i]=(float)((int)(i%9)-4)/32;
    }
    for (size_t i=0;i<td;++i) x[i]=(float)((int)((i*11+5)%37)-18)/16;
    vt_weights w={q,k,v,o,up,down,g1,b1,g2,b2};
    vt_workspace *ws=NULL;
    int rc=0;
    if (reuse) {
        rc=vt_workspace_create(&c,&ws);
        if (!rc) {
            /* Prime scratch with another input and the opposite mask mode. */
            c.causal=!c.causal;
            for (size_t i=0;i<td;++i) x[i]=-x[i];
            rc=vt_forward_workspace(&c,&w,x,y,ws,NULL);
            c.causal=!c.causal;
            for (size_t i=0;i<td;++i) x[i]=-x[i];
            if (!rc) rc=vt_forward_workspace(&c,&w,x,y,ws,NULL);
            if (!rc) rc=vt_forward_workspace(&c,&w,x,y,ws,NULL);
        }
    } else rc=vt_forward(&c,&w,x,y);
    vt_workspace_destroy(ws);
    if (!rc && fwrite(y,sizeof(float),td,stdout)!=td) rc=1;
    free(mem); return rc ? 1 : 0;
}
