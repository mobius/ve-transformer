#include "transformer.h"
#include <stdio.h>
#include <string.h>
int main(int argc, char **argv) {
    if (argc>2 || (argc==2 && strcmp(argv[1],"--causal"))) {
        fprintf(stderr,"usage: %s [--causal]\n",argv[0]); return 2;
    }
    vt_config c={4,8,2,16,argc==2};
    float q[64],k[64],v[64],o[64],up[128],down[128];
    float gain[8],bias[8]={0},x[32],y[32];
    for (int i=0;i<64;++i) {
        q[i]=(float)(i%11-5)/32; k[i]=(float)(i%7-3)/24;
        v[i]=(float)(i%13-6)/40; o[i]=(float)(i%5-2)/16;
    }
    for (int i=0;i<128;++i) {
        up[i]=(float)(i%9-4)/32; down[i]=(float)(i%7-3)/32;
    }
    for (int i=0;i<8;++i) gain[i]=1;
    for (int i=0;i<32;++i) x[i]=(float)(i%17-8)/8;
    vt_weights w={q,k,v,o,up,down,gain,bias,gain,bias};
    if (vt_forward(&c,&w,x,y)) return 1;
    for (int i=0;i<32;++i) printf("%.9g%c",y[i],i==31?'\n':' ');
    return 0;
}
