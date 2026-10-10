#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern void sd_ve_binary_plane_f32(const float*,float*,float,int64_t,int);
static float from_bits(uint32_t bits) { float x; memcpy(&x,&bits,4); return x; }
static uint32_t bits(float x) { uint32_t v; memcpy(&v,&x,4); return v; }
int main(void) {
    const int lengths[]={0,1,3,7,63,64,127,128,255,256,257,511,512,513,4096,16384,65536,262144};
    const uint32_t special[]={0,0x80000000,1,0x80000001,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc01234};
    const uint32_t scalars[]={0,0x80000000,0x3f000000,0xbf800000,0x7f7fffff,0x7f800000,0x7fc04321};
    float * input=malloc(262146*4),*output=malloc(262146*4),*original=malloc(262146*4);
    if(!input || !output || !original) return 2;
    int cases=0;
    for(unsigned size=0;size<sizeof(lengths)/sizeof(lengths[0]);++size)
    for(unsigned scalar=0;scalar<sizeof(scalars)/sizeof(scalars[0]);++scalar)
    for(int multiply=0;multiply<2;++multiply)
    for(int inplace=0;inplace<2;++inplace) {
        const int n=lengths[size];
        for(int i=0;i<n+2;++i) {
            input[i]=i%32<9 ? from_bits(special[i%32]) : (float)(i%101-50)*0.125f;
            output[i]=from_bits(0x4b123456);
        }
        input[0]=input[n+1]=from_bits(0x4b123456);
        memcpy(original,input,(n+2)*4);
        float * destination=inplace ? input+1 : output+1;
        const float value=from_bits(scalars[scalar]);
        sd_ve_binary_plane_f32(input+1,destination,value,n,multiply);
        for(int i=0;i<n;++i) {
            volatile float x=original[i+1],v=value;
            const float expected=multiply ? x*v : x+v;
            const float actual=destination[i];
            if((isnan(expected) && !isnan(actual)) || (!isnan(expected) && bits(expected)!=bits(actual))) {
                fprintf(stderr,"BINARY_FAIL n=%d scalar=%u multiply=%d inplace=%d index=%d\n",n,scalar,multiply,inplace,i);
                return 1;
            }
        }
        const float * guarded=inplace ? input : output;
        if(bits(guarded[0])!=0x4b123456 || bits(guarded[n+1])!=0x4b123456 ||
           (!inplace && memcmp(input,original,(n+2)*4))) return 1;
        ++cases;
    }
    free(input);free(output);free(original);
    printf("BINARY_PLANE_PASS cases=%d finite_bitwise=1 nan_classification=1 guards=1 independent_scalar_oracle=1\n",cases);
    return 0;
}
