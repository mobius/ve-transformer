// NEC VE long-vector quantized dot kernels; upstream packed block layout preserved.
#include "ggml-quants.h"
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

_Static_assert(offsetof(block_q4_K,qs)==16,"q4_K layout changed");
_Static_assert(offsetof(block_q5_K,qs)==48,"q5_K layout changed");
_Static_assert(offsetof(block_q6_K,scales)==192,"q6_K layout changed");
_Static_assert(offsetof(block_q8_K,bsums)==260 && sizeof(block_q8_K)==292,"q8_K layout changed");

static inline float bits_float(uint32_t bits) {
    union {uint32_t bits;float value;} u;u.bits=bits;return u.value;
}
static inline unsigned half_word(const uint32_t *p,int byte) {
    return (p[byte/4]>>((byte&3)*8))&65535;
}
static inline float half_to_float(uint32_t h) {
    h&=65535;
    uint32_t sign=(h&32768)<<16,exponent=(h>>10)&31,mantissa=h&1023;
    if(exponent==0) {
        float denormal=(float)mantissa*0x1p-24f;
        return sign?-denormal:denormal;
    }
    uint32_t bits=sign|(exponent==31?0x7f800000u:(exponent+112)<<23)|(mantissa<<13);
    return bits_float(bits);
}

float ve_fp16_to_fp32(uint32_t h) {return half_to_float(h);}

static inline unsigned byte_word(const uint32_t *p,int byte) {
    return (p[byte/4]>>((byte&3)*8))&255;
}
static inline int sbyte_word(const uint32_t *p,int byte) {
    return (int)(byte_word(p,byte)^128)-128;
}
static inline int sshort_word(const uint32_t *p,int byte) {
    unsigned v=(p[byte/4]>>((byte&3)*8))&65535;
    return (int)(v^32768)-32768;
}
static inline int scale_at(const uint32_t *p,int base,int group) {
    if(group<4) return byte_word(p,base+group)&63;
    return (byte_word(p,base+group+4)&15)|((byte_word(p,base+group-4)>>6)<<4);
}
static inline int min_at(const uint32_t *p,int base,int group) {
    if(group<4) return byte_word(p,base+group+4)&63;
    return (byte_word(p,base+group+4)>>4)|((byte_word(p,base+group)>>6)<<4);
}
#define DOT_ARGS int n,float * restrict result,size_t bs,const void * restrict vx,size_t bx,const void * restrict vy,size_t by,int nrc
#define CHECK_ARGS assert(n>0 && n%256==0 && n<=32768 && nrc==1);(void)bs;(void)bx;(void)by
#define COPY_ROWS(T) \
    const int blocks=n/256; \
    uint32_t weights[(sizeof(T)*blocks+3)/4],activation[(sizeof(block_q8_K)*blocks+3)/4]; \
    float d[blocks]; \
    weights[(sizeof(T)*blocks+3)/4-1]=0;activation[(sizeof(block_q8_K)*blocks+3)/4-1]=0; \
    memcpy(weights,vx,sizeof(T)*blocks);memcpy(activation,vy,sizeof(block_q8_K)*blocks); \
    for(int b=0;b<blocks;++b) d[b]=half_to_float(half_word(weights,b*sizeof(T)+offsetof(T,d)))*bits_float(activation[b*73])

void ve_vec_dot_q4_K_q8_K(DOT_ARGS) {
    CHECK_ARGS;COPY_ROWS(block_q4_K);
    float dm[blocks];for(int b=0;b<blocks;++b) dm[b]=half_to_float(half_word(weights,b*sizeof(block_q4_K)+2))*bits_float(activation[b*73]);
    float total=0,correction=0;
    for(int k=0;k<n;++k) {
        int b=k/256,l=k%256,base=b*sizeof(block_q4_K);
        unsigned packed=byte_word(weights,base+16+(l/64)*32+l%32);
        int q=(packed>>((l/32%2)*4))&15;
        int scale=scale_at(weights,base+4,l/32);
        int a=sbyte_word(activation,b*sizeof(block_q8_K)+4+l);
        total+=d[b]*(scale*q*a);
    }
    for(int j=0;j<n/16;++j) {
        int b=j/16,g=(j%16)/2;
        int m=min_at(weights,b*sizeof(block_q4_K)+4,g);
        int a=sshort_word(activation,b*sizeof(block_q8_K)+260+(j%16)*2);
        correction+=dm[b]*(m*a);
    }
    *result=total-correction;
}
void ve_vec_dot_q5_K_q8_K(DOT_ARGS) {
    CHECK_ARGS;COPY_ROWS(block_q5_K);
    float dm[blocks];for(int b=0;b<blocks;++b) dm[b]=half_to_float(half_word(weights,b*sizeof(block_q5_K)+2))*bits_float(activation[b*73]);
    float total=0,correction=0;
    for(int k=0;k<n;++k) {
        int b=k/256,l=k%256,base=b*sizeof(block_q5_K);
        unsigned packed=byte_word(weights,base+48+(l/64)*32+l%32);
        unsigned high=byte_word(weights,base+16+l%32);
        int q=((packed>>((l/32%2)*4))&15)|(((high>>(l/32))&1)<<4);
        int scale=scale_at(weights,base+4,l/32);
        int a=sbyte_word(activation,b*sizeof(block_q8_K)+4+l);
        total+=d[b]*(scale*q*a);
    }
    for(int j=0;j<n/16;++j) {
        int b=j/16,g=(j%16)/2;
        int m=min_at(weights,b*sizeof(block_q5_K)+4,g);
        int a=sshort_word(activation,b*sizeof(block_q8_K)+260+(j%16)*2);
        correction+=dm[b]*(m*a);
    }
    *result=total-correction;
}
void ve_vec_dot_q6_K_q8_K(DOT_ARGS) {
    CHECK_ARGS;COPY_ROWS(block_q6_K);
    float total=0;
    for(int k=0;k<n;++k) {
        int b=k/256,l=k%256,g=(l%128)/32,low=l%32,base=b*sizeof(block_q6_K);
        unsigned packed=byte_word(weights,base+(l/128)*64+(g%2)*32+low);
        unsigned high=byte_word(weights,base+128+(l/128)*32+low);
        int q=(((packed>>((g/2)*4))&15)|(((high>>(2*g))&3)<<4))-32;
        int scale=sbyte_word(weights,base+192+l/16);
        int a=sbyte_word(activation,b*sizeof(block_q8_K)+4+l);
        total+=d[b]*(scale*q*a);
    }
    *result=total;
}

void ve_vec_dot_q8_0_q8_0(DOT_ARGS) {
    assert(n>0 && n%32==0 && n<=32768 && nrc==1);(void)bs;(void)bx;(void)by;
        const int blocks=n/32;
    uint32_t weights[(sizeof(block_q8_0)*blocks+3)/4],activation[(sizeof(block_q8_0)*blocks+3)/4];
    float d[blocks];
    weights[(sizeof(block_q8_0)*blocks+3)/4-1]=0;activation[(sizeof(block_q8_0)*blocks+3)/4-1]=0;
    memcpy(weights,vx,sizeof(block_q8_0)*blocks);memcpy(activation,vy,sizeof(block_q8_0)*blocks);
    for(int b=0;b<blocks;++b) d[b]=half_to_float(half_word(weights,b*sizeof(block_q8_0)))*half_to_float(half_word(activation,b*sizeof(block_q8_0)));
    float total=0;
    for(int k=0;k<n;++k) {
        int b=k/32,l=k%32,offset=b*sizeof(block_q8_0)+2+l;
        total+=d[b]*(sbyte_word(weights,offset)*sbyte_word(activation,offset));
    }
    *result=total;
}

// Decode a bounded contiguous row tile. Caller owns aligned scratch storage.
void ve_dequant_rows(enum ggml_type type,const void *restrict src,float *restrict out,int rows,int width,uint32_t *restrict words) {
    assert(rows>0 && rows<=128 && width>0 && width<=16384);
    size_t bytes=ggml_row_size(type,width)*(size_t)rows;
    words[(bytes+3)/4-1]=0;memcpy(words,src,bytes);
    int n=rows*width;
    if(type==GGML_TYPE_Q4_K || type==GGML_TYPE_Q5_K) {
        int blocks=n/256,size=type==GGML_TYPE_Q4_K?144:176,offset=type==GGML_TYPE_Q4_K?16:48;
        assert(n%256==0);
        float d[blocks],dm[blocks];int sc[blocks*8],mi[blocks*8];
        for(int b=0;b<blocks;++b) {
            int base=b*size;d[b]=half_to_float(half_word(words,base));dm[b]=half_to_float(half_word(words,base+2));
            for(int g=0;g<8;++g) {sc[b*8+g]=scale_at(words,base+4,g);mi[b*8+g]=min_at(words,base+4,g);}
        }
        for(int k=0;k<n;++k) {
            int b=k/256,l=k%256,base=b*size,g=b*8+l/32;
            unsigned packed=byte_word(words,base+offset+(l/64)*32+l%32);
            int q=(packed>>((l/32%2)*4))&15;
            if(type==GGML_TYPE_Q5_K) q|=((byte_word(words,base+16+l%32)>>(l/32))&1)<<4;
            out[k]=d[b]*(sc[g]*q)-dm[b]*mi[g];
        }
    } else if(type==GGML_TYPE_Q6_K) {
        int blocks=n/256;assert(n%256==0);
        float d[blocks];int sc[blocks*16];
        for(int b=0;b<blocks;++b) {
            d[b]=half_to_float(half_word(words,b*210+208));
            for(int g=0;g<16;++g)sc[b*16+g]=sbyte_word(words,b*210+192+g);
        }
        for(int k=0;k<n;++k) {
            int b=k/256,l=k%256,g=(l%128)/32,low=l%32,base=b*210;
            unsigned packed=byte_word(words,base+(l/128)*64+(g%2)*32+low);
            unsigned high=byte_word(words,base+128+(l/128)*32+low);
            int q=(((packed>>((g/2)*4))&15)|(((high>>(g*2))&3)<<4))-32;
            out[k]=d[b]*(sc[b*16+l/16]*q);
        }
    } else if(type==GGML_TYPE_Q8_0) {
        int blocks=n/32;assert(n%32==0);float d[blocks];
        for(int b=0;b<blocks;++b)d[b]=half_to_float(half_word(words,b*34));
        for(int k=0;k<n;++k) {
            int b=k/32,base=b*34;
            out[k]=d[b]*sbyte_word(words,base+2+k%32);
        }
    } else {
        assert(!"unsupported dequantization type");
    }
}
