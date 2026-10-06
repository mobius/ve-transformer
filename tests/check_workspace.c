/* API lifetime, shape validation, and stale-scratch regression. */
#include "transformer.h"
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr,"workspace assertion failed line %d\n",__LINE__); return 1; } } while (0)
int main(void) {
    vt_config c={3,4,2,5,0};
    float weights[4*16+2*20], gain[4]={1,1,1,1}, bias[4]={0};
    float x[12], y[12], ref[12];
    for (size_t i=0;i<sizeof(weights)/sizeof(*weights);++i)
        weights[i]=(float)((int)(i%11)-5)/16;
    vt_weights w={weights,weights+16,weights+32,weights+48,
                  weights+64,weights+84,gain,bias,gain,bias};
    vt_workspace *ws=NULL;
    CHECK(vt_workspace_create(&c,&ws)==0 && ws);
    for (int round=0;round<8;++round) {
        c.causal=round%2;
        for (size_t i=0;i<12;++i) x[i]=(float)((int)(i*3+round)%17-8)/8;
        CHECK(vt_forward(&c,&w,x,ref)==0);
        vt_profile p;
        CHECK(vt_forward_workspace(&c,&w,x,y,ws,&p)==0);
        for (size_t i=0;i<12;++i) CHECK(isfinite(y[i]) && fabsf(y[i]-ref[i])<1e-6f);
    }
    float up_bias[5]={0.1f,-0.2f,0.3f,0.05f,-0.1f};
    vt_layer_options opts={gain,gain,gain,gain,up_bias,gain,1,1,2,1e-4f};
    c.causal=1;
    for (int round=0;round<6;++round) {
        opts.window=(size_t)(round%4);
        for (size_t i=0;i<12;++i) x[i]=(float)((int)(i*3+round)%17-8)/8;
        CHECK(vt_forward_extended(&c,&w,&opts,x,ref,NULL,NULL)==0);
        CHECK(vt_forward_extended(&c,&w,&opts,x,y,ws,NULL)==0);
        for (size_t i=0;i<12;++i) CHECK(isfinite(y[i]) && fabsf(y[i]-ref[i])<1e-6f);
    }
    opts.activation=2;
    CHECK(vt_forward_extended(&c,&w,&opts,x,y,ws,NULL)==-1);
    opts.activation=1; opts.norm_epsilon=NAN;
    CHECK(vt_forward_extended(&c,&w,&opts,x,y,ws,NULL)==-1);
    opts.norm_epsilon=1e-4f; opts.window=2; c.causal=0;
    CHECK(vt_forward_extended(&c,&w,&opts,x,y,ws,NULL)==-1);
    c.causal=1;
    vt_config bad=c; bad.tokens++;
    CHECK(vt_forward_workspace(&bad,&w,x,y,ws,NULL)==-1);
    bad=c; bad.heads=1;
    CHECK(vt_forward_workspace(&bad,&w,x,y,ws,NULL)==-1);
    CHECK(vt_forward_workspace(&c,&w,x,y,NULL,NULL)==-1);
    CHECK(vt_forward_workspace(&c,&w,x,x,ws,NULL)==-1);
    vt_workspace *invalid=ws;
    bad=c; bad.tokens=SIZE_MAX;
    CHECK(vt_workspace_create(&bad,&invalid)==-1 && invalid==NULL);
    bad=c; bad.causal=2;
    CHECK(vt_workspace_create(&bad,&invalid)==-1 && invalid==NULL);
    CHECK(vt_workspace_create(&c,NULL)==-1);
    vt_workspace_destroy(ws); vt_workspace_destroy(NULL);
    CHECK(vt_workspace_create(&c,&ws)==0);
    CHECK(vt_forward_workspace(&c,&w,x,y,ws,NULL)==0);
    vt_workspace_destroy(ws);
    float varying_x[28], varying_y[28], varying_ref[28];
    c.tokens=1;
    CHECK(vt_workspace_create_capacity(&c,7,&ws)==0);
    for (int round=0;round<10;++round) {
        c.tokens=(size_t)((round*3)%7+1); c.causal=round%2;
        for (size_t i=0;i<c.tokens*4;++i) varying_x[i]=(float)((int)(i+round)%11-5)/8;
        CHECK(vt_forward(&c,&w,varying_x,varying_ref)==0);
        CHECK(vt_forward_workspace(&c,&w,varying_x,varying_y,ws,NULL)==0);
        for (size_t i=0;i<c.tokens*4;++i)
            CHECK(isfinite(varying_y[i]) && fabsf(varying_y[i]-varying_ref[i])<1e-6f);
    }
    c.tokens=8;
    CHECK(vt_forward_workspace(&c,&w,varying_x,varying_y,ws,NULL)==-1);
    vt_workspace_destroy(ws);
    CHECK(vt_workspace_create_capacity(&c,7,&ws)==-1 && !ws);
    puts("workspace lifetime/shape/mask/input-reuse/decoder-options/capacity: PASS");
    return 0;
}
