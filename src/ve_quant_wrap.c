// Link-time substitution; CPU reference and upstream checkout remain unchanged.
#include <stddef.h>
#define ARGS int n,float *s,size_t bs,const void *x,size_t bx,const void *y,size_t by,int nrc
#define VALUES n,s,bs,x,bx,y,by,nrc
#define WRAP(NAME,BLOCK) \
extern void ve_vec_dot_##NAME(ARGS); \
extern void __real_ggml_vec_dot_##NAME(ARGS); \
void __wrap_ggml_vec_dot_##NAME(ARGS) { \
    if(n>0 && n<=32768 && n%BLOCK==0 && nrc==1) ve_vec_dot_##NAME(VALUES); \
    else __real_ggml_vec_dot_##NAME(VALUES); \
}
WRAP(q4_K_q8_K,256)
WRAP(q5_K_q8_K,256)
WRAP(q6_K_q8_K,256)

WRAP(q8_0_q8_0,32)
