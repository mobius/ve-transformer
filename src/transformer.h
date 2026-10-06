#ifndef VE_TRANSFORMER_H
#define VE_TRANSFORMER_H
#include <stddef.h>
/* Row-major float32 arrays. Caller owns all buffers. No global state. */
typedef struct {
    size_t tokens, width, heads, hidden;
    int causal;
} vt_config;
typedef struct {
    const float *q, *k, *v, *out; /* width x width */
    const float *up, *down;      /* width x hidden, hidden x width */
    const float *gain1, *bias1, *gain2, *bias2; /* width */
} vt_weights;
/* Optional extended decoder semantics, independent of scratch layout.
   activation: 0=ReLU, 1=GELU tanh (gelu_new).
   window: 0=global; positive=last window positions, requires causal=1.
   Bias pointers may be NULL for no bias. Explicit epsilon must be >0 and <1. */
typedef struct {
    const float *q_bias, *k_bias, *v_bias, *out_bias, *up_bias, *down_bias;
    unsigned activation, unscaled_attention;
    size_t window;
    float norm_epsilon;
} vt_layer_options;
/* Optional per-call stage timings; clock overhead is included in these values.
   Pass NULL through vt_forward() for ordinary uninstrumented execution. */
typedef struct {
    double allocation_ms, norm1_ms, qkv_ms, attention_ms;
    double projection_ms, norm2_ms, ffn_ms, release_ms;
} vt_profile;
/* Returns 0 on success; -1 invalid arguments, -2 allocation failure.
   Input/output must be separate buffers. Linear layers have no biases.
   Convenience calls allocate scratch memory per call. */
int vt_forward(const vt_config *cfg, const vt_weights *w,
               const float *input, float *output);
int vt_forward_profile(const vt_config *cfg, const vt_weights *w,
                       const float *input, float *output, vt_profile *profile);
/* Fixed-shape reusable scratch. Create sets *result=NULL on failure.
   Same tokens/width/heads/hidden required; causal may change per call.
   One workspace per concurrent caller. No weights or results are cached.
   Creation allocates memory but first use may incur page-touch costs.
   Destroy accepts NULL. Optional profile has the same semantics as above. */
typedef struct vt_workspace vt_workspace;
int vt_workspace_create(const vt_config *cfg, vt_workspace **result);
void vt_workspace_destroy(vt_workspace *workspace);
/* Allocate for max_tokens, accepting 1..max_tokens per forward call.
   Width/heads/hidden remain fixed. Original create remains exact-shape. */
int vt_workspace_create_capacity(const vt_config *cfg, size_t max_tokens,
                                 vt_workspace **result);
int vt_forward_workspace(const vt_config *cfg, const vt_weights *w,
                         const float *input, float *output,
                         vt_workspace *workspace, vt_profile *profile);
/* NULL options preserves the original semantics. NULL workspace allocates
   per call; otherwise the same fixed-shape workspace contract applies. */
int vt_forward_extended(const vt_config *cfg, const vt_weights *w,
                        const vt_layer_options *options,
                        const float *input, float *output,
                        vt_workspace *workspace, vt_profile *profile);
#endif
