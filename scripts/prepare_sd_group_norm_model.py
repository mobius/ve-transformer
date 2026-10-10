"""Apply only independently verified GroupNorm shapes to the actual model.

The strict tile and fallback blocks are transcribed unchanged from round 91.
This helper prepares source; hardware verification remains a separate gate.
"""
CENTER_VARIANCE = r"""            ggml_float sum2 = 0.0;
            if (center_candidate && ggml_is_contiguous(src0) && ggml_is_contiguous(dst) &&
                    ((uintptr_t)dst->data>(uintptr_t)src0->data ?
                        (uintptr_t)dst->data-(uintptr_t)src0->data>=ggml_nbytes(src0) :
                        (uintptr_t)src0->data-(uintptr_t)dst->data>=ggml_nbytes(dst)) && ne00>0 && ne00<=2048 && 2048%ne00==0 &&
                    ne01>0 && step>0 && ne01<=2147483647LL/ne00 &&
                    step<=2147483647LL/(ne00*ne01)) {
                const int64_t group_count=ne00*ne01*step;
                const float * group_x=(const float *)((const char *)src0->data + start*nb02 + i03*nb03);
                float * group_y=(float *)((char *)dst->data + start*nb2 + i03*nb3);
                for (int64_t offset=0; offset<group_count; offset+=2048) {
                    const int count=(int)(group_count-offset<2048 ? group_count-offset : 2048);
                    sd_ve_group_norm_center_square_f32(count,group_y+offset,group_squares,group_x+offset,mean);
                    for (int row=0; row<count; row+=(int)ne00) {
                        ggml_float sumr=0.0;
                        for (int col=0; col<(int)ne00; ++col) {
                            sumr += (ggml_float)group_squares[row+col];
                        }
                        sum2 += sumr;
                    }
                }
            } else {
            for (int64_t i02 = start; i02 < end; i02++) {
                for (int64_t i01 = 0; i01 < ne01; i01++) {
                    const float * x = (float *)((char *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    float * y = (float *)((char *) dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

                    ggml_float sumr = 0.0;
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        float v = x[i00] - mean;
                        y[i00] = v;
                        sumr += (ggml_float)(v * v);
                    }
                    sum2 += sumr;
                }
            }
            }
"""
GROUP_SCALE = r"""            if (scale_candidate && ggml_is_contiguous(dst) && ne00>0 && ne01>0 && step>0 &&
                    ne00<=2147483647LL && ne01<=2147483647LL/ne00 && step<=2147483647LL/(ne00*ne01)) {
                float * group_y=(float *)((char *)dst->data + start*nb2 + i03*nb3);
                sd_ve_softmax_scale_f32((int)(ne00*ne01*step),group_y,scale);
            } else {
            for (int64_t i02 = start; i02 < end; i02++) {
                for (int64_t i01 = 0; i01 < ne01; i01++) {
                    float * y = (float *)((char *) dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }
            }
"""
SELECTION = r"""
    const char * group_stage=sd_group_norm_enabled ? getenv("SD_PROFILE_STAGE") : NULL;
    const bool group_unet=group_stage && strcmp(group_stage,"unet")==0;
    const bool group_vae=group_stage && strcmp(group_stage,"vae")==0;
    const bool group_shape=n_groups==32 && ne03==1 &&
        ((group_unet && (
            (ne00==64 && ne01==64 && ne02==320 && eps==1e-5f) ||
            (ne00==64 && ne01==64 && ne02==320 && eps==1e-6f) ||
            (ne00==32 && ne01==32 && ne02==320 && eps==1e-5f) ||
            (ne00==32 && ne01==32 && ne02==640 && eps==1e-5f) ||
            (ne00==32 && ne01==32 && ne02==640 && eps==1e-6f) ||
            (ne00==16 && ne01==16 && ne02==640 && eps==1e-5f) ||
            (ne00==16 && ne01==16 && ne02==1280 && eps==1e-5f) ||
            (ne00==16 && ne01==16 && ne02==1280 && eps==1e-6f) ||
            (ne00==8 && ne01==8 && ne02==1280 && eps==1e-5f) ||
            (ne00==8 && ne01==8 && ne02==1280 && eps==1e-6f) ||
            (ne00==8 && ne01==8 && ne02==2560 && eps==1e-5f) ||
            (ne00==16 && ne01==16 && ne02==2560 && eps==1e-5f) ||
            (ne00==16 && ne01==16 && ne02==1920 && eps==1e-5f) ||
            (ne00==32 && ne01==32 && ne02==1920 && eps==1e-5f) ||
            (ne00==32 && ne01==32 && ne02==1280 && eps==1e-5f) ||
            (ne00==32 && ne01==32 && ne02==960 && eps==1e-5f) ||
            (ne00==64 && ne01==64 && ne02==960 && eps==1e-5f) ||
            (ne00==64 && ne01==64 && ne02==640 && eps==1e-5f))) || (group_vae && (
            (ne00==64 && ne01==64 && ne02==512 && eps==1e-6f) ||
            (ne00==128 && ne01==128 && ne02==512 && eps==1e-6f) ||
            (ne00==256 && ne01==256 && ne02==512 && eps==1e-6f) ||
            (ne00==256 && ne01==256 && ne02==256 && eps==1e-6f) ||
            (ne00==512 && ne01==512 && ne02==256 && eps==1e-6f) ||
            (ne00==512 && ne01==512 && ne02==128 && eps==1e-6f))));
    const bool scale_candidate=sd_group_norm_enabled && group_shape &&
        src0->type==GGML_TYPE_F32 && dst->type==GGML_TYPE_F32 &&
        ggml_is_contiguous(src0) && ggml_is_contiguous(dst);
    const bool group_disjoint=scale_candidate &&
        ((uintptr_t)dst->data>(uintptr_t)src0->data ?
            (uintptr_t)dst->data-(uintptr_t)src0->data>=ggml_nbytes(src0) :
            (uintptr_t)src0->data-(uintptr_t)dst->data>=ggml_nbytes(dst));
    const bool center_candidate=scale_candidate && group_disjoint && ne00>=32;
    float group_squares[2048];
    if (ith==0) {
        if (scale_candidate) ++sd_group_norm_scale_optimized; else ++sd_group_norm_scale_fallback;
        if (center_candidate) ++sd_group_norm_center_optimized; else ++sd_group_norm_center_fallback;
    }
"""


def instrument(text):
    function='static void ggml_compute_forward_group_norm_f32('
    start=text.index(function)
    end=text.index('static void ggml_compute_forward_group_norm(',start)
    body=text[start:end]
    needle='    int n_groups = dst->op_params[0];'
    if body.count(needle)!=1 or 'sd_group_norm_enabled' in body:
        raise RuntimeError('unique original GroupNorm function required')
    body=body.replace(needle,needle+SELECTION)
    begin=body.index('            ggml_float sum2 = 0.0;')
    finish=body.index('            const float variance =',begin)
    body=body[:begin]+CENTER_VARIANCE+body[finish:]
    begin=body.index('            for (int64_t i02 = start;',body.index('            const float scale ='))
    finish=body.rindex('        }\n    }\n}')
    body=body[:begin]+GROUP_SCALE+body[finish:]
    text=text[:start]+body+text[end:]
    text='#include <stdint.h>\nextern void sd_ve_group_norm_center_square_f32(int,float*,float*,const float*,float);\nstatic int sd_group_norm_enabled;\nstatic int64_t sd_group_norm_scale_optimized,sd_group_norm_scale_fallback;\nstatic int64_t sd_group_norm_center_optimized,sd_group_norm_center_fallback;\n'+text
    needle='    sd_cont_optimized=sd_cont_fallback=0;'
    if text.count(needle)!=1:raise RuntimeError('unique stage reset required')
    text=text.replace(needle,needle+'\n    sd_group_norm_scale_optimized=sd_group_norm_scale_fallback=0;\n    sd_group_norm_center_optimized=sd_group_norm_center_fallback=0;\n    const char * group_flag=getenv("SD_VE_GROUP_NORM");\n    sd_group_norm_enabled=group_flag && strcmp(group_flag,"1")==0;')
    needle='void sd_ve_op_profile_end(const char* stage) {'
    if text.count(needle)!=1:raise RuntimeError('unique stage end required')
    diagnostic='\n    fprintf(stderr,"SD_GROUP_NORM_DISPATCH stage=%s scale_optimized=%" PRId64 " scale_fallback=%" PRId64 " center_optimized=%" PRId64 " center_fallback=%" PRId64 " enabled=%d'+chr(92)+'n",stage,sd_group_norm_scale_optimized,sd_group_norm_scale_fallback,sd_group_norm_center_optimized,sd_group_norm_center_fallback,sd_group_norm_enabled);'
    return text.replace(needle,needle+diagnostic)
