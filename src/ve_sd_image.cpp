// Restricted SD2 text-to-image orchestration, adapted from stable-diffusion.cpp
// a1ded76da5818803fca97a3b433669ef727d32cf (MIT; see THIRD_PARTY.md).
// Neural execution remains in the actual CLIP, UNet, VAE and sampler runners.
#include "generation.h"
#include "diffusion_engine.h"
#include "request.h"
#include "core/rng.hpp"
#include "../build/sd-overlay/sd_validation.h"
#include <cstdlib>

namespace sd::pipeline {
static bool nonempty(const char* value) { return value && *value; }

sd::Tensor<float> ensure_image_tensor_channels(sd::Tensor<float> image, int channels) {
    if (image.empty() || (image.dim()==4 && image.shape()[2]==channels)) return image;
    LOG_ERROR("Channel conversion is unsupported in this restricted VE image build");
    return {};
}

bool generate_image(StableDiffusionGGML* sd, const sd_img_gen_params_t* p,
                    sd_image_t** images_out, int* num_images_out) {
    if (images_out) *images_out=nullptr;
    if (num_images_out) *num_images_out=0;
    if (!sd || !p) return false;
    if (sd->version!=VERSION_SD2 || p->batch_count!=1 ||
        p->init_image.data || p->mask_image.data || p->control_image.data ||
        p->ip_adapter_image.data || p->ref_images_count || p->lora_count ||
        p->hires.enabled || p->vae_tiling_params.enabled ||
        p->circular_x || p->circular_y || p->pm_params.id_images_count ||
        nonempty(p->pm_params.id_embed_path) || nonempty(p->pulid_params.id_embedding_path) ||
        nonempty(p->ref_image_args) || nonempty(p->image_preprocess.rules) ||
        nonempty(p->sample_params.extra_sample_args) ||
        p->sample_params.shifted_timestep!=0 || p->cache.mode!=SD_CACHE_DISABLED) {
        LOG_ERROR("VE image entry supports single SD2 text-to-image without optional adapters, edits or cache");
        return false;
    }
    sd->reset_cancel_flag();
    GenerationRequest request(sd,p);
    SamplePlan plan(sd,p,request);
    if (request.use_uncond || request.use_img_uncond || request.use_high_noise_uncond ||
        request.use_high_noise_img_uncond || plan.sample_method!=EULER_SAMPLE_METHOD ||
        plan.eta!=0 || plan.sigmas.size()<2 || request.width!=512 || request.height!=512) {
        LOG_ERROR("VE image validation requires 512x512 Euler without guidance mixing");
        return false;
    }
    sd->vae_tiling_params=p->vae_tiling_params;
    sd->apply_circular_axes(false,false);
    sd->rng->manual_seed(request.seed);
    sd->sampler_rng->manual_seed(request.seed);
    sd->set_flow_shift(p->sample_params.flow_shift);
    sd::Tensor<float> init_latent=sd->generate_init_latent(request.width,request.height);
    if (init_latent.empty()) return false;
    SDCondition cond;
    {
        ConditionerRunnerEndOnExit release{sd->cond_stage_model.get()};
        ConditionerParams params;
        params.text=request.prompt;
        params.clip_skip=request.clip_skip;
        params.width=request.width;
        params.height=request.height;
        params.zero_out_masked=false;
        cond=sd->get_learned_condition(params);
    }
    if (cond.empty()) return false;
    sd::Tensor<float> noise=sd::randn_like<float>(init_latent,sd->rng);
    sd_load_fixed_noise(noise);
    sd_trace_tensor("native-noise",noise);
    sd_trace_tensor("native-embeddings",cond.c_crossattn);
    sd::Tensor<float> mask=sd::full<float>(
        {request.width/request.vae_scale_factor,request.height/request.vae_scale_factor,1,1},1.f);
    const RefImageParams ref_params=sd->resolve_ref_image_params(nullptr);
    SDCondition empty_condition;
    sd::Tensor<float> empty_tensor;
    std::vector<sd::Tensor<float>> empty_refs;
    sd::Tensor<float> latent=sd->sample(sd->diffusion_model,true,init_latent,std::move(noise),
        cond,empty_condition,empty_condition,empty_tensor,request.control_strength,
        request.guidance,plan.eta,request.shifted_timestep,plan.sample_method,
        sd->is_flow_denoiser(),plan.extra_sample_args,plan.sigmas,empty_refs,ref_params,
        mask,empty_tensor,1.f,0,static_cast<float>(request.fps),request.cache_params,true);
    if (latent.empty() || sd->get_cancel_flag()==SD_CANCEL_ALL) return false;
    sd_trace_tensor("native-latent",latent);
    sd::Tensor<float> image=sd->decode_first_stage(latent);
    if (image.empty() || sd->get_cancel_flag()==SD_CANCEL_ALL) return false;
    sd_trace_tensor("native-decoded",image);
    sd_image_t* result=static_cast<sd_image_t*>(std::calloc(1,sizeof(sd_image_t)));
    if (!result) return false;
    result[0]=tensor_to_sd_image(image);
    if (!result[0].data) { std::free(result);return false; }
    if (images_out) *images_out=result;
    else free_sd_images(result,1);
    if (num_images_out) *num_images_out=1;
    return true;
}
} // namespace sd::pipeline
