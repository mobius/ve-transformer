#include "ve_sd_turbo_stages.h"
#include "ve_sd_turbo_support.h"
#include "vae.hpp"
#include <memory>

struct VeSdVae {
    VeSdBackend backend;
    std::unique_ptr<AutoEncoderKL> runner;
    explicit VeSdVae(const std::string& root) {
    ModelLoader loader;
    ve_sd_prepare(loader,root+"/vae/diffusion_pytorch_model.safetensors","vae.");
    const std::string prefix="first_stage_model";
    runner.reset(new AutoEncoderKL(backend.value,loader.tensor_storages_types,prefix,true,false,VERSION_SD2));
    ve_sd_weights(loader,*runner,prefix,backend.value);
    }
};
void* ve_sd_vae_open(const std::string& root) { return new VeSdVae(root); }
void ve_sd_vae_close(void* stage) { delete static_cast<VeSdVae*>(stage); }
ggml_tensor* ve_sd_vae_run(void* stage, ggml_tensor* latent,
                         int threads, ggml_context* work) {
    ggml_tensor* output=NULL;
    static_cast<VeSdVae*>(stage)->runner->compute(threads,latent,true,&output,work);
    return output;
}
ggml_tensor* ve_sd_vae(const std::string& root, ggml_tensor* latent,
                     int threads, ggml_context* work) {
    std::unique_ptr<void,decltype(&ve_sd_vae_close)> stage(ve_sd_vae_open(root),ve_sd_vae_close);
    return ve_sd_vae_run(stage.get(),latent,threads,work);
}
