#include "ve_sd_turbo_stages.h"
#include "ve_sd_turbo_support.h"
#include "unet.hpp"
#include <memory>

struct VeSdUnet {
    VeSdBackend backend;
    std::unique_ptr<UNetModelRunner> runner;
    explicit VeSdUnet(const std::string& root) {
        ModelLoader loader;
        ve_sd_prepare(loader,root+"/unet/diffusion_pytorch_model.safetensors","unet.");
        const std::string prefix="model.diffusion_model";
        runner.reset(new UNetModelRunner(backend.value,loader.tensor_storages_types,prefix,VERSION_SD2,false));
        ve_sd_weights(loader,*runner,prefix,backend.value);
    }
};
void* ve_sd_unet_open(const std::string& root) { return new VeSdUnet(root); }
ggml_tensor* ve_sd_unet_run(void* stage, ggml_tensor* input, ggml_tensor* timestep,
                          ggml_tensor* embeddings, int threads, ggml_context* work) {
    auto value=static_cast<VeSdUnet*>(stage);
    ggml_tensor* output=NULL;
    value->runner->compute(threads,input,timestep,embeddings,NULL,NULL,-1,{},0.f,&output,work);
    return output;
}
void ve_sd_unet_close(void* stage) { delete static_cast<VeSdUnet*>(stage); }
void ve_sd_unet_release_compute(void* stage) {
    static_cast<VeSdUnet*>(stage)->runner->free_compute_buffer();
}
