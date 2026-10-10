#ifndef VE_SD_TURBO_STAGES_H
#define VE_SD_TURBO_STAGES_H
#include "ggml.h"
#include <string>

ggml_tensor* ve_sd_clip(const std::string& root, const std::string& prompt,
                      int threads, ggml_context* work);
void* ve_sd_clip_open(const std::string& root);
ggml_tensor* ve_sd_clip_run(void* stage, const std::string& prompt,
                          int threads, ggml_context* work);
void ve_sd_clip_close(void* stage);
void* ve_sd_unet_open(const std::string& root);
ggml_tensor* ve_sd_unet_run(void* stage, ggml_tensor* input, ggml_tensor* timestep,
                          ggml_tensor* embeddings, int threads, ggml_context* work);
void ve_sd_unet_close(void* stage);
void ve_sd_unet_release_compute(void* stage);
ggml_tensor* ve_sd_vae(const std::string& root, ggml_tensor* latent,
                     int threads, ggml_context* work);
void* ve_sd_vae_open(const std::string& root);
ggml_tensor* ve_sd_vae_run(void* stage, ggml_tensor* latent,
                         int threads, ggml_context* work);
void ve_sd_vae_close(void* stage);
#endif
