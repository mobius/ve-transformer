#include "ve_sd_turbo_stages.h"
#include "ve_sd_turbo_support.h"
#include "clip.hpp"
#include <memory>

struct VeSdClip {
    VeSdBackend backend;
    std::unique_ptr<CLIPTextModelRunner> runner;
    std::unique_ptr<CLIPTokenizer> tokenizer;
    explicit VeSdClip(const std::string& root) {
    ModelLoader loader;
    ve_sd_prepare(loader,root+"/text_encoder/model.safetensors","te.");
    const std::string prefix="cond_stage_model.transformer.text_model";
    runner.reset(new CLIPTextModelRunner(backend.value,loader.tensor_storages_types,
                              prefix,OPEN_CLIP_VIT_H_14,2,true));
    ve_sd_weights(loader,*runner,prefix,backend.value);
    }
};
void* ve_sd_clip_open(const std::string& root) { return new VeSdClip(root); }
void ve_sd_clip_close(void* stage) { delete static_cast<VeSdClip*>(stage); }
ggml_tensor* ve_sd_clip_run(void* stage, const std::string& prompt,
                          int threads, ggml_context* work) {
    SdProfileTimer tokenizer_init;
    VeSdClip* component=static_cast<VeSdClip*>(stage);
    const char* flag=std::getenv("SD_REUSE_TOKENIZER");
    const bool reuse=flag && std::strcmp(flag,"1")==0;
    std::unique_ptr<CLIPTokenizer> temporary;
    CLIPTokenizer* tokenizer=NULL;
    bool initialized=false;
    if (reuse) {
        if (!component->tokenizer) {
            component->tokenizer.reset(new CLIPTokenizer(0));
            initialized=true;
        }
        tokenizer=component->tokenizer.get();
    } else {
        temporary.reset(new CLIPTokenizer(0));
        tokenizer=temporary.get();
        initialized=true;
    }
    std::fprintf(stderr,"SD_TOKENIZER reuse=%d initialized=%d\n",reuse ? 1 : 0,initialized ? 1 : 0);
    tokenizer_init.emit("clip","tokenizer_init");
    SdProfileTimer tokenizer_encode;
    auto plain_token=[](std::string&,std::vector<int32_t>&) { return false; };
    std::vector<int> tokens=tokenizer->encode(prompt,plain_token);
    std::vector<float> weights(tokens.size(),1.f);
    tokenizer->pad_tokens(tokens,weights,77,true);
    if (tokens.size()!=77) throw std::runtime_error("only one CLIP token chunk is supported");
    tokenizer_encode.emit("clip","tokenizer_encode");
    SdProfileTimer tokenizer_tensor;
    auto ids=vector_to_ggml_tensor_i32(work,tokens);
    tokenizer_tensor.emit("clip","tokenizer_tensor");
    ggml_tensor* output=NULL;
    static_cast<VeSdClip*>(stage)->runner->compute(threads,ids,0,NULL,0,false,&output,work);
    return output;
}
ggml_tensor* ve_sd_clip(const std::string& root, const std::string& prompt,
                      int threads, ggml_context* work) {
    std::unique_ptr<void,decltype(&ve_sd_clip_close)> stage(ve_sd_clip_open(root),ve_sd_clip_close);
    return ve_sd_clip_run(stage.get(),prompt,threads,work);
}
