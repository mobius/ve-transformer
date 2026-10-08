// Factory adapter for the pinned llama.cpp ABI, for target and native MTP.
#include "llama-model.h"
#include "llama-model-loader.h"
#include "models/models.h"
#include <stdexcept>

static llama_model *create(llm_arch arch, const llama_model_params &params) {
    if (arch != LLM_ARCH_QWEN35MOE || params.split_mode == LLAMA_SPLIT_MODE_TENSOR)
        throw std::runtime_error("This isolated executor requires qwen35moe without tensor split");
    auto *model = new llama_model_qwen35moe(params);
    model->arch = arch;
    return model;
}
extern "C" llama_model *__wrap__Z18llama_model_create8llm_archRK18llama_model_params(llm_arch arch, const llama_model_params &params) {
    return create(arch, params);
}
extern "C" llama_model *__wrap__Z18llama_model_createR18llama_model_loaderRK18llama_model_params(llama_model_loader &loader, const llama_model_params &params) {
    return create(loader.get_arch(), params);
}
