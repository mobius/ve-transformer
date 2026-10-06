// Small factory adapter for the pinned llama.cpp ABI; no numerical kernels changed.
#include "llama-model.h"
#include "llama-model-loader.h"
#include "models/models.h"
#include <stdexcept>

llama_model *qwen15_create(llm_arch arch, const llama_model_params &params) {
    if (arch != LLM_ARCH_QWEN2)
        throw std::runtime_error("This isolated executor supports only qwen2");
    if (params.split_mode == LLAMA_SPLIT_MODE_TENSOR && !llm_arch_supports_sm_tensor(arch))
        throw std::runtime_error("Tensor split mode unsupported for qwen2");
    auto *model = new llama_model_qwen2(params);
    model->arch = arch;
    return model;
}
extern "C" llama_model *__wrap__Z18llama_model_create8llm_archRK18llama_model_params(llm_arch arch, const llama_model_params &params) {
    return qwen15_create(arch, params);
}
extern "C" llama_model *__wrap__Z18llama_model_createR18llama_model_loaderRK18llama_model_params(llama_model_loader &loader, const llama_model_params &params) {
    return qwen15_create(loader.get_arch(), params);
}
