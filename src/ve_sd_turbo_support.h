#ifndef VE_SD_TURBO_SUPPORT_H
#define VE_SD_TURBO_SUPPORT_H
#include "model.h"
#include "ggml-cpu.h"
#include "ve_sd_turbo_profile.h"
#include <stdexcept>
extern "C" void* sd_shared_threadpool_acquire(int threads);
extern "C" void sd_shared_threadpool_release(void* pool);

struct VeSdBackend {
    ggml_backend_t value;
    ggml_threadpool_t pool;
    bool shared;
    VeSdBackend() : value(ggml_backend_cpu_init()),pool(NULL),shared(false) {
        if (!value) throw std::runtime_error("native VE backend initialization failed");
        try {
            const char* mode=std::getenv("SD_PERSISTENT_THREADPOOL");
            if (mode && std::strcmp(mode,"1")==0) {
                const char* count=std::getenv("SD_POOL_THREADS");
                char* end=NULL;
                long threads=count ? std::strtol(count,&end,10) : 0;
                if (!count || !end || *end || (threads!=1 && threads!=2 && threads!=4 && threads!=8))
                    throw std::runtime_error("validated pool thread count required");
                const char* shared_flag=std::getenv("SD_SHARED_THREADPOOL");
                shared=shared_flag && std::strcmp(shared_flag,"1")==0;
                if (shared) pool=static_cast<ggml_threadpool_t>(sd_shared_threadpool_acquire(int(threads)));
                else {
                    ggml_threadpool_params params=ggml_threadpool_params_default(int(threads));
                    params.poll=0; // Block while NLC executes instead of spinning.
                    params.paused=true;
                    pool=ggml_threadpool_new(&params);
                }
                if (!pool) throw std::runtime_error("native VE threadpool initialization failed");
                ggml_backend_cpu_set_threadpool(value,pool);
                std::fprintf(stderr,"SD_THREADPOOL stage=%s threads=%ld poll=0 shared=%d\n",sd_profile_stage(),threads,int(shared));
            }
        } catch (...) {
            if (pool) { if (shared) sd_shared_threadpool_release(pool); else ggml_threadpool_free(pool); }
            ggml_backend_free(value);
            throw;
        }
    }
    ~VeSdBackend() {
        ggml_backend_free(value);
        if (pool) { if (shared) sd_shared_threadpool_release(pool); else ggml_threadpool_free(pool); }
    }
    VeSdBackend(const VeSdBackend&) = delete;
    VeSdBackend& operator=(const VeSdBackend&) = delete;
};
inline void ve_sd_prepare(ModelLoader& loader, const std::string& file,
                          const std::string& prefix) {
    if (!loader.init_from_file(file, prefix))
        throw std::runtime_error("verified FP32 component index load failed");
    loader.set_wtype_override(GGML_TYPE_F32);
}
template<class Runner>
void ve_sd_weights(ModelLoader& loader, Runner& runner,
                   const std::string& prefix, ggml_backend_t backend) {
    SdProfileTimer timer;
    if (!runner.alloc_params_buffer()) throw std::runtime_error("component allocation failed");
    std::map<std::string,ggml_tensor*> tensors;
    runner.get_param_tensors(tensors,prefix);
    if (!loader.load_tensors(tensors,backend)) throw std::runtime_error("component weight loading failed");
    timer.emit(sd_profile_stage(),"weights_load");
}
#endif
