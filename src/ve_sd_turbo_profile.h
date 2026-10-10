#ifndef VE_SD_TURBO_PROFILE_H
#define VE_SD_TURBO_PROFILE_H
#include "ggml.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

inline bool sd_profile_enabled() {
    const char* flag=std::getenv("SD_PROFILE");
    return flag && std::strcmp(flag,"1")==0;
}
inline const char* sd_profile_stage() {
    const char* stage=std::getenv("SD_PROFILE_STAGE");
    return stage ? stage : "unknown";
}
inline void sd_profile_emit(const char* stage,const char* part,double seconds,int calls=1) {
    if (sd_profile_enabled())
        std::fprintf(stderr,"SD_PROFILE stage=%s part=%s seconds=%.9f calls=%d\n",stage,part,seconds,calls);
}
struct SdProfileTimer {
    bool enabled;
    int64_t start;
    SdProfileTimer() : enabled(sd_profile_enabled()),start(enabled ? ggml_time_us() : 0) {}
    double seconds() const { return enabled ? double(ggml_time_us()-start)/1e6 : 0.; }
    void emit(const char* stage,const char* part) const {
        if (enabled) sd_profile_emit(stage,part,seconds());
    }
};
#endif
