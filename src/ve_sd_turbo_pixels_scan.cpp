#include <algorithm>
#include <cmath>
#include <cstddef>

// Scan first, then write only the original valid prefix. The externally
// observable return value and every byte match the early-return baseline.
extern "C" int ve_sd_turbo_pixel_clamp(float* pixels, size_t count) {
    if (!pixels || count<1 || count>512*512*3) return 0;
    size_t valid=count;
    for (size_t i=0;i<count;++i)
        valid=std::min(valid,std::isfinite(pixels[i]) ? count : i);
    for (size_t i=0;i<valid;++i)
        pixels[i]=std::min(1.f,std::max(0.f,(pixels[i]+1.f)*.5f));
    return valid==count;
}

extern "C" int ve_sd_turbo_pixel_pack(const float* pixels, unsigned char* rgb, size_t spatial) {
    if (!pixels || !rgb || spatial<1 || spatial>512*512) return 0;
    for (int c=0;c<3;++c) for (size_t i=0;i<spatial;++i)
        rgb[3*i+c]=static_cast<unsigned char>(std::round(pixels[c*spatial+i]*255.f));
    return 1;
}
