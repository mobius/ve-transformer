#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

static_assert(sizeof(float)==sizeof(uint32_t) &&
              std::numeric_limits<float>::is_iec559 &&
              std::numeric_limits<float>::radix==2 &&
              std::numeric_limits<float>::digits==24 &&
              std::numeric_limits<float>::max_exponent==128,
              "pixel classification requires IEEE binary32");

static inline bool pixel_finite_bits(const float* value) {
    uint32_t word;
    std::memcpy(&word,value,sizeof word);
    return (word & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

// Scan first, then write only the original valid prefix. The externally
// observable return value and every byte match the early-return baseline.
extern "C" int ve_sd_turbo_pixel_clamp(float* pixels, size_t count) {
    if (!pixels || count<1 || count>512*512*3) return 0;
    int invalid=0;
    for (size_t i=0;i<count;++i)
        invalid |= int(!pixel_finite_bits(pixels+i));
    size_t valid=count;
    if (invalid) {
        valid=0;
        while (valid<count && pixel_finite_bits(pixels+valid)) ++valid;
    }
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
