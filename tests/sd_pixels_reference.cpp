#include <algorithm>
#include <cmath>
#include <cstddef>
extern "C" int sd_pixels_reference_clamp(float* pixels,size_t count) {
    for (size_t i=0;i<count;++i) {
        if(!std::isfinite(pixels[i])) return 0;
        pixels[i]=std::min(1.f,std::max(0.f,(pixels[i]+1.f)*.5f));
    }
    return 1;
}
extern "C" int sd_pixels_reference_pack(const float* pixels,unsigned char* rgb,size_t spatial) {
    for(size_t i=0;i<spatial;++i) for(int c=0;c<3;++c)
        rgb[3*i+c]=static_cast<unsigned char>(std::round(pixels[c*spatial+i]*255.f));
    return 1;
}
