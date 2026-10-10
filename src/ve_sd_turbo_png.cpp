/* Same STB implementation/settings as the native pipeline, isolated for O2. */
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
extern "C" int ve_sd_turbo_png_write(const char *path,int width,int height,const unsigned char *rgb,int stride) {
    if(!path || !rgb || width<1 || width>512 || height<1 || height>512 || stride<width*3) return 0;
    return stbi_write_png(path,width,height,3,rgb,stride);
}
