/* Independent O1 C++ compilation of the pipeline's original STB encoder. */
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
extern "C" int sd_png_reference(const char *path,int width,int height,const unsigned char *rgb,int stride) {
    return stbi_write_png(path,width,height,3,rgb,stride);
}
