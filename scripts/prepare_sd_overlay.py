"""Generate narrowly scoped image validation sources; never edit upstream."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'build/vendor/stable-diffusion.cpp'
OUT=ROOT/'build/sd-overlay'
OUT.mkdir(exist_ok=True)

def write(name,text):
    path=OUT/name
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists() or path.read_text()!=text:path.write_text(text)

text=(SOURCE/'ggml/src/ggml.c').read_text()
old='a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16'
start=text.index('struct ggml_tensor * ggml_conv_2d(')
end=text.index('// a: [OC*IC',start)
part=text[start:end]
assert part.count(old)==1
part=part.replace(old,'(a->type == GGML_TYPE_BF16 || a->type == GGML_TYPE_F32) ? GGML_TYPE_F32 : GGML_TYPE_F16')
write('sd-ggml.c',text[:start]+part+text[end:])
text=(SOURCE/'ggml/src/gguf.cpp').read_text()
start=text.index('    template <typename T>\n    const T & get_val(')
end=text.index('\n    void cast(',start)
part=text[start:end]
old='''        }
        const size_t type_size = gguf_type_size(type);
        GGML_ASSERT(data.size() % type_size == 0);
        GGML_ASSERT(data.size() >= (i+1)*type_size);
        return reinterpret_cast<const T *>(data.data())[i];
'''
new='''        } else {
            const size_t type_size = gguf_type_size(type);
            GGML_ASSERT(data.size() % type_size == 0);
            GGML_ASSERT(data.size() >= (i+1)*type_size);
            return reinterpret_cast<const T *>(data.data())[i];
        }
'''
assert part.count(old)==1
write('sd-gguf.cpp',text[:start]+part.replace(old,new)+text[end:])
text=(SOURCE/'src/core/ggml_extend.cpp').read_text()
# Official SD-Turbo CLIP and U-Net GEGLU use erf GELU. The upstream
# helper uses tanh GELU with FP16 lookup on the generic backend.
assert text.count('x = ggml_gelu_inplace(ctx, x);')==1
assert text.count('x = ggml_gelu(ctx, x);')==1
text=text.replace('x = ggml_gelu_inplace(ctx, x);','x = ggml_gelu_erf_inplace(ctx, x);')
text=text.replace('x = ggml_gelu(ctx, x);','x = ggml_gelu_erf(ctx, x);')
write('sd-ggml-extend.cpp',text)
text=(SOURCE/'src/convert.cpp').read_text()
disabled='#include "stable-diffusion.h"\n#include <cstdio>\n'
for signature in ('bool convert_with_components(', 'bool convert('):
    start=text.index(signature)
    end=text.index(' {',start)
    disabled+=text[start:end]+''' {
    std::fputs("Model conversion is disabled in this VE inference build; use a host converter.\\n",stderr);
    return false;
}
'''
write('sd-convert-disabled.cpp',disabled)
text=(SOURCE/'src/detailer.cpp').read_text()
disabled='#include "stable-diffusion.h"\n#include <cstdio>\n'
bodies={
    'adetailer_ctx_t* new_adetailer_ctx(':''' {
    std::fputs("Automatic detailing is disabled in this VE inference build.\\n",stderr);
    return nullptr;
}
''',
    'void free_adetailer_ctx(':''' { (void)context; }
''',
    'bool adetail_image(':''' {
    if (images_out) *images_out=nullptr;
    if (num_images_out) *num_images_out=0;
    std::fputs("Automatic detailing is disabled in this VE inference build.\\n",stderr);
    return false;
}
'''}
for signature,body in bodies.items():
    start=text.index(signature);end=text.index(' {',start)
    disabled+=text[start:end]+body
write('sd-detailer-disabled.cpp',disabled)
text=(SOURCE/'src/pipeline/model_builders.cpp').read_text()
# Retain the actual upstream SD2 constructors, excluding unrelated factories
# which trigger NCC's front-end crash when instantiated together.
import re
allowed={'model/diffusion/control.hpp','model/diffusion/model.hpp',
         'model/diffusion/unet.hpp','model/vae/auto_encoder_kl.hpp','model/vae/vae.hpp'}
text=re.sub(r'^#include "(model/(?:diffusion|vae|audio)/[^\"]+)"\n',
            lambda m:m[0] if m[1] in allowed else '',text,flags=re.M)
start=text.index('        if (sd_version_is_sd3(version)) {')
branch=text.index('        } else {  // SD1.x SD2.x SDXL',start)
end=text.index('\n        if (strlen(SAFE_STR(sd_ctx_params->ip_adapter_path))',branch)
classic=text[branch:end].split('\n',1)[1]
assert classic.endswith('        }\n')
classic=classic[:-len('        }\n')]
text=text[:start]+'''        if (version != VERSION_SD2) {
            LOG_ERROR("This VE image build supports SD2 runners only");
            return false;
        }
'''+classic+text[end:]
start=text.index('        auto create_tae = ')
end=text.index('        if (sd_ctx_params->vae_conv_direct)',start)
text=text[:start]+'''        if (version != VERSION_SD2 || options.use_tae || options.use_audio_vae ||
            sd_ctx_params->vae_format != SD_VAE_FORMAT_AUTO) {
            LOG_ERROR("This VE image build requires the standard SD2 VAE");
            return false;
        }
        LOG_INFO("using VAE for encoding / decoding");
        result.vae = std::make_shared<AutoEncoderKL>(
            ctx.backends.runtime_backend(SDBackendModule::VAE), tensor_storage_map,
            "first_stage_model", false, false, version, weight_manager);
'''+text[end:]
write('sd-model-builders.cpp',text)
text=(SOURCE/'src/pipeline/video.cpp').read_text()
disabled='#include "generation.h"\n#include <cstdio>\nnamespace sd::pipeline {\n'
for signature,body in {
 '    bool generate_video(':''' {
    if (frames_out) *frames_out=nullptr;
    if (num_frames_out) *num_frames_out=0;
    if (audio_out) *audio_out=nullptr;
    if (fps_out) *fps_out=0;
    std::fputs("Video generation is disabled in this VE image build.\\n",stderr);
    return false;
}
''',
 '    sd::Tensor<float> upscale_ltx_spatial_video_latent(':''' {
    std::fputs("Video latent upscaling is disabled in this VE image build.\\n",stderr);
    return {};
}
'''}.items():
    start=text.index(signature);end=text.index(' {',start)
    disabled+=text[start:end]+body
write('sd-video-disabled.cpp',disabled+'}\n')
text=(SOURCE/'src/model/common/ggml_block.hpp').read_text()
old='''    GroupNorm32(int64_t num_channels)
        : GroupNorm(32, num_channels, 1e-06f) {}'''
new='''    GroupNorm32(int64_t num_channels, float eps = 1e-06f)
        : GroupNorm(32, num_channels, eps) {}'''
assert text.count(old)==1
write('include/model/common/ggml_block.hpp',text.replace(old,new))
text=(SOURCE/'src/model/common/block.hpp').read_text()
for old,new in [('new GroupNorm32(channels));','new GroupNorm32(channels, 1e-05f));'),
                ('new GroupNorm32(out_channels));','new GroupNorm32(out_channels, 1e-05f));')]:
    assert text.count(old)==1
    text=text.replace(old,new)
write('include/model/common/block.hpp',text)
text=(SOURCE/'src/model/diffusion/unet.hpp').read_text()
old='new GroupNorm32(ch));  // ch == model_channels'
assert text.count(old)==1
write('include/model/diffusion/unet.hpp',text.replace(old,'new GroupNorm32(ch, 1e-05f));  // official SD-Turbo norm_eps'))
helper=r'''
#ifndef SD_VALIDATION_H
#define SD_VALIDATION_H
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include "core/tensor.hpp"
static void sd_load_fixed_noise(sd::Tensor<float>& tensor) {
    const char *path=std::getenv("SD_FIXED_NOISE");
    if (!path || !*path) return;
    FILE *f=std::fopen(path,"rb");
    if (!f) throw std::runtime_error("fixed noise open failed");
    const size_t n=static_cast<size_t>(tensor.numel());
    const bool good=std::fread(tensor.data(),sizeof(float),n,f)==n &&
                    std::fgetc(f)==EOF && !std::ferror(f);
    std::fclose(f);
    if (!good) throw std::runtime_error("fixed noise shape or read failure");
    for (size_t i=0;i<n;++i) if (!std::isfinite(tensor.data()[i]))
        throw std::runtime_error("nonfinite fixed noise");
}
static void sd_trace_tensor(const char *name,const sd::Tensor<float>& tensor) {
    const char *dir=std::getenv("SD_TRACE_DIR");
    if (!dir || !*dir) return;
    std::string path=std::string(dir)+"/"+name+".f32";
    FILE *f=std::fopen(path.c_str(),"wb");
    if (!f) throw std::runtime_error("trace open failed");
    const size_t n=static_cast<size_t>(tensor.numel());
    bool good=std::fwrite(tensor.data(),sizeof(float),n,f)==n;
    if (std::fclose(f)) good=false;
    if (!good) throw std::runtime_error("trace write failed");
}
#endif
'''
write('sd_validation.h',helper)
text=(SOURCE/'src/pipeline/image.cpp').read_text()
text='#include "sd_validation.h"\n'+text
old='sd::Tensor<float> noise = sd::randn_like<float>(latents.init_latent, sd->rng);'
assert text.count(old)==1
text=text.replace(old,old+'\n            sd_load_fixed_noise(noise);\n            sd_trace_tensor("native-noise",noise);\n            sd_trace_tensor("native-embeddings",embeds.cond.c_crossattn);')
old='if (!x_0.empty()) {'
assert text.count(old)==2
text=text.replace(old,old+'\n                sd_trace_tensor("native-latent",x_0);',1)
old='''                decoded_images.push_back(std::move(image));
            }
            int64_t t2'''
new='''                sd_trace_tensor("native-decoded",image);
                decoded_images.push_back(std::move(image));
            }
            int64_t t2'''
assert text.count(old)==1
text=text.replace(old,new)
write('sd-image.cpp',text)
text=(SOURCE/'src/pipeline/diffusion_engine.cpp').read_text()
text='#include "sd_validation.h"\n'+text
old='''            auto output_opt = work_diffusion_model->compute(n_threads, diffusion_params);'''
new='''            const std::string trace_step="native-step"+std::to_string(step - 1);
            sd_trace_tensor((trace_step+"-input").c_str(),noised_input);
            sd_trace_tensor((trace_step+"-timestep").c_str(),timesteps_tensor);
            auto output_opt = work_diffusion_model->compute(n_threads, diffusion_params);'''
assert text.count(old)==1
text=text.replace(old,new)
old='''            step_cache.after_condition(&condition, noised_input, output_opt);'''
assert text.count(old)==1
text=text.replace(old,'''            sd_trace_tensor((trace_step+"-epsilon").c_str(),output_opt);
'''+old)
write('sd-diffusion-engine.cpp',text)
print('Generated FP32 convolution and bounded validation overlays; original checkout unchanged')
