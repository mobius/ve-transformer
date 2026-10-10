// Fixed SD-Turbo validation entry: actual CLIP, UNet and VAE execute on VE.
#include "ve_sd_turbo_stages.h"
#include "ve_sd_turbo_profile.h"
#include "sd_baseline_validation.h"
#include "stable-diffusion.h"
#include <algorithm>
#include <memory>
#include <cstdio>
#include <map>
#include <fstream>
#include "ggml-cpu.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

extern "C" int ve_sd_turbo_png_write(const char*,int,int,const unsigned char*,int);
extern "C" int ve_sd_turbo_pixel_clamp(float*,size_t);
extern "C" int ve_sd_turbo_pixel_pack(const float*,unsigned char*,size_t);

// This validation pipeline executes components sequentially on one main
// thread. References own the pool; backend attachments never destroy it.
static ggml_threadpool_t shared_pool=NULL;
static int shared_references=0,shared_threads=0;
extern "C" void* sd_shared_threadpool_acquire(int threads) {
    if (!shared_pool) {
        ggml_threadpool_params params=ggml_threadpool_params_default(threads);
        params.poll=0;params.paused=true;
        shared_pool=ggml_threadpool_new(&params);
        if (!shared_pool) throw std::runtime_error("shared pool allocation failed");
        shared_threads=threads;
        std::fprintf(stderr,"SD_SHARED_POOL created=1 threads=%d poll=0\n",threads);
    }
    if (threads!=shared_threads) throw std::runtime_error("shared pool thread count mismatch");
    ++shared_references;
    return shared_pool;
}
extern "C" void sd_shared_threadpool_release(void* pool) {
    if (!pool) return;
    if (pool!=shared_pool || shared_references<=0) std::abort();
    if (--shared_references==0) {
        ggml_threadpool_free(shared_pool);shared_pool=NULL;shared_threads=0;
        std::fprintf(stderr,"SD_SHARED_POOL released=1 references=0\n");
    }
}

int main(int argc,char** argv) {
    try {
        SdProfileTimer total_timer;
        std::fprintf(stderr,"SD_NATIVE_BOOT argc=%d\n",argc);
        sd_set_log_callback([](sd_log_level_t,const char* message,void*) {
            std::fputs(message,stderr);
        },NULL);
        std::map<std::string,std::string> args;
        const std::vector<std::string> allowed={"-m","-p","-W","-H","--steps","--cfg-scale",
            "--clip-skip","--sampling-method","--schedule","--type","-t","-s","-o"};
        for(int i=1;i<argc;i+=2) {
            if(i+1>=argc || std::find(allowed.begin(),allowed.end(),argv[i])==allowed.end()
                || !args.insert({argv[i],argv[i+1]}).second)
                throw std::runtime_error("unsupported or duplicate validation argument");
        }
        for(auto& name:allowed) if(!args.count(name)) throw std::runtime_error("missing validation argument");
        std::fprintf(stderr,"SD_NATIVE_CONFIG parsed\n");
        int steps=std::stoi(args["--steps"]),threads=std::stoi(args["-t"]);
        if(args["-W"]!="512" || args["-H"]!="512" || (steps!=1 && steps!=4)
            || (threads!=1 && threads!=2 && threads!=4 && threads!=8)
            || args["--cfg-scale"]!="1" || args["--clip-skip"]!="2"
            || args["--sampling-method"]!="euler" || args["--schedule"]!="discrete"
            || args["--type"]!="f32") throw std::runtime_error("unsupported SD-Turbo validation configuration");
        if (setenv("SD_POOL_THREADS",args["-t"].c_str(),1))
            throw std::runtime_error("pool configuration failed");
        for(const char* name:{"SD_FIXED_NOISE","SD_FIXED_SIGMAS","SD_TRACE_DIR"})
            if(!std::getenv(name) || !*std::getenv(name)) throw std::runtime_error("fixed validation arrays required");
        const char* resident_flag=std::getenv("SD_RESIDENT_WEIGHTS");
        if (resident_flag && std::strcmp(resident_flag,"0") && std::strcmp(resident_flag,"1"))
            throw std::runtime_error("invalid residency configuration");
        bool resident=resident_flag && std::strcmp(resident_flag,"1")==0;
        // Five tab-separated fields: prompt, noise, sigmas, trace directory, PNG.
        // The host verifier creates and validates these local request files.
        std::vector<std::vector<std::string> > requests;
        const char* request_file=std::getenv("SD_REQUEST_MANIFEST");
        std::fprintf(stderr,"SD_NATIVE_CONFIG manifest_begin\n");
        if (request_file && *request_file) {
            std::ifstream stream(request_file);
            if (!stream) throw std::runtime_error("request manifest unavailable");
            std::string line;
            while (std::getline(stream,line)) {
                std::fprintf(stderr,"SD_NATIVE_CONFIG manifest_row\n");
                std::vector<std::string> fields;
                std::fprintf(stderr,"SD_NATIVE_CONFIG fields_ready\n");
                size_t begin=0;
                bool carriage_return=false;
                for (size_t i=0;i<line.size();++i) if (line[i]=='\r') carriage_return=true;
                for (;;) {
                    // NCC O0: string::find(char) crashes in a standalone VE
                    // reproducer. Direct scanning preserves the TSV grammar.
                    size_t end=begin;
                    while (end<line.size() && line[end]!='\t') ++end;
                    std::fprintf(stderr,"SD_NATIVE_CONFIG field_found\n");
                    fields.push_back(line.substr(begin,end-begin));
                    std::fprintf(stderr,"SD_NATIVE_CONFIG field_copied\n");
                    if (end==line.size()) break;
                    begin=end+1;
                }
                if (fields.size()!=5 || carriage_return)
                    throw std::runtime_error("invalid request manifest row");
                for (const auto& value:fields) if (value.empty())
                    throw std::runtime_error("empty request manifest field");
                requests.push_back(fields);
                std::fprintf(stderr,"SD_NATIVE_CONFIG manifest_row_ready\n");
                if (requests.size()>16) throw std::runtime_error("too many validation requests");
            }
            if (!stream.eof() || requests.empty()) throw std::runtime_error("request manifest incomplete");
        } else {
            requests.push_back({args["-p"],std::getenv("SD_FIXED_NOISE"),
                std::getenv("SD_FIXED_SIGMAS"),std::getenv("SD_TRACE_DIR"),args["-o"]});
        }
        const char* shared_flag=std::getenv("SD_SHARED_THREADPOOL");
        if (shared_flag && std::strcmp(shared_flag,"0") && std::strcmp(shared_flag,"1"))
            throw std::runtime_error("invalid shared pool configuration");
        bool shared=shared_flag && std::strcmp(shared_flag,"1")==0;
        if (resident && !shared)
            throw std::runtime_error("resident validation requires the shared pipeline pool");
        const char* persistent_flag=std::getenv("SD_PERSISTENT_THREADPOOL");
        if (shared && (!persistent_flag || std::strcmp(persistent_flag,"1")))
            throw std::runtime_error("shared pool requires persistent mode");
        std::unique_ptr<void,decltype(&sd_shared_threadpool_release)> pipeline_pool(NULL,sd_shared_threadpool_release);
        if (shared) pipeline_pool.reset(sd_shared_threadpool_acquire(threads));
        std::unique_ptr<void,decltype(&ve_sd_clip_close)> clip(NULL,ve_sd_clip_close);
        std::fprintf(stderr,"SD_NATIVE_CONFIG requests_ready\n");
        std::unique_ptr<void,decltype(&ve_sd_unet_close)> unet(NULL,ve_sd_unet_close);
        std::unique_ptr<void,decltype(&ve_sd_vae_close)> vae(NULL,ve_sd_vae_close);
        const char* rgb_flag=std::getenv("SD_REUSE_RGB_BUFFER");
        if (rgb_flag && std::strcmp(rgb_flag,"0") && std::strcmp(rgb_flag,"1"))
            throw std::runtime_error("invalid RGB buffer configuration");
        const bool reuse_rgb=rgb_flag && std::strcmp(rgb_flag,"1")==0;
        std::vector<unsigned char> resident_rgb;
        const char* pixels_flag=std::getenv("SD_VE_PIXELS");
        if (pixels_flag && std::strcmp(pixels_flag,"0") && std::strcmp(pixels_flag,"1"))
            throw std::runtime_error("invalid pixel kernel configuration");
        const bool pixels_ve=pixels_flag && std::strcmp(pixels_flag,"1")==0;
        for (size_t request=0;request<requests.size();++request) {
        const int64_t request_started=ggml_time_us();
        const auto& config=requests[request];
        if (setenv("SD_FIXED_NOISE",config[1].c_str(),1) ||
            setenv("SD_FIXED_SIGMAS",config[2].c_str(),1) ||
            setenv("SD_TRACE_DIR",config[3].c_str(),1))
            throw std::runtime_error("request environment configuration failed");
        std::fprintf(stderr,"SD_REQUEST_BEGIN index=%zu resident=%d\n",request,int(resident));
        ggml_init_params params={32*1024*1024,NULL,false};
        std::unique_ptr<ggml_context,decltype(&ggml_free)> work(ggml_init(params),ggml_free);
        if(!work) throw std::runtime_error("validation workspace allocation failed");
        std::fprintf(stderr,"SD_NATIVE_STAGE clip\n");
        setenv("SD_PROFILE_STAGE","clip",1);
        SdProfileTimer clip_timer;
        if (!clip) {
            clip.reset(ve_sd_clip_open(args["-m"]));
            std::fprintf(stderr,"SD_WEIGHT_LOAD request=%zu stage=clip\n",request);
        }
        auto embeddings=ve_sd_clip_run(clip.get(),config[0],threads,work.get());
        if (!resident) clip.reset();
        clip_timer.emit("clip","stage_total");
        sd_trace("native-embeddings",embeddings);
        auto noise=ggml_new_tensor_4d(work.get(),GGML_TYPE_F32,64,64,4,1);
        sd_load_noise(noise);sd_trace("native-noise",noise);
        std::vector<float> sigmas(steps+1);sd_load_sigmas(sigmas,steps);
        auto latent=ggml_dup_tensor(work.get(),noise);
        float* values=static_cast<float*>(latent->data);
        float* initial=static_cast<float*>(noise->data);
        size_t count=ggml_nelements(noise);
        for(size_t i=0;i<count;++i) values[i]=initial[i]*sigmas[0];
        setenv("SD_PROFILE_STAGE","unet",1);
        SdProfileTimer unet_load_timer;
        if (!unet) {
            unet.reset(ve_sd_unet_open(args["-m"]));
            std::fprintf(stderr,"SD_WEIGHT_LOAD request=%zu stage=unet\n",request);
        }
        unet_load_timer.emit("unet","load_total");
        for(int step=0;step<steps;++step) {
            float sigma=sigmas[step];
            auto input=ggml_dup_tensor(work.get(),noise);
            float* scaled=static_cast<float*>(input->data);
            float scale=1.f/std::sqrt(sigma*sigma+1.f);
            for(size_t i=0;i<count;++i) scaled[i]=values[i]*scale;
            auto timestep=ggml_new_tensor_1d(work.get(),GGML_TYPE_F32,1);
            // Official trailing schedule: 999 for one step; 999,749,499,249 for four.
            static_cast<float*>(timestep->data)[0]=999.f-float(step*(1000/steps));
            std::string prefix="native-step"+std::to_string(step);
            sd_trace((prefix+"-input").c_str(),input);
            sd_trace((prefix+"-timestep").c_str(),timestep);
            SdProfileTimer unet_step_timer;
            auto epsilon=ve_sd_unet_run(unet.get(),input,timestep,embeddings,threads,work.get());
            unet_step_timer.emit("unet","step_total");
            sd_trace((prefix+"-epsilon").c_str(),epsilon);
            float* predicted=static_cast<float*>(epsilon->data);
            for(size_t i=0;i<count;++i) {
                float denoised=values[i]-sigma*predicted[i];
                float derivative=(values[i]-denoised)/sigma;
                values[i]=values[i]+derivative*(sigmas[step+1]-sigma);
            }
        }
        if (resident) ve_sd_unet_release_compute(unet.get());
        else unet.reset();
        sd_trace("native-latent",latent);
        for(size_t i=0;i<count;++i) values[i]=values[i]/.18215f;
        setenv("SD_PROFILE_STAGE","vae",1);
        SdProfileTimer vae_timer;
        if (!vae) {
            vae.reset(ve_sd_vae_open(args["-m"]));
            std::fprintf(stderr,"SD_WEIGHT_LOAD request=%zu stage=vae\n",request);
        }
        auto decoded=ve_sd_vae_run(vae.get(),latent,threads,work.get());
        if (!resident) vae.reset();
        vae_timer.emit("vae","stage_total");
        SdProfileTimer output_timer;
        const char* output_flag=std::getenv("SD_IMAGE_OUTPUT_PROFILE");
        const bool output_profile=output_flag && std::strcmp(output_flag,"1")==0;
        int64_t output_part_start=output_profile ? ggml_time_us() : 0;
        auto output_part=[&](const char* part) {
            if (output_profile) {
                std::fprintf(stderr,"SD_IMAGE_OUTPUT_PROFILE part=%s seconds=%.9f calls=1\n",
                             part,double(ggml_time_us()-output_part_start)/1e6);
                output_part_start=ggml_time_us();
            }
        };

        if(ggml_nelements(decoded)!=512*512*3) throw std::runtime_error("decoded shape mismatch");
        float* pixels=static_cast<float*>(decoded->data);
        const bool rgb_reused=reuse_rgb && !resident_rgb.empty();
        std::vector<unsigned char> request_rgb(reuse_rgb ? 0 : 512*512*3);
        if (reuse_rgb && resident_rgb.empty()) resident_rgb.resize(512*512*3);
        std::vector<unsigned char>& rgb=reuse_rgb ? resident_rgb : request_rgb;
        std::fprintf(stderr,"SD_RGB_BUFFER mode=%s allocated=%d reused=%d bytes=786432\n",
                     reuse_rgb?"resident":"request",int(!rgb_reused),int(rgb_reused));
        output_part("prepare");
        if (pixels_ve) {
            if (!ve_sd_turbo_pixel_clamp(pixels,512*512*3))
                throw std::runtime_error("nonfinite decoded pixels");
        } else for(size_t i=0;i<512*512*3;++i) {
            if(!std::isfinite(pixels[i])) throw std::runtime_error("nonfinite decoded pixels");
            pixels[i]=std::min(1.f,std::max(0.f,(pixels[i]+1.f)*.5f));
        }
        output_part("pixel_clamp");
        sd_trace("native-decoded",decoded);
        output_part("decoded_trace");
        if (pixels_ve) {
            if (!ve_sd_turbo_pixel_pack(pixels,rgb.data(),512*512))
                throw std::runtime_error("pixel packing failed");
        } else for(size_t i=0;i<512*512;++i) for(int c=0;c<3;++c)
            rgb[3*i+c]=static_cast<unsigned char>(std::round(pixels[c*512*512+i]*255.f));
        std::fprintf(stderr,"SD_PIXEL_KERNEL mode=%s clamp_calls=1 pack_calls=1 spatial=262144 bytes=786432\n",
                     pixels_ve?"ve":"generic");
        output_part("pixel_pack");
        const char* png_flag=std::getenv("SD_VE_PNG");
        const bool png_ve=png_flag && std::strcmp(png_flag,"1")==0;
        const int png_ok=png_ve ? ve_sd_turbo_png_write(config[4].c_str(),512,512,rgb.data(),512*3)
                               : stbi_write_png(config[4].c_str(),512,512,3,rgb.data(),512*3);
        if(!png_ok) throw std::runtime_error("image write failed");
        std::fprintf(stderr,"SD_PNG_ENCODER mode=%s width=512 height=512 channels=3 stride=1536\n",png_ve?"ve":"generic");
        output_part("png_encode");
        output_timer.emit("pipeline","image_output");
        // Include the per-request ggml output workspace cleanup in latency.
        // Every saved trace and PNG has finished consuming its tensors here.
        work.reset();
        std::fprintf(stderr,"SD_REQUEST_END index=%zu seconds=%.9f\n",request,
                     double(ggml_time_us()-request_started)/1e6);
        }
        clip.reset();unet.reset();vae.reset();
        pipeline_pool.reset();
        total_timer.emit("pipeline","total");
        std::fprintf(stderr,"SD_TURBO_NATIVE_COMPLETE steps=%d threads=%d\n",steps,threads);
        return 0;
    } catch(const std::exception& error) {
        std::fprintf(stderr,"SD-Turbo validation failed: %s\n",error.what());return 1;
    }
}
