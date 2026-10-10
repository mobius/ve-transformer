"""Generate a fixed compact SD-Turbo port, without editing vendor sources."""
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'build/vendor/sd-turbo-baseline'
OUT=ROOT/'build/sd-baseline-overlay'
OUT.mkdir(exist_ok=True)
def write(name,text):
    p=OUT/name
    if not p.exists() or p.read_text()!=text:p.write_text(text)
write('sd_profile.h',(ROOT/'src/ve_sd_turbo_profile.h').read_text())
# Optional per-call dimensions/timing, leaving the vendor and GEMM arguments intact.
blas=(SOURCE/'ggml/src/ggml-blas/ggml-blas.cpp').read_text()
thread_control=r'''#include "sd_profile.h"
#ifdef _OPENMP
#include <omp.h>
#include <stdexcept>
struct SdNlcGraphThreads {
    int configured;
    int observed;
    int restore;
    bool emit;
    bool fixed;
    const char* stage;
    SdNlcGraphThreads() : configured(1),observed(1),restore(0),emit(false),fixed(false),stage(sd_profile_stage()) {
        SdProfileTimer prepare;
        const char* mode=std::getenv("SD_NLC_STAGE_THREADS");
        const char* unified=std::getenv("SD_NLC_FIXED_THREADS");
        fixed=unified && std::strcmp(unified,"8")==0;
        if (fixed) {
            configured=8;
            const char* vae=std::getenv("SD_NLC_VAE_THREADS");
            if (vae && std::strcmp(vae,"4")==0 && std::strcmp(stage,"vae")==0) configured=4;
            const char* selective=std::getenv("SD_NLC_VAE_BLAS_THREADS");
            if (selective && std::strcmp(selective,"4")==0 && std::strcmp(stage,"vae")==0) {
                if (vae && std::strcmp(vae,"4")==0) throw std::runtime_error("selective VAE requires generic eight");
                if (omp_get_max_threads()!=8) throw std::runtime_error("selective VAE must enter from eight");
                configured=4;
                restore=8;
            }
        }
        else if (mode && std::strcmp(mode,"1")==0) {
            if (std::strcmp(stage,"unet")==0) configured=8;
            else if (std::strcmp(stage,"vae")==0) configured=4;
        }
        if (omp_get_max_threads()!=configured) omp_set_num_threads(configured);
        const char* profile=std::getenv("SD_NLC_THREAD_PROFILE");
        emit=profile && std::strcmp(profile,"1")==0;
        if (emit) {
            int actual=1;
#pragma omp parallel shared(actual)
            {
#pragma omp single
                actual=omp_get_num_threads();
            }
            observed=actual;
            if (observed!=configured) {
                omp_set_num_threads(1);
                throw std::runtime_error("NLC parallel region thread count differs");
            }
        }
        prepare.emit(stage,"nlc_thread_prepare");
    }
    ~SdNlcGraphThreads() {
        SdProfileTimer release;
        if (restore && omp_get_max_threads()!=restore) omp_set_num_threads(restore);
        else if (!fixed && omp_get_max_threads()!=1) omp_set_num_threads(1);
        if (emit) std::fprintf(stderr,"SD_NLC_THREADS stage=%s configured=%d actual=%d idle_after=%d\n",
                              stage,configured,observed,omp_get_max_threads());
        release.emit(stage,"nlc_thread_release");
    }
};
#endif
'''
blas=thread_control+blas
entry='static enum ggml_status ggml_backend_blas_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {'
assert blas.count(entry)==1
blas=blas.replace(entry,entry+'\n#ifdef _OPENMP\n    SdNlcGraphThreads nlc_threads;\n#endif')
needle='''            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        ne1, ne01, ne10,
                        1.0f,   y, ne10,
                                x, ne00,
                        0.0f,   d, ne01);'''
assert blas.count(needle)==1
instrumented='''            const char* profile_flag=std::getenv("SD_NLC_PROFILE");
            const bool profile_nlc=profile_flag && std::strcmp(profile_flag,"1")==0;
            const int64_t profile_start=profile_nlc ? ggml_time_us() : 0;
'''+'            bool tile_vae=false;\n#ifdef _OPENMP\n            const char* spatial_flag=std::getenv("SD_NLC_VAE_SPATIAL_TILE");\n            tile_vae=spatial_flag && std::strcmp(spatial_flag,"8192")==0 &&\n                std::strcmp(sd_profile_stage(),"vae")==0 && omp_get_max_threads()==4 &&\n                ((ne1==256 && ne01==262144 && ne10==2304) ||\n                 (ne1==512 && ne01==65536 && ne10==4608) ||\n                 (ne1==256 && ne01==65536 && (ne10==2304 || ne10==4608)) ||\n                 (ne1==128 && ne01==262144 && ne10==2304));\n\n#endif\n            if (tile_vae) {\n                for (int64_t offset=0; offset<ne01; offset+=8192) {\n                    const int64_t width=std::min<int64_t>(8192,ne01-offset);\n                    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,\n                        ne1,width,ne10,1.0f,y,ne10,x+offset*ne00,ne00,0.0f,d+offset,ne01);\n                }\n                std::fprintf(stderr,"SD_NLC_SPATIAL_TILE stage=vae m=%lld n=%lld k=%lld tile=8192 calls=%lld\\n",\n                    (long long)ne1,(long long)ne01,(long long)ne10,(long long)((ne01+8191)/8192));\n            } else {\n            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,\n                        ne1, ne01, ne10,\n                        1.0f,   y, ne10,\n                                x, ne00,\n                        0.0f,   d, ne01);\n            }\n'+'''
            if (profile_nlc) {
                const double elapsed=double(ggml_time_us()-profile_start)/1e6;
                std::fprintf(stderr,"SD_NLC_GEMM stage=%s m=%lld n=%lld k=%lld lda=%lld ldb=%lld ldc=%lld trans_a=N trans_b=T seconds=%.9f\\n",
                    sd_profile_stage(),(long long)ne1,(long long)ne01,(long long)ne10,
                    (long long)ne10,(long long)ne00,(long long)ne01,elapsed);
            }'''
blas=blas.replace(needle,instrumented)
from prepare_sd_gemm_spatial_small_model import instrument as instrument_small_spatial
from prepare_sd_gemm_untiled_model import instrument as instrument_untiled_spatial
from prepare_sd_gemm_pretranspose_model import instrument as instrument_pretranspose
write('sd-ggml-blas.cpp',instrument_pretranspose(instrument_untiled_spatial(instrument_small_spatial(blas))))
modified={'ggml_extend.hpp','common.hpp','unet.hpp','stable-diffusion.cpp','util.h','model.cpp','model.h','t5.hpp'}
for p in SOURCE.iterdir():
    if p.is_file() and p.suffix in ('.h','.hpp','.cpp') and 'vocab' not in p.name.lower() and p.name not in modified:
        write(p.name,p.read_text())
# Keep the actual CLIP bytes unchanged, but out of the template-heavy loader.
# SD-Turbo never uses the much larger embedded T5 tokenizer.
vocab=(SOURCE/'vocab.hpp').read_text()
clip=vocab[:vocab.index('static unsigned char t5_tokenizer_json_str[]')]
count=len(re.findall(r'0x[0-9a-fA-F]{2}',clip))
assert count>500000
write('sd_clip_vocab.c',clip.replace('static unsigned char merges_utf8_c_str[]',
                                   'unsigned char merges_utf8_c_str[]'))
write('sd_clip_vocab.h',f'extern "C" {{ extern unsigned char merges_utf8_c_str[{count}]; }}\n')
text=(SOURCE/'model.cpp').read_text().replace('#include "vocab.hpp"','#include "sd_clip_vocab.h"\n#include <stdexcept>')
old='''    std::string json_str(reinterpret_cast<const char*>(t5_tokenizer_json_str), sizeof(t5_tokenizer_json_str));
    return json_str;'''
assert text.count(old)==1
text=text.replace(old,'    throw std::runtime_error("T5 tokenizer is unavailable in the SD-Turbo validation baseline");')
start=text.index('bool is_safetensors_file(');end=text.index('\nbool ModelLoader::init_from_file',start)
text=text[:start]+'''bool is_safetensors_file(const std::string& file_path) {
    std::ifstream index(file_path + ".ve-index");
    std::string magic;
    return bool(index >> magic) && magic == "VE_SDTURBO_INDEX_V1";
}
'''+text[end:]
start=text.index('bool ModelLoader::init_from_safetensors_file(')
end=text.index('/*================================================= DiffusersModelLoader',start)
text=text[:start]+'''bool ModelLoader::init_from_safetensors_file(const std::string& file_path, const std::string& prefix) {
    std::ifstream file(file_path,std::ios::binary);
    std::ifstream index(file_path+".ve-index");
    if (!file || !index) return false;
    std::string magic;
    size_t bytes=0,header=0,count=0;
    if (!(index>>magic>>bytes>>header>>count) || magic!="VE_SDTURBO_INDEX_V1" ||
        header<2 || header>16*1024*1024 || count==0 || count>100000) return false;
    file.seekg(0,std::ios::end);
    if (size_t(file.tellg())!=bytes || header+8>=bytes) return false;
    file.seekg(0);
    uint8_t header_buf[8];
    if (!file.read(reinterpret_cast<char*>(header_buf),8) || read_u64(header_buf)!=header) return false;
    file_paths_.push_back(file_path);
    size_t file_index=file_paths_.size()-1;
    std::set<std::string> names;
    for(size_t row=0;row<count;++row) {
        std::string name;int dims=0;int64_t ne[SD_MAX_DIMS]={1,1,1,1,1};
        if (!(index>>name>>dims) || dims<1 || dims>4 || !names.insert(name).second) return false;
        size_t elements=1;
        for(int i=0;i<dims;++i) {
            if (!(index>>ne[i]) || ne[i]<=0 || size_t(ne[i])>(bytes/4)/elements) return false;
            elements*=size_t(ne[i]);
        }
        size_t begin=0,end=0;
        if (!(index>>begin>>end) || end<begin || end>bytes-header-8 || end-begin!=elements*4) return false;
        if (is_unused_tensor(name)) continue;
        TensorStorage tensor(prefix+name,GGML_TYPE_F32,ne,dims,file_index,header+8+begin);
        tensor.reverse_ne();
        if (tensor.nbytes()!=end-begin) return false;
        tensor_storages.push_back(tensor);
        tensor_storages_types[tensor.name]=tensor.type;
    }
    index>>std::ws;
    return index.eof();
}

'''+text[end:]
assert 'nlohmann::' not in text
write('model.cpp',text)
text=(SOURCE/'model.h').read_text().replace('#include "json.hpp"','')
write('model.h',text)
text=(SOURCE/'t5.hpp').read_text().replace('#include "json.hpp"','#include <stdexcept>')
start=text.index('    void InitializePieces(');end=text.index('    // Builds a Trie index.',start)
text=text[:start]+'''    void InitializePieces(const std::string&) {
        throw std::runtime_error("T5 is unavailable in the SD-Turbo validation baseline");
    }

'''+text[end:]
write('t5.hpp',text)
text=(SOURCE/'util.h').read_text()
for level in ('DEBUG','INFO','WARN','ERROR'):
    old=f'#define LOG_{level}(format, ...) log_printf(SD_LOG_{level}, __FILE__, __LINE__, format, ##__VA_ARGS__)'
    assert text.count(old)==1
    text=text.replace(old,f'#define LOG_{level}(...) log_printf(SD_LOG_{level}, __FILE__, __LINE__, __VA_ARGS__)')
write('util.h',text)
text=(SOURCE/'ggml_extend.hpp').read_text()
text='#include "sd_profile.h"\n'+text
text=text.replace('#include "ggml-cpu.h"','#include "ggml-cpu.h"\n#include "ggml-blas.h"')
start=text.index('class Conv2d :');end=text.index('class Conv3dnx1x1 :',start)
part=text[start:end];assert part.count('enum ggml_type wtype = GGML_TYPE_F16;')==1
text=text[:start]+part.replace('enum ggml_type wtype = GGML_TYPE_F16;','enum ggml_type wtype = GGML_TYPE_F32;')+text[end:]
old='''    GroupNorm32(int64_t num_channels)
        : GroupNorm(32, num_channels, 1e-06f) {}'''
assert text.count(old)==1
text=text.replace(old,'''    GroupNorm32(int64_t num_channels, float eps = 1e-06f)
        : GroupNorm(32, num_channels, eps) {}''')
old='int num_groups = 32) {';assert text.count(old)==1
text=text.replace(old,'int num_groups = 32, float eps = 1e-6f) {')
old='    const float eps = 1e-6f;  // default eps parameter\n    x               = ggml_group_norm';assert text.count(old)==1
text=text.replace(old,'    x               = ggml_group_norm')
old='return ggml_nn_group_norm(ctx, x, w, b, num_groups);';assert text.count(old)==1
text=text.replace(old,'return ggml_nn_group_norm(ctx, x, w, b, num_groups, eps);')
old='    struct ggml_gallocr* compute_allocr = NULL;';assert text.count(old)==1
text=text.replace(old,old+'\n    ggml_backend_t nlc_backend = NULL;\n    ggml_backend_sched_t compute_sched = NULL;')
start=text.index('    bool alloc_compute_buffer(');end=text.index('\n    void cpy_data_to_backend_tensor()',start)
text=text[:start]+'''    bool alloc_compute_buffer(get_graph_cb_t get_graph) {
        (void)get_graph;
        if (compute_sched != NULL) return true;
        GGML_ASSERT(ggml_backend_is_cpu(backend));
        nlc_backend=ggml_backend_blas_init();
        GGML_ASSERT(nlc_backend != NULL);
        ggml_backend_t backends[]={nlc_backend,backend};
        compute_sched=ggml_backend_sched_new(backends,NULL,2,MAX_GRAPH_SIZE,false);
        GGML_ASSERT(compute_sched != NULL);
        return true;
    }
'''+text[end:]
start=text.index('    void free_compute_buffer()');end=text.index('\n    // do copy after alloc graph',start)
text=text[:start]+'''    void free_compute_buffer() {
        if (compute_sched) { ggml_backend_sched_free(compute_sched);compute_sched=NULL; }
        if (nlc_backend) { ggml_backend_free(nlc_backend);nlc_backend=NULL; }
    }
'''+text[end:]
old='        size_t params_buffer_size = ggml_backend_buffer_get_size(params_buffer);'
assert text.count(old)==1
text=text.replace(old,'        ggml_backend_buffer_set_usage(params_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);\n'+old)
old='''        alloc_compute_buffer(get_graph);
        reset_compute_ctx();''';assert text.count(old)==1
text=text.replace(old,'''        alloc_compute_buffer(get_graph);
        ggml_backend_sched_reset(compute_sched);
        reset_compute_ctx();''')
old='        struct ggml_cgraph* gf = get_graph();';assert text.count(old)==1
text=text.replace(old,'''        SdProfileTimer graph_timer;
        struct ggml_cgraph* gf = get_graph();
        graph_timer.emit(get_desc().c_str(),"graph_build");''')
old='GGML_ASSERT(ggml_gallocr_alloc_graph(compute_allocr, gf));';assert text.count(old)==1
text=text.replace(old,'GGML_ASSERT(ggml_backend_sched_alloc_graph(compute_sched, gf));')
old='        ggml_backend_graph_compute(backend, gf);';assert text.count(old)==1
text=text.replace(old,'''        int nlc_nodes=0;
        for (int i=0;i<ggml_graph_n_nodes(gf);++i)
            if (ggml_backend_sched_get_tensor_backend(compute_sched,ggml_graph_node(gf,i))==nlc_backend)
                ++nlc_nodes;
        std::fprintf(stderr,"SD_NLC_GRAPH stage=%s nodes=%d\\n",get_desc().c_str(),nlc_nodes);
        GGML_ASSERT(ggml_backend_sched_graph_compute(compute_sched,gf)==GGML_STATUS_SUCCESS);''')
write('ggml_extend.hpp',text)
text=(SOURCE/'common.hpp').read_text()
for old,new in [('new GroupNorm32(channels));','new GroupNorm32(channels, 1e-05f));'),
                ('new GroupNorm32(out_channels));','new GroupNorm32(out_channels, 1e-05f));')]:
    assert text.count(old)==1;text=text.replace(old,new)
write('common.hpp',text)
text=(SOURCE/'unet.hpp').read_text();old='new GroupNorm32(ch));  // ch == model_channels'
assert text.count(old)==1
write('unet.hpp',text.replace(old,'new GroupNorm32(ch, 1e-05f));  // SD-Turbo norm_eps'))
text=(SOURCE/'ggml/src/ggml-cpu/ggml-cpu.c').read_text()
old='#define GGML_GELU_FP16';assert text.count(old)==1
text=text.replace(old,'#undef GGML_GELU_FP16 // fixed native FP32 erf GELU')
old='return 0.5f*x*(1.0f + tanhf(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));';assert text.count(old)==1
text=text.replace(old,'return 0.5f*x*(1.0f + erff(x*0.7071067811865475244f));')
dispatch=r'''
static bool sd_binary_enabled=false;
static int64_t sd_binary_optimized[2],sd_binary_fallback[2];
static bool sd_binary_overlap(const struct ggml_tensor *,const struct ggml_tensor *);
extern void sd_ve_binary_plane_f32(const float*,float*,float,int64_t,int);
static bool sd_binary_dispatch(const struct ggml_compute_params * params, struct ggml_tensor * dst,int multiply) {
    const struct ggml_tensor * a=dst->src[0],*b=dst->src[1];
    const bool eligible=sd_binary_enabled && a->type==GGML_TYPE_F32 && b->type==GGML_TYPE_F32 && dst->type==GGML_TYPE_F32
        && ggml_is_contiguous(a) && ggml_is_contiguous(dst) && b->ne[0]==1 && b->ne[1]==1 && b->nb[0]==sizeof(float)
        && !sd_binary_overlap(dst,b) && (!sd_binary_overlap(dst,a) || dst->data==a->data);
    if(params->ith==0) {
        if(eligible) ++sd_binary_optimized[multiply]; else ++sd_binary_fallback[multiply];
    }
    if(!eligible) return false;
    const int64_t size=a->ne[0]*a->ne[1],channels=a->ne[2],planes=channels*a->ne[3];
    for(int64_t p=params->ith;p<planes;p+=params->nth) {
        const int64_t channel=p%channels,batch=p/channels;
        const float value=*(const float*)((const char*)b->data+(channel%b->ne[2])*b->nb[2]+(batch%b->ne[3])*b->nb[3]);
        sd_ve_binary_plane_f32((const float*)a->data+p*size,(float*)dst->data+p*size,value,size,multiply);
    }
    return true;
}
'''
for operation,multiply in (('add',0),('mul',1)):
    start=text.index('static void ggml_compute_forward_'+operation+'_f32(')
    end=text.index('\nstatic void ',start+1)
    part=text[start:end]
    assertion='    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));'
    assert part.count(assertion)==1
    part=part.replace(assertion,assertion+'\n    if(sd_binary_dispatch(params,dst,'+str(multiply)+')) return;')
    text=text[:start]+part+text[end:]
needle='static void ggml_compute_forward_add_f32('
text=text.replace(needle,dispatch+'\n'+needle,1)
stats=r'''
// Fixed SD-Turbo has one synchronous graph at a time. Only worker zero
// updates these counters; other workers never read the profiling state.
static bool sd_op_enabled=false;
int sd_ve_stage_threads(const char * stage) {
#ifdef GGML_USE_OPENMP
    const char * mode=getenv("SD_NLC_VAE_THREADS"),*fixed=getenv("SD_NLC_FIXED_THREADS");
    if(!mode || strcmp(mode,"4") || !fixed || strcmp(fixed,"8"))return 0;
    const int wanted=strcmp(stage,"vae")==0 ? 4 : 8;
    const int before=omp_get_max_threads();
    const int64_t start=ggml_time_us();
    if(before!=wanted)omp_set_num_threads(wanted);
    const int after=omp_get_max_threads();
    GGML_ASSERT(after==wanted);
    fprintf(stderr,"SD_STAGE_THREADS stage=%s before=%d configured=%d after=%d seconds=%.9f\n",
            stage,before,wanted,after,(double)(ggml_time_us()-start)/1e6);
    return wanted;
#else
    (void)stage;
    return 0;
#endif
}
static bool sd_binary_shape_enabled=false;
static bool sd_cont_shape_enabled=false;
static bool sd_binary_overlap(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    uintptr_t x=(uintptr_t)a->data,y=(uintptr_t)b->data;
    return x<=y ? y-x<ggml_nbytes(a) : x-y<ggml_nbytes(b);
}
static void sd_binary_shape(const struct ggml_tensor * node) {
    const struct ggml_tensor * a=node->src[0],*b=node->src[1];
    const char * stage=getenv("SD_PROFILE_STAGE");
    fprintf(stderr,"SD_BINARY_SHAPE stage=%s op=%s type=%d,%d,%d a=%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64
        " b=%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 " an=%zu,%zu,%zu,%zu bn=%zu,%zu,%zu,%zu dn=%zu,%zu,%zu,%zu exact_a=%d overlap_a=%d overlap_b=%d\n",
        stage ? stage : "unknown",ggml_op_name(node->op),(int)a->type,(int)b->type,(int)node->type,
        a->ne[0],a->ne[1],a->ne[2],a->ne[3],b->ne[0],b->ne[1],b->ne[2],b->ne[3],
        a->nb[0],a->nb[1],a->nb[2],a->nb[3],b->nb[0],b->nb[1],b->nb[2],b->nb[3],node->nb[0],node->nb[1],node->nb[2],node->nb[3],
        node->data==a->data,sd_binary_overlap(node,a),sd_binary_overlap(node,b));
}
static void sd_cont_shape(const struct ggml_tensor * node, int index, int threads, int64_t elapsed) {
    const struct ggml_tensor * a=node->src[0];
    const char * stage=getenv("SD_PROFILE_STAGE");
    fprintf(stderr,"SD_CONT_SHAPE stage=%s node=%d threads=%d type=%d,%d a=%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64
        " d=%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 " an=%zu,%zu,%zu,%zu dn=%zu,%zu,%zu,%zu"
        " elements=%" PRId64 " source_span=%zu destination_span=%zu source_contiguous=%d destination_contiguous=%d exact=%d overlap=%d seconds=%.9f\n",
        stage ? stage : "unknown",index,threads,(int)a->type,(int)node->type,
        a->ne[0],a->ne[1],a->ne[2],a->ne[3],node->ne[0],node->ne[1],node->ne[2],node->ne[3],
        a->nb[0],a->nb[1],a->nb[2],a->nb[3],node->nb[0],node->nb[1],node->nb[2],node->nb[3],
        ggml_nelements(a),ggml_nbytes(a),ggml_nbytes(node),ggml_is_contiguous(a),ggml_is_contiguous(node),
        node->data==a->data,sd_binary_overlap(node,a),(double)elapsed/1e6);
}
static int64_t sd_op_us[GGML_OP_COUNT],sd_op_nodes[GGML_OP_COUNT];
static int64_t sd_unary_us[GGML_UNARY_OP_COUNT],sd_unary_nodes[GGML_UNARY_OP_COUNT];
static bool sd_gelu_enabled=false;
static int64_t sd_gelu_optimized,sd_gelu_fallback;
void sd_ve_op_profile_begin(void) {
    sd_cont_optimized=sd_cont_fallback=0;
    const char* cont=getenv("SD_VE_CONT");
    sd_cont_extended=cont && strcmp(cont,"extended")==0;
    sd_cont_enabled=cont && (strcmp(cont,"1")==0 || sd_cont_extended);
    sd_gelu_optimized=sd_gelu_fallback=0;
    const char* gelu=getenv("SD_VE_GELU");
    sd_gelu_enabled=gelu && strcmp(gelu,"1")==0;
    sd_im2col_optimized=sd_im2col_fallback=0;
    sd_im2col_rows=sd_im2col_channels=0;
    const char* rows=getenv("SD_VE_IM2COL_ROWS");
    sd_im2col_rows_enabled=rows && strcmp(rows,"1")==0;
    const char* extended=getenv("SD_VE_IM2COL_256");
    sd_im2col_rows_extended=sd_im2col_rows_enabled && extended && strcmp(extended,"1")==0;
    memset(sd_binary_optimized,0,sizeof(sd_binary_optimized));
    memset(sd_binary_fallback,0,sizeof(sd_binary_fallback));
    const char* binary=getenv("SD_VE_BINARY_SCALAR");
    sd_binary_enabled=binary && strcmp(binary,"1")==0;
    const char* shapes=getenv("SD_BINARY_SHAPE_PROFILE");
    sd_binary_shape_enabled=shapes && strcmp(shapes,"1")==0;
    const char* flag=getenv("SD_OP_PROFILE");
    sd_op_enabled=flag && strcmp(flag,"1")==0;
    const char* cont_shapes=getenv("SD_CONT_SHAPE_PROFILE");
    sd_cont_shape_enabled=sd_op_enabled && cont_shapes && strcmp(cont_shapes,"1")==0;
    if (!sd_op_enabled) return;
    memset(sd_op_us,0,sizeof(sd_op_us));memset(sd_op_nodes,0,sizeof(sd_op_nodes));
    memset(sd_unary_us,0,sizeof(sd_unary_us));memset(sd_unary_nodes,0,sizeof(sd_unary_nodes));
}
void sd_ve_op_profile_end(const char* stage) {
    fprintf(stderr,"SD_CONT_DISPATCH stage=%s optimized=%" PRId64 " fallback=%" PRId64 " enabled=%d\n",
            stage,sd_cont_optimized,sd_cont_fallback,sd_cont_enabled);
    fprintf(stderr,"SD_GELU_DISPATCH stage=%s optimized=%" PRId64 " fallback=%" PRId64 " enabled=%d\n",
            stage,sd_gelu_optimized,sd_gelu_fallback,sd_gelu_enabled);
    for(int op=0;op<2;++op)
        fprintf(stderr,"SD_BINARY_DISPATCH stage=%s op=%s optimized=%" PRId64 " fallback=%" PRId64 " enabled=%d\n",
                stage,op ? "MUL" : "ADD",sd_binary_optimized[op],sd_binary_fallback[op],sd_binary_enabled);
    const char * packing=getenv("SD_VE_IM2COL");
    if (packing && strcmp(packing,"1")==0)
        fprintf(stderr,"SD_IM2COL stage=%s optimized_nodes=%" PRId64 " fallback_nodes=%" PRId64 "\n",
                stage,sd_im2col_optimized,sd_im2col_fallback);
    fprintf(stderr,"SD_IM2COL_ROWS stage=%s rows=%" PRId64 " channels=%" PRId64 " enabled=%d\n",
            stage,sd_im2col_rows,sd_im2col_channels,sd_im2col_rows_enabled);
    fprintf(stderr,"SD_IM2COL_SCOPE stage=%s extended=%d\n",stage,sd_im2col_rows_extended);
    if (!sd_op_enabled) return;
    for (int op=0;op<GGML_OP_COUNT;++op) {
        if (op==GGML_OP_UNARY || !sd_op_nodes[op]) continue;
        fprintf(stderr,"SD_OP_PROFILE stage=%s op=%s seconds=%.9f nodes=%" PRId64 "\n",
                stage,ggml_op_name((enum ggml_op)op),(double)sd_op_us[op]/1e6,sd_op_nodes[op]);
    }
    for (int op=0;op<GGML_UNARY_OP_COUNT;++op) {
        if (!sd_unary_nodes[op]) continue;
        fprintf(stderr,"SD_OP_PROFILE stage=%s op=UNARY_%s seconds=%.9f nodes=%" PRId64 "\n",
                stage,ggml_unary_op_name((enum ggml_unary_op)op),(double)sd_unary_us[op]/1e6,sd_unary_nodes[op]);
    }
    sd_op_enabled=false;
}
'''
old='static thread_ret_t ggml_graph_compute_thread(void * data) {';assert text.count(old)==1
omp_single='''                atomic_store_explicit(&threadpool->n_threads_cur, n_threads, memory_order_relaxed);'''
assert text.count(omp_single)==1
text=text.replace(omp_single,omp_single+'''
                const char * sd_stage=getenv("SD_PROFILE_STAGE");
                fprintf(stderr,"SD_GGML_THREADS stage=%s actual=%d maximum=%d\\n",
                        sd_stage ? sd_stage : "unknown",n_threads,omp_get_max_threads());''')
text=text.replace(old,stats+'\n'+old)
old='''        ggml_compute_forward(&params, node);''';assert text.count(old)==1
text=text.replace(old,'''        if (state->ith==0 && sd_binary_shape_enabled && (node->op==GGML_OP_ADD || node->op==GGML_OP_MUL)) sd_binary_shape(node);
        if (state->ith==0 && node->op==GGML_OP_UNARY && ggml_get_unary_op(node)==GGML_UNARY_OP_GELU) {
            if(sd_gelu_enabled) ++sd_gelu_optimized; else ++sd_gelu_fallback;
        }
        const int64_t sd_started=(state->ith==0 && sd_op_enabled) ? ggml_time_us() : 0;
        ggml_compute_forward(&params, node);''')
old='''        ggml_barrier(state->threadpool);
    }

    return 0;''';assert text.count(old)==1
text=text.replace(old,'''        ggml_barrier(state->threadpool);
        if (sd_started) {
            const int64_t elapsed=ggml_time_us()-sd_started;
            if (sd_cont_shape_enabled && node->op==GGML_OP_CONT) sd_cont_shape(node,node_n,params.nth,elapsed);
            if (node->op==GGML_OP_UNARY) {
                const int op=ggml_get_unary_op(node);
                sd_unary_us[op]+=elapsed;++sd_unary_nodes[op];
            } else {
                sd_op_us[node->op]+=elapsed;++sd_op_nodes[node->op];
            }
        }
    }

    return 0;''')
old='''inline static void ggml_vec_gelu_f32(const int n, float * y, const float * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = ggml_gelu_f32(x[i]);'''
assert text.count(old)==1
text=text.replace(old,'''extern void sd_ve_gelu_f32(int,float*,const float*);
inline static void ggml_vec_gelu_f32(const int n, float * y, const float * x) {
    const char * flag=getenv("SD_VE_GELU");
    if (flag && strcmp(flag,"1")==0) { sd_ve_gelu_f32(n,y,x); return; }
    for (int i = 0; i < n; ++i) {
        y[i] = ggml_gelu_f32(x[i]);''')
old='static void ggml_vec_silu_f32(const int n, float * y, const float * x) {'
assert text.count(old)==1
text=text.replace(old,'''extern void sd_ve_silu_f32(int n, float * y, const float * x);
static void ggml_vec_silu_f32(const int n, float * y, const float * x) {
    const char * flag=getenv("SD_VE_SILU");
    if (flag && strcmp(flag,"1")==0) { sd_ve_silu_f32(n,y,x); return; }
''')
old='static ggml_float ggml_vec_soft_max_f32(const int n, float * y, const float * x, float max) {'
assert text.count(old)==1
text=text.replace(old,'''extern void sd_ve_softmax_exp_f32(int,float*,const float*,float);
extern double sd_ve_softmax_sum_f32(int,const float*);
static ggml_float ggml_vec_soft_max_f32(const int n, float * y, const float * x, float max) {
    const char * flag=getenv("SD_VE_SOFTMAX");
    if (flag && strcmp(flag,"1")==0) {
        sd_ve_softmax_exp_f32(n,y,x,max);
        return sd_ve_softmax_sum_f32(n,y);
    }
''')
# Counters are written only by worker zero; graphs execute synchronously.
text='#include <stdint.h>\nstatic int64_t sd_cont_optimized,sd_cont_fallback;\nstatic int sd_cont_enabled,sd_cont_extended;\nstatic int64_t sd_im2col_optimized,sd_im2col_fallback,sd_im2col_rows,sd_im2col_channels;\nstatic int sd_im2col_rows_enabled,sd_im2col_rows_extended;\n'+text
cont_dispatch=r'''
extern int sd_ve_cont_transpose_f32(float*,const float*,size_t,size_t,int,int);
static int sd_cont_transpose_dispatch(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    if (dst->op!=GGML_OP_CONT) return 0;
    const struct ggml_tensor * a=dst->src[0];
    const char * stage=sd_cont_enabled ? getenv("SD_PROFILE_STAGE") : NULL;
    const bool unet=stage && strcmp(stage,"unet")==0;
    const bool vae=stage && strcmp(stage,"vae")==0;
    const bool clip=stage && strcmp(stage,"clip")==0;
    const bool original=unet &&
        ((a->ne[0]==320 && a->ne[1]==64 && a->ne[2]==64) ||
         (a->ne[0]==4096 && a->ne[1]==64 && a->ne[2]==5) ||
         (a->ne[0]==4096 && a->ne[1]==320 && a->ne[2]==1));
    const bool extra=sd_cont_extended &&
        ((vae && ((a->ne[0]==512 && a->ne[1]==64 && a->ne[2]==64) ||
                  (a->ne[0]==4096 && a->ne[1]==512 && a->ne[2]==1))) ||
         (clip && a->ne[0]==77 && a->ne[1]==64 && a->ne[2]==16) ||
         (unet && ((a->ne[0]==640 && a->ne[1]==32 && a->ne[2]==32) ||
                   (a->ne[0]==1024 && a->ne[1]==640 && a->ne[2]==1) ||
                   (a->ne[0]==1024 && a->ne[1]==64 && a->ne[2]==10) ||
                   (a->ne[0]==1280 && a->ne[1]==16 && a->ne[2]==16) ||
                   (a->ne[0]==256 && a->ne[1]==1280 && a->ne[2]==1) ||
                   (a->ne[0]==256 && a->ne[1]==64 && a->ne[2]==20))));
    const bool shape=a->ne[3]==1 && (original || extra);
    const size_t n=shape ? (size_t)a->ne[1]*(size_t)a->ne[2] : 0;
    const bool eligible=sd_cont_enabled && shape &&
        a->type==GGML_TYPE_F32 && dst->type==GGML_TYPE_F32 && ggml_are_same_shape(a,dst) &&
        ggml_is_contiguous(dst) && a->nb[0]==n*sizeof(float) && a->nb[1]==sizeof(float) &&
        (a->ne[2]==1 || a->nb[2]==(size_t)a->ne[1]*sizeof(float));
    const int done=eligible ? sd_ve_cont_transpose_f32((float*)dst->data,(const float*)a->data,
        (size_t)a->ne[0],n,params->ith,params->nth) : 0;
    if (params->ith==0) { if(done) ++sd_cont_optimized; else ++sd_cont_fallback; }
    return done;
}
'''
needle='static void ggml_compute_forward_dup_f32('
assert text.count(needle)==1
text=text.replace(needle,cont_dispatch+'\n'+needle,1)
start=text.index('static void ggml_compute_forward_dup(')
end=text.index('\nstatic void ',start+1)
part=text[start:end]
assertion='    const struct ggml_tensor * src0 = dst->src[0];'
assert part.count(assertion)==1
# Equal-type CONT uses dup_bytes; dispatch must precede that public branch.
assertion='    const struct ggml_tensor * src0 = dst->src[0];'
part=part.replace(assertion,assertion+'\n    if (sd_cont_transpose_dispatch(params,dst)) return;')
text=text[:start]+part+text[end:]
start=text.index('static void ggml_compute_forward_im2col_f32(')
end=text.index('// ggml_compute_forward_im2col_f16',start)
part=text[start:end]
old='    GGML_ASSERT(nb10 == sizeof(float));'
assert part.count(old)==1
part=part.replace(old,old+'''
    const char * packing=getenv("SD_VE_IM2COL");
    if (packing && strcmp(packing,"1")==0) {
        const bool eligible=is_2D && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)
            && s0>0 && s1>0 && d0>0 && d1>0 && p0>=0 && p1>=0;
        if (ith==0) { if(eligible) ++sd_im2col_optimized; else ++sd_im2col_fallback; }
        const char * shape_profile=getenv("SD_IM2COL_SHAPE_PROFILE");
        if (ith==0 && shape_profile && strcmp(shape_profile,"1")==0) {
            const char * stage=getenv("SD_PROFILE_STAGE");
            fprintf(stderr,"SD_IM2COL_SHAPE stage=%s eligible=%d N=%" PRId64 " IC=%" PRId64 " IH=%" PRId64 " IW=%" PRId64
                " KH=%" PRId64 " KW=%" PRId64 " OH=%" PRId64 " OW=%" PRId64
                " s0=%d s1=%d p0=%d p1=%d d0=%d d1=%d threads=%d\\n",
                stage ? stage : "unknown",(int)eligible,N,IC,IH,IW,KH,KW,OH,OW,s0,s1,p0,p1,d0,d1,nth);
        }
        if (eligible) {
            const bool shape512=(IC==128 || IC==256) && IH==512 && IW==512 && OH==512 && OW==512;
            const bool shape256=sd_im2col_rows_extended && (IC==256 || IC==512)
                && IH==256 && IW==256 && OH==256 && OW==256;
            const bool use_rows=sd_im2col_rows_enabled && N==1 && (shape512 || shape256) && KH==3 && KW==3
                && s0==1 && s1==1 && p0==1 && p1==1 && d0==1 && d1==1 && nth==8;
            if (ith==0) { if(use_rows) ++sd_im2col_rows; else ++sd_im2col_channels; }
            if (use_rows) {
                sd_ve_im2col_rows_f32(src1->data,dst->data,N,IC,IH,IW,KH,KW,OH,OW,
                                     s0,s1,p0,p1,d0,d1,ith,nth);
                return;
            }
            sd_ve_im2col_f32(src1->data,dst->data,N,IC,IH,IW,KH,KW,OH,OW,
                             s0,s1,p0,p1,d0,d1,ith,nth);
            return;
        }
    }
''')
prototype='extern void sd_ve_im2col_f32(const float*,float*,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int,int,int,int,int,int,int,int);\n'
prototype+=prototype.replace('sd_ve_im2col_f32','sd_ve_im2col_rows_f32')
text=text[:start]+prototype+part+text[end:]
# Default-off diagnostic: each worker owns its counters until the graph barrier.
text='#include <stdint.h>\nstatic int sd_softmax_shape_enabled;\nstatic int64_t sd_softmax_phase_us[8][5],sd_softmax_phase_rows[8];\n'+text
needle='    sd_cont_shape_enabled=sd_op_enabled && cont_shapes && strcmp(cont_shapes,"1")==0;'
assert text.count(needle)==1
text=text.replace(needle,needle+'\n    const char * soft_shapes=getenv("SD_SOFTMAX_SHAPE_PROFILE");\n    sd_softmax_shape_enabled=sd_op_enabled && soft_shapes && strcmp(soft_shapes,"1")==0;')
start=text.index('static void ggml_compute_forward_soft_max_f32(')
end=text.index('\nstatic void ggml_compute_forward_soft_max(',start)
part=text[start:end]
needle='    for (int i1 = ir0; i1 < ir1; i1++) {'
assert part.count(needle)==1
part=part.replace(needle,'''    const char * soft_flag=getenv("SD_VE_SOFTMAX");
    const bool phase=sd_softmax_shape_enabled && nth<=8 && soft_flag && strcmp(soft_flag,"1")==0;
    if (phase) {
        for (int p=0;p<5;++p) sd_softmax_phase_us[ith][p]=0;
        sd_softmax_phase_rows[ith]=ir1>ir0 ? ir1-ir0 : 0;
    }
    for (int i1 = ir0; i1 < ir1; i1++) {
        int64_t phase_start=phase ? ggml_time_us() : 0;''')
needle='        float max = -INFINITY;\n        ggml_vec_max_f32(nc, &max, wp);\n\n        ggml_float sum = ggml_vec_soft_max_f32(nc, dp, wp, max);'
assert part.count(needle)==1
part=part.replace(needle,'''        if (phase) { const int64_t now=ggml_time_us();sd_softmax_phase_us[ith][0]+=now-phase_start;phase_start=now; }
        float max = -INFINITY;
        ggml_vec_max_f32(nc, &max, wp);
        if (phase) { const int64_t now=ggml_time_us();sd_softmax_phase_us[ith][1]+=now-phase_start;phase_start=now; }
        ggml_float sum;
        if (phase) {
            sd_ve_softmax_exp_f32(nc,dp,wp,max);
            const int64_t now=ggml_time_us();sd_softmax_phase_us[ith][2]+=now-phase_start;phase_start=now;
            sum=sd_ve_softmax_sum_f32(nc,dp);
            const int64_t done=ggml_time_us();sd_softmax_phase_us[ith][3]+=done-phase_start;phase_start=done;
        } else sum = ggml_vec_soft_max_f32(nc, dp, wp, max);''')
needle='        ggml_vec_scale_f32(nc, dp, sum);'
assert part.count(needle)==1
part=part.replace(needle,needle+'\n        if (phase) sd_softmax_phase_us[ith][4]+=ggml_time_us()-phase_start;')
text=text[:start]+part+text[end:]
needle='static void sd_cont_shape(const struct ggml_tensor * node, int index, int threads, int64_t elapsed) {'
assert text.count(needle)==1
soft_shape=r'''
static void sd_softmax_shape(const struct ggml_tensor * node,int index,int threads,int64_t elapsed) {
    const struct ggml_tensor * a=node->src[0],*mask=node->src[1];
    const char * stage=getenv("SD_PROFILE_STAGE");
    float scale,bias;memcpy(&scale,node->op_params,4);memcpy(&bias,(const char *)node->op_params+4,4);
    fprintf(stderr,"SD_SOFTMAX_SHAPE stage=%s node=%d threads=%d type=%d ne=%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64
        " nb=%zu,%zu,%zu,%zu rows=%" PRId64 " columns=%" PRId64 " mask_type=%d scale=%.9g bias=%.9g seconds=%.9f\n",
        stage ? stage : "unknown",index,threads,(int)a->type,a->ne[0],a->ne[1],a->ne[2],a->ne[3],
        a->nb[0],a->nb[1],a->nb[2],a->nb[3],ggml_nrows(a),a->ne[0],mask ? (int)mask->type : -1,scale,bias,(double)elapsed/1e6);
    const char * flag=getenv("SD_VE_SOFTMAX");
    if (threads<=8 && flag && strcmp(flag,"1")==0)
        for (int ith=0;ith<threads;++ith)
            fprintf(stderr,"SD_SOFTMAX_PHASE stage=%s node=%d thread=%d rows=%" PRId64 " prepare_us=%" PRId64 " max_us=%" PRId64 " exp_us=%" PRId64 " sum_us=%" PRId64 " normalize_us=%" PRId64 "\n",
                stage ? stage : "unknown",index,ith,sd_softmax_phase_rows[ith],sd_softmax_phase_us[ith][0],sd_softmax_phase_us[ith][1],
                sd_softmax_phase_us[ith][2],sd_softmax_phase_us[ith][3],sd_softmax_phase_us[ith][4]);
}
'''
text=text.replace(needle,soft_shape+'\n'+needle,1)
needle='            if (sd_cont_shape_enabled && node->op==GGML_OP_CONT) sd_cont_shape(node,node_n,params.nth,elapsed);'
assert text.count(needle)==1
text=text.replace(needle,needle+'\n            if (sd_softmax_shape_enabled && node->op==GGML_OP_SOFT_MAX) sd_softmax_shape(node,node_n,params.nth,elapsed);')
text='#include <stdint.h>\nstatic int sd_softmax_scale_enabled;\nstatic int64_t sd_softmax_scale_optimized,sd_softmax_scale_fallback;\n'+text
needle='    sd_cont_optimized=sd_cont_fallback=0;'
assert text.count(needle)==1
text=text.replace(needle,needle+'''\n    sd_softmax_scale_optimized=sd_softmax_scale_fallback=0;
    const char * scale_flag=getenv("SD_VE_SOFTMAX_SCALE");
    sd_softmax_scale_enabled=scale_flag && strcmp(scale_flag,"1")==0;''')
needle='void sd_ve_op_profile_end(const char* stage) {'
assert text.count(needle)==1
text=text.replace(needle,needle+'''\n    fprintf(stderr,"SD_SOFTMAX_SCALE_DISPATCH stage=%s optimized=%" PRId64 " fallback=%" PRId64 " enabled=%d\\n",
            stage,sd_softmax_scale_optimized,sd_softmax_scale_fallback,sd_softmax_scale_enabled);''')
start=text.index('static void ggml_compute_forward_soft_max_f32(')
end=text.index('static void ggml_compute_forward_soft_max(',start)
soft=text[start:end]
needle='    const bool use_f16 = (src1 && src1->type == GGML_TYPE_F16);'
assert soft.count(needle)==1
soft=soft.replace(needle,needle+'''
    const char * scale_stage=sd_softmax_scale_enabled ? getenv("SD_PROFILE_STAGE") : NULL;
    const bool scale_unet=scale_stage && strcmp(scale_stage,"unet")==0;
    const bool scale_vae=scale_stage && strcmp(scale_stage,"vae")==0;
    const bool scale_shape=src0->ne[3]==1 &&
        ((scale_unet &&
          ((ne00==4096 && ne01==4096 && ne02==5) ||
           (ne00==77 && ne01==4096 && ne02==5) ||
           (ne00==1024 && ne01==1024 && ne02==10) ||
           (ne00==77 && ne01==1024 && ne02==10) ||
           (ne00==256 && ne01==256 && ne02==20) ||
           (ne00==77 && ne01==256 && ne02==20))) ||
         (scale_vae && ne00==4096 && ne01==4096 && ne02==1));
    const bool scale_candidate=sd_softmax_scale_enabled && scale_shape &&
        src0->type==GGML_TYPE_F32 && dst->type==GGML_TYPE_F32 &&
        ggml_is_contiguous(src0) && ggml_is_contiguous(dst) &&
        !src1 && scale==1.0f && max_bias==0.0f;
    if (ith==0) {
        if (scale_candidate) ++sd_softmax_scale_optimized;
        else ++sd_softmax_scale_fallback;
    }
''')
needle='        ggml_vec_cpy_f32  (nc, wp, sp);\n        ggml_vec_scale_f32(nc, wp, scale);'
assert soft.count(needle)==1
soft=soft.replace(needle,'        if (scale_candidate) sd_ve_softmax_copy_scale_f32(nc,wp,sp,scale);\n        else {\n'+needle+'\n        }')
needle='        ggml_vec_scale_f32(nc, dp, sum);'
assert soft.count(needle)==1
soft=soft.replace(needle,'        if (scale_candidate) sd_ve_softmax_scale_f32(nc,dp,(float)sum);\n        else '+needle.strip())
text=text[:start]+soft+text[end:]
text='extern void sd_ve_softmax_scale_f32(int,float*,float);\nextern void sd_ve_softmax_copy_scale_f32(int,float*,const float*,float);\n'+text
from prepare_sd_group_norm_profile import instrument as group_norm_instrument
text=group_norm_instrument(text)
from prepare_sd_group_norm_model import instrument as group_norm_model_instrument
text=group_norm_model_instrument(text)
from prepare_sd_current_vae_gate import instrument as current_vae_gate_instrument
text=current_vae_gate_instrument(text)
write('sd-ggml-cpu.c',text)
# The older model runner passes tensors allocated by ggml_init(), without a
# backend buffer or INPUT flag. Recognize those existing native-VE data leaves.
text=(SOURCE/'ggml/src/ggml-backend.cpp').read_text()
text='#include "sd_profile.h"\n#include "ggml-cpu.h"\nextern "C" int sd_ve_stage_threads(const char*);\nextern "C" void sd_ve_op_profile_begin(void);\nextern "C" void sd_ve_op_profile_end(const char*);\n'+text
old='if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {'
assert text.count(old)==1
text=text.replace(old,'''if ((tensor->flags & GGML_TENSOR_FLAG_INPUT) ||
        (tensor->op == GGML_OP_NONE && tensor->data != NULL && tensor->buffer == NULL)) {''')
old='''bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT''';assert text.count(old)==1
text=text.replace(old,'''bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    SdProfileTimer profile_timer;
    GGML_ASSERT''')
old='''    sched->is_alloc = true;

    return true;''';assert text.count(old)==1
text=text.replace(old,'''    sched->is_alloc = true;
    profile_timer.emit(sd_profile_stage(),"graph_alloc");
    return true;''')
old='''enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    enum ggml_status err''';assert text.count(old)==1
text=text.replace(old,'''enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    SdProfileTimer profile_timer;
    const int stage_threads=sd_ve_stage_threads(sd_profile_stage());
    if(stage_threads) {
        for(int i=0;i<sched->n_backends;++i)
            if(ggml_backend_is_cpu(sched->backends[i])) ggml_backend_cpu_set_n_threads(sched->backends[i],stage_threads);
    }
    sd_ve_op_profile_begin();
    enum ggml_status err''')
old='''    ggml_backend_sched_synchronize(sched);
    return err;''';assert text.count(old)==1
text=text.replace(old,'''    ggml_backend_sched_synchronize(sched);
    profile_timer.emit(sd_profile_stage(),"graph_compute");
    sd_ve_op_profile_end(sd_profile_stage());
    return err;''')
old='''static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    struct ggml_backend_sched_split * splits = sched->splits;''';assert text.count(old)==1
text=text.replace(old,old+'''
    const bool profiling=sd_profile_enabled();
    int64_t backend_us[GGML_SCHED_MAX_BACKENDS]={0};
    int backend_calls[GGML_SCHED_MAX_BACKENDS]={0};
''')
old='''            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);''';assert text.count(old)==1
text=text.replace(old,'''            const int64_t started=profiling ? ggml_time_us() : 0;
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            if (profiling) {
                // This fixed port uses synchronous native VE CPU and NLC backends.
                backend_us[split_backend_id]+=ggml_time_us()-started;
                ++backend_calls[split_backend_id];
            }''')
old='''    sched->cur_copy = (sched->cur_copy + 1) % sched->n_copies;''';assert text.count(old)==1
text=text.replace(old,'''    if (profiling) {
        for (int b=0;b<sched->n_backends;++b) {
            std::string part=std::string("backend_")+ggml_backend_name(sched->backends[b]);
            sd_profile_emit(sd_profile_stage(),part.c_str(),double(backend_us[b])/1e6,backend_calls[b]);
        }
    }
'''+old)
write('sd-ggml-backend.cpp',text)
helper=r'''
#ifndef SD_BASELINE_VALIDATION_H
#define SD_BASELINE_VALIDATION_H
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include "ggml.h"
static void sd_read_f32(const char* path,float* data,size_t n) {
    FILE* f=std::fopen(path,"rb");
    if (!f) throw std::runtime_error("fixed array open failed");
    bool good=std::fread(data,sizeof(float),n,f)==n && std::fgetc(f)==EOF && !std::ferror(f);
    if (std::fclose(f)) good=false;
    if (!good) throw std::runtime_error("fixed array size or read failure");
    for (size_t i=0;i<n;++i) if (!std::isfinite(data[i])) throw std::runtime_error("nonfinite fixed array");
}
static void sd_load_noise(ggml_tensor* t) {
    const char* p=std::getenv("SD_FIXED_NOISE");if (!p || !*p) return;
    GGML_ASSERT(t->type==GGML_TYPE_F32 && ggml_is_contiguous(t));
    sd_read_f32(p,static_cast<float*>(t->data),ggml_nelements(t));
}
static void sd_trace(const char* name,const ggml_tensor* t) {
    SdProfileTimer timer;
    const char* dir=std::getenv("SD_TRACE_DIR");if (!dir || !*dir) return;
    GGML_ASSERT(t && t->type==GGML_TYPE_F32 && ggml_is_contiguous(t));
    std::string path=std::string(dir)+"/"+name+".f32";
    FILE* f=std::fopen(path.c_str(),"wb");if (!f) throw std::runtime_error("trace open failed");
    bool good=std::fwrite(t->data,1,ggml_nbytes(t),f)==ggml_nbytes(t);
    if (std::fclose(f)) good=false;
    if (!good) throw std::runtime_error("trace write failed");
    timer.emit("trace","write");
}
static void sd_load_sigmas(std::vector<float>& sigmas,int steps) {
    const char* p=std::getenv("SD_FIXED_SIGMAS");if (!p || !*p) return;
    sigmas.resize(static_cast<size_t>(steps)+1);sd_read_f32(p,sigmas.data(),sigmas.size());
    if (sigmas.back()!=0) throw std::runtime_error("terminal sigma must be zero");
    for (size_t i=1;i<sigmas.size();++i)
        if (!(sigmas[i-1]>sigmas[i]) || sigmas[i]<0) throw std::runtime_error("sigmas must decrease");
}
#endif
'''
write('sd_baseline_validation.h',helper)
text=(SOURCE/'stable-diffusion.cpp').read_text()
text='#include "sd_baseline_validation.h"\n'+text
old='if (sd_version_is_sd2(version)) {\n            if (is_using_v_parameterization_for_sd2';assert text.count(old)==1
text=text.replace(old,'if (sd_version_is_sd2(version) && !std::getenv("SD_TURBO_EPS")) {\n            if (is_using_v_parameterization_for_sd2')
old='            ggml_tensor_scale(noised_input, c_in);';assert text.count(old)==1
text=text.replace(old,old+'''
            const std::string trace_step="native-step"+std::to_string(step-1);
            sd_trace((trace_step+"-input").c_str(),noised_input);
            sd_trace((trace_step+"-timestep").c_str(),timesteps);''')
old='            float* negative_data = NULL;';assert text.count(old)==1
text=text.replace(old,'            sd_trace((trace_step+"-epsilon").c_str(),out_cond);\n'+old)
start=text.index('sd_image_t* generate_image(');end=text.index('sd_image_t* txt2img(',start)
part=text[start:end]
old='        ggml_tensor_set_f32_randn(noise, sd_ctx->sd->rng);';assert part.count(old)==1
part=part.replace(old,old+'\n        sd_load_noise(noise);\n        sd_trace("native-noise",noise);\n        sd_trace("native-embeddings",cond.c_crossattn);')
old='        final_latents.push_back(x_0);';assert part.count(old)==1
part=part.replace(old,'        sd_trace("native-latent",x_0);\n'+old)
old='            decoded_images.push_back(img);';assert part.count(old)==1
part=part.replace(old,'            sd_trace("native-decoded",img);\n'+old)
text=text[:start]+part+text[end:]
start=text.index('sd_image_t* txt2img(');end=text.index('sd_image_t* img2img(',start)
part=text[start:end];old='    std::vector<float> sigmas = sd_ctx->sd->denoiser->get_sigmas(sample_steps);';assert part.count(old)==1
part=part.replace(old,old+'\n    sd_load_sigmas(sigmas,sample_steps);')
write('stable-diffusion.cpp',text[:start]+part+text[end:])
ops=(ROOT/'tests/check_sd_ops.cpp').read_text()
ops=ops.replace('ggml_gelu_erf(', 'ggml_gelu(')
ops=ops.replace('false,true)', 'false)')
ops=ops.replace(',GGML_SCALE_MODE_NEAREST', '')
ops=ops.replace('bool require_blas=false) {','bool require_blas=false, bool external_data=false) {')
ops=ops.replace('for (const auto &item:inputs) ggml_set_input(item.first);',
'''for (const auto &item:inputs) {
            if (external_data) item.first->data=const_cast<float*>(item.second.data());
            else ggml_set_input(item.first);
        }''')
ops=ops.replace('ggml_backend_tensor_set(item.first,item.second.data(),0,ggml_nbytes(item.first));',
                'if (!external_data) ggml_backend_tensor_set(item.first,item.second.data(),0,ggml_nbytes(item.first));')
ops=ops.replace('static void matmul(int k,int m,int n) {',
                'static void matmul(int k,int m,int n,bool external_data=false) {')
ops=ops.replace('g.run(out,{{a,av},{b,bv}},true);','g.run(out,{{a,av},{b,bv}},true,external_data);')
ops=ops.replace('check("NLC SGEMM",actual,expected);',
                'check(external_data ? "NLC external-data SGEMM" : "NLC SGEMM",actual,expected);')
ops=ops.replace('matmul(257,65,77);matmul(1024,1024,77);',
                'matmul(257,65,77);matmul(1024,1024,77);matmul(257,65,77,true);')
ops=ops.replace('IMAGE_OPERATOR_PASS cases=11;', 'IMAGE_OPERATOR_PASS cases=12;')
write('check_sd_ops.cpp',ops)
print('Generated compact FP32 SD-Turbo overlay and NLC scheduler')
