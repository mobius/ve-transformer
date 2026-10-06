// Minimal reproducible runner: real model, incremental state, complete logit traces.
#include "llama.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef QWEN_ACCUM_FP64
#include "qwen_dense_cache.h"
#endif

using Clock = std::chrono::steady_clock;
extern "C" bool qwen_test_abort(void *data) {
    static_cast<std::atomic<unsigned>*>(data)->fetch_add(1,std::memory_order_relaxed);
    return true;
}
static double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now()-start).count();
}
struct ProfileEntry {size_t calls=0;double seconds=0;};
struct Profile {
    Clock::time_point start;
    std::map<std::string,ProfileEntry> entries;
    std::ofstream node_data,node_meta;
    std::string phase;
    int repeat=0;
    size_t trace_offset=0;
};
static std::string json_string(const std::string & s);
static bool profile_node(ggml_tensor *tensor,bool ask,void *data) {
    auto &profile=*static_cast<Profile*>(data);
    if(ask) {profile.start=Clock::now();return true;}
    double seconds=elapsed(profile.start);
    std::string key=ggml_op_name(tensor->op);
    if(tensor->src[0]) {
        auto *input=tensor->src[0];
        key+=":"+std::string(ggml_type_name(input->type))+":"+
             std::to_string(input->ne[0])+"x"+std::to_string(input->ne[1]);
    }
    key+=":"+std::to_string(tensor->ne[0])+"x"+std::to_string(tensor->ne[1])+"x"+std::to_string(tensor->ne[2]);
    auto &entry=profile.entries[key];++entry.calls;entry.seconds+=seconds;
    if(profile.node_data.is_open() && tensor->data && ggml_is_contiguous(tensor) &&
       (tensor->type==GGML_TYPE_F32 || tensor->type==GGML_TYPE_I32) &&
       tensor->op!=GGML_OP_NONE && tensor->op!=GGML_OP_VIEW && tensor->op!=GGML_OP_RESHAPE &&
       ggml_nbytes(tensor)<=16*1024*1024) {
        size_t bytes=ggml_nbytes(tensor);
        profile.node_data.write(static_cast<const char*>(tensor->data),bytes);
        profile.node_meta<<"{\"phase\":"<<json_string(profile.phase)<<",\"repeat\":"<<profile.repeat
            <<",\"name\":"<<json_string(tensor->name)<<",\"op\":"<<json_string(ggml_op_name(tensor->op))
            <<",\"type\":"<<json_string(ggml_type_name(tensor->type))<<",\"ne\":["
            <<tensor->ne[0]<<","<<tensor->ne[1]<<","<<tensor->ne[2]<<","<<tensor->ne[3]
            <<"],\"offset\":"<<profile.trace_offset<<",\"bytes\":"<<bytes<<"}\n";
        if(!profile.node_data || !profile.node_meta) {
            std::fprintf(stderr,"node trace write failed\n");std::abort();
        }
        profile.trace_offset+=bytes;
    }
    return true;
}
static int integer(const std::string & s, int lo, int hi) {
    size_t end=0; int v=std::stoi(s,&end);
    if (end!=s.size() || v<lo || v>hi) throw std::runtime_error("argument out of range");
    return v;
}
static void set_batch(llama_batch_ext * batch, const llama_token * ids, int n, int pos) {
    llama_batch_ext_clear(batch);
    for (int i=0;i<n;++i) {
        int idx=llama_batch_ext_add_token(batch,0,ids[i]);
        llama_pos p=pos+i;
        if (!llama_batch_ext_set_pos(batch,idx,&p)) throw std::runtime_error("batch position failed");
    }
    if (!llama_batch_ext_set_output_logits(batch,n-1,true)) throw std::runtime_error("batch output failed");
}
static std::string json_string(const std::string & s) {
    std::string out="\"";
    for (size_t i=0;i<s.size();++i) {
        unsigned char c=s[i];
        if (c=='"' || c=='\\') {out+='\\';out+=c;}
        else if(c<32) {char b[8];std::snprintf(b,sizeof(b),"\\u%04x",c);out+=b;}
        else if(c<128) out+=c;
        else {
            int n=c>=0xc2 && c<=0xdf?2:c>=0xe0 && c<=0xef?3:c>=0xf0 && c<=0xf4?4:0;
            bool valid=n && i+n<=s.size();
            for(int j=1;valid && j<n;++j) valid=(static_cast<unsigned char>(s[i+j])&0xc0)==0x80;
            if(valid) {
                unsigned char second=s[i+1];
                if((c==0xe0 && second<0xa0) || (c==0xed && second>=0xa0) ||
                   (c==0xf0 && second<0x90) || (c==0xf4 && second>=0x90)) valid=false;
            }
            if(valid) {out.append(s,i,n);i+=n-1;}
            else out+="\\ufffd"; // A token limit can end in the middle of a UTF-8 character.
        }
    }
    return out+'"';
}
static void report_profile(const Profile &profile,const char *phase,int repeat) {
    for(const auto &item:profile.entries)
        std::fprintf(stderr,"PROFILE {\"repeat\":%d,\"phase\":%s,\"operator\":%s,\"calls\":%zu,\"seconds\":%.9f}\n",
                     repeat,json_string(phase).c_str(),json_string(item.first).c_str(),item.second.calls,item.second.seconds);
}
// Length-delimited stdin protocol; EOF between frames closes the session cleanly.
static bool read_request(std::string &prompt,int &count) {
    unsigned char header[8];
    size_t n=std::fread(header,1,8,stdin);
    if(n==0 && std::feof(stdin)) return false;
    if(n!=8) throw std::runtime_error("incomplete request header");
    auto u32=[&](int i) {return (uint32_t)header[i]|((uint32_t)header[i+1]<<8)|((uint32_t)header[i+2]<<16)|((uint32_t)header[i+3]<<24);};
    uint32_t bytes=u32(0),tokens=u32(4);
    if(bytes>1024*1024 || tokens<1 || tokens>1024) throw std::runtime_error("request bounds exceeded");
    prompt.resize(bytes);
    if(bytes && std::fread(&prompt[0],1,bytes,stdin)!=bytes) throw std::runtime_error("incomplete request body");
    count=(int)tokens;return true;
}
int main(int argc,char **argv) {
    try {
        std::string model_path,prompt="Hello",trace,forced_path,node_trace;
        int count=32,threads=8,context=2048,repeats=1;
#ifdef QWEN_ACCUM_FP64
        int dense_cache_mib=0;
#endif
        bool force_count=false,use_mmap=true,profiling=false,fresh_threads=false,test_abort=false,serve=false;
        Profile profile;
        for(int i=1;i<argc;++i) {
            std::string a=argv[i];
            if(a=="--serve") {serve=true;continue;}
            if(a=="--test-abort") {test_abort=true;continue;}
            if(a=="--fresh-threads") {fresh_threads=true;continue;}
            if(a=="--profile") {profiling=true;continue;}
            if(a=="--force-count") {force_count=true;continue;}
            if(a=="--no-mmap") {use_mmap=false;continue;}
            if(i+1>=argc) throw std::runtime_error("missing argument value");
            std::string v=argv[++i];
            if(a=="--model") model_path=v;
            else if(a=="--prompt") prompt=v;
            else if(a=="--tokens") count=integer(v,1,1024);
            else if(a=="--threads") threads=integer(v,1,96);
            else if(a=="--context") context=integer(v,64,32768);
            else if(a=="--repeats") repeats=integer(v,1,20);
#ifdef QWEN_ACCUM_FP64
            else if(a=="--dense-cache-mib") dense_cache_mib=integer(v,0,24576);
#endif
            else if(a=="--trace") trace=v;
            else if(a=="--node-trace") node_trace=v;
            else if(a=="--forced-tokens") forced_path=v;
            else throw std::runtime_error("unknown argument");
        }
        if(model_path.empty() || prompt.size()>1024*1024) throw std::runtime_error("model required or prompt too large");
        if(serve && (repeats!=1 || !forced_path.empty() || test_abort)) throw std::runtime_error("serve does not support repeats, forced tokens or abort smoke");
        std::vector<llama_token> forced;
        if(!forced_path.empty()) {
            std::ifstream f(forced_path); int64_t id;
            while(f>>id) {
                if(id<0 || id>INT32_MAX || forced.size()>=1024) throw std::runtime_error("invalid forced tokens");
                forced.push_back(static_cast<llama_token>(id));
            }
            if(!f.eof() || forced.size()!=static_cast<size_t>(count)) throw std::runtime_error("forced token count mismatch");
        }
#ifdef QWEN_ACCUM_FP64
        // Ensure RMSNorm is handled by the shared high-precision hook.
        if(setenv("GGML_CPU_DISABLE_FUSION","1",1)) throw std::runtime_error("fusion configuration failed");
        qwen_dense_cache_configure((uint64_t)dense_cache_mib*1024*1024);
#endif
        llama_backend_init();
        auto t=Clock::now();
        auto mp=llama_model_default_params();
        mp.n_gpu_layers=0;mp.load_mode=use_mmap?LLAMA_LOAD_MODE_MMAP:LLAMA_LOAD_MODE_NONE;
        mp.use_extra_bufts=false;mp.load_mtp=false;mp.lazy_mode=LLAMA_LAZY_MODE_OFF;
        std::unique_ptr<llama_model,decltype(&llama_model_free)> model(
            llama_model_load_from_file(model_path.c_str(),mp),llama_model_free);
        if(!model) throw std::runtime_error("model load failed");
        double load=elapsed(t);
        const auto *vocab=llama_model_get_vocab(model.get());
        int nv=llama_vocab_n_tokens(vocab);
        for(auto id:forced) if(id>=nv) throw std::runtime_error("forced token exceeds vocabulary");
        int np=0;
        std::vector<llama_token> ids;
        auto tokenize=[&]() {
            np=-llama_tokenize(vocab,prompt.data(),prompt.size(),nullptr,0,true,true);
            if(np<=0 || np+count>context) throw std::runtime_error("prompt exceeds requested context");
            ids.resize(np);
            if(llama_tokenize(vocab,prompt.data(),prompt.size(),ids.data(),np,true,true)!=np)
                throw std::runtime_error("tokenization failed");
        };
        auto cp=llama_context_default_params();
        cp.n_ctx=context;cp.n_batch=512;cp.n_ubatch=128;cp.n_seq_max=1;
        cp.n_threads=threads;cp.n_threads_batch=threads;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.offload_kqv=false;cp.op_offload=false;cp.no_perf=false;
        if(!node_trace.empty()) {
            profile.node_data.open(node_trace+".nodes.bin",std::ios::binary);
            profile.node_meta.open(node_trace+".nodes.jsonl");
            if(!profile.node_data || !profile.node_meta) throw std::runtime_error("node trace open failed");
        }
        if(profiling || !node_trace.empty()) {cp.cb_eval=profile_node;cp.cb_eval_user_data=&profile;}
        auto pool_params=ggml_threadpool_params_default(threads);
        std::unique_ptr<ggml_threadpool,decltype(&ggml_threadpool_free)> pool(
            fresh_threads?nullptr:ggml_threadpool_new(&pool_params),ggml_threadpool_free);
        if(!fresh_threads && !pool) throw std::runtime_error("threadpool creation failed");
        t=Clock::now();
        std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);
        if(!ctx) throw std::runtime_error("context creation failed");
        double init=elapsed(t);
        if(pool) llama_attach_threadpool(ctx.get(),pool.get(),pool.get());
        std::unique_ptr<llama_batch_ext,decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx.get()),llama_batch_ext_free);
        if(!batch) throw std::runtime_error("batch allocation failed");
        if(test_abort) {
            tokenize();
            std::atomic<unsigned> calls{0};
            llama_set_abort_callback(ctx.get(),qwen_test_abort,&calls);
            set_batch(batch.get(),ids.data(),1,0);
            int status=llama_process(ctx.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get());
            llama_set_abort_callback(ctx.get(),nullptr,nullptr);
            if(status!=2 || calls.load(std::memory_order_relaxed)==0)
                throw std::runtime_error("abort callback was not invoked or decode was not aborted");
            std::fprintf(stderr,"ABORT_CALLBACK_PASS status=%d calls=%u; callback cleared before normal inference\n",
                         status,calls.load(std::memory_order_relaxed));
        }
        if(serve) {std::printf("{\"event\":\"ready\",\"protocol\":1,\"load_seconds\":%.9f,\"context_init_seconds\":%.9f}\n",load,init);std::fflush(stdout);}
        for(int r=0;serve || r<repeats;++r) {
            if(serve && !read_request(prompt,count)) break;
            auto request_start=Clock::now();
            t=Clock::now();tokenize();double tokenize_seconds=elapsed(t);
            t=Clock::now();llama_memory_clear(llama_get_memory(ctx.get()),true);double reset_seconds=elapsed(t);
#ifdef QWEN_TIMING
            auto cache_before=qwen_dense_cache_stats();
#endif
            std::ofstream log;
            if(!trace.empty()) {
                log.open(trace+"-"+std::to_string(r)+".f32",std::ios::binary);
                if(!log) throw std::runtime_error("trace open failed");
            }
            profile.entries.clear();
            profile.repeat=r;profile.phase="prefill";
            t=Clock::now();
            for(int pos=0;pos<np;pos+=128) {
                int n=std::min(128,np-pos);set_batch(batch.get(),ids.data()+pos,n,pos);
                if(llama_process(ctx.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get())) throw std::runtime_error("prefill failed");
            }
            double pp=elapsed(t);
            if(profiling) report_profile(profile,"prefill",r);
            profile.entries.clear();
            profile.phase="decode";
            std::vector<llama_token> chosen,greedy;
            std::string text;
            double select_seconds=0,piece_seconds=0,model_decode_seconds=0;
            t=Clock::now();
            for(int step=0;step<count;++step) {
                auto step_start=Clock::now();
                float *scores=llama_get_logits_ith(ctx.get(),-1);
                if(!scores) throw std::runtime_error("missing logits");
                for(int k=0;k<nv;++k) if(!std::isfinite(scores[k])) throw std::runtime_error("nonfinite logits");
                if(log.is_open()) {log.write(reinterpret_cast<char*>(scores),nv*sizeof(float));if(!log) throw std::runtime_error("trace write failed");}
                llama_token predicted=std::max_element(scores,scores+nv)-scores;
                llama_token next=forced.empty()?predicted:forced[step];
                select_seconds+=elapsed(step_start);
                if(!force_count && forced.empty() && llama_vocab_is_eog(vocab,next)) break;
                greedy.push_back(predicted);chosen.push_back(next);
                auto piece_start=Clock::now();
                std::vector<char> piece(256);
                int n=llama_token_to_piece(vocab,next,piece.data(),piece.size(),0,true);
                if(n<0) {piece.resize(-n);n=llama_token_to_piece(vocab,next,piece.data(),piece.size(),0,true);}
                if(n<0) throw std::runtime_error("token piece failed");
                text.append(piece.data(),n);
                piece_seconds+=elapsed(piece_start);
                auto model_start=Clock::now();
                set_batch(batch.get(),&next,1,np+step);
                if(llama_process(ctx.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get())) throw std::runtime_error("incremental decode failed");
                model_decode_seconds+=elapsed(model_start);
            }
            double decode=elapsed(t);
            if(log.is_open()) {log.flush();if(!log) throw std::runtime_error("trace flush failed");}
            if(profiling) report_profile(profile,"decode",r);
            const double request_seconds=elapsed(request_start);
            std::printf("{\"repeat\":%d,\"threads\":%d,\"context\":%d,\"vocab\":%d,\"prompt_tokens\":%d,\"generated_tokens\":%zu,\"model_bytes\":%llu,\"model_parameters\":%llu,\"load_seconds\":%.6f,\"context_init_seconds\":%.6f,\"prefill_seconds\":%.6f,\"decode_seconds\":%.6f,\"decode_tokens_per_second\":%.6f,\"tokens\":[",
                r,threads,context,nv,np,chosen.size(),(unsigned long long)llama_model_size(model.get()),(unsigned long long)llama_model_n_params(model.get()),load,init,pp,decode,chosen.size()/decode);
            for(size_t k=0;k<chosen.size();++k) std::printf("%s%d",k?",":"",chosen[k]);
            std::printf("],\"greedy_tokens\":[");
            for(size_t k=0;k<greedy.size();++k) std::printf("%s%d",k?",":"",greedy[k]);
            std::printf("],\"text\":%s",json_string(text).c_str());
            std::printf(",\"tokenize_seconds\":%.9f,\"reset_seconds\":%.9f,\"selection_seconds\":%.9f,\"text_conversion_seconds\":%.9f,\"model_decode_seconds\":%.9f,\"request_seconds\":%.9f",tokenize_seconds,reset_seconds,select_seconds,piece_seconds,model_decode_seconds,request_seconds);
#ifdef QWEN_ACCUM_FP64
            std::printf(",\"math_mode\":\"fp64_accumulation\",\"activation_storage\":\"float32\"");
            auto cache=qwen_dense_cache_stats();
#ifdef QWEN_TIMING
            std::printf(",\"cache_fill_thread_seconds\":%.9f,\"cache_new_entries\":%llu",cache.fill_thread_seconds-cache_before.fill_thread_seconds,(unsigned long long)(cache.entries-cache_before.entries));
#endif
            std::printf(",\"dense_cache_budget_bytes\":%llu,\"dense_cache_retained_bytes\":%llu,\"dense_cache_entries\":%llu,\"dense_cache_hits\":%llu,\"dense_cache_misses\":%llu,\"dense_cache_rejected\":%llu",
                (unsigned long long)cache.budget_bytes,(unsigned long long)cache.retained_bytes,
                (unsigned long long)cache.entries,(unsigned long long)cache.hits,
                (unsigned long long)cache.misses,(unsigned long long)cache.rejected);
            std::printf(",\"dense_cache_reserved_bytes\":%llu",(unsigned long long)cache.reserved_bytes);
#endif
            std::printf("}\n");std::fflush(stdout);
        }
#ifdef QWEN_ACCUM_FP64
        qwen_dense_cache_configure(0);
#endif
        return 0;
    } catch(const std::exception &e) {std::fprintf(stderr,"qwen-infer: %s\n",e.what());return 1;}
}
