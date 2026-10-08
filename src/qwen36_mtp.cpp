// Isolated native MTP experiment; greedy decoding only. Keep traces in build/.
#include "llama.h"
#include "llama-ext.h"
#include "llama-model.h"
#include "ggml-cpu.h"
#include "qwen_dense_cache.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point t) {return std::chrono::duration<double>(Clock::now()-t).count();}
static int number(const char *s,int lo,int hi) {
    size_t end=0;int n=std::stoi(s,&end);
    if(end!=std::strlen(s) || n<lo || n>hi) throw std::runtime_error("invalid numeric argument");
    return n;
}
static llama_token greedy(const float *logits,int nv) {
    if(!logits) throw std::runtime_error("missing logits");
    int best=0;
    for(int i=0;i<nv;++i) {
        if(!std::isfinite(logits[i])) throw std::runtime_error("nonfinite logits");
        if(logits[i]>logits[best]) best=i;
    }
    return best;
}
static std::string json_text(const std::string &s) {
    std::string out="\"";
    for(size_t i=0;i<s.size();++i) {
        unsigned char c=s[i];
        if(c=='"' || c=='\\') {out+='\\';out+=c;}
        else if(c<32) {char b[8];std::snprintf(b,sizeof(b),"\\u%04x",c);out+=b;}
        else if(c<128) out+=c;
        else {
            int n=c>=0xc2 && c<=0xdf?2:c>=0xe0 && c<=0xef?3:c>=0xf0 && c<=0xf4?4:0;
            bool valid=n && i+n<=s.size();
            for(int j=1;valid && j<n;++j) valid=(static_cast<unsigned char>(s[i+j])&0xc0)==0x80;
            if(valid) {
                unsigned char second=s[i+1];
                if((c==0xe0 && second<0xa0) || (c==0xed && second>=0xa0) || (c==0xf0 && second<0x90) || (c==0xf4 && second>=0x90)) valid=false;
            }
            if(valid) {out.append(s,i,n);i+=n-1;} else out+="\\ufffd";
        }
    }
    return out+'"';
}
// Same bounded protocol as the existing resident executor; no network listener.
static bool read_request(std::string &prompt,int &count) {
    unsigned char header[8];size_t n=std::fread(header,1,8,stdin);
    if(n==0 && std::feof(stdin)) return false;
    if(n!=8) throw std::runtime_error("incomplete request header");
    auto u32=[&](int i) {return (uint32_t)header[i]|((uint32_t)header[i+1]<<8)|((uint32_t)header[i+2]<<16)|((uint32_t)header[i+3]<<24);};
    uint32_t bytes=u32(0),tokens=u32(4);
    if(bytes>1024*1024 || tokens<1 || tokens>1024) throw std::runtime_error("request bounds exceeded");
    prompt.resize(bytes);
    if(bytes && std::fread(&prompt[0],1,bytes,stdin)!=bytes) throw std::runtime_error("incomplete request body");
    count=(int)tokens;return true;
}

struct Decoder {
    std::unique_ptr<llama_context,decltype(&llama_free)> ctx{nullptr,llama_free};
    std::unique_ptr<llama_batch_ext,decltype(&llama_batch_ext_free)> batch{nullptr,llama_batch_ext_free};
    Decoder(llama_model *model,llama_context_params cp,ggml_threadpool *pool) {
        ctx.reset(llama_init_from_model(model,cp));
        if(!ctx) throw std::runtime_error("context creation failed");
        llama_attach_threadpool(ctx.get(),pool,pool);
        batch.reset(llama_batch_ext_init(ctx.get()));
        if(!batch) throw std::runtime_error("batch allocation failed");
    }
    void decode(const std::vector<llama_token> &ids,int pos,bool all,const float *state=nullptr,int width=0,bool output=true) {
        llama_batch_ext_clear(batch.get());
        for(size_t i=0;i<ids.size();++i) {
            int idx=llama_batch_ext_add_token(batch.get(),0,ids[i]);llama_pos p=pos+i;
            if(idx<0 || !llama_batch_ext_set_pos(batch.get(),idx,&p) ||
               !llama_batch_ext_set_output_logits(batch.get(),idx,output && (all || i+1==ids.size())))
                throw std::runtime_error("invalid batch");
            // The pinned state's setter is a TODO returning false. Its own
            // common MTP driver uses this token-embedding slot for hidden state
            // when ctx_type == MTP, while retaining the token ID alongside it.
            if(state && !llama_batch_ext_set_embd_token(batch.get(),idx,{state+i*width,1,(size_t)width}))
                throw std::runtime_error("MTP state input failed");
        }
        if(llama_process(ctx.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get())) throw std::runtime_error("decode failed");
    }
    void remove(int pos) {
        if(!llama_memory_seq_rm(llama_get_memory(ctx.get()),0,pos,-1))
            throw std::runtime_error("state rollback failed; refusing to continue with stale state");
    }
    std::vector<float> hidden(int rows,int width) {
        const float *h=llama_get_embeddings_nextn(ctx.get());
        if(!h) throw std::runtime_error("missing MTP hidden states");
        return std::vector<float>(h,h+(size_t)rows*width);
    }
};

int main(int argc,char **argv) {
    try {
        std::string target_path,head_path,prompt="Hello";
        int count=16,drafts=2,threads=4,context=512,repeats=1,cache_mib=16384;
        bool force_count=false,reject_first=false,serve=false,auto_mtp=false;
        for(int i=1;i<argc;++i) {
            std::string arg=argv[i];
            if(arg=="--serve") {serve=true;continue;}
            if(arg=="--auto-mtp") {auto_mtp=true;continue;}
            if(arg=="--force-count") {force_count=true;continue;}
            if(arg=="--reject-first") {reject_first=true;continue;}
            if(i+1>=argc) throw std::runtime_error("missing option value");
            const char *v=argv[++i];
            if(arg=="--model") target_path=v;
            else if(arg=="--mtp-model") head_path=v;
            else if(arg=="--prompt") prompt=v;
            else if(arg=="--tokens") count=number(v,1,1024);
            else if(arg=="--draft-tokens") drafts=number(v,1,2);
            else if(arg=="--threads") threads=number(v,1,8);
            else if(arg=="--context") context=number(v,16,2048);
            else if(arg=="--repeats") repeats=number(v,1,16);
            else if(arg=="--dense-cache-mib") cache_mib=number(v,0,16384);
            else throw std::runtime_error("unknown option");
        }
        if(target_path.empty() || prompt.size()>1024*1024 || ((reject_first || auto_mtp) && head_path.empty()))
            throw std::runtime_error("invalid model or prompt options");
        if(setenv("GGML_CPU_DISABLE_FUSION","1",1)) throw std::runtime_error("fusion configuration failed");
        qwen_dense_cache_configure((uint64_t)cache_mib*1024*1024);
        llama_backend_init();
        auto mp=llama_model_default_params();
        mp.n_gpu_layers=0;mp.load_mode=LLAMA_LOAD_MODE_NONE;mp.use_extra_bufts=false;
        mp.lazy_mode=LLAMA_LAZY_MODE_OFF;mp.load_mtp=false;
        auto start=Clock::now();
        std::unique_ptr<llama_model,decltype(&llama_model_free)> target(llama_model_load_from_file(target_path.c_str(),mp),llama_model_free);
        if(!target) throw std::runtime_error("target load failed");
        std::unique_ptr<llama_model,decltype(&llama_model_free)> head(nullptr,llama_model_free);
        if(!head_path.empty()) {mp.load_mtp=true;head.reset(llama_model_load_from_file(head_path.c_str(),mp));if(!head) throw std::runtime_error("MTP load failed");}
        double load=seconds(start);
        auto *vocab=llama_model_get_vocab(target.get());int nv=llama_vocab_n_tokens(vocab);
        int width=llama_model_n_embd_out(target.get());
        if(head) {
            auto *hv=llama_model_get_vocab(head.get());
            if(llama_model_n_layer_nextn(head.get())!=1 || llama_model_n_embd_out(head.get())!=width || llama_vocab_n_tokens(hv)!=nv)
                throw std::runtime_error("incompatible native MTP configuration");
            for(int i=0;i<nv;++i) if(std::strcmp(llama_vocab_get_text(vocab,i),llama_vocab_get_text(hv,i)) ||
                llama_vocab_get_attr(vocab,i)!=llama_vocab_get_attr(hv,i) || llama_vocab_is_eog(vocab,i)!=llama_vocab_is_eog(hv,i))
                throw std::runtime_error("MTP vocabulary mismatch");
            // The prepared head carries exact copies of the target's quantized
            // shared tensors. Borrow their live tensors to share cache keys as
            // well as arithmetic. Target lifetime encloses the head lifetime.
            for(auto pair : {std::make_pair(head->tok_embd,target->tok_embd),std::make_pair(head->output,target->output)}) {
                if(!pair.first || !pair.second || pair.first->type!=pair.second->type ||
                   ggml_nbytes(pair.first)!=ggml_nbytes(pair.second) ||
                   std::memcmp(pair.first->ne,pair.second->ne,sizeof(pair.first->ne)) ||
                   std::memcmp(pair.first->data,pair.second->data,ggml_nbytes(pair.first)))
                    throw std::runtime_error("shared MTP weights differ from the target");
            }
            head->tok_embd=target->tok_embd;head->output=target->output;
        }
        auto cp=llama_context_default_params();
        cp.n_ctx=context;cp.n_batch=128;cp.n_ubatch=128;cp.n_seq_max=1;
        cp.n_threads=threads;cp.n_threads_batch=threads;cp.offload_kqv=false;cp.op_offload=false;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.n_rs_seq=head?3:0;
        auto pp=ggml_threadpool_params_default(threads);
        std::unique_ptr<ggml_threadpool,decltype(&ggml_threadpool_free)> pool(ggml_threadpool_new(&pp),ggml_threadpool_free);
        if(!pool) throw std::runtime_error("threadpool creation failed");
        Decoder tgt(target.get(),cp,pool.get());
        std::unique_ptr<Decoder> dft;
        if(head) {
            cp.ctx_type=LLAMA_CONTEXT_TYPE_MTP;cp.n_rs_seq=0;
            dft.reset(new Decoder(head.get(),cp,pool.get()));
            llama_set_embeddings_nextn(tgt.ctx.get(),true,false);
            llama_set_embeddings_nextn(dft->ctx.get(),true,true);
        }
        if(serve) {std::printf("{\"event\":\"ready\",\"protocol\":1}\n");std::fflush(stdout);}
        for(int repeat=0;serve || repeat<repeats;++repeat) {
            if(serve && !read_request(prompt,count)) break;
        int np=-llama_tokenize(vocab,prompt.data(),prompt.size(),nullptr,0,true,true);
        if(np<=0 || np+count>context) throw std::runtime_error("prompt/context bounds exceeded");
        std::vector<llama_token> input(np);
        if(llama_tokenize(vocab,prompt.data(),prompt.size(),input.data(),np,true,true)!=np) throw std::runtime_error("tokenization failed");
            bool mtp_active=bool(dft),auto_disabled=false;
            double scalar_seconds=0;int scalar_steps=0;

            llama_memory_clear(llama_get_memory(tgt.ctx.get()),true);
            if(dft) llama_memory_clear(llama_get_memory(dft->ctx.get()),true);
            std::vector<float> pending(width,0);
            start=Clock::now();
            for(int pos=0;pos<np;pos+=128) {
                int n=std::min(128,np-pos);std::vector<llama_token> chunk(input.begin()+pos,input.begin()+pos+n);
                tgt.decode(chunk,pos,false);
                if(dft) {
                    auto h=tgt.hidden(n,width);std::vector<float> shifted((size_t)n*width);
                    std::copy(pending.begin(),pending.end(),shifted.begin());
                    if(n>1) std::copy(h.begin(),h.end()-width,shifted.begin()+width);
                    dft->decode(chunk,pos,false,shifted.data(),width,false);
                    pending.assign(h.end()-width,h.end());
                }
            }
            double prefill=seconds(start),draft_seconds=0,verify_seconds=0,catchup_seconds=0,rollback_seconds=0;
            int pos=np,proposed=0,accepted=0,rounds=0,rollbacks=0;
            int hist[3]={0,0,0};std::vector<llama_token> result;
            auto next=greedy(llama_get_logits_ith(tgt.ctx.get(),-1),nv);
            start=Clock::now();
            while((int)result.size()<count) {
                if(!force_count && llama_vocab_is_eog(vocab,next)) {result.push_back(next);break;}
                int n=mtp_active && !(auto_mtp && scalar_steps==0)?std::max(0,std::min(drafts,count-(int)result.size()-2)):0;
                if(n==0) {
                    result.push_back(next);
                    if((int)result.size()==count) break;
                    auto t=Clock::now();tgt.decode({next},pos,false);double elapsed=seconds(t);verify_seconds+=elapsed;
                    scalar_seconds+=elapsed;++scalar_steps;
                    if(mtp_active) {
                        auto h=tgt.hidden(1,width);t=Clock::now();
                        dft->decode({next},pos,false,pending.data(),width,false);
                        catchup_seconds+=seconds(t);pending=std::move(h);
                    }
                    ++pos;
                    next=greedy(llama_get_logits_ith(tgt.ctx.get(),-1),nv);continue;
                }
                ++rounds;std::vector<llama_token> candidate{next};auto carry=pending;
                auto t=Clock::now();
                for(int step=0;step<n;++step) {
                    dft->decode({candidate.back()},pos+step,false,carry.data(),width);
                    candidate.push_back(greedy(llama_get_logits_ith(dft->ctx.get(),-1),nv));
                    carry=dft->hidden(1,width);
                }
                if(reject_first) candidate[1]=(candidate[1]+1)%nv;
                draft_seconds+=seconds(t);proposed+=n;
                t=Clock::now();tgt.decode(candidate,pos,true);verify_seconds+=seconds(t);
                auto h=tgt.hidden(n+1,width);int keep=1;
                while(keep<=n && candidate[keep]==greedy(llama_get_logits_ith(tgt.ctx.get(),keep-1),nv)) {
                    ++keep;
                    if(!force_count && llama_vocab_is_eog(vocab,candidate[keep-1])) break;
                }
                accepted+=keep-1;++hist[keep-1];
                next=greedy(llama_get_logits_ith(tgt.ctx.get(),keep-1),nv);
                t=Clock::now();
                if(keep<n+1) {tgt.remove(pos+keep);++rollbacks;}
                // Rebuild this region from true target states, not MTP predictions.
                dft->remove(pos);rollback_seconds+=seconds(t);
                std::vector<float> shifted((size_t)keep*width);
                std::copy(pending.begin(),pending.end(),shifted.begin());
                if(keep>1) std::copy(h.begin(),h.begin()+(size_t)(keep-1)*width,shifted.begin()+width);
                candidate.resize(keep);t=Clock::now();dft->decode(candidate,pos,false,shifted.data(),width,false);catchup_seconds+=seconds(t);
                pending.assign(h.begin()+(size_t)(keep-1)*width,h.begin()+(size_t)keep*width);
                result.insert(result.end(),candidate.begin(),candidate.end());pos+=keep;
                // Three trials avoid deciding from a single first-use sample.
                if(auto_mtp && rounds>=3 && scalar_steps &&
                   (draft_seconds+verify_seconds-scalar_seconds+catchup_seconds+rollback_seconds)/
                       (rounds+accepted) > scalar_seconds/scalar_steps) {
                    mtp_active=false;auto_disabled=true;
                }
                if(!force_count && llama_vocab_is_eog(vocab,result.back())) break;
            }
            double generation=seconds(start);
            std::string text;std::vector<char> piece(32);
            for(auto id:result) {
                int n=llama_token_to_piece(vocab,id,piece.data(),piece.size(),0,false);
                if(n<0) {piece.resize(-n);n=llama_token_to_piece(vocab,id,piece.data(),piece.size(),0,false);}
                if(n<0) throw std::runtime_error("token rendering failed");
                text.append(piece.data(),n);
            }
            auto cache=qwen_dense_cache_stats();
            std::printf("{\"repeat\":%d,\"mtp\":%s,\"reject_first\":%s,\"prompt_tokens\":%d,\"generated_tokens\":%zu,\"load_seconds\":%.9f,\"prefill_seconds\":%.9f,\"generation_seconds\":%.9f,\"tokens_per_second\":%.9f,\"draft_seconds\":%.9f,\"verify_seconds\":%.9f,\"catchup_seconds\":%.9f,\"rollback_seconds\":%.9f,\"rounds\":%d,\"proposed\":%d,\"accepted\":%d,\"rollbacks\":%d,\"accept_histogram\":[%d,%d,%d],\"cache_bytes\":%llu,\"greedy\":[",
                repeat,dft?"true":"false",reject_first?"true":"false",np,result.size(),load,prefill,generation,result.size()/generation,
                draft_seconds,verify_seconds,catchup_seconds,rollback_seconds,rounds,proposed,accepted,rollbacks,hist[0],hist[1],hist[2],(unsigned long long)cache.retained_bytes);
            for(size_t i=0;i<result.size();++i) std::printf("%s%d",i?",":"",result[i]);
            std::printf("],\"auto_mtp\":%s,\"auto_disabled\":%s,\"text\":%s}\n",auto_mtp?"true":"false",auto_disabled?"true":"false",json_text(text).c_str());std::fflush(stdout);
        }
        qwen_dense_cache_configure(0); // All workers quiescent; clear before model lifetimes end.
        return 0;
    } catch(const std::exception &error) {std::fprintf(stderr,"MTP_ERROR %s\n",error.what());return 1;}
}
