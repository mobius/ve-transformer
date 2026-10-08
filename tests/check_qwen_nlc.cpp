#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>
#ifdef QWEN_ACCUM_FP64
#include "../src/qwen_dense_cache.h"
#endif
extern "C" bool __wrap_ggml_cpu_extra_compute_forward(ggml_compute_params *,ggml_tensor *);
extern "C" void ve_dequant_rows(ggml_type,const void *,float *,int,int,uint32_t *);
static int check_decode_bounds(ggml_type type,int rows,int width) {
    size_t count=(size_t)rows*width,bytes=ggml_row_size(type,width)*(size_t)rows;
    std::vector<float> input(count),decoded(count),reference(width);
    std::vector<unsigned char> packed(bytes);
    std::vector<uint32_t> scratch((bytes+3)/4);
    for(size_t k=0;k<count;++k)input[k]=((int)((k*13+7)%101)-50)*0.03125f;
    ggml_quantize_chunk(type,input.data(),packed.data(),0,rows,width,nullptr);
    ve_dequant_rows(type,packed.data(),decoded.data(),rows,width,scratch.data());
    double maximum=0;
    for(int row=0;row<rows;++row) {
        ggml_get_type_traits(type)->to_float(packed.data()+row*ggml_row_size(type,width),reference.data(),width);
        for(int k=0;k<width;++k) {
            float got=decoded[(size_t)row*width+k],ref=reference[k];
            double error=std::abs((double)got-ref);maximum=std::max(maximum,error);
            if(!std::isfinite(got) || error>0.000001+0.000001*std::abs(ref))return 1;
        }
    }
    std::printf("decode_bound type=%s rows=%d width=%d max_abs_error=%.9g PASS\n",
                ggml_type_name(type),rows,width,maximum);
    return 0;
}
static int check(ggml_type type,int width,bool moe,int tokens,int expanded,bool cache_checks=false,int m=257,int input_pad=0) {
#ifdef QWEN_ACCUM_FP64
    qwen_dense_cache_configure(0);
#endif
    int experts=moe?6:1,used=moe?4:1,input_routes=expanded?used:1;
    auto *ctx=ggml_init({4*1024*1024,nullptr,true});if(!ctx)return 1;
    auto *a=ggml_new_tensor_3d(ctx,type,width,m,experts);
    auto *b=moe?ggml_new_tensor_3d(ctx,GGML_TYPE_F32,width,input_routes,tokens):ggml_new_tensor_2d(ctx,GGML_TYPE_F32,width,tokens);
    GGML_ASSERT(input_pad>=0 && (!moe || input_pad==0));
    if(input_pad) {
        b->nb[1]=(width+input_pad)*sizeof(float);
        b->nb[2]=b->nb[1]*b->ne[1];b->nb[3]=b->nb[2]*b->ne[2];
    }
    auto *ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,used,tokens);
    auto *dst=moe?ggml_mul_mat_id(ctx,a,b,ids):ggml_mul_mat(ctx,a,b);
    std::vector<float> weights((size_t)width*m*experts),input((size_t)(width+input_pad)*input_routes*tokens),output(m*used*tokens,-12345),decoded(width);
    std::vector<int32_t> selection(used*tokens);
    std::vector<unsigned char> packed(ggml_nbytes(a));
    for(size_t k=0;k<weights.size();++k)weights[k]=((int)((k*13+7)%101)-50)*0.03125f;
    for(size_t k=0;k<input.size();++k)input[k]=((int)((k*17+3)%97)-48)*0.03125f;
    for(int t=0;t<tokens;++t)for(int j=0;j<used;++j)selection[t*used+j]=(t*3+(j==2?0:j)*5)%experts;
    if(type==GGML_TYPE_F32)std::memcpy(packed.data(),weights.data(),packed.size());
    else ggml_quantize_chunk(type,weights.data(),packed.data(),0,m*experts,width,nullptr);
    a->data=packed.data();b->data=input.data();ids->data=selection.data();dst->data=output.data();
    int failures=0;
#pragma omp parallel for num_threads(3) reduction(+:failures)
    for(int worker=0;worker<3;++worker) {
        ggml_compute_params p={};p.ith=worker;p.nth=3;
        if(!__wrap_ggml_cpu_extra_compute_forward(&p,dst))failures++;
    }
    double maximum=0;
    for(int t=0;t<tokens;++t)for(int j=0;j<used;++j)for(int r=0;r<m;++r) {
        int expert=moe?selection[t*used+j]:0;
        const void *row=packed.data()+expert*a->nb[2]+r*a->nb[1];
        if(type==GGML_TYPE_F32)std::memcpy(decoded.data(),row,width*4);
        else ggml_get_type_traits(type)->to_float(row,decoded.data(),width);
        double ref=0;for(int k=0;k<width;++k)ref+=(double)decoded[k]*input[(t*input_routes+j%input_routes)*(width+input_pad)+k];
        float got=output[(t*used+j)*m+r];double error=std::abs(got-ref);maximum=std::max(maximum,error);
        if(!std::isfinite(got)||error>0.002+0.00002*std::abs(ref))failures++;
    }
    std::printf("type=%s width=%d moe=%d tokens=%d input_routes=%d input_pad=%d max_abs_error=%.9g %s\n",ggml_type_name(type),width,moe,tokens,input_routes,input_pad,maximum,failures?"FAIL":"PASS");
#ifdef QWEN_ACCUM_FP64
    if(cache_checks && !failures) {
        const auto baseline=output;
        auto compute=[&]() {
            int errors=0;
#pragma omp parallel for num_threads(3) reduction(+:errors)
            for(int worker=0;worker<3;++worker) {
                ggml_compute_params p={};p.ith=worker;p.nth=3;
                if(!__wrap_ggml_cpu_extra_compute_forward(&p,dst))++errors;
            }
            return errors;
        };
        const uint64_t required=(uint64_t)width*m*sizeof(double);
#ifdef QWEN_PROJECTION_TILE_ROWS
        constexpr int block_rows=QWEN_PROJECTION_TILE_ROWS;
#else
        constexpr int block_rows=128;
#endif
        uint64_t expected_entries=0;
        for(int worker=0;worker<3;++worker)expected_entries+=(m*(worker+1)/3-m*worker/3+block_rows-1)/block_rows;
        for(uint64_t budget:{uint64_t(1),required/2,required}) {
            qwen_dense_cache_configure(budget);
            if(compute() || output!=baseline)return 1;
            auto cold=qwen_dense_cache_stats();
            if(cold.retained_bytes>budget || cold.reserved_bytes)return 1;
            if(moe && (cold.entries || cold.hits || cold.misses))return 1;
            if(!moe && budget==1 && (cold.retained_bytes || !cold.rejected))return 1;
            if(!moe && budget==required/2 && (!cold.retained_bytes || !cold.rejected))return 1;
            if(!moe && budget==required && (cold.retained_bytes!=required || cold.entries!=expected_entries))return 1;
            if(compute() || output!=baseline)return 1;
            auto warm=qwen_dense_cache_stats();
            if(warm.reserved_bytes)return 1;
            if(!moe && budget==required && warm.hits<cold.hits+expected_entries)return 1;
        }
#ifdef QWEN_REUSE_INPUT
        // Same address, new activation values: operator-local preparation must refresh.
        for(float &value:input)value*=-0.5f;
        if(compute())return 1;
        for(size_t k=0;k<output.size();++k)if(output[k]!=-0.5f*baseline[k])return 1;
        for(float &value:input)value*=-2.0f;
        if(compute() || output!=baseline)return 1;
        std::printf("input refresh same-address values and restore PASS\n");
#endif
        // A new model may reuse addresses; its boundary must invalidate old tiles.
        qwen_dense_cache_configure(0);
        for(float &value:weights)value*=-0.5f;
        if(type==GGML_TYPE_F32)std::memcpy(packed.data(),weights.data(),packed.size());
        else ggml_quantize_chunk(type,weights.data(),packed.data(),0,m*experts,width,nullptr);
        if(compute())return 1;
        const auto new_model=output;
        if(new_model==baseline)return 1;
        qwen_dense_cache_configure(required);
        if(compute() || output!=new_model)return 1;
        qwen_dense_cache_configure(0);
        auto empty=qwen_dense_cache_stats();
        if(empty.entries || empty.retained_bytes || empty.budget_bytes)return 1;
        std::printf("cache contract type=%s expert=%d capacity, reuse, reset, output equality PASS\n",ggml_type_name(type),moe);
    }
#endif
    ggml_free(ctx);return failures?1:0;
}
int main() {
    ggml_cpu_init();
#ifdef QWEN_PROJECTION_SMOKE
    if(check(GGML_TYPE_Q4_K,256,false,3,0,true,4103))return 1;
    if(check(GGML_TYPE_Q6_K,512,false,1,0,true,4103))return 1;
    if(check(GGML_TYPE_Q8_0,256,false,17,0,true,4103))return 1;
    if(check(GGML_TYPE_F32,512,false,3,0,true,4103))return 1;
#ifdef QWEN_REUSE_INPUT
    if(check(GGML_TYPE_Q4_K,256,false,129,0,true,65))return 1;
    if(check(GGML_TYPE_F32,256,false,257,0,true,65))return 1;
    if(check(GGML_TYPE_Q4_K,256,true,3,1,true))return 1;
    if(check(GGML_TYPE_Q8_0,256,false,129,0,true,65,7))return 1;
    std::printf("INPUT_REUSE_COLUMN_TAIL_PASS cases=2; columns 129 and 257\n");
#endif
    std::printf("PROJECTION_LARGE_TILE_PASS cases=4; bounded decode and cache contracts\n");
    return 0;
#endif
    for(auto type:{GGML_TYPE_Q4_K,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K,GGML_TYPE_Q8_0})
        for(int rows:{1,128})for(int width:{256,16384})
            if(check_decode_bounds(type,rows,width))return 1;
    for(auto type:{GGML_TYPE_F32,GGML_TYPE_Q4_K,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K,GGML_TYPE_Q8_0})
        for(int width:{256,512,2048})
            for(int mode=0;mode<4;++mode)
                if(check(type,width,mode>0,mode==1?1:3,mode==3))return 1;
#ifdef QWEN_ACCUM_FP64
    for(auto type:{GGML_TYPE_F32,GGML_TYPE_Q4_K,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K,GGML_TYPE_Q8_0})
        if(check(type,512,false,3,0,true))return 1;
    if(check(GGML_TYPE_Q4_K,512,true,3,1,true))return 1;
#endif
    return 0;
}
