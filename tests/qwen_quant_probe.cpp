// Quantized dot kernels checked against double accumulation of dequantized inputs.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef VE_QUANT_CANDIDATE
extern "C" {
float ve_fp16_to_fp32(uint32_t);
void ve_vec_dot_q4_K_q8_K(int,float *,size_t,const void *,size_t,const void *,size_t,int);
void ve_vec_dot_q5_K_q8_K(int,float *,size_t,const void *,size_t,const void *,size_t,int);
void ve_vec_dot_q6_K_q8_K(int,float *,size_t,const void *,size_t,const void *,size_t,int);
void ve_vec_dot_q8_0_q8_0(int,float *,size_t,const void *,size_t,const void *,size_t,int);
}
#endif
int main(int argc,char **argv) {
    int rows=argc>1?std::atoi(argv[1]):512;
    int width=argc>2?std::atoi(argv[2]):2048;
    int repeats=argc>3?std::atoi(argv[3]):5;
    if(rows<1 || rows>16384 || width<256 || width>16384 || width%256 || repeats<1 || repeats>100) return 2;
    ggml_cpu_init();
#ifdef VE_QUANT_CANDIDATE
    for(uint32_t h=0;h<65536;++h) {
        float got=ve_fp16_to_fp32(h),ref=ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(h));
        uint32_t gb,rb;std::memcpy(&gb,&got,4);std::memcpy(&rb,&ref,4);
        if(!(std::isnan(got)&&std::isnan(ref)) && gb!=rb) {
            std::fprintf(stderr,"half conversion mismatch h=%u FAIL\n",h);return 1;
        }
    }
    std::fprintf(stderr,"All 65536 half encodings checked against ggml reference: PASS\n");
#endif
    std::vector<float> w(static_cast<size_t>(rows)*width),a(width),aq(width),row(width),out(rows);
    // Deterministic arithmetic fixture: no host/VE libm random differences.
    for(size_t k=0;k<w.size();++k) w[k]=static_cast<int>((k*13+7)%101)-50;
    for(int k=0;k<width;++k) a[k]=(static_cast<int>((k*17+3)%97)-48)*0.03125f;
    for(ggml_type type:{GGML_TYPE_Q4_K,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K,GGML_TYPE_Q8_0}) {
        auto ct=ggml_get_type_traits_cpu(type); auto qt=ggml_get_type_traits_cpu(ct->vec_dot_type);
        ggml_vec_dot_t dot=ct->vec_dot;
#ifdef VE_QUANT_CANDIDATE
        if(type==GGML_TYPE_Q4_K) dot=ve_vec_dot_q4_K_q8_K;
        if(type==GGML_TYPE_Q5_K) dot=ve_vec_dot_q5_K_q8_K;
        if(type==GGML_TYPE_Q6_K) dot=ve_vec_dot_q6_K_q8_K;
        if(type==GGML_TYPE_Q8_0) dot=ve_vec_dot_q8_0_q8_0;
#endif
        auto wt=ggml_get_type_traits(type); auto at=ggml_get_type_traits(ct->vec_dot_type);
        size_t stride=ggml_row_size(type,width);
        std::vector<unsigned char> qw(stride*rows),qa(ggml_row_size(ct->vec_dot_type,width));
        ggml_quantize_chunk(type,w.data(),qw.data(),0,rows,width,nullptr);
        qt->from_float(a.data(),qa.data(),width);
        if(ct->vec_dot_type==GGML_TYPE_Q8_K) dequantize_row_q8_K(reinterpret_cast<const block_q8_K *>(qa.data()),aq.data(),width);
        else at->to_float(qa.data(),aq.data(),width);
        double max_error=0,max_relative=0;
        for(int r=0;r<rows;++r) {
            dot(width,&out[r],0,qw.data()+r*stride,0,qa.data(),0,1);
            wt->to_float(qw.data()+r*stride,row.data(),width);
            double ref=0;for(int k=0;k<width;++k) ref+=double(row[k])*aq[k];
            double error=std::abs(out[r]-ref);max_error=std::max(max_error,error);
            max_relative=std::max(max_relative,error/(1+std::abs(ref)));
            if(!std::isfinite(out[r]) || error>0.02+0.0002*std::abs(ref)) {
                std::fprintf(stderr,"type=%s row=%d error=%.9g reference=%.9g FAIL\n",ggml_type_name(type),r,error,ref);return 1;
            }
        }
        // Warm kernel and OpenMP thread team before timing.
        for(int rep=-1;rep<repeats;++rep) {
            auto start=std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static)
            for(int r=0;r<rows;++r) dot(width,&out[r],0,qw.data()+r*stride,0,qa.data(),0,1);
            double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            if(rep>=0) {
                double checksum=0;for(float x:out) checksum+=x;
                std::printf("type=%s rows=%d width=%d repeat=%d seconds=%.9f weight_GBps=%.6f max_abs_error=%.9g max_relative_error=%.9g checksum=%.9g PASS\n",
                    ggml_type_name(type),rows,width,rep,sec,qw.size()/sec/1e9,max_error,max_relative,checksum);
            }
        }
    }
    return 0;
}
