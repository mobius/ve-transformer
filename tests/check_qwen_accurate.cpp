// Independent long-double checks for strided precise normalization and activations.
#include "ggml.h"
#include "ggml-cpu-impl.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
extern "C" bool __wrap_ggml_cpu_extra_compute_forward(ggml_compute_params *,ggml_tensor *);
int main() {
    for(int width:{31,256,4096})for(float scale:{0.0f,0.125f,1.0f}) {
        auto *ctx=ggml_init({1024*1024,nullptr,true});
        auto *src=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,width,17,2);
        auto *out=ggml_soft_max_ext(ctx,src,nullptr,scale,0);
        // The public builder requires contiguous input. Adjust strides afterwards
        // to exercise the hook directly without claiming public strided support.
        src->nb[1]=(width+5)*sizeof(float);src->nb[2]=17*src->nb[1];src->nb[3]=2*src->nb[2];
        std::vector<float> input((width+5)*34,12345),result(width*34,-12345);
        for(int row=0;row<34;++row)for(int i=0;i<width;++i)
            input[row*(width+5)+i]=1000+((row*7+i*13)%101-50)*0.25f;
        src->data=input.data();out->data=result.data();
        for(int worker=0;worker<3;++worker) {
            ggml_compute_params params={};params.ith=worker;params.nth=3;
            if(!__wrap_ggml_cpu_extra_compute_forward(&params,out))return 1;
        }
        double maximum_error=0;
        for(int row=0;row<34;++row) {
            long double maximum=-INFINITY,sum=0,total=0;
            for(int i=0;i<width;++i)maximum=std::max(maximum,(long double)input[row*(width+5)+i]*scale);
            for(int i=0;i<width;++i)sum+=std::exp((long double)input[row*(width+5)+i]*scale-maximum);
            for(int i=0;i<width;++i) {
                long double reference=std::exp((long double)input[row*(width+5)+i]*scale-maximum)/sum;
                float value=result[row*width+i];double error=(double)std::abs((long double)value-reference);
                maximum_error=std::max(maximum_error,error);total+=value;
                if(!std::isfinite(value) || value<0 || error>2e-8+6e-8*std::abs(reference)) {
                    std::fprintf(stderr,"softmax FAIL width=%d scale=%.3g row=%d column=%d reference=%.12Lg value=%.12g error=%.9g\n",width,scale,row,i,reference,value,error);return 1;
                }
            }
            if(std::abs(total-1)>2e-7L)return 1;
        }
        std::printf("precise softmax width=%d scale=%.3g max_abs_error=%.9g PASS\n",width,scale,maximum_error);
        ggml_free(ctx);
    }

    for(int width:{8,128})for(int tokens:{1,3})for(int vector_gate:{0,1}) {
        auto *ctx=ggml_init({1024*1024,nullptr,true});
        const int H=4,B=2,K=2,G=vector_gate?width:1;
        auto *q=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,width,1,tokens,1);
        auto *k=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,width,2,tokens,B);
        auto *v=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,width,H,tokens,B);
        auto *g=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,G,H,tokens,B);
        auto *beta=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,1,H,tokens,B);
        auto *state=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,width,width,H,B);
        auto *out=ggml_gated_delta_net(ctx,q,k,v,g,beta,state,K);
        std::vector<float> data[6];ggml_tensor *inputs[]={q,k,v,g,beta,state};
        for(int n=0;n<6;++n) {
            data[n].resize(ggml_nelements(inputs[n]));
            for(size_t i=0;i<data[n].size();++i)data[n][i]=((int)((i*13+n*7)%37)-18)*0.015625f;
            inputs[n]->data=data[n].data();
        }
        for(float &x:data[3])x=-std::abs(x)-0.125f;
        for(float &x:data[4])x=0.25f+std::abs(x);
        std::vector<float> result(ggml_nelements(out),-12345),reference(result);
        out->data=result.data();
        for(int worker=0;worker<11;++worker) {
            ggml_compute_params params={};params.ith=worker;params.nth=11;
            if(!__wrap_ggml_cpu_extra_compute_forward(&params,out))return 1;
        }
        const int attn=width*H*tokens*B,snapshot=width*width*H*B;
        // Logical state [key,value] is independent of the operator's transposed storage.
        for(int b=0;b<B;++b)for(int h=0;h<H;++h) {
            std::vector<float> logical(width*width),delta(width);
            for(int i=0;i<width;++i)for(int j=0;j<width;++j)
                logical[i*width+j]=data[5][(b*H+h)*width*width+j*width+i];
            for(int t=0;t<tokens;++t) {
                for(int i=0;i<width;++i)for(int j=0;j<width;++j) {
                    long double gate=data[3][((b*tokens+t)*H+h)*G+(vector_gate?i:0)];
                    logical[i*width+j]=(float)((long double)logical[i*width+j]*std::exp(gate));
                }
                for(int j=0;j<width;++j) {
                    long double sum=0;
                    for(int i=0;i<width;++i)sum+=(long double)logical[i*width+j]*data[1][((b*tokens+t)*2+h%2)*width+i];
                    delta[j]=(float)(((long double)data[2][((b*tokens+t)*H+h)*width+j]-sum)*data[4][(b*tokens+t)*H+h]);
                }
                for(int j=0;j<width;++j) {
                    long double sum=0;
                    for(int i=0;i<width;++i) {
                        logical[i*width+j]=(float)((long double)logical[i*width+j]+(long double)data[1][((b*tokens+t)*2+h%2)*width+i]*delta[j]);
                        sum+=(long double)logical[i*width+j]*data[0][t*width+i];
                    }
                    reference[((b*tokens+t)*H+h)*width+j]=(float)(sum/std::sqrt((long double)width));
                }
                int slot=tokens-1-t;
                if(slot<K)for(int i=0;i<width;++i)for(int j=0;j<width;++j)
                    reference[attn+slot*snapshot+(b*H+h)*width*width+j*width+i]=logical[i*width+j];
            }
        }
        double maximum=0;
        for(size_t i=0;i<result.size();++i) {
            double error=std::abs((double)result[i]-reference[i]);maximum=std::max(maximum,error);
            if(!std::isfinite(result[i]) || error>2e-6+2e-6*std::abs(reference[i])) {
                std::fprintf(stderr,"delta net FAIL width=%d tokens=%d vector_gate=%d index=%zu error=%.9g\n",width,tokens,vector_gate,i,error);return 1;
            }
            if(reference[i]==-12345 && result[i]!=-12345)return 1;
        }
        std::printf("precise delta net width=%d tokens=%d vector_gate=%d snapshots=%d max_abs_error=%.9g PASS\n",width,tokens,vector_gate,K,maximum);
        ggml_free(ctx);
    }

    for(int tokens:{1,3}) {
        auto *ctx=ggml_init({1024*1024,nullptr,true});
        const int channels=17,sequences=2,kernel=4,length=kernel-1+tokens;
        auto *src=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,length,channels,sequences);
        auto *weights=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,kernel,channels);
        auto *out=ggml_ssm_conv(ctx,src,weights);
        std::vector<float> input(length*channels*sequences),coefficients(kernel*channels),result(channels*tokens*sequences,-12345);
        for(size_t k=0;k<input.size();++k)input[k]=((int)(k*13%101)-50)*0.03125f;
        for(size_t k=0;k<coefficients.size();++k)coefficients[k]=((int)(k*17%97)-48)*0.003125f;
        src->data=input.data();weights->data=coefficients.data();out->data=result.data();
        for(int worker=0;worker<3;++worker) {
            ggml_compute_params params={};params.ith=worker;params.nth=3;
            if(!__wrap_ggml_cpu_extra_compute_forward(&params,out))return 1;
        }
        for(int seq=0;seq<sequences;++seq)for(int token=0;token<tokens;++token)for(int channel=0;channel<channels;++channel) {
            long double reference=0;
            for(int k=0;k<kernel;++k)reference+=(long double)input[(seq*channels+channel)*length+token+k]*coefficients[channel*kernel+k];
            float value=result[(seq*tokens+token)*channels+channel];
            if(!std::isfinite(value)||std::abs((long double)value-reference)>0.000001L)return 1;
        }
        std::printf("precise convolution tokens=%d channels=%d sequences=%d PASS\n",tokens,channels,sequences);
        ggml_free(ctx);
    }
    for(int width:{128,2048})for(int operation=0;operation<4;++operation) {
        auto *ctx=ggml_init({1024*1024,nullptr,true});
        auto *src=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,width,19,3);
        src->nb[1]=(width+7)*sizeof(float);src->nb[2]=src->nb[1]*19;src->nb[3]=src->nb[2]*3;
        auto *out=operation==0?ggml_rms_norm(ctx,src,1e-6f):operation==1?ggml_silu(ctx,src):
                  operation==2?ggml_sigmoid(ctx,src):ggml_softplus(ctx,src);
        std::vector<float> input((width+7)*57,12345),result(width*57,-12345);
        for(int row=0;row<57;++row)for(int k=0;k<width;++k)
            input[row*(width+7)+k]=((row*17+k*13)%201-100)*0.3125f;
        src->data=input.data();out->data=result.data();
        for(int worker=0;worker<3;++worker) {
            ggml_compute_params params={};params.ith=worker;params.nth=3;
            if(!__wrap_ggml_cpu_extra_compute_forward(&params,out))return 1;
        }
        long double maximum=0;
        for(int row=0;row<57;++row) {
            long double scale=0;
            if(operation==0) {
                for(int k=0;k<width;++k) {long double x=input[row*(width+7)+k];scale+=x*x;}
                scale=1/std::sqrt(scale/width+(long double)1e-6f);
            }
            for(int k=0;k<width;++k) {
                long double x=input[row*(width+7)+k],reference;
                if(operation==0)reference=x*scale;
                else if(operation==3)reference=std::log1p(std::exp(x));
                else {
                    long double sigmoid=1/(1+std::exp(-x));
                    reference=operation==1?x*sigmoid:sigmoid;
                }
                long double error=std::abs((long double)result[row*width+k]-reference);
                maximum=std::max(maximum,error);
                if(!std::isfinite(result[row*width+k]) || error>0.000004L+0.0000003L*std::abs(reference)) {
                    std::fprintf(stderr,"precise primitive FAIL width=%d op=%d row=%d col=%d\n",width,operation,row,k);
                    return 1;
                }
            }
        }
        std::printf("precise primitive width=%d op=%d max_abs_error=%.9g PASS\n",width,operation,(double)maximum);
        ggml_free(ctx);
    }
    return 0;
}
