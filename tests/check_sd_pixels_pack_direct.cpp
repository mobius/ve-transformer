#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <time.h>
extern "C" int sd_pixels_reference_pack(const float*,unsigned char*,size_t);
extern "C" int ve_sd_turbo_pixel_pack(const float*,unsigned char*,size_t);
static double now(){timespec t;if(clock_gettime(CLOCK_MONOTONIC,&t))std::abort();return t.tv_sec+t.tv_nsec*1e-9;}
int main(int argc,char**argv){
 if(argc!=3)return 2;
 std::ifstream input(argv[1]);if(!input)return 3;std::string line;int index=0;
 while(std::getline(input,line)){
  std::istringstream stream(line);int id;size_t spatial;std::string path;
  if(!(stream>>id>>spatial>>path)||id!=index||spatial<1||spatial>262144)return 4;
  const size_t count=spatial*3;
  std::vector<float> source(count+2,12345.f),reference(count+2,12345.f),candidate(count+2,12345.f);
  std::vector<unsigned char> ref_rgb(count+2,173),rgb(count+2,173);
  FILE*f=std::fopen(path.c_str(),"rb");if(!f)return 5;
  if(std::fread(source.data()+1,sizeof(float),count,f)!=count||std::fgetc(f)!=EOF)return 6;std::fclose(f);
  for(size_t i=1;i<=count;i++)if(!(source[i]>=0.f&&source[i]<=1.f))return 7;
  std::memcpy(reference.data(),source.data(),(count+2)*sizeof(float));std::memcpy(candidate.data(),source.data(),(count+2)*sizeof(float));
  if(!sd_pixels_reference_pack(reference.data()+1,ref_rgb.data()+1,spatial)||!ve_sd_turbo_pixel_pack(candidate.data()+1,rgb.data()+1,spatial)||std::memcmp(ref_rgb.data(),rgb.data(),count+2)||std::memcmp(source.data(),reference.data(),(count+2)*sizeof(float))||std::memcmp(source.data(),candidate.data(),(count+2)*sizeof(float))||rgb.front()!=173||rgb.back()!=173)return 8;
  char name[64];std::snprintf(name,sizeof name,"direct%02d.rgb",id);std::string dest=std::string(argv[2])+"/"+name;
  f=std::fopen(dest.c_str(),"wb");if(!f)return 9;
  if(std::fwrite(rgb.data()+1,1,count,f)!=count||std::fclose(f))return 10;
  std::printf("DIRECT_PACK_CHECK index=%d spatial=%zu full_rgb_bytes=1 input_unchanged=1 canaries=1\n",id,spatial);
  for(int arm=0;arm<4;arm++)for(int rep=0;rep<4;rep++){
   bool opt=arm==1||arm==2;std::memset(rgb.data(),173,count+2);
   const float* pixels=opt?candidate.data()+1:reference.data()+1;
   double start=now();int ok=opt?ve_sd_turbo_pixel_pack(pixels,rgb.data()+1,spatial):sd_pixels_reference_pack(pixels,rgb.data()+1,spatial);double seconds=now()-start;
   if(!ok||std::memcmp(ref_rgb.data(),rgb.data(),count+2)||std::memcmp(source.data(),reference.data(),(count+2)*sizeof(float))||std::memcmp(source.data(),candidate.data(),(count+2)*sizeof(float)))return 11;
   std::printf("DIRECT_PACK_TIME index=%d arm=%d rep=%d kernel=%s seconds=%.9f full_rgb_bytes=1 input_unchanged=1 canaries=1\n",id,arm,rep,opt?"candidate":"reference",seconds);std::fflush(stdout);
  }
  index++;
 }
 if(index!=12)return 12;
 std::printf("DIRECT_PACK_COMPLETE cases=12 timed_calls=192 PASS\n");return 0;
}
