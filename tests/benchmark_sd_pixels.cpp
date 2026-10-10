#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <time.h>
extern "C" int sd_pixels_reference_clamp(float*,size_t);
extern "C" int sd_pixels_reference_pack(const float*,unsigned char*,size_t);
extern "C" int ve_sd_turbo_pixel_clamp(float*,size_t);
extern "C" int ve_sd_turbo_pixel_pack(const float*,unsigned char*,size_t);
static double now(){timespec t;if(clock_gettime(CLOCK_MONOTONIC,&t))std::abort();return t.tv_sec+t.tv_nsec*1e-9;}
static void save(const std::string& path,const void*data,size_t bytes){
 FILE*f=std::fopen(path.c_str(),"wb");if(!f)std::abort();
 if(std::fwrite(data,1,bytes,f)!=bytes||std::fclose(f))std::abort();
}
int main(int argc,char**argv){
 if(argc!=3)return 2;
 float f=0;unsigned char b=0;
 if(ve_sd_turbo_pixel_clamp(NULL,1)||ve_sd_turbo_pixel_clamp(&f,0)||ve_sd_turbo_pixel_clamp(&f,786433)||
    ve_sd_turbo_pixel_pack(NULL,&b,1)||ve_sd_turbo_pixel_pack(&f,NULL,1)||ve_sd_turbo_pixel_pack(&f,&b,0)||ve_sd_turbo_pixel_pack(&f,&b,262145))return 3;
 std::printf("PIXEL_INVALID_ARGUMENTS checks=7 PASS\n");
 std::ifstream input(argv[1]);if(!input)return 4;std::string line;int index=0,finite_cases=0;
 while(std::getline(input,line)){
  std::istringstream stream(line);int id,invalid;size_t spatial;std::string path;
  if(!(stream>>id>>spatial>>invalid>>path)||id!=index||spatial<1||spatial>262144)return 5;
  const size_t count=spatial*3;std::vector<float> source(count),reference(count+2),candidate(count+2);
  std::vector<unsigned char> ref_rgb(count+2,173),rgb(count+2,173);
  FILE*fp=std::fopen(path.c_str(),"rb");if(!fp)return 6;
  if(std::fread(source.data(),sizeof(float),count,fp)!=count||std::fgetc(fp)!=EOF)return 7;std::fclose(fp);
  reference.front()=reference.back()=candidate.front()=candidate.back()=12345.f;
  std::memcpy(reference.data()+1,source.data(),count*sizeof(float));std::memcpy(candidate.data()+1,source.data(),count*sizeof(float));
  int expected=sd_pixels_reference_clamp(reference.data()+1,count),actual=ve_sd_turbo_pixel_clamp(candidate.data()+1,count);
  if(expected!=(invalid<0)||actual!=expected||std::memcmp(reference.data(),candidate.data(),(count+2)*sizeof(float))||candidate.front()!=12345.f||candidate.back()!=12345.f)return 8;
  if(expected){
   if(!sd_pixels_reference_pack(reference.data()+1,ref_rgb.data()+1,spatial)||!ve_sd_turbo_pixel_pack(candidate.data()+1,rgb.data()+1,spatial)||std::memcmp(ref_rgb.data(),rgb.data(),count+2)||rgb.front()!=173||rgb.back()!=173)return 9;
  }
  char name[64];std::snprintf(name,sizeof(name),"case%02d.clamped.f32",id);save(std::string(argv[2])+"/"+name,candidate.data()+1,count*sizeof(float));
  if(expected){std::snprintf(name,sizeof(name),"case%02d.rgb",id);save(std::string(argv[2])+"/"+name,rgb.data()+1,count);}
  std::printf("PIXEL_CHECK index=%d spatial=%zu invalid=%d clamp_ok=%d full_float_bytes=1 full_rgb_bytes=%d canaries=1\n",id,spatial,invalid,actual,expected);
  if(expected){
   finite_cases++;
   for(int arm=0;arm<4;arm++)for(int rep=0;rep<4;rep++){
    bool opt=arm==1||arm==2;std::memcpy(candidate.data()+1,source.data(),count*sizeof(float));
    double start=now();int ok=opt?ve_sd_turbo_pixel_clamp(candidate.data()+1,count):sd_pixels_reference_clamp(candidate.data()+1,count);double clamp=now()-start;
    if(!ok||std::memcmp(reference.data(),candidate.data(),(count+2)*sizeof(float)))return 10;
    start=now();ok=opt?ve_sd_turbo_pixel_pack(candidate.data()+1,rgb.data()+1,spatial):sd_pixels_reference_pack(candidate.data()+1,rgb.data()+1,spatial);double pack=now()-start;
    if(!ok||std::memcmp(rgb.data(),ref_rgb.data(),count+2)||std::memcmp(reference.data(),candidate.data(),(count+2)*sizeof(float)))return 11;
    std::printf("PIXEL_TIME index=%d arm=%d rep=%d kernel=%s clamp_seconds=%.9f pack_seconds=%.9f full_bytes=1\n",id,arm,rep,opt?"candidate":"reference",clamp,pack);std::fflush(stdout);
   }
  }
  index++;
 }
 if(index!=27||finite_cases!=18)return 12;
 std::printf("PIXEL_COMPLETE cases=27 finite=18 timed_pairs=288 invalid=9\n");return 0;
}
