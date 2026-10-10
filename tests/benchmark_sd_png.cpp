#define _POSIX_C_SOURCE 200809L
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <time.h>
extern "C" int sd_png_reference(const char*,int,int,const unsigned char*,int);
extern "C" int ve_sd_turbo_png_write(const char*,int,int,const unsigned char*,int);
static double now(){struct timespec t;if(clock_gettime(CLOCK_MONOTONIC,&t))std::abort();return t.tv_sec+t.tv_nsec*1e-9;}
int main(int argc,char**argv){
 if(argc!=3)return 2;
 std::ifstream input(argv[1]);if(!input)return 3;std::string line;int index=0;
 while(std::getline(input,line)){
  std::istringstream stream(line);int id,width,height;std::string path;
  if(!(stream>>id>>width>>height>>path)||id!=index||width<1||width>512||height<1||height>512)return 4;
  const size_t bytes=(size_t)width*height*3;std::vector<unsigned char> rgb(bytes),copy(bytes);
  FILE*f=std::fopen(path.c_str(),"rb");if(!f)return 5;
  if(std::fread(rgb.data(),1,bytes,f)!=bytes||std::fgetc(f)!=EOF){std::fclose(f);return 6;}std::fclose(f);copy=rgb;
  std::printf("PNG_CONFIG index=%d width=%d height=%d channels=3 bytes=%zu\n",id,width,height,bytes);
  for(int arm=0;arm<4;arm++)for(int rep=0;rep<4;rep++){
   const bool candidate=arm==1||arm==2;char name[128];std::snprintf(name,sizeof(name),"case%02d-arm%d-rep%d.png",id,arm,rep);std::string output=std::string(argv[2])+"/"+name;
   double start=now();int ok=candidate?ve_sd_turbo_png_write(output.c_str(),width,height,rgb.data(),width*3):sd_png_reference(output.c_str(),width,height,rgb.data(),width*3);double elapsed=now()-start;
   if(ok!=1||std::memcmp(rgb.data(),copy.data(),bytes))return 7;
   std::printf("PNG_TIME index=%d arm=%d rep=%d codec=%s seconds=%.9f file=%s input_unchanged=1\n",id,arm,rep,candidate?"candidate":"reference",elapsed,name);std::fflush(stdout);
  }
  index++;
 }
 if(index!=15)return 8;std::printf("PNG_COMPLETE cases=%d calls=240\n",index);return 0;
}
