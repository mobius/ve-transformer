#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/resource.h>
#include <sys/mman.h>
static int stage(const char *name) {
 struct rusage usage;int rc=getrusage(RUSAGE_SELF,&usage);
 printf("VE_MEMORY_STAGE %s total_pages=%ld free_pages=%ld page_bytes=%ld maxrss_kib=%ld\n",name,
        sysconf(_SC_PHYS_PAGES),sysconf(_SC_AVPHYS_PAGES),sysconf(_SC_PAGE_SIZE),rc?-1L:usage.ru_maxrss);
 fflush(stdout);return getchar()=='a'?0:1;
}
int main(void) {
 const size_t bytes=512UL*1024*1024;
 if(stage("baseline"))return 2;
 unsigned char *p=mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);if(p==MAP_FAILED)return 3;
 long pages=sysconf(_SC_PAGE_SIZE);if(pages<=0){munmap(p,bytes);return 4;}
 volatile unsigned char *touch=p;
 for(size_t i=0;i<bytes;i+=(size_t)pages)touch[i]=0x5a;
 touch[bytes-1]=0xa5;
 if(stage("allocated")){munmap(p,bytes);return 5;}
 if(munmap(p,bytes))return 6;
 if(stage("released"))return 7;
 return 0;
}
