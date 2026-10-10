"""Apply only the independently measured selective spatial-block delta."""
ORIGINAL = r'''            bool tile_vae=false;
#ifdef _OPENMP
            const char* spatial_flag=std::getenv("SD_NLC_VAE_SPATIAL_TILE");
            tile_vae=spatial_flag && std::strcmp(spatial_flag,"8192")==0 &&
                std::strcmp(sd_profile_stage(),"vae")==0 && omp_get_max_threads()==4 &&
                ((ne1==256 && ne01==262144 && ne10==2304) ||
                 (ne1==512 && ne01==65536 && ne10==4608) ||
                 (ne1==256 && ne01==65536 && (ne10==2304 || ne10==4608)) ||
                 (ne1==128 && ne01==262144 && ne10==2304));

#endif
            if (tile_vae) {
                for (int64_t offset=0; offset<ne01; offset+=8192) {
                    const int64_t width=std::min<int64_t>(8192,ne01-offset);
                    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                        ne1,width,ne10,1.0f,y,ne10,x+offset*ne00,ne00,0.0f,d+offset,ne01);
                }
                std::fprintf(stderr,"SD_NLC_SPATIAL_TILE stage=vae m=%lld n=%lld k=%lld tile=8192 calls=%lld\n",
                    (long long)ne1,(long long)ne01,(long long)ne10,(long long)((ne01+8191)/8192));
            } else {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        ne1, ne01, ne10,
                        1.0f,   y, ne10,
                                x, ne00,
                        0.0f,   d, ne01);
            }

'''
REPLACEMENT = r'''            bool tile_vae=false;
            int64_t spatial_block=8192;
#ifdef _OPENMP
            const char* spatial_flag=std::getenv("SD_NLC_VAE_SPATIAL_TILE");
            tile_vae=spatial_flag && (std::strcmp(spatial_flag,"8192")==0 ||
                std::strcmp(spatial_flag,"mixed4096")==0) &&
                std::strcmp(sd_profile_stage(),"vae")==0 && omp_get_max_threads()==4 &&
                ((ne1==256 && ne01==262144 && ne10==2304) ||
                 (ne1==512 && ne01==65536 && ne10==4608) ||
                 (ne1==256 && ne01==65536 && (ne10==2304 || ne10==4608)) ||
                 (ne1==128 && ne01==262144 && ne10==2304));

            if (tile_vae && std::strcmp(spatial_flag,"mixed4096")==0 &&
                ne1==256 && ne01==65536 && (ne10==2304 || ne10==4608)) spatial_block=4096;
#endif
            if (tile_vae) {
                for (int64_t offset=0; offset<ne01; offset+=spatial_block) {
                    const int64_t width=std::min<int64_t>(spatial_block,ne01-offset);
                    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
                        ne1,width,ne10,1.0f,y,ne10,x+offset*ne00,ne00,0.0f,d+offset,ne01);
                }
                std::fprintf(stderr,"SD_NLC_SPATIAL_TILE stage=vae m=%lld n=%lld k=%lld tile=%lld calls=%lld\n",
                    (long long)ne1,(long long)ne01,(long long)ne10,(long long)spatial_block,(long long)((ne01+spatial_block-1)/spatial_block));
            } else {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        ne1, ne01, ne10,
                        1.0f,   y, ne10,
                                x, ne00,
                        0.0f,   d, ne01);
            }

'''
def instrument(text):
    if text.count(ORIGINAL)!=1:
        raise RuntimeError('one unchanged original spatial block required')
    return text.replace(ORIGINAL,REPLACEMENT)
