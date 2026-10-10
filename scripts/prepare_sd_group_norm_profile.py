"""Generate default-off, per-node GroupNorm geometry diagnostics.

This preparation helper does not run hardware or change accepted model inputs.
The caller must include it in build provenance before model integration.
"""
import argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def instrument(text):
 start=text.index('static void ggml_compute_forward_group_norm_f32(')
 end=text.index('static void ggml_compute_forward_group_norm(',start)
 body=text[start:end]
 needle='    int n_groups = dst->op_params[0];'
 if body.count(needle)!=1 or 'SD_GROUP_NORM_SHAPE ' in text:raise RuntimeError('unique uninstrumented GroupNorm function required')
 diagnostic=r'''
    const char * group_flag=getenv("SD_GROUP_NORM_SHAPE_PROFILE");
    if (ith==0 && group_flag && strcmp(group_flag,"1")==0) {
        const char * group_stage=getenv("SD_PROFILE_STAGE");
        fprintf(stderr,"SD_GROUP_NORM_SHAPE stage=%s threads=%d src_type=%d dst_type=%d ne=%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 " src_nb=%zu,%zu,%zu,%zu dst_nb=%zu,%zu,%zu,%zu groups=%d eps=%.9g\n",
                group_stage ? group_stage : "unknown",nth,(int)src0->type,(int)dst->type,
                src0->ne[0],src0->ne[1],src0->ne[2],src0->ne[3],
                src0->nb[0],src0->nb[1],src0->nb[2],src0->nb[3],
                dst->nb[0],dst->nb[1],dst->nb[2],dst->nb[3],n_groups,(double)eps);
    }
'''
 modified=body.replace(needle,needle+diagnostic)
 return text[:start]+modified+text[end:]
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('build/sd-baseline-overlay/sd-ggml-cpu.c'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();source=a.source.resolve();output=a.output.resolve();source.relative_to(ROOT/'build');output.relative_to(ROOT/'build')
 if output==source:raise RuntimeError('standalone preparation must preserve source')
 output.parent.mkdir(parents=True,exist_ok=True);output.write_text(instrument(source.read_text()));print('Default-off GroupNorm diagnostic source prepared:',output.relative_to(ROOT))
if __name__=='__main__':main()
