"""Prepare a native MTP head using the target's exact quantized shared tensors.

Run with temperature_guard.py. No upstream edits or global downloads/install.
Configuration/vocabulary/norm equality is compatibility evidence, not proof
that every quantized target tensor came from the selected official revision.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'build/vendor/llama.cpp/gguf-py'))
from gguf import GGUFReader,GGUFWriter,GGUFValueType

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()

def main():
    folder=ROOT/'build/models/qwen36-mtp'
    source=folder/'mtp-f32.gguf'
    target=ROOT/'build/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
    target_sha=sha(target)
    if target_sha!='ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61':
        raise RuntimeError('target checksum mismatch')
    head=GGUFReader(str(source));base=GGUFReader(str(target))
    def value(r,name):return r.fields[name].contents()
    if value(head,'general.architecture')!=value(base,'general.architecture') or value(head,'qwen35moe.nextn_predict_layers')!=1:
        raise RuntimeError('architecture/MTP mismatch')
    keys=('embedding_length','attention.head_count','attention.head_count_kv','attention.key_length','attention.value_length',
          'expert_count','expert_used_count','expert_feed_forward_length','attention.layer_norm_rms_epsilon','rope.freq_base','rope.dimension_count')
    for suffix in keys:
        key='qwen35moe.'+suffix
        if key not in head.fields or key not in base.fields or value(head,key)!=value(base,key):
            raise RuntimeError('configuration mismatch: '+key)
    if value(head,'qwen35moe.block_count')!=value(base,'qwen35moe.block_count')+1:
        raise RuntimeError('block count mismatch')
    vocab_keys=[key for key in base.fields if key.startswith('tokenizer.') and key!='tokenizer.chat_template']
    tokenizer_adjustments=[]
    for key in vocab_keys:
        if key not in head.fields or value(head,key)!=value(base,key):
            # Padding is never supplied to the single-sequence draft graph.
            # Preserve the verified token-ID mapping; use target padding policy.
            if key=='tokenizer.ggml.padding_token_id':tokenizer_adjustments.append(key)
            else:raise RuntimeError('tokenizer mismatch: '+key)
    for key in head.fields:
        if key.startswith('tokenizer.') and key not in base.fields:
            if key=='tokenizer.ggml.add_eos_token' and value(head,key) is False:
                tokenizer_adjustments.append(key)
            else:raise RuntimeError('unexpected tokenizer metadata: '+key)
    ht={t.name:t for t in head.tensors};bt={t.name:t for t in base.tensors}
    if not np.array_equal(ht['output_norm.weight'].data,bt['output_norm.weight'].data):
        raise RuntimeError('official/quantized target output norm mismatch')
    replacements={'token_embd.weight','output.weight'}
    output=folder/'mtp-shared-q4km.gguf';temporary=folder/'mtp-shared-q4km.gguf.partial'
    writer=GGUFWriter(str(temporary),value(head,'general.architecture'))
    for key,field in head.fields.items():
        if key.startswith(('GGUF.','tokenizer.')) or key=='general.architecture':continue
        writer.add_key_value(key,field.contents(),field.types[0],field.types[-1] if field.types[0]==GGUFValueType.ARRAY else None)
    for key,field in base.fields.items():
        if key.startswith('tokenizer.'):
            writer.add_key_value(key,field.contents(),field.types[0],field.types[-1] if field.types[0]==GGUFValueType.ARRAY else None)
    for tensor in head.tensors:
        if tensor.name in replacements:
            other=bt[tensor.name]
            if not np.array_equal(tensor.shape,other.shape):raise RuntimeError('shared tensor shape mismatch')
            tensor=other
        writer.add_tensor(tensor.name,tensor.data,raw_dtype=tensor.tensor_type)
    # This pinned writer emits tensor-info itself inside write_tensors_to_file.
    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
    temporary.replace(output)
    actual=GGUFReader(str(output));at={t.name:t for t in actual.tensors}
    for name in replacements:
        if at[name].tensor_type!=bt[name].tensor_type or not np.array_equal(at[name].shape,bt[name].shape) or not np.array_equal(at[name].data,bt[name].data):
            raise RuntimeError('written shared tensor mismatch')
    report={'target_sha256':target_sha,'source_f32_sha256':sha(source),'mtp_sha256':sha(output),
            'mtp_bytes':output.stat().st_size,'native_mtp_layers':1,'tensor_count':len(actual.tensors),
            'shared_tensor_types':{name:bt[name].tensor_type.name for name in sorted(replacements)},
            'configuration_keys_checked':list(keys),'tokenizer_keys_checked':vocab_keys,'tokenizer_settings_from_target':True,
            'tokenizer_metadata_adjustments':tokenizer_adjustments,'output_norm_exact_match':True,
            'target_official_revision_identity':'not established for all quantized tensors',
            'official_revision':'995ad96eacd98c81ed38be0c5b274b04031597b0'}
    (folder/'compatibility.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Native MTP GGUF prepared; bytes',report['mtp_bytes'],flush=True)

if __name__=='__main__':main()
