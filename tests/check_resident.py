"""State reuse, stream rejection, timeouts, and single-card generation timings."""
from pathlib import Path
import argparse
import json
import os
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'python'))
import torch
from gpt_neo_ve import GPTNeoNative, load_model, greedy_generate
import resident_ve
from resident_ve import GPTNeoResident, ResidentSession, export_bundle


def protocol_checks(model):
    with tempfile.TemporaryDirectory(dir=resident_ve.ROOT/'build') as tmp:
        tmp = Path(tmp)
        bundle = tmp/'model.bundle'
        export_bundle(model, bundle, 8)
        original = bundle.read_bytes()
        runner = str(resident_ve.ROOT/'build/resident-cpu')
        for offset, value in [(8, 0), (8, 33), (12, 2049), (16, 1025),
                              (28, 2), (40, 8193), (48, 2)]:
            changed = bytearray(original)
            struct.pack_into('<I', changed, offset, value)
            bundle.write_bytes(changed)
            p = subprocess.run([runner, str(bundle)], input=b'', capture_output=True, timeout=10)
            assert p.returncode == 1 and not p.stdout
        for bad in (original[:19], original[:-1], original+b'junk'):
            bundle.write_bytes(bad)
            p = subprocess.run([runner, str(bundle)], input=b'', capture_output=True, timeout=10)
            assert p.returncode == 1
        bundle.write_bytes(original)
        d = model.config.hidden_size
        for bad in (struct.pack('<2I', 9, 0), struct.pack('<2I', 1, 2), b'\1',
                    struct.pack('<2I', 1, 0)+b'\0'*4,
                    struct.pack('<2I', 1, 0)+struct.pack('<{}f'.format(d), *([float('nan')]*d))):
            p = subprocess.run([runner, str(bundle)], input=bad, capture_output=True, timeout=10)
            assert p.returncode == 1 and p.stdout[:8] == b'VTREADY1'
        p = subprocess.run([runner, str(bundle)], input=struct.pack('<2I', 0, 0),
                           capture_output=True, timeout=10)
        assert p.returncode == 0 and len(p.stdout) == 16
    print('resident model/request bounds/truncation/nonfinite/close: PASS', flush=True)


def lifecycle_checks(model):
    x = torch.zeros(1, model.config.hidden_size)
    with ResidentSession(model, backend='cpu', capacity=8) as session:
        temp = session._tmp.name
        process = session.proc
        assert os.getpgid(process.pid) == os.getpgrp()
        for invalid in (torch.zeros(9, x.shape[1]), torch.full_like(x, float('inf')),
                        x.double(), torch.zeros(1, x.shape[1]+1)):
            try:
                session.forward(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError('invalid features accepted')
        session.forward(x)
    assert process.poll() == 0 and not Path(temp).exists()
    session.close()  # Idempotent close.
    try:
        session.forward(x)
    except RuntimeError:
        pass
    else:
        raise AssertionError('closed session accepted a request')
    session = ResidentSession(model, backend='cpu', capacity=8)
    process, temp = session.proc, session._tmp.name
    process.terminate()
    process.wait(timeout=2)
    try:
        session.forward(x)
    except RuntimeError:
        pass
    else:
        raise AssertionError('exited worker accepted a request')
    assert session.closed and not Path(temp).exists()
    # A controlled local fake worker handshakes but never answers requests.
    original_popen = subprocess.Popen
    script = ('import sys,struct,time; sys.stdout.buffer.write(struct.pack("<8sd",'
              'b"VTREADY1",0.0)); sys.stdout.buffer.flush(); time.sleep(10)')
    def stalled(command, **kwargs):
        return original_popen([sys.executable, '-c', script], **kwargs)
    with patch.object(resident_ve.subprocess, 'Popen', side_effect=stalled):
        session = ResidentSession(model, backend='cpu', capacity=8, timeout=0.3)
        process, temp = session.proc, session._tmp.name
        try:
            session.forward(x)
        except TimeoutError:
            pass
        else:
            raise AssertionError('stalled worker was not stopped')
        assert session.closed and process.poll() is not None and not Path(temp).exists()
    print('resident validation/recovery/exit/timeout/process-group cleanup: PASS', flush=True)


def run(model, tokenizer, node, repeats, include_old):
    prompt = 'Once upon a time there was a little girl'
    ids = tokenizer.encode(prompt, return_tensors='pt')[0]
    def cpu_last(tokens):
        return model(tokens.unsqueeze(0), use_cache=False).logits[0,-1]
    cpu_last(ids)
    cpu_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        ref, ref_ids = greedy_generate(cpu_last, ids, 16, model.config.eos_token_id)
        cpu_times.append(time.perf_counter()-start)
    print(json.dumps(dict(mode='torch-default-all-head', threads=1, times_s=cpu_times,
                          mean_tokens_per_second=16/statistics.mean(cpu_times),
                          generated_ids=ref_ids)), flush=True)
    if include_old:
        old = GPTNeoNative(model, node, 'fast')
        old.logits(ids, last_only=True)
        start = time.perf_counter()
        output, generated = greedy_generate(lambda tokens: old.logits(tokens,last_only=True)[0],
                                             ids, 16, model.config.eos_token_id)
        elapsed = time.perf_counter()-start
        assert generated == ref_ids and torch.equal(output, ref)
        print(json.dumps(dict(mode='old', node=node, time_s=elapsed,
                              tokens_per_second=16/elapsed, ids_match=True)), flush=True)
    # Full and last-only outputs, varying lengths, different inputs and local boundary.
    checks = [ids, tokenizer.encode('One day, a small dog',return_tensors='pt')[0],
              torch.tensor([i*37 % model.config.vocab_size for i in range(259)]), ids[:1], ids]
    for backend in ('cpu', 'fast'):
        with GPTNeoResident(model, node, backend, capacity=512) as native:
            for index, tokens in enumerate(checks):
                expected = model(tokens.unsqueeze(0),use_cache=False).logits[0]
                actual = native.logits(tokens)
                torch.testing.assert_close(actual,expected,atol=2e-4,rtol=2e-4)
                assert torch.equal(actual.argmax(-1),expected.argmax(-1))
                last = native.logits(tokens,last_only=True)
                torch.testing.assert_close(last,actual[-1:],atol=2e-4,rtol=2e-4)
                print('resident backend={} node={} length={} repeat={} error={:.3g} PASS'.format(
                    backend,node,len(tokens),index,(actual-expected).abs().max().item()),flush=True)
    for threads in (1,2,4,8):
        with GPTNeoResident(model,node,'fast',capacity=512,threads=threads) as native:
            native.logits(ids,last_only=True)
            times, compute, roundtrip = [], [], []
            for _ in range(repeats):
                native.session.trace.clear()
                start = time.perf_counter()
                output, generated = greedy_generate(
                    lambda tokens:native.logits(tokens,last_only=True)[0], ids,16,
                    model.config.eos_token_id)
                times.append(time.perf_counter()-start)
                assert generated == ref_ids and torch.equal(output,ref)
                compute.append(sum(t['compute_ms'] for t in native.session.trace))
                roundtrip.append(sum(t['roundtrip_ms'] for t in native.session.trace))
            print(json.dumps(dict(mode='resident',node=node,threads=threads,times_s=times,
                                  mean_tokens_per_second=16/statistics.mean(times),
                                  compute_ms=compute,roundtrip_ms=roundtrip,
                                  export_ms=native.session.export_ms,start_ms=native.session.start_ms,
                                  load_ms=native.session.load_ms,ids_match=True)),flush=True)
    print('resident single-card generation/reference/state-reuse: PASS',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--node',type=int,default=1)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--include-old',action='store_true')
    args=parser.parse_args()
    if args.node<0 or not 1<=args.repeats<=5:
        parser.error('invalid node or repeat count')
    torch.set_num_threads(1)
    model,tokenizer=load_model()
    with torch.inference_mode():
        protocol_checks(model)
        lifecycle_checks(model)
        run(model,tokenizer,args.node,args.repeats,args.include_old)
