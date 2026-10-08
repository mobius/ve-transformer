"""Sequence-only boundary regression against the accepted VE executor.

This does not extend the independent CPU full-score precision claim.
Run under temperature_guard.py; output contains aggregate metrics and hashes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from qwen_session import QwenSession


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, choices=(2, 4, 8), default=4)
    parser.add_argument('--candidate-dir',type=Path,default=ROOT/'build/qwen15-projection')
    args = parser.parse_args()
    folder = ROOT / 'build/results' / time.strftime('%Y%m%dT%H%M%SZ-qwen15-boundaries', time.gmtime())
    folder.mkdir()
    executor = ROOT / 'build/qwen15-projection/qwen-infer-ve'
    model = ROOT / 'build/models/qwen25-1.5b/qwen2.5-1.5b-instruct-q4_k_m.gguf'
    assert hashlib.sha256(executor.read_bytes()).hexdigest() == '9499eff5e001fdcf281516887444c798c5ba8934095cc42591f3c4f43877231b'
    candidate=args.candidate_dir/'qwen-infer-ve'
    candidate_digest=hashlib.sha256(candidate.read_bytes()).hexdigest()
    manifest=json.loads((args.candidate_dir/'manifest.json').read_text())
    assert candidate_digest==manifest['sha256'][str(candidate.resolve().relative_to(ROOT))]
    digest = hashlib.sha256()
    with model.open('rb') as handle:
        for block in iter(lambda: handle.read(16 * 1024**2), b''):
            digest.update(block)
    assert digest.hexdigest() == '6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e'
    env = dict(os.environ, OMP_NUM_THREADS='1', OMP_DYNAMIC='FALSE',
               VE_LD_LIBRARY_PATH='/opt/nec/ve/ncc/5.4.1/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/nlc/3.1.0/lib')
    gold = json.loads((ROOT / 'build/results/20261006T031326Z-qwen15/accurate-reset-cpu.jsonl').read_text().splitlines()[0])['tokens']
    expected = []
    cases = []
    report = {'completed': False, 'reference': 'accepted VE, four threads; sequence-only',
              'candidate_directory':str(args.candidate_dir.resolve().relative_to(ROOT)),
              'candidate_binary_sha256':candidate_digest,'candidate_threads': args.threads, 'binary_sha256': hashlib.sha256(executor.read_bytes()).hexdigest(),
              'model_sha256': digest.hexdigest(), 'runs': []}
    for run, threads in enumerate((4, args.threads)):
        command = ['ve_exec', '-N', '1', str(executor if run==0 else candidate), '--model', str(model),
                   '--threads', str(threads), '--context', '2048', '--no-mmap',
                   '--dense-cache-mib', '16384', '--force-count', '--serve']
        with (folder / f'{run}.stderr.log').open('w') as err:
            session = QwenSession(command, err, env)
            try:
                if run == 0:
                    for target in (129, 257):
                        length = target
                        for _ in range(4):
                            prompt = ' x' * length
                            row = session.request(prompt, 1, raw=True)
                            if row['prompt_tokens'] == target:
                                cases.append((prompt, 8, True, target))
                                break
                            length += target - row['prompt_tokens']
                        else:
                            raise AssertionError('cannot establish exact token boundary')
                    cases.extend([('Write a Python function that adds two integers.', 128, False, None),
                                  ('Write a Python function that adds two integers.', 32, False, None)])
                rows = []
                for i, (prompt, count, raw, target) in enumerate(cases):
                    row = session.request(prompt, count, raw=raw)
                    if target is not None:
                        assert row['prompt_tokens'] == target
                    assert row['generated_tokens'] == count
                    if target is None:
                        assert row['tokens'][:32] == gold
                    assert row['dense_cache_reserved_bytes'] == 0
                    assert row['dense_cache_retained_bytes'] <= row['dense_cache_budget_bytes']
                    if run == 0:
                        expected.append(row['tokens'])
                    else:
                        assert row['tokens'] == expected[i]
                    safe = {k: v for k, v in row.items() if k not in ('tokens', 'greedy_tokens', 'text')}
                    safe['token_sequence_sha256'] = hashlib.sha256(json.dumps(row['tokens']).encode()).hexdigest()
                    rows.append(safe)
                    print('boundary', run, i, row['prompt_tokens'], count, 'PASS', flush=True)
                report['runs'].append({'threads': threads, 'requests': rows})
                (folder / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
            finally:
                code = session.close()
                if code and sys.exc_info()[0] is None:
                    raise RuntimeError('native executor failed; inspect local stderr log')
    report['completed'] = True
    (folder / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Completed', folder.relative_to(ROOT), flush=True)


if __name__ == '__main__':
    main()
