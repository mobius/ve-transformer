"""Aggregate diagnostic-only worker-time snapshots; never publish generated text."""
import argparse
import json
from pathlib import Path
import statistics

FIELDS = ('lock_wait', 'lookup', 'pack', 'blas', 'cast', 'scatter', 'weight', 'lookups', 'multiplies', 'sampled_lookups', 'sampled_multiplies')
TIMES = FIELDS[:7]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--allow-partial', action='store_true', help='local progress only; mark report incomplete')
    args = parser.parse_args()
    summary = json.loads((args.folder / 'summary.json').read_text())
    assert summary['completed'] or args.allow_partial, 'incomplete benchmark'
    report = {'completed':summary['completed'],'scope': 'sampling estimates corrected by quiescent median clock-pair cost; not wall time or all inference operators; correction under contention remains approximate',
              'single_column_scope': 'mainly generation; also includes single-token prefill tails',
              'runs': [], 'warm_medians': {}}
    for run_index, run in enumerate(summary['benchmarks']):
        if run['backend'] != 'attention':
            continue
        snapshots = [json.loads(line.removeprefix('QWEN_PROFILE '))
                     for line in (args.folder / f'bench-{run_index}.stderr.log').read_text().splitlines()
                     if line.startswith('QWEN_PROFILE ')]
        clocks=[json.loads(line.removeprefix('QWEN_CLOCK '))
                for line in (args.folder / f'bench-{run_index}.stderr.log').read_text().splitlines()
                if line.startswith('QWEN_CLOCK ')]
        requests = run['requests'] + [run['long_request'], run['after_long']]
        assert len(snapshots) == 4 * len(requests), 'snapshot/request mismatch'
        assert len(clocks)==2*len(requests)
        rows = []
        for i, request in enumerate(requests):
            bins = []
            for bin_index, columns in enumerate((1, 2)):
                before = snapshots[i * 4 + bin_index]
                after = snapshots[i * 4 + 2 + bin_index]
                assert before['columns'] == after['columns'] == columns
                delta = {key: after[key] - before[key] for key in FIELDS}
                assert all(value >= -1e-8 for value in delta.values()), 'non-monotonic counter'
                bits=after['sample_bits']
                assert before['sample_bits']==bits
                delta['sample_bits']=bits
                delta['raw_sampled_thread_seconds']={key:delta[key] for key in TIMES}
                counts={key:after['timer_counts'][key]-before['timer_counts'][key] for key in TIMES}
                assert all(value>=0 for value in counts.values())
                delta['sampled_timer_counts']=counts
                floor=(clocks[2*i]['median_seconds']+clocks[2*i+1]['median_seconds'])/2
                delta['clock_floor_seconds']=floor
                delta['estimated_uncorrected_thread_seconds']={key:delta[key]*2**bits for key in TIMES}
                for key in TIMES:
                    delta[key]=max(0,delta[key]-counts[key]*floor)*2**bits
                total = sum(delta[key] for key in TIMES)
                delta.update(columns=columns, measured_worker_seconds=total,
                             percent_of_measured={key: 100 * delta[key] / total if total else 0 for key in TIMES})
                bins.append(delta)
            rows.append({'request_index': i, 'prompt_tokens': request['prompt_tokens'],
                         'generated_tokens': request['generated_tokens'],
                         'host_roundtrip_seconds': request['host_roundtrip_seconds'], 'bins': bins})
        report['runs'].append({'benchmark_index': run_index, 'requests': rows})
    assert len(report['runs']) > 0
    for bin_index, columns in enumerate((1, 2)):
        bins = [row['bins'][bin_index] for run in report['runs'] for row in run['requests'][1:4]]
        report['warm_medians'][str(columns)] = {
            'thread_seconds': {key: statistics.median(row[key] for row in bins) for key in TIMES},
            'percent_of_measured': {key: statistics.median(row['percent_of_measured'][key] for row in bins) for key in TIMES}}
    (args.folder / 'profile-analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['warm_medians'], indent=2))


if __name__ == '__main__':
    main()
