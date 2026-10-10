"""Run a guarded command while recording discrete whole-node VE memory samples."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from ve_memory_info import VeMemoryInfo

ROOT = Path(__file__).resolve().parents[1]


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt()

    # The sampled command owns a separate group so counter failures can stop
    # all descendants. Forward guard termination through finally before the
    # outer temperature guard's two-second SIGKILL deadline.
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    parser = argparse.ArgumentParser()
    parser.add_argument('--slot', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--interval', type=float, default=.2)
    parser.add_argument('--timeout', type=float, default=3600)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or not .05 <= args.interval <= 5 or not 0 < args.timeout <= 7200:
        parser.error('command, interval in [0.05,5], timeout in (0,7200] required')
    reader = VeMemoryInfo()  # refuses execution without temperature supervision
    folder = ROOT/'build/results'/time.strftime('%Y%m%dT%H%M%SZ-ve-memory-sampled', time.gmtime())
    folder.mkdir()
    report = {'completed': False, 'runtime_slot': args.slot, 'library_version': reader.version,
              'requested_interval_seconds': args.interval,
              'scope': 'discrete whole-node VEOS usage including runtime and other processes; not instantaneous peak or model-exclusive usage',
              'sampler_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'adapter_sha256': hashlib.sha256((ROOT/'scripts/ve_memory_info.py').read_bytes()).hexdigest()}

    def save():
        (folder/'summary.json').write_text(json.dumps(report, indent=2)+'\n')

    save()
    child = None
    samples = []
    start = time.monotonic()
    try:
        with (folder/'memory.csv').open('w') as trace:
            writer = csv.DictWriter(trace, fieldnames=['elapsed_seconds', 'total_kib', 'used_kib', 'free_kib'])
            writer.writeheader()

            def sample():
                value = dict(elapsed_seconds=time.monotonic()-start, **reader.sample(args.slot))
                samples.append(value)
                writer.writerow(value)
                trace.flush()

            sample()
            child = subprocess.Popen(command, start_new_session=True)
            while child.poll() is None:
                if time.monotonic()-start > args.timeout:
                    raise TimeoutError('memory-sampled command timeout')
                sample()
                time.sleep(args.interval)
            sample()
            report['returncode'] = child.returncode
            # Observe delayed cleanup without assuming immediate reclamation.
            for _ in range(10):
                time.sleep(args.interval)
                sample()
            if child.returncode:
                raise RuntimeError('memory-sampled command failed')
            report['completed'] = True
    except BaseException as error:
        report['error_type'] = type(error).__name__
        raise
    finally:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        if samples:
            gaps = [b['elapsed_seconds']-a['elapsed_seconds'] for a, b in zip(samples, samples[1:])]
            highest = max(samples, key=lambda row: row['used_kib'])
            report.update(sample_count=len(samples), baseline_used_kib=samples[0]['used_kib'],
                          final_used_kib=samples[-1]['used_kib'],
                          sampled_highest_used_kib=highest['used_kib'],
                          sampled_highest_elapsed_seconds=highest['elapsed_seconds'],
                          max_sample_gap_seconds=max(gaps) if gaps else None)
        save()
        print('VE memory samples:', folder.relative_to(ROOT), flush=True)


if __name__ == '__main__':
    main()
