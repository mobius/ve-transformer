"""Run under temperature_guard: validate shape edges and measure warm quant kernels."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
env = os.environ.copy()
env.update(VE_LD_LIBRARY_PATH='/opt/nec/ve/ncc/5.4.1/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/nlc/3.1.0/lib',
           OMP_NUM_THREADS='8', OMP_DYNAMIC='FALSE')


def run(binary, rows, width, repeats, ve=True):
    cmd = ['/opt/nec/ve/bin/ve_exec', '-N', '1'] if ve else []
    cmd += [str(ROOT/binary), str(rows), str(width), str(repeats)]
    result = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True, timeout=180)
    if result.returncode or result.stdout.count(' PASS') != 4*repeats:
        raise RuntimeError('quant probe failed: '+result.stdout+result.stderr)
    print('binary={} rows={} width={}'.format(binary, rows, width), flush=True)
    print(result.stdout, end='', flush=True)


if __name__ == '__main__':
    for width in (256, 512, 2048, 8192, 16384):
        run('build/qwen-quant-probe-ve-candidate', 17, width, 1)
    for threads in (1, 2, 4, 8):
        env['OMP_NUM_THREADS'] = str(threads)
        print('THREADS={}'.format(threads), flush=True)
        run('build/qwen-quant-probe-ve', 4096, 8192, 3)
        run('build/qwen-quant-probe-ve-candidate', 4096, 8192, 3)
