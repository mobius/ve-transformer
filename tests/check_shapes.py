"""Compare VE elements to an independent NumPy float64 oracle."""
import argparse
import math
import struct
import subprocess
from numpy_reference import reference


def output(command, shape, causal, workspace=False):
    args = command + [str(n) for n in shape] + [str(causal)]
    if workspace:
        args.append('--workspace')
    data = subprocess.check_output(args, timeout=30)
    count = shape[0]*shape[1]
    if len(data) != count*4:
        raise AssertionError('unexpected output byte count')
    return struct.unpack('<{}f'.format(count), data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--node', type=int, required=True)
    parser.add_argument('--nlc', action='store_true')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--workspace', action='store_true')
    parser.add_argument('--cpu-diagnostic', action='store_true',
                        help='also run the slow CPU fixture for drift diagnosis')
    args = parser.parse_args()
    if args.node < 0:
        parser.error('node must be nonnegative')
    if args.fast and args.nlc:
        parser.error('choose one backend')
    shapes = [(1, 1, 1, 1), (7, 15, 3, 31), (32, 64, 4, 128),
              (128, 256, 8, 1024), (256, 512, 8, 2048)]
    binary = './build/shape-ve-nlc' if args.nlc else './build/shape-ve'
    if args.fast:
        binary = './build/shape-ve-fast'
    runner = ['/opt/nec/ve/bin/ve_exec', '-N', str(args.node), binary]
    for shape in shapes:
        for causal in (0, 1):
            ref = reference(shape, causal).ravel()
            cpu = output(['./build/shape-cpu'], shape, causal) if args.cpu_diagnostic else None
            actual = output(runner, shape, causal, args.workspace)
            error = max(abs(a-b) for a, b in zip(actual, ref))
            for a, b in zip(actual, ref):
                assert math.isfinite(a) and math.isfinite(b)
                assert abs(a-b) <= 5e-5 + 1e-4*abs(b), (shape, causal, a, b)
            cpu_error = max(abs(a-b) for a, b in zip(cpu, ref)) if cpu is not None else None
            print('node={} backend={} workspace={} shape={} causal={} max_abs_error={:.3g} '
                  'cpu_max_abs_error={} PASS'
                  .format(args.node, 'fast' if args.fast else ('nlc' if args.nlc else 'base'), args.workspace,
                          'x'.join(str(n) for n in shape), causal, error,
                          '{:.3g}'.format(cpu_error) if cpu_error is not None else 'not_requested'),
                  flush=True)
