"""Independent float64 reference using only Python's standard library."""
import math
import subprocess
import sys


def matmul(a, b):
    return [[sum(x*y for x, y in zip(row, col)) for col in zip(*b)]
            for row in a]


def matrix(m, n, modulus, offset, scale):
    return [[((i*n+j) % modulus-offset)/scale for j in range(n)]
            for i in range(m)]


def norm(a):
    result = []
    for row in a:
        mean = sum(row)/len(row)
        inv = 1/math.sqrt(sum((x-mean)**2 for x in row)/len(row)+1e-5)
        result.append([(x-mean)*inv for x in row])
    return result


def reference(causal):
    x = matrix(4, 8, 17, 8, 8)
    n = norm(x)
    q = matmul(n, matrix(8, 8, 11, 5, 32))
    k = matmul(n, matrix(8, 8, 7, 3, 24))
    v = matmul(n, matrix(8, 8, 13, 6, 40))
    attn = [[0.0]*8 for _ in range(4)]
    for head in range(2):
        lo, hi = head*4, head*4+4
        for i in range(4):
            limit = i+1 if causal else 4
            scores = [sum(a*b for a, b in zip(q[i][lo:hi], k[j][lo:hi]))/2
                      for j in range(limit)]
            probs = [math.exp(s-max(scores)) for s in scores]
            total = sum(probs)
            for p in range(lo, hi):
                attn[i][p] = sum(probs[j]/total*v[j][p] for j in range(limit))
    projected = matmul(attn, matrix(8, 8, 5, 2, 16))
    residual = [[a+b for a, b in zip(row, orig)]
                for row, orig in zip(projected, x)]
    ff = matmul(norm(residual), matrix(8, 16, 9, 4, 32))
    ff = [[max(0, value) for value in row] for row in ff]
    out = matmul(ff, matrix(16, 8, 7, 3, 32))
    return [a+b for row, res in zip(out, residual) for a, b in zip(row, res)]


if __name__ == '__main__':
    for causal in (False, True):
        args = sys.argv[1:] + (['--causal'] if causal else [])
        actual = [float(x) for x in subprocess.check_output(args).split()]
        expected = reference(causal)
        assert len(actual) == len(expected)
        error = max(abs(a-b) for a, b in zip(actual, expected))
        assert all(math.isfinite(a) for a in actual)
        assert error < 2e-6, error
        print('causal={} max_abs_error={:.3g} PASS'.format(causal, error))
