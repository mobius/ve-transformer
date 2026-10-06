"""Float64 oracle starting from exactly the fixture's float32 inputs."""
import numpy as np


def fixture(shape):
    t, d, h, f = shape
    count = 4*d*d+2*d*f
    packed = (((np.arange(count, dtype=np.int64)*7+3) % 31-15)/256
              ).astype(np.float32).astype(np.float64)
    sizes = [d*d]*4+[d*f]*2
    split = np.split(packed, np.cumsum(sizes)[:-1])
    matrices = [a.reshape(d, d) for a in split[:4]]
    matrices += [split[4].reshape(d, f), split[5].reshape(f, d)]
    i = np.arange(d)
    vectors = [1+(i % 3)/16, (i % 7-3)/32,
               1-(i % 5)/32, (i % 9-4)/32]
    x = (((np.arange(t*d)*11+5) % 37-18)/16).reshape(t, d)
    return x, matrices, vectors


def reference(shape, causal):
    t, d, h, f = shape
    x, (wq, wk, wv, wo, up, down), (g1, b1, g2, b2) = fixture(shape)

    def norm(a, g, b):
        centered = a-a.mean(axis=-1, keepdims=True)
        return centered/np.sqrt((centered**2).mean(axis=-1, keepdims=True)+1e-5)*g+b

    n = norm(x, g1, b1)
    q, k, v = [(n @ w).reshape(t, h, d//h).transpose(1, 0, 2)
               for w in (wq, wk, wv)]
    scores = q @ k.transpose(0, 2, 1)/np.sqrt(d//h)
    if causal:
        scores[:, np.triu_indices(t, 1)[0], np.triu_indices(t, 1)[1]] = -np.inf
    probs = np.exp(scores-scores.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    a = (probs @ v).transpose(1, 0, 2).reshape(t, d)
    r = x+a @ wo
    return r+np.maximum(norm(r, g2, b2) @ up, 0) @ down
