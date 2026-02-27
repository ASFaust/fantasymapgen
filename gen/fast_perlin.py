import math
import numpy as np
from numba import njit


@njit(cache=True)
def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@njit(cache=True)
def _lerp(a, b, t):
    return a + t * (b - a)


@njit(cache=True)
def _grad3(h, x, y, z):
    h = h & 15
    u = x if h < 8 else y
    v = y if h < 4 else (x if (h == 12 or h == 14) else z)
    return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)


@njit(cache=True)
def perlin3(x, y, z, perm):
    xi = int(math.floor(x)) & 255
    yi = int(math.floor(y)) & 255
    zi = int(math.floor(z)) & 255

    xf = x - math.floor(x)
    yf = y - math.floor(y)
    zf = z - math.floor(z)

    u = _fade(xf)
    v = _fade(yf)
    w = _fade(zf)

    a  = perm[xi    ] + yi
    aa = perm[a     ] + zi
    ab = perm[a + 1 ] + zi
    b  = perm[xi + 1] + yi
    ba = perm[b     ] + zi
    bb = perm[b + 1 ] + zi

    return _lerp(
        _lerp(
            _lerp(_grad3(perm[aa    ], xf,       yf,       zf      ),
                  _grad3(perm[ba    ], xf - 1.0, yf,       zf      ), u),
            _lerp(_grad3(perm[ab    ], xf,       yf - 1.0, zf      ),
                  _grad3(perm[bb    ], xf - 1.0, yf - 1.0, zf      ), u), v),
        _lerp(
            _lerp(_grad3(perm[aa + 1], xf,       yf,       zf - 1.0),
                  _grad3(perm[ba + 1], xf - 1.0, yf,       zf - 1.0), u),
            _lerp(_grad3(perm[ab + 1], xf,       yf - 1.0, zf - 1.0),
                  _grad3(perm[bb + 1], xf - 1.0, yf - 1.0, zf - 1.0), u), v), w)


def make_permutation(seed):
    rng = np.random.default_rng(seed)
    p = np.arange(256, dtype=np.int32)
    rng.shuffle(p)
    perm = np.empty(512, dtype=np.int32)
    perm[:256] = p
    perm[256:] = p
    return perm
