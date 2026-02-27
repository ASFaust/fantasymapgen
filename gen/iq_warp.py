import numpy as np
from numba import njit
from .fast_perlin import make_permutation, perlin3


# ==========================================================
# Core Noise
# ==========================================================

@njit(cache=True)
def fbm3(x, y, z, perm, octaves, base_freq, lacunarity, gain):
    freq = base_freq
    amp = 1.0
    total = 0.0
    for _ in range(octaves):
        total += amp * perlin3(x * freq, y * freq, z * freq, perm)
        freq *= lacunarity
        amp  *= gain
    return total


@njit(cache=True)
def iq_double_warp3d(
    x, y, z,
    perm,
    octaves,
    base_freq,
    lacunarity,
    gain,
    warp1,
    warp2,
):
    # First warp field q
    qx = fbm3(x + 1.7,  y + 9.2,  z + 3.5,  perm, octaves, base_freq, lacunarity, gain)
    qy = fbm3(x + 8.3,  y - 2.8,  z + 11.1, perm, octaves, base_freq, lacunarity, gain)
    qz = fbm3(x - 4.6,  y + 6.7,  z - 7.3,  perm, octaves, base_freq, lacunarity, gain)

    # Second warp field r (warped by q)
    rx = fbm3(x + warp1 * qx + 5.2,  y + warp1 * qy - 3.1,  z + warp1 * qz + 14.8, perm, octaves, base_freq, lacunarity, gain)
    ry = fbm3(x + warp1 * qx - 11.4, y + warp1 * qy + 7.6,  z + warp1 * qz - 1.9,  perm, octaves, base_freq, lacunarity, gain)
    rz = fbm3(x + warp1 * qx + 2.9,  y + warp1 * qy - 8.5,  z + warp1 * qz + 6.1,  perm, octaves, base_freq, lacunarity, gain)

    # Final sample (warped by r)
    return fbm3(
        x + warp2 * rx,
        y + warp2 * ry,
        z + warp2 * rz,
        perm, octaves, base_freq, lacunarity, gain,
    )


# ==========================================================
# Utilities
# ==========================================================

def normalize(img):
    m = img.min()
    M = img.max()
    if M - m < 1e-8:
        return np.zeros_like(img)
    return (img - m) / (M - m)


def sphere_coords(size):
    u = np.linspace(0, 1, size, endpoint=False)
    v = np.linspace(0, 1, size)

    lon = 2 * np.pi * u
    lat = np.pi * (v - 0.5)

    lon, lat = np.meshgrid(lon, lat)

    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    return (
        x.astype(np.float32),
        y.astype(np.float32),
        z.astype(np.float32),
        lat.astype(np.float32),
        lon.astype(np.float32),
    )


def weighted_quantile(values, weights, q):
    sorter = np.argsort(values)
    values  = values[sorter]
    weights = weights[sorter]

    cumulative = np.cumsum(weights)
    total = cumulative[-1]

    idx = np.searchsorted(cumulative, q * total)
    idx = min(idx, len(values) - 1)

    return values[idx]


def enforce_land_percentage_spherical(img, lat, land_percentage, sea_level):
    weights = np.cos(lat)

    flat_vals    = img.ravel()
    flat_weights = weights.ravel()

    t = weighted_quantile(flat_vals, flat_weights, 1.0 - land_percentage)

    eps = 1e-8
    t = np.clip(t, eps, 1.0 - eps)

    out = np.empty_like(img)

    mask_low  = img <= t
    out[mask_low] = sea_level * (img[mask_low] / t)

    mask_high = ~mask_low
    out[mask_high] = sea_level + (1.0 - sea_level) * (
        (img[mask_high] - t) / (1.0 - t)
    )

    return out


def apply_polar_exclusion(img, z, strength, sharpness):
    if strength <= 0.0:
        return img
    m = np.abs(z) ** sharpness
    suppression = 1.0 - strength * m
    return img * suppression


def apply_backside_exclusion(img, lon, seam_lon, strength, sharpness):
    if strength <= 0.0:
        return img
    m = 0.5 * (1.0 + np.cos(lon - seam_lon))
    m = m ** sharpness
    suppression = 1.0 - strength * m
    return img * suppression


# ==========================================================
# Public API
# ==========================================================

def create_planet_heightmap(
    size=512,
    seed=42,

    # FBM
    octaves=6,
    base_freq=1.5,
    lacunarity=2.0,
    gain=0.5,

    # IQ double warp strengths
    warp1=0.8,
    warp2=0.8,

    # Ridge
    ridge=False,

    # Land shaping
    land_percentage=0.5,
    sea_level=0.2,

    # Polar exclusion
    polar_strength=0.0,
    polar_sharpness=1.0,

    # Backside seam suppression
    backside_strength=0.0,
    backside_sharpness=1.0,
    seam_lon=0.0,
):
    """
    Generate a spherical planet heightmap using IQ-style double
    domain-warped 3D FBM.
    """

    perm = make_permutation(seed)
    x, y, z, lat, lon = sphere_coords(size)

    h, w = x.shape
    out = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            val = iq_double_warp3d(
                x[i, j], y[i, j], z[i, j],
                perm,
                octaves,
                base_freq,
                lacunarity,
                gain,
                warp1,
                warp2,
            )

            if ridge:
                val = 1.0 - abs(val)
                val = val * val

            out[i, j] = val

    out = normalize(out)

    # --- Spherical modifiers ---
    out = apply_polar_exclusion(out, z, polar_strength, polar_sharpness)
    out = apply_backside_exclusion(out, lon, seam_lon,
                                   backside_strength, backside_sharpness)

    out = normalize(out)

    # --- Land percentage enforcement ---
    out = enforce_land_percentage_spherical(
        out, lat, land_percentage, sea_level
    )

    return out, lat, lon


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    heightmap, lat, lon = create_planet_heightmap(
        size=512,
        seed=12345,
        warp1=0.8,
        warp2=0.8,
        ridge=False,
        land_percentage=0.45,
        polar_strength=0.6,
    )

    plt.imshow(heightmap, cmap="terrain")
    plt.axis("off")
    plt.show()
