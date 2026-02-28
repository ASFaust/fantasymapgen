import numpy as np
from scipy.ndimage import gaussian_filter

from gen.fast_perlin import make_permutation, perlin3_grid

def compute_base_wind(
    sim_lat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the base (terrain-independent) wind field via cubic spline
    interpolation through known circulation control points.

    Control points encode the six main latitude belts:
      polar easterlies → westerlies → horse latitudes →
      trade winds → doldrums (ITCZ)
    … mirrored across the equator.

    The spline is evaluated once per row (latitude) and broadcast over
    all columns, because the base circulation is purely zonal.

    Returns
    -------
    wind_u : (H, W)  eastward component in m/s   (positive = blowing east)
    wind_v : (H, W)  northward component in m/s  (positive = blowing north)
    wind_speed : (H, W)  magnitude in m/s
    """
    from scipy.interpolate import PchipInterpolator
    from scipy.ndimage import gaussian_filter1d

    # Control points  (latitude in degrees, wind in m/s)
    #
    # Zero crossings in u are placed explicitly at the three belt
    # boundaries: polar front (±65°), horse latitudes (±30°), ITCZ (0°).
    # Peak u values sit at the midpoints of each belt.
    #
    # lat°  |  u (east+)  |  v (north+)  | regime
    # ------+-------------+--------------+----------------------------
    #  -90  |    -8       |     0        | S polar easterlies (peak)
    #  -65  |     0       |     1        | S polar front (u zero crossing)
    #  -50  |    10       |     0        | S westerlies (peak)
    #  -30  |     0       |     0        | S horse latitudes (calm)
    #  -18  |    -8       |     4        | SE trades (peak, toward equator)
    #    0  |     0       |     0        | ITCZ / doldrums
    #   18  |    -8       |    -4        | NE trades (peak, toward equator)
    #   30  |     0       |     0        | N horse latitudes (calm)
    #   50  |    10       |     0        | N westerlies (peak)
    #   65  |     0       |    -1        | N polar front (u zero crossing)
    #   90  |    -8       |     0        | N polar easterlies (peak)
    ctrl_lat_deg = np.array([-90, -65, -50, -30, -18,   0,  18,  30,  50,  65,  90], dtype=float)
    ctrl_u       = np.array([ -8,   0,  10,   0,  -8,   0,  -8,   0,  10,   0,  -8], dtype=float)
    ctrl_v       = np.array([  0,   1,   0,   0,   4,   0,  -4,   0,   0,  -1,   0], dtype=float)

    ctrl_lat_rad = np.deg2rad(ctrl_lat_deg)

    # PchipInterpolator (monotone piecewise cubic Hermite): never overshoots
    # between tightly-packed turning points, giving physically plausible profiles.
    cs_u = PchipInterpolator(ctrl_lat_rad, ctrl_u)
    cs_v = PchipInterpolator(ctrl_lat_rad, ctrl_v)

    H, W = sim_lat.shape
    lat_col = sim_lat[:, W // 2]
    lat_col_clamped = np.clip(lat_col, ctrl_lat_rad[0], ctrl_lat_rad[-1])

    wind_u_row = cs_u(lat_col_clamped)
    wind_v_row = cs_v(lat_col_clamped)

    # Gaussian blur along the latitude profile to smooth Pchip C1 kinks
    dlat_deg = np.degrees(float(np.abs(np.median(np.diff(lat_col_clamped)))))
    sigma_rows = max(1.0, 5.0 / dlat_deg)
    wind_u_row = gaussian_filter1d(wind_u_row, sigma=sigma_rows)
    wind_v_row = gaussian_filter1d(wind_v_row, sigma=sigma_rows)

    wind_u = np.broadcast_to(wind_u_row[:, np.newaxis], (H, W)).copy()
    wind_v = np.broadcast_to(wind_v_row[:, np.newaxis], (H, W)).copy()
    wind_speed = np.sqrt(wind_u**2 + wind_v**2)

    return wind_u, wind_v, wind_speed


def compute_base_temperature(
    sim_lat: np.ndarray,
    sim_heightmap: np.ndarray,
    sea_level: float,
    mountain_height_km: float,
    polar_temp: float,
    equatorial_temp: float,
    lapse_rate: float,
) -> np.ndarray:
    """
    Compute the base surface temperature field (°C).

    Two contributions:
      1. Latitude-driven insolation: T(φ) = polar + (equatorial − polar) · cos(φ)
      2. Lapse rate: −lapse_rate °C / km above sea level (land only).

    Returns
    -------
    temperature : (H, W)  surface temperature in °C
    """
    temp = polar_temp + (equatorial_temp - polar_temp) * np.cos(sim_lat)

    # Lapse rate correction — land cells only, clamped to ≥ 0 km ASL
    elevation_asl_km = np.maximum(0.0, sim_heightmap - sea_level) / (1.0 - sea_level) * mountain_height_km
    temp -= lapse_rate * elevation_asl_km

    return temp


def compute_ocean_temp_noise(
    sim_lat: np.ndarray,
    sim_lon: np.ndarray,
    sim_ocean: np.ndarray,
    seed: int = 42,
    amplitude: float = 4.0,
    frequency: float = 2.0,
) -> np.ndarray:
    """Low-frequency spherical Perlin noise applied to ocean cells only.

    Two octaves of 3D Perlin noise sampled on the unit sphere, giving
    smooth ±amplitude °C perturbations that break the zonal symmetry of
    ocean surface temperature (and therefore ice cap shape).

    Returns (H, W) float array — zero on land, noise on ocean.
    """
    # Map lat/lon to unit-sphere 3D coords
    x = np.cos(sim_lat) * np.cos(sim_lon)
    y = np.cos(sim_lat) * np.sin(sim_lon)
    z = np.sin(sim_lat)

    perm = make_permutation(seed + 7777)

    # Two octaves: base freq + 2× freq at half amplitude
    noise = perlin3_grid(x * frequency, y * frequency, z * frequency, perm)
    noise += 0.5 * perlin3_grid(x * frequency * 2, y * frequency * 2, z * frequency * 2, perm)
    noise *= (2.0 / 3.0)  # normalize back to roughly [-1, 1]

    result = (noise * amplitude).astype(np.float32)
    result[~sim_ocean] = 0.0
    return result


import numpy as np
import numba


@numba.njit
def _moisture_loop(
    humidity,             # (H, W) float32, in-place
    precip,               # (H, W) float32, in-place
    sim_ocean,            # (H, W) bool
    sim_elevation_km,     # (H, W) float32
    max_saturation,       # (H, W) float32
    wind_u,               # (H, W) float32, m/s
    wind_v,               # (H, W) float32, m/s
    sim_lat,              # (H, W) float32, radians
    sim_lon,              # (H, W) float32, radians
    lat_min,              # scalar float — sim_lat[0, 0]
    lat_range,            # scalar float — sim_lat[-1, 0] - sim_lat[0, 0]
    lon_min,              # scalar float — sim_lon[0, 0]
    lon_range,            # scalar float — sim_lon[0, -1] - sim_lon[0, 0]
    radius_m,             # scalar float — radius_km * 1000
    dt,                   # scalar float
    sigma_diffusion,      # scalar float, m/s
    n_steps,              # int
    max_transport_km,     # scalar float, e.g. 2000.0 km — controls how much moisture is lost to precipitation during transport 
    total_orog_loss_km,   # scalar float, e.g. 8.0 km — controls how much moisture is lost to orographic precip during transport over mountains
    precip_gamma,         
    hum_gamma,
    seed,                 # int
):
    H = humidity.shape[0]
    W = humidity.shape[1]
    np.random.seed(seed)
    #precip is 0 at the beginning
    
    # Phase 1: seed ocean cells to saturation
    for i in range(H):
        for j in range(W):
            if sim_ocean[i, j]:
                humidity[i, j] = max_saturation[i, j]

    new_humidity = np.zeros_like(humidity)
    for step in range(n_steps):
        for i in range(H):
            for j in range(W):
                h = humidity[i, j]
                if h < 1e-12:
                    continue

                # Noisy wind
                noise_u = np.random.randn() * sigma_diffusion
                noise_v = np.random.randn() * sigma_diffusion
                total_u = wind_u[i, j] + noise_u
                total_v = wind_v[i, j] + noise_v

                # Target lat/lon
                cos_lat = np.cos(sim_lat[i, j])
                if cos_lat < 1e-6:
                    cos_lat = 1e-6

                t_lat = sim_lat[i, j] + (total_v * dt) / radius_m
                t_lon = sim_lon[i, j] + (total_u * dt) / (radius_m * cos_lat)

                # Convert to pixel coordinates
                tr = int((t_lat - lat_min) / lat_range * (H - 1) + 0.5)
                tc = int((t_lon - lon_min) / lon_range * (W - 1) + 0.5)

                # Sphere-aware wrapping
                if tr < 0:
                    tr = -tr
                    tc = tc + W // 2
                elif tr >= H:
                    tr = 2 * H - 1 - tr
                    tc = tc + W // 2
                tc = tc % W

                # Clamp in case of extreme overshoot past both poles
                if tr < 0:
                    tr = 0
                elif tr >= H:
                    tr = H - 1
                
                dy_m = (tr - i) * (lat_range / (H - 1)) * radius_m
                # dx in meters: account for longitude spacing and cos(lat)
                # handle wraparound for dx
                dc = tc - j
                if dc > W // 2:
                    dc -= W
                elif dc < -W // 2:
                    dc += W
                dx_m = dc * (lon_range / (W - 1)) * radius_m * cos_lat
                dist_km = np.sqrt(dy_m * dy_m + dx_m * dx_m) / 1000.0

                elevation_gain_km = max(0.0, sim_elevation_km[tr, tc] - sim_elevation_km[i, j])

                transport_eff = np.exp(-dist_km / max_transport_km) #how much is lost to precip during transport
                #max_transport_km is around ~2000 km, meaning after 2000 km, only about 37% of the original moisture remains, and after 4000 km, only about 14% remains.
                transport_precip = h * (1.0 - transport_eff)

                orog_eff = np.exp(-elevation_gain_km / total_orog_loss_km) #total orog loss is a float around 8km, 
                orog_precip = h * (1.0 - orog_eff)

                arrival_hum = h * transport_eff * orog_eff #how much humidity arrives at the target cell
                supersat_precip = max(0.0, humidity[tr, tc] + arrival_hum - max_saturation[tr, tc]) 
                new_humidity[tr, tc] += arrival_hum - supersat_precip
                precip[i, j] = precip_gamma * precip[i, j] + (1 - precip_gamma) * (orog_precip + transport_precip * 0.5)
                precip[tr, tc] = precip_gamma * precip[tr, tc] + (1 - precip_gamma) * (supersat_precip + transport_precip * 0.5)

        for i in range(H):
            for j in range(W):
                if sim_ocean[i, j]:
                    humidity[i, j] = max_saturation[i, j]                    
                else:
                    humidity[i, j] = hum_gamma * humidity[i, j] + new_humidity[i, j] * (1 - hum_gamma)
                new_humidity[i, j] = 0.0


def compute_moisture_grid(
    sim_ocean: np.ndarray,
    sim_lat: np.ndarray,
    sim_lon: np.ndarray,
    sim_elevation_km: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    radius_km: float,
    temperature: np.ndarray,
    dt: float,
    sigma_diffusion: float = 2.0,
    n_steps: int = 100,
    max_transport_km: float = 2000.0,
    total_orog_loss_km: float = 8.0,
    precip_gamma: float = 0.5,
    hum_gamma: float = 0.5,
    precip_hum_ratio: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    H, W = sim_lat.shape
    humidity = np.zeros((H, W), dtype=np.float32)
    precip = np.zeros((H, W), dtype=np.float32)

    # Clausius-Clapeyron saturation
    T0 = 273.15
    L = 2.5e6
    R = 461.5
    max_saturation = (
        0.0038 * np.exp((L / R) * (1.0 / T0 - 1.0 / (temperature + 273.15)))
    ).astype(np.float32)
    humidity[sim_ocean] = max_saturation[sim_ocean]

    # Grid geometry
    lat_min = np.float64(sim_lat[0, 0])
    lat_range = np.float64(sim_lat[-1, 0] - sim_lat[0, 0])
    lon_min = np.float64(sim_lon[0, 0])
    lon_range = np.float64(sim_lon[0, -1] - sim_lon[0, 0])
    radius_m = np.float64(radius_km * 1000.0)

    print(f"Grid: {H}x{W}, lat [{np.degrees(sim_lat[0,0]):.1f}°, {np.degrees(sim_lat[-1,0]):.1f}°], "
          f"lon [{np.degrees(sim_lon[0,0]):.1f}°, {np.degrees(sim_lon[0,-1]):.1f}°]")
    print(f"max/min wind_u: {wind_u.max():.2f} / {wind_u.min():.2f} m/s")
    print(f"max/min wind_v: {wind_v.max():.2f} / {wind_v.min():.2f} m/s")
    print(f"sigma_diffusion: {sigma_diffusion:.2f} m/s")
    print(f"dt: {dt:.1f} s, n_steps: {n_steps}")

    max_wind = np.sqrt(wind_u**2 + wind_v**2).max()
    max_travel_km = (max_wind + 3 * sigma_diffusion) * dt / 1000.0
    print(f"Max possible travel per step (wind + 3σ): {max_travel_km:.1f} km")

    print("Running moisture simulation...")
    _moisture_loop(
        humidity, precip,
        sim_ocean,
        sim_elevation_km.astype(np.float32),
        max_saturation,
        wind_u.astype(np.float32),
        wind_v.astype(np.float32),
        sim_lat.astype(np.float32),
        sim_lon.astype(np.float32),
        lat_min, lat_range, lon_min, lon_range,
        radius_m, np.float64(dt),
        np.float64(sigma_diffusion),
        n_steps, 
        max_transport_km,
        total_orog_loss_km,
        precip_gamma,
        hum_gamma,
        seed
    )
    

    land_precip = precip[~sim_ocean]
    land_hum = humidity[~sim_ocean]
    print(f"Land precip: mean {land_precip.mean():.2f}, max {land_precip.max():.2f}")
    print(f"Land humidity: mean {land_hum.mean():.2f}, max {land_hum.max():.2f}")
    moisture = np.zeros_like(precip) #(H, W) float32
    moisture[~sim_ocean] = precip[~sim_ocean] * precip_hum_ratio + humidity[~sim_ocean] * (1 - precip_hum_ratio)
    moisture[sim_ocean] = moisture[~sim_ocean].mean()  # assign ocean cells the mean land moisture
    #normalize to [0, 1]
    moisture /= moisture.max()
    moisture -= moisture.min()
    return moisture