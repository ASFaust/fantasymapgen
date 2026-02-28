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
    reevaporation_factor: float = 0.5,
    precipitation_factor: float = 0.01,
    orographic_scale_km: float = 1.0,
    n_steps: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    H, W = sim_lat.shape
    humidity = np.zeros((H, W), dtype=np.float32)
    precip = np.zeros((H, W), dtype=np.float32)

    # Correct Clausius-Clapeyron: reference at T0=0°C, q_sat0 ~ 0.0038 kg/kg
    T0 = 273.15  # reference temp in K
    L = 2.5e6
    R = 461.5
    q_sat = 0.0038 * np.exp((L / R) * (1.0 / T0 - 1.0 / (temperature + 273.15)))
    humidity[sim_ocean] = q_sat[sim_ocean]
    humidity[~sim_ocean] = 0.0

    max_saturation = np.zeros_like(humidity)
    max_saturation[sim_ocean] = q_sat[sim_ocean]
    max_saturation[~sim_ocean] = q_sat[~sim_ocean] * np.exp(-sim_elevation_km[~sim_ocean] / 8.0)

    dlat = (wind_v * dt) / (radius_km * 1000)
    dlon = (wind_u * dt) / (radius_km * 1000 * np.cos(sim_lat))
    target_lat = sim_lat + dlat
    target_lon = sim_lon + dlon
    target_lon = (target_lon + np.pi) % (2 * np.pi) - np.pi
    target_x = ((target_lon + np.pi) / (2 * np.pi) * W).astype(int)
    target_y = ((target_lat + np.pi / 2) / np.pi * H).astype(int)
    target_x = np.clip(target_x, 0, W - 1)
    target_y = np.clip(target_y, 0, H - 1)
    distance = np.sqrt((dlat * radius_km * 1000)**2 + (dlon * radius_km * 1000 * np.cos(sim_lat))**2)

    print(f"Min distance advected per step: {distance.min():.2f} m")
    print(f"Max distance advected per step: {distance.max():.2f} m")
    print(f"Mean distance advected per step: {distance.mean():.2f} m")
    # compute also in pixel space: how many pixels does the average air parcel move per step?
    pixel_distance = np.sqrt((target_y - np.arange(H)[:, None])**2 + (target_x - np.arange(W)[None, :])**2)
    print(f"Min pixel distance advected per step: {pixel_distance.min():.2f} pixels")
    print(f"Max pixel distance advected per step: {pixel_distance.max():.2f} pixels")
    print(f"Mean pixel distance advected per step: {pixel_distance.mean():.2f} pixels")

    elev_source = sim_elevation_km
    elev_target = sim_elevation_km[target_y, target_x]
    elev_gain = np.maximum(0.0, elev_target - elev_source)
    orographic_loss = 1.0 - np.exp(-elev_gain / orographic_scale_km)
    distance_km = distance / 1000.0
    static_loss = precipitation_factor * distance_km
    max_sat_target = max_saturation[target_y, target_x]

    # --- Accumulators for stats ---
    total_orog_precip   = 0.0  # precipitation due to orographic lifting
    total_super_precip  = 0.0  # precipitation due to supersaturation
    total_static_precip = 0.0  # precipitation due to static/advective rainout

    total_orog_hum_loss   = 0.0  # net humidity lost to orographic (after reevap)
    total_super_hum_loss  = 0.0  # net humidity lost to supersaturation (after reevap)
    total_static_hum_loss = 0.0  # net humidity lost to static rainout (after reevap)

    for _ in range(n_steps):
        # Orographic loss happens at the source, during ascent
        orog_precip = orographic_loss * humidity
        humidity -= orog_precip * (1.0 - reevaporation_factor)
        precip += orog_precip  # precipitation attributed to source cell (windward slope)
        total_orog_precip   += float(orog_precip.sum())
        total_orog_hum_loss += float(net_orog_hum_loss.sum())

        # 1. Advect
        new_humidity = np.zeros((H, W), dtype=np.float32)
        np.add.at(new_humidity, (target_y, target_x), humidity)
        
        # 3. Supersaturation check - no reevaporation
        excess = np.maximum(0.0, new_humidity - max_sat_target)
        new_humidity -= excess
        precip += excess
        total_super_precip   += float(excess.sum())
        total_super_hum_loss += float(excess.sum())  # reevaporated humidity is not lost

        # 4. Static / advective rainout
        static_precip = static_loss * new_humidity
        net_static_hum_loss = static_precip * (1 - reevaporation_factor)
        new_humidity -= net_static_hum_loss
        precip += static_precip
        total_static_precip   += float(static_precip.sum())
        total_static_hum_loss += float(net_static_hum_loss.sum())

        humidity = new_humidity

    # --- Normalize outputs ---
    q_ref = 0.622 * np.exp(L * 30.0 / (R * (30.0 + 273.15)))
    moisture = np.clip(humidity / q_ref, 0.0, 1.0).astype(np.float32)

    land_precip = precip[~sim_ocean]
    if land_precip.max() > 0:
        precip = precip / land_precip.max()
    precip = np.clip(precip, 0.0, 1.0).astype(np.float32)

    # --- Print stats ---
    total_precip = total_orog_precip + total_super_precip + total_static_precip
    total_hum_loss = total_orog_hum_loss + total_super_hum_loss + total_static_hum_loss

    def pct(part, whole):
        return 100.0 * part / whole if whole > 0 else 0.0

    print("=== Moisture Grid Stats ===")
    print(f"  Precipitation breakdown (sum over all cells & steps):")
    print(f"    Orographic:      {total_orog_precip:12.2f}  ({pct(total_orog_precip,   total_precip):.1f}%)")
    print(f"    Supersaturation: {total_super_precip:12.2f}  ({pct(total_super_precip,  total_precip):.1f}%)")
    print(f"    Static/advective:{total_static_precip:12.2f}  ({pct(total_static_precip, total_precip):.1f}%)")
    print(f"    Total:           {total_precip:12.2f}")
    print(f"  Net humidity loss breakdown (after reevaporation, sum over all cells & steps):")
    print(f"    Orographic:      {total_orog_hum_loss:12.2f}  ({pct(total_orog_hum_loss,   total_hum_loss):.1f}%)")
    print(f"    Supersaturation: {total_super_hum_loss:12.2f}  ({pct(total_super_hum_loss,  total_hum_loss):.1f}%)")
    print(f"    Static/advective:{total_static_hum_loss:12.2f}  ({pct(total_static_hum_loss, total_hum_loss):.1f}%)")
    print(f"    Total:           {total_hum_loss:12.2f}")
    print("===========================")

    return moisture, precip