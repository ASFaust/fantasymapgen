import numpy as np
from scipy.ndimage import gaussian_filter

from gen.fast_perlin import make_permutation, perlin3_grid

# ---------------------------------------------------------------------------
# Optional Numba acceleration for the moisture time-stepping loop.
# On first import the JIT function is compiled (a few seconds); subsequent
# runs reuse the on-disk cache (cache=True).  Falls back to pure NumPy if
# numba is not installed.
# ---------------------------------------------------------------------------
try:
    import numba as _nb

    @_nb.njit(parallel=True, cache=True)
    def _moisture_step_nb(q, q_new, precip_step,
                          q_sat, decay,
                          u_pos, u_neg, v_pos, v_neg,
                          inv_dx, inv_dx2, inv_dy, inv_dy2,
                          dt, diff_coef, sim_ocean):
        """One upwind-advection step (JIT-compiled, parallel over rows).

        Reads *q*, writes into *q_new* and *precip_step* in-place.
        Uses direct index arithmetic instead of np.roll/concatenate, so
        no temporary arrays are allocated during the hot loop.
        """
        H, W = q.shape
        for i in _nb.prange(H):
            i_N = i - 1 if i > 0 else 0
            i_S = i + 1 if i < H - 1 else H - 1
            for j in range(W):
                j_W = j - 1 if j > 0 else W - 1
                j_E = j + 1 if j < W - 1 else 0

                qc = q[i, j]
                qW = q[i, j_W]
                qE = q[i, j_E]
                qN = q[i_N, j]
                qS = q[i_S, j]
                qs = q_sat[i, j]

                # Upwind advection
                flux_x = (u_pos[i, j] * (qc - qW) +
                           u_neg[i, j] * (qE - qc)) * inv_dx[i, j]
                flux_y = (v_pos[i, j] * (qc - qS) +
                           v_neg[i, j] * (qN - qc)) * inv_dy

                # 5-point diffusion
                laplacian = ((qE - 2.0 * qc + qW) * inv_dx2[i, j] +
                             (qN - 2.0 * qc + qS) * inv_dy2)

                q_adv = qc - dt * (flux_x + flux_y) + diff_coef * laplacian

                # Decay → precipitation
                p = q_adv * (1.0 - decay[i, j])
                q_val = q_adv * decay[i, j]

                # Supersaturation
                excess = q_val - qs
                if excess > 0.0:
                    p += excess
                    q_val = qs

                # Clamp and ocean pin
                if q_val < 0.0:
                    q_val = 0.0
                elif q_val > 1.0:
                    q_val = 1.0
                if sim_ocean[i, j]:
                    q_val = qs

                q_new[i, j] = q_val
                precip_step[i, j] = p

    _HAVE_NUMBA = True

except ImportError:
    _HAVE_NUMBA = False


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
    temperature: np.ndarray | None = None,
    sim_time_hours: float = 500.0,
    diffusion_km2_hr: float = 100.0,
    orographic_factor: float = 0.5,
    land_penetration_km: float = 2000.0,
    max_outflow_fraction: float = 0.8,
    convergence_threshold: float = 1e-6,
    progress_cb=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Upwind advection moisture simulation with CFL time stepping.

    Replaces the 8-neighbor CA with a first-order upwind finite-difference
    scheme.  Moisture is advected strictly along the wind direction without
    directional leakage to off-wind neighbors.  This, combined with a
    CFL-derived physical dt, makes inland penetration resolution-independent.

    Each step:
      1. Upwind advection  — moves moisture along wind
      2. Diffusion         — small isotropic smoothing (5-point Laplacian)
      3. Land loss         — exponential decay with e-folding = land_penetration_km
      4. Orographic loss   — moisture removed when wind pushes air upslope
      5. Supersaturation   — precipitate excess above q_sat
      6. Ocean reset       — ocean cells pinned to q_sat

    Parameters
    ----------
    sim_ocean           : (H, W) bool   — True where ocean
    sim_lat             : (H, W) float  — latitude in radians, row 0 = north
    sim_lon             : (H, W) float  — longitude in radians
    sim_elevation_km    : (H, W) float  — terrain elevation above sea level [km]
    wind_u              : (H, W) float  — eastward wind [m/s]
    wind_v              : (H, W) float  — northward wind [m/s]
    radius_km           : float         — planetary radius in km
    temperature         : (H, W) float  — surface temperature in °C, or None
    sim_time_hours      : float         — total physical simulation time [hours]
    diffusion_km2_hr    : float         — isotropic diffusion coefficient [km²/hr]
    orographic_factor   : float         — moisture loss fraction per km elevation gain
    land_penetration_km : float         — e-folding distance over flat land [km]
    max_outflow_fraction: float         — (unused, kept for API compat)
    convergence_threshold: float        — stop if max per-cell |Δq| < this

    Returns
    -------
    moisture      : (H, W) float32  — steady-state atmospheric moisture [0, 1]
    precipitation : (H, W) float32  — precipitation at final step [0, 1], normalized
                                      so the land maximum = 1.0
    """
    H, W = sim_lat.shape

    # ------------------------------------------------------------------
    # 1. Temperature-dependent saturation capacity
    # ------------------------------------------------------------------
    if temperature is not None:
        t_max = float(temperature.max())
        q_sat = np.exp(0.067 * (temperature - t_max)).astype(np.float64)
    else:
        q_sat = np.ones((H, W), dtype=np.float64)

    # ------------------------------------------------------------------
    # 2. Grid geometry
    # ------------------------------------------------------------------
    lat_col = sim_lat[:, W // 2]
    dlat = float(np.abs(np.median(np.diff(lat_col))))
    dlon = float(np.abs(np.median(np.diff(sim_lon[H // 2, :]))))

    dy = radius_km * dlat                                # N-S spacing [km], scalar
    dx = np.maximum(radius_km * dlon * np.cos(sim_lat),  # E-W spacing [km], (H,W)
                    dy * 0.25)                            # cap so poles don't strangle CFL

    # ------------------------------------------------------------------
    # 3. Wind in km/hr, CFL time step
    # ------------------------------------------------------------------
    u_kmh = wind_u * 3.6                                 # eastward [km/hr]
    v_kmh = wind_v * 3.6                                 # northward [km/hr]

    cfl = 0.45
    # Full 2D CFL: dt * (|u|/dx + |v|/dy) < 1
    max_courant = float((np.abs(u_kmh) / dx + np.abs(v_kmh) / dy).max())
    dt = cfl / max(max_courant, 1e-6)                    # hours
    n_steps = min(int(np.ceil(sim_time_hours / dt)), 10000)

    print(f"Moisture sim: dy={dy:.1f} km, v_max={float(np.sqrt(u_kmh**2+v_kmh**2).max()):.1f} km/h, "
          f"dt={dt:.2f} hr, n_steps={n_steps}")

    # ------------------------------------------------------------------
    # 4. Precompute upwind helpers (static)
    # ------------------------------------------------------------------
    u_pos = np.maximum(u_kmh, 0.0)
    u_neg = np.minimum(u_kmh, 0.0)
    v_pos = np.maximum(v_kmh, 0.0)   # northward = row-decreasing
    v_neg = np.minimum(v_kmh, 0.0)   # southward = row-increasing

    inv_dx = 1.0 / dx               # (H, W)
    inv_dy = 1.0 / dy               # scalar

    # Diffusion stability: D*dt*(1/dx² + 1/dy²) < 0.5  (always satisfied
    # at our CFL because advection is the stiff term, not diffusion)
    inv_dx2 = inv_dx ** 2
    inv_dy2 = inv_dy ** 2

    # ------------------------------------------------------------------
    # 5. Terrain gradient → orographic uplift rate [km/hr]
    # ------------------------------------------------------------------
    elev = sim_elevation_km
    # Central differences, wrapping EW, clamping NS
    elev_E = np.roll(elev, -1, axis=1)
    elev_W = np.roll(elev, 1, axis=1)
    elev_N = np.concatenate([elev[:1], elev[:-1]], axis=0)
    elev_S = np.concatenate([elev[1:], elev[-1:]], axis=0)

    dz_dx = (elev_E - elev_W) / (2.0 * dx)              # [km/km]
    dz_dy = (elev_N - elev_S) / (2.0 * dy)              # [km/km]

    # Uplift = wind·∇z (positive when air is pushed upslope)
    uplift_kmh = np.maximum(0.0, u_kmh * dz_dx + v_kmh * dz_dy)  # [km/hr]

    # ------------------------------------------------------------------
    # 6. Precompute per-step decay factors (static)
    # ------------------------------------------------------------------
    land = (~sim_ocean).astype(np.float64)
    land_loss_per_km = 1.0 / max(land_penetration_km, 1.0)
    wind_speed_kmh = np.sqrt(u_kmh ** 2 + v_kmh ** 2)
    # Minimum effective speed for loss: ensures calm regions still lose
    # moisture over land (convective precip, radiation cooling)
    effective_speed = np.maximum(wind_speed_kmh, 1.0)     # km/hr

    land_decay = np.exp(-effective_speed * dt * land_loss_per_km * land)
    oro_decay  = np.exp(-orographic_factor * uplift_kmh * dt)
    decay = land_decay * oro_decay
    decay[sim_ocean] = 1.0                                # no loss over ocean

    # ------------------------------------------------------------------
    # 7. Main loop
    # ------------------------------------------------------------------
    q = np.where(sim_ocean, q_sat, 0.0)
    diff_coef = diffusion_km2_hr * dt

    if _HAVE_NUMBA:
        # --- Numba path: JIT-compiled parallel step, double-buffered ---
        # No temporary arrays are allocated inside the loop; q and q_new
        # are swapped each step instead of copied.
        q_new = np.empty_like(q)
        precip_step = np.zeros((H, W), dtype=np.float64)
        for step in range(n_steps):
            _moisture_step_nb(q, q_new, precip_step,
                              q_sat, decay,
                              u_pos, u_neg, v_pos, v_neg,
                              inv_dx, inv_dx2, inv_dy, inv_dy2,
                              dt, diff_coef, sim_ocean)
            delta = float(np.max(np.abs(q_new - q)))
            q, q_new = q_new, q                          # double-buffer swap

            if (step + 1) % max(1, n_steps // 20) == 0 or delta < convergence_threshold:
                t_cur = (step + 1) * dt
                print(f"\r  {step+1}/{n_steps} ({t_cur:.0f}/{sim_time_hours:.0f} h) "
                      f"Δq={delta:.2e}", end="", flush=True)
                if progress_cb is not None:
                    progress_cb(step + 1, n_steps)
            if delta < convergence_threshold:
                break

    else:
        # --- NumPy fallback (original algorithm, unchanged) ---
        precip_step = np.zeros((H, W), dtype=np.float64)
        for step in range(n_steps):
            q_W = np.roll(q, 1, axis=1)
            q_E = np.roll(q, -1, axis=1)
            q_N = np.concatenate([q[:1], q[:-1]], axis=0)
            q_S = np.concatenate([q[1:], q[-1:]], axis=0)

            flux_x = (u_pos * (q - q_W) + u_neg * (q_E - q)) * inv_dx
            flux_y = (v_pos * (q - q_S) + v_neg * (q_N - q)) * inv_dy
            q_adv = q - dt * (flux_x + flux_y)

            laplacian = ((q_E - 2.0 * q + q_W) * inv_dx2 +
                         (q_N - 2.0 * q + q_S) * inv_dy2)
            q_adv += diff_coef * laplacian

            precip_step[:] = q_adv * (1.0 - decay)
            q_new = q_adv * decay
            excess = np.maximum(0.0, q_new - q_sat)
            precip_step += excess
            q_new -= excess

            q_new = np.clip(q_new, 0.0, 1.0)
            q_new[sim_ocean] = q_sat[sim_ocean]

            delta = float(np.max(np.abs(q_new - q)))
            q = q_new

            if (step + 1) % max(1, n_steps // 20) == 0 or delta < convergence_threshold:
                t_cur = (step + 1) * dt
                print(f"\r  {step+1}/{n_steps} ({t_cur:.0f}/{sim_time_hours:.0f} h) "
                      f"Δq={delta:.2e}", end="", flush=True)
                if progress_cb is not None:
                    progress_cb(step + 1, n_steps)
            if delta < convergence_threshold:
                break

    print()
    precip = precip_step

    # ------------------------------------------------------------------
    # 8. Post-process precipitation
    # ------------------------------------------------------------------
    precip = gaussian_filter(precip, sigma=2.0)

    land_mask = ~sim_ocean
    if land_mask.any():
        land_max = precip[land_mask].max()
        if land_max > 0.0:
            precip /= land_max

    moisture = np.clip(q, 0.0, 1.0).astype(np.float32)
    moisture[sim_ocean] = 1.0
    precip = np.clip(precip, 0.0, 1.0).astype(np.float32)
    precip[sim_ocean] = 1.0

    return moisture, precip
