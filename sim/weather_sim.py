import heapq

import numpy as np
from scipy.ndimage import zoom as _zoom

EARTH_RADIUS_KM = 6371.0


class WeatherSim:
    """
    Scaffolding for a planetary weather simulation.

    Parameters
    ----------
    heightmap : np.ndarray  (H, W) float32, values in [0, 1]
    lat : np.ndarray        (H, W) float32, latitude in radians [-π/2, π/2]
    lon : np.ndarray        (H, W) float32, longitude in radians [0, 2π]
    earth_radius_factor : float  multiplier on Earth's radius (6371 km)
    mountain_height_km : float   elevation corresponding to heightmap == 1
    sim_resolution : float       downsample factor for internal sim grids (0 < x ≤ 1)
    sea_level : float            heightmap value below which a cell is ocean [0, 1]
    """

    def __init__(
        self,
        heightmap: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        earth_radius_factor: float = 1.0,
        mountain_height_km: float = 8.0,
        sim_resolution: float = 1.0,
        sea_level: float = 0.2,
        shore_decay_km: float = 600.0,
        lapse_rate: float = 6.5,
        onshore_strength: float = 0.4,
        orographic_strength: float = 0.5,
        orographic_n_steps: int = 20,
        polar_temp: float = -25.0,
        equatorial_temp: float = 27.0,
    ):
        self.heightmap = heightmap
        self.lat = lat
        self.lon = lon
        self.earth_radius_factor = earth_radius_factor
        self.mountain_height_km = mountain_height_km
        self.sim_resolution = sim_resolution
        self.sea_level = sea_level
        self.shore_decay_km = shore_decay_km
        self.lapse_rate = lapse_rate
        self.onshore_strength = onshore_strength
        self.orographic_strength = orographic_strength
        self.orographic_n_steps = orographic_n_steps
        self.polar_temp = polar_temp
        self.equatorial_temp = equatorial_temp

        self.radius_km = EARTH_RADIUS_KM * earth_radius_factor
        # Full-resolution elevation in km
        self.elevation_km: np.ndarray = heightmap * mountain_height_km

        # ------------------------------------------------------------------
        # Sim-resolution grids  (prefix _sim_)
        # All subsequent texture maps are computed at this resolution and
        # can be upsampled back to (H, W) for rendering.
        # ------------------------------------------------------------------
        if sim_resolution == 1.0:
            self._sim_heightmap = heightmap
            self._sim_lat = lat
            self._sim_lon = lon
        else:
            # order=1 (bilinear) avoids ringing; good enough for a heightmap
            self._sim_heightmap = _zoom(heightmap, sim_resolution, order=1)
            self._sim_lat = _zoom(lat, sim_resolution, order=1)
            self._sim_lon = _zoom(lon, sim_resolution, order=1)

        self._sim_elevation_km: np.ndarray = self._sim_heightmap * mountain_height_km
        self._sim_ocean: np.ndarray = self._sim_heightmap < sea_level

        # Texture 1: distance to nearest ocean cell, in km, on the sphere
        self.sim_shore_distance_km: np.ndarray = self._compute_shore_distance()

        # Texture 2: terrain gradient (dimensionless slope = km elev / km horizontal)
        # grad_x : E/W component  (positive = uphill eastward)
        # grad_y : N/S component  (positive = uphill in +row direction, i.e. southward)
        self.sim_grad_x: np.ndarray
        self.sim_grad_y: np.ndarray
        self.sim_grad_magnitude: np.ndarray
        self.sim_grad_x, self.sim_grad_y, self.sim_grad_magnitude = (
            self._compute_terrain_gradient()
        )

        # Texture 3: base wind field (m/s) — global circulation only, no terrain effects
        # wind_u : eastward component   (positive = blowing east)
        # wind_v : northward component  (positive = blowing north)
        # wind_speed : magnitude
        self.sim_wind_u: np.ndarray
        self.sim_wind_v: np.ndarray
        self.sim_wind_speed: np.ndarray
        self.sim_wind_u, self.sim_wind_v, self.sim_wind_speed = (
            self._compute_base_wind()
        )

        # Texture 4: base temperature (°C) — insolation + lapse rate, no circulation
        self.sim_temperature: np.ndarray = self._compute_base_temperature()

        # Texture 5: moisture / precipitation potential [0, 1]
        self.sim_moisture: np.ndarray = self._compute_moisture()

    # ------------------------------------------------------------------
    # Spherical distance field
    # ------------------------------------------------------------------

    def _compute_shore_distance(self) -> np.ndarray:
        """
        Fast shore distance using scipy's distance transform + per-row haversine scaling.

        Strategy:
        1. Compute a pixel-unit Euclidean distance transform from ocean cells.
        2. Convert to km by multiplying by the local arc-length per pixel.
        - N/S: constant = R * Δlat  (one row step)
        - E/W: shrinks with cos(lat) = R * cos(lat) * Δlon  (one col step)
        3. Use the geometric mean of the two as the effective pixel size, then
        multiply by the pixel-distance. This is an approximation but is
        accurate to within a few % and runs in milliseconds.

        For a true geodesic field, use the Dijkstra path below; for most
        weather-sim purposes this fast version is indistinguishable.
        """
        from scipy.ndimage import distance_transform_edt

        H, W = self._sim_lat.shape
        R = self.radius_km

        # --- pixel spacing in radians ---
        # Use median spacing to handle non-uniform grids gracefully
        dlat = float(np.abs(np.median(np.diff(self._sim_lat[:, W // 2]))))
        dlon = float(np.abs(np.median(np.diff(self._sim_lon[H // 2, :]))))

        # N/S km per pixel: constant across the grid
        km_per_pixel_ns = R * dlat  # arc length for one row step

        # E/W km per pixel: varies with latitude (cos factor)
        # shape (H, 1) so it broadcasts over columns
        cos_lat = np.cos(self._sim_lat[:, W // 2])[:, np.newaxis]  # (H, 1)
        km_per_pixel_ew = R * np.abs(dlon) * cos_lat               # (H, W) after broadcast

        # Geometric mean pixel size at each row — used as the isotropic scale
        #pixel_km = np.sqrt(km_per_pixel_ns * km_per_pixel_ew)      # (H, 1)
        pixel_km = np.sqrt(np.maximum(km_per_pixel_ns * km_per_pixel_ew, 0.0))

        # EDT returns distance in "pixels" with anisotropic sampling supported.
        # We pass sampling=(ns, ew_equator) and then correct per-row below.
        # Simpler: run unitless EDT then multiply by pixel_km.
        land_mask = (~self._sim_ocean).astype(np.uint8)

        # distance_transform_edt handles the periodic longitude wrap poorly, so
        # we tile the grid horizontally (×3) and crop the centre strip.
        land_tiled = np.tile(land_mask, 3)
        pixel_dist_tiled = distance_transform_edt(land_tiled)
        # Crop centre
        pixel_dist = pixel_dist_tiled[:, W:2*W]

        dist_km = pixel_dist * pixel_km   # broadcast (H, W) * (H, 1)

        # Ocean cells must be exactly 0
        dist_km[self._sim_ocean] = 0.0

        return dist_km

    # ------------------------------------------------------------------
    # Terrain gradient
    # ------------------------------------------------------------------

    def _compute_terrain_gradient(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the terrain gradient in physical units (km elevation / km horizontal).

        Uses central differences with longitude wrap handled by tiling one
        column of padding on each side, matching the approach in
        _compute_shore_distance.

        Returns
        -------
        grad_x : (H, W)  E/W slope  — positive = uphill eastward
        grad_y : (H, W)  N/S slope  — positive = uphill southward (row +)
        grad_magnitude : (H, W)  |∇h|
        """
        H, W = self._sim_lat.shape
        R = self.radius_km

        dlat = float(np.abs(np.median(np.diff(self._sim_lat[:, W // 2]))))
        dlon = float(np.abs(np.median(np.diff(self._sim_lon[H // 2, :]))))

        km_per_pixel_ns = R * dlat                                        # scalar
        cos_lat = np.cos(self._sim_lat[:, W // 2])[:, np.newaxis]        # (H, 1)
        km_per_pixel_ew = R * np.abs(dlon) * np.maximum(cos_lat, 1e-6)  # (H, 1)

        elev = self._sim_elevation_km  # (H, W), km

        # E/W gradient — pad one column on each side for longitude wrap
        elev_padded = np.concatenate([elev[:, -1:], elev, elev[:, :1]], axis=1)
        # np.gradient uses central differences for interior points; the padded
        # edge columns will get forward/backward differences but we discard them.
        grad_x_padded = np.gradient(elev_padded, axis=1)
        grad_x = grad_x_padded[:, 1:-1] / km_per_pixel_ew  # km/km, (H, W)

        # N/S gradient — no wrap (poles are boundaries)
        grad_y = np.gradient(elev, axis=0) / km_per_pixel_ns  # km/km, (H, W)

        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        return grad_x, grad_y, grad_magnitude

    # ------------------------------------------------------------------
    # Base wind field
    # ------------------------------------------------------------------

    def _compute_base_wind(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        # ----------------------------------------------------------------
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
        #  -30  |     0       |     0        | S horse latitudes (u zero crossing, calm)
        #  -18  |    -8       |     4        | SE trades (peak, toward equator)
        #    0  |     0       |     0        | ITCZ / doldrums (u zero crossing)
        #   18  |    -8       |    -4        | NE trades (peak, toward equator)
        #   30  |     0       |     0        | N horse latitudes (u zero crossing, calm)
        #   50  |    10       |     0        | N westerlies (peak)
        #   65  |     0       |    -1        | N polar front (u zero crossing)
        #   90  |    -8       |     0        | N polar easterlies (peak)
        # ----------------------------------------------------------------
        ctrl_lat_deg = np.array([-90, -65, -50, -30, -18,   0,  18,  30,  50,  65,  90], dtype=float)
        ctrl_u       = np.array([ -8,   0,  10,   0,  -8,   0,  -8,   0,  10,   0,  -8], dtype=float)
        ctrl_v       = np.array([  0,   1,   0,   0,   4,   0,  -4,   0,   0,  -1,   0], dtype=float)

        ctrl_lat_rad = np.deg2rad(ctrl_lat_deg)

        # PchipInterpolator (monotone piecewise cubic Hermite) is used instead
        # of CubicSpline because the u/v control points oscillate sign across
        # narrow latitude bands.  CubicSpline (not-a-knot) is C2 but has no
        # monotonicity guarantee, so it overshoots between tightly-packed
        # turning points and produces spurious wind reversals.  Pchip is C1,
        # never exceeds the control-point envelope on any interval, and gives
        # smooth, physically plausible wind profiles.
        cs_u = PchipInterpolator(ctrl_lat_rad, ctrl_u)
        cs_v = PchipInterpolator(ctrl_lat_rad, ctrl_v)

        # Latitude per row — use the centre column, same as elsewhere
        H, W = self._sim_lat.shape
        lat_col = self._sim_lat[:, W // 2]          # (H,)

        # Clamp to spline domain [-π/2, π/2] to avoid edge extrapolation
        lat_col_clamped = np.clip(lat_col, ctrl_lat_rad[0], ctrl_lat_rad[-1])

        wind_u_row = cs_u(lat_col_clamped)  # (H,)
        wind_v_row = cs_v(lat_col_clamped)  # (H,)

        # Pchip is only C1: curvature is discontinuous at each knot, producing
        # visible kinks in the rendered field.  A Gaussian blur along the 1-D
        # latitude profile smooths these out before broadcasting.
        # sigma = 5° expressed in grid rows, clamped to ≥1.
        from scipy.ndimage import gaussian_filter1d
        dlat_deg = np.degrees(float(np.abs(np.median(np.diff(lat_col_clamped)))))
        sigma_rows = max(1.0, 5.0 / dlat_deg)
        wind_u_row = gaussian_filter1d(wind_u_row, sigma=sigma_rows)
        wind_v_row = gaussian_filter1d(wind_v_row, sigma=sigma_rows)

        # Broadcast to (H, W) — base circulation is longitude-independent
        wind_u = np.broadcast_to(wind_u_row[:, np.newaxis], (H, W)).copy()
        wind_v = np.broadcast_to(wind_v_row[:, np.newaxis], (H, W)).copy()

        wind_speed = np.sqrt(wind_u**2 + wind_v**2)

        return wind_u, wind_v, wind_speed

    # ------------------------------------------------------------------
    # Base temperature field
    # ------------------------------------------------------------------

    def _compute_base_temperature(self) -> np.ndarray:
        """
        Compute the base surface temperature field (°C).

        Two contributions:
          1. Latitude-driven insolation: cubic spline through annual-mean
             sea-level temperature control points.
          2. Lapse rate: −6.5 °C / km above sea level (land only; ocean
             cells are treated as being at sea level).

        No wind advection or ocean heat-capacity effects — those belong to
        later layers once moisture is available.

        Returns
        -------
        temperature : (H, W)  surface temperature in °C
        """
        # T(φ) = polar_temp + (equatorial_temp − polar_temp) · cos(φ)
        # Insolation scales as cos(lat); this gives the correct endpoints and a
        # physically motivated shape with no free intermediate knots.
        temp = (self.polar_temp
                + (self.equatorial_temp - self.polar_temp) * np.cos(self._sim_lat))

        # Lapse rate correction — land cells only, clamped to ≥ 0 km ASL.
        # (heightmap − sea_level) is the normalised height above the ocean
        # threshold; multiplying by mountain_height_km converts to physical km.
        elevation_asl_km = np.maximum(0.0, self._sim_heightmap - self.sea_level) * self.mountain_height_km
        temp -= self.lapse_rate * elevation_asl_km

        return temp

    # ------------------------------------------------------------------
    # Moisture / precipitation potential
    # ------------------------------------------------------------------

    def _compute_moisture(self) -> np.ndarray:
        """
        Compute a moisture / precipitation-potential field in [0, 1].

        Four multiplicative contributions:
          1. Shore-distance decay: exponential decay from ocean cells
             (~600 km e-folding scale for continental drying).
          2. Onshore wind alignment: ∇(shore_distance) points inland, so
             wind · ∇(shore_dist) > 0 means the wind is carrying moisture
             inland (onshore); < 0 means it is blowing dry air offshore.
          3. Orographic effect: semi-Lagrangian moisture advection for
             ``orographic_n_steps`` steps.  Air ascending over terrain
             loses moisture proportionally to the lift; descending air
             retains whatever the depleted upstream supply provided (rain
             shadow from upwind depletion, not from explicit Föhn drying).
          4. Temperature cap (Clausius-Clapeyron proxy): cold air holds less
             moisture. Logistic centred at 0 °C with a 15 °C scale:
               −30 °C → ~0.12,   0 °C → 0.50,  +30 °C → ~0.88.

        Ocean cells are forced to 1.0 after normalisation.

        Returns
        -------
        moisture : (H, W)  precipitation potential in [0, 1]
        """
        H, W = self._sim_lat.shape
        R = self.radius_km

        dlat = float(np.abs(np.median(np.diff(self._sim_lat[:, W // 2]))))
        dlon = float(np.abs(np.median(np.diff(self._sim_lon[H // 2, :]))))
        km_per_pixel_ns = R * dlat
        cos_lat = np.cos(self._sim_lat[:, W // 2])[:, np.newaxis]        # (H, 1)
        km_per_pixel_ew = R * np.abs(dlon) * np.maximum(cos_lat, 1e-6)   # (H, 1)

        # --- 1. Shore-distance exponential decay ---
        moisture = np.exp(-self.sim_shore_distance_km / self.shore_decay_km)  # (H, W)

        # --- 2. Onshore wind alignment ---
        # Gradient of shore_distance points inland (away from ocean).
        # wind · ∇(shore_dist) > 0  →  onshore (moistening)
        # wind · ∇(shore_dist) < 0  →  offshore (drying)
        sd = self.sim_shore_distance_km
        sd_padded = np.concatenate([sd[:, -1:], sd, sd[:, :1]], axis=1)
        dsd_dx = np.gradient(sd_padded, axis=1)[:, 1:-1] / km_per_pixel_ew  # (H, W)
        dsd_dy = np.gradient(sd, axis=0) / km_per_pixel_ns                  # (H, W)

        # Normalise by wind speed so the factor reflects direction only
        dot = self.sim_wind_u * dsd_dx + self.sim_wind_v * dsd_dy
        onshore = dot / (self.sim_wind_speed + 1e-6)                         # dimensionless ~[-1, 1]
        onshore_factor = 1.0 + self.onshore_strength * np.tanh(onshore)      # [1-s, 1+s]
        moisture *= onshore_factor

        # --- 3. Orographic effect via iterated semi-Lagrangian advection ---
        moisture = self._advect_orographic(moisture)

        # --- 4. Temperature cap (Clausius-Clapeyron via Magnus formula) ---
        # e_s(T) ∝ exp(17.67·T / (T + 243.5)) — the correct ~7 %/°C scaling.
        # Normalised to 1.0 at 30 °C; anything warmer is clipped to 1.0
        # (moisture becomes source-limited, not capacity-limited, above that).
        # Spot-check vs logistic:  T=0 °C → 0.14,  T=15 °C → 0.40,  T=30 °C → 1.0
        _T = self.sim_temperature
        _cc_ref = float(np.exp(17.67 * 30.0 / (30.0 + 243.5)))
        temp_factor = np.clip(np.exp(17.67 * _T / (_T + 243.5)) / _cc_ref, 0.0, 1.0)
        moisture *= temp_factor

        # --- 5. Normalise and clamp ---
        # Scale so the mean of ocean cells sits at 1.0
        ocean_mean = moisture[self._sim_ocean].mean() if self._sim_ocean.any() else moisture.max()
        if ocean_mean > 0.0:
            moisture /= ocean_mean
        moisture = np.clip(moisture, 0.0, 1.0)

        # Ocean cells are fully saturated by definition
        moisture[self._sim_ocean] = 1.0

        return moisture

    # ------------------------------------------------------------------
    # Semi-Lagrangian orographic advection
    # ------------------------------------------------------------------

    def _advect_orographic(self, moisture: np.ndarray) -> np.ndarray:
        """
        Replace the local tanh orographic proxy with a backward semi-Lagrangian
        advection loop (``orographic_n_steps`` iterations).

        Each step:
          1. Back-trace every cell one half-pixel upstream along the wind.
          2. Bilinear-interpolate the moisture at that upstream location
             (longitude wraps periodically; latitude clamps at the poles).
          3. Multiply by a per-cell *retain* factor that removes moisture
             whenever the parcel ascends (windward precipitation).
             Descending air keeps what it carries — the rain shadow on the
             leeward side emerges from the already-depleted upstream supply.
          4. Pin ocean cells back to 1.0 (permanent moisture sources).

        The retain factor is calibrated so that a parcel rising
        ``mountain_height_km`` in total loses ``orographic_strength`` of
        its moisture:

            retain = exp(−orographic_strength / mountain_height_km
                         × max(elevation_gain_per_step, 0))

        After enough steps the pattern converges; 20–40 steps typically
        lets moisture travel several hundred to ~1 000 km inland.
        """
        from scipy.ndimage import map_coordinates

        H, W = self._sim_lat.shape

        dlat = float(np.abs(np.median(np.diff(self._sim_lat[:, W // 2]))))
        dlon = float(np.abs(np.median(np.diff(self._sim_lon[H // 2, :]))))
        km_ns = self.radius_km * dlat                                          # scalar, km / row
        cos_lat = np.cos(self._sim_lat[:, W // 2])[:, np.newaxis]             # (H, 1)
        km_ew   = self.radius_km * np.abs(dlon) * np.maximum(cos_lat, 1e-6)  # (H, 1), km / col

        # ---- pixel-space velocity ----------------------------------------
        # wind_u > 0 → eastward  → +col   (vel_col > 0)
        # wind_v > 0 → northward → −row   (vel_row < 0, row 0 = north)
        vel_col_raw =  self.sim_wind_u / km_ew   # (H, W)  proportional to col/step
        vel_row_raw = -self.sim_wind_v / km_ns   # (H, W)  proportional to row/step

        # Normalise to CFL ≤ 0.5 px / step (keeps bilinear interp well-behaved)
        max_vel = float(np.max(np.sqrt(vel_col_raw ** 2 + vel_row_raw ** 2))) + 1e-6
        scale   = 0.5 / max_vel
        vel_col = vel_col_raw * scale   # (H, W), pixels per step
        vel_row = vel_row_raw * scale   # (H, W), pixels per step

        # ---- elevation gain per step (km) --------------------------------
        # step_e / step_s : physical distance moved eastward / southward (km)
        step_e = vel_col * km_ew   # (H, W), km eastward  per step
        step_s = vel_row * km_ns   # (H, W), km southward per step
        # grad_x = d(elev)/d(east), grad_y = d(elev)/d(south), both km/km
        elevation_gain = self.sim_grad_x * step_e + self.sim_grad_y * step_s  # km

        # ---- per-step moisture retention ---------------------------------
        # Calibrated: rising mountain_height_km total → ×(1 − orographic_strength)
        precip_rate = self.orographic_strength / max(self.mountain_height_km, 1e-6)
        retain = np.exp(-precip_rate * np.maximum(elevation_gain, 0.0))  # (H, W), (0, 1]

        # ---- coordinate grids for interpolation --------------------------
        row_grid = np.broadcast_to(
            np.arange(H, dtype=np.float32)[:, np.newaxis], (H, W)
        ).copy()
        col_grid = np.broadcast_to(
            np.arange(W, dtype=np.float32)[np.newaxis, :], (H, W)
        ).copy()

        m = moisture.copy()

        for _ in range(self.orographic_n_steps):
            # Upstream source position — handle boundaries before interpolation
            src_row = np.clip(row_grid - vel_row, 0.0, H - 1.0)  # clamp at poles
            src_col = (col_grid - vel_col) % W                    # wrap longitude

            coords = np.array([src_row.ravel(), src_col.ravel()])
            m_up = map_coordinates(
                m, coords, order=1, mode='nearest', prefilter=False
            ).reshape(H, W)

            m = np.clip(m_up * retain, 0.0, 1.0)
            m[self._sim_ocean] = 1.0   # ocean: fixed moisture source

        return m