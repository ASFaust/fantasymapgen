import numpy as np
from scipy.ndimage import distance_transform_edt, zoom as _zoom

from sim.terrain import compute_shore_distance
from sim.atmosphere import (
    compute_base_wind, compute_base_temperature, compute_ocean_temp_noise,
    compute_moisture_grid,
)

EARTH_RADIUS_KM = 6371.0


class WeatherSim:
    """
    Scaffolding for a planetary weather simulation.

    Parameters
    ----------
    heightmap : np.ndarray  (H, W) float32, values in [0, 1]
    lat : np.ndarray        (H, W) float32, latitude in radians [-π/2, π/2]
    lon : np.ndarray        (H, W) float32, longitude in radians [-π, π]
    earth_radius_factor : float  multiplier on Earth's radius (6371 km)
    mountain_height_km : float   elevation corresponding to heightmap == 1
    sim_resolution : int         internal sim grid size (e.g. 128, 256, 512, 1024)
    sea_level : float            heightmap value below which a cell is ocean [0, 1]
    """

    def __init__(
        self,
        heightmap: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        earth_radius_factor: float = 1.0,
        mountain_height_km: float = 8.0,
        sim_resolution: int = 256,
        sea_level: float = 0.2,
        lapse_rate: float = 6.5,
        polar_temp: float = -25.0,
        equatorial_temp: float = 27.0,
        dt: float = 3600.0,
        reevaporation_factor: float = 0.5,
        precipitation_factor: float = 0.01,
        orographic_scale_km: float = 1.0,
        n_steps: int = 100,
        seed: int = 42,
        ocean_temp_noise: float = 4.0,
    ):
        self.heightmap = heightmap
        self.lat = lat
        self.lon = lon
        self.earth_radius_factor = earth_radius_factor
        self.mountain_height_km = mountain_height_km
        self.sim_resolution = sim_resolution
        self.sea_level = sea_level
        self.lapse_rate = lapse_rate
        self.polar_temp = polar_temp
        self.equatorial_temp = equatorial_temp
        self.dt = dt
        self.reevaporation_factor = reevaporation_factor
        self.precipitation_factor = precipitation_factor
        self.orographic_scale_km = orographic_scale_km
        self.n_steps = n_steps

        self.radius_km = EARTH_RADIUS_KM * earth_radius_factor
        self.elevation_km: np.ndarray = np.maximum(0.0, heightmap - sea_level) / (1.0 - sea_level) * mountain_height_km

        # ------------------------------------------------------------------
        # Sim-resolution grids  (prefix _sim_)
        # Downsample to (sim_resolution × sim_resolution) for computation,
        # then upsample results back to (H, W) for rendering.
        # ------------------------------------------------------------------
        self._full_shape = heightmap.shape
        H, W = self._full_shape
        if H == sim_resolution and W == sim_resolution:
            self._sim_heightmap = heightmap
            self._sim_lat = lat
            self._sim_lon = lon
        else:
            _zf = (sim_resolution / H, sim_resolution / W)
            self._sim_heightmap = _zoom(heightmap, _zf, order=1)
            self._sim_lat = _zoom(lat, _zf, order=1)
            self._sim_lon = _zoom(lon, _zf, order=1)

        # Elevation above sea level — used for orographic moisture loss.
        # Raw heightmap × mountain_height includes ocean-floor depth, which
        # would create a phantom cliff at every coastline.
        self._sim_elevation_km: np.ndarray = (
            np.maximum(0.0, self._sim_heightmap - sea_level) / (1.0 - sea_level) * mountain_height_km
        )
        self._sim_ocean: np.ndarray = self._sim_heightmap < sea_level

        # Texture 1: distance to nearest ocean cell, in km, on the sphere
        self.sim_shore_distance_km: np.ndarray = self._upsample(
            compute_shore_distance(
                self._sim_ocean, self._sim_lat, self._sim_lon, self.radius_km,
            )
        )

        # Texture 2: base wind field (m/s) — global circulation only, no terrain effects
        # wind_u : eastward component   (positive = blowing east)
        # wind_v : northward component  (positive = blowing north)
        # wind_speed : magnitude
        self._sim_wind_u, self._sim_wind_v, self._sim_wind_speed = (
            compute_base_wind(self._sim_lat)
        )
        self.sim_wind_u: np.ndarray = self._upsample(self._sim_wind_u)
        self.sim_wind_v: np.ndarray = self._upsample(self._sim_wind_v)
        self.sim_wind_speed: np.ndarray = self._upsample(self._sim_wind_speed)

        # Texture 3: base temperature (°C) — insolation + lapse rate, no circulation
        self._sim_temperature: np.ndarray = compute_base_temperature(
            self._sim_lat, self._sim_heightmap, sea_level, mountain_height_km,
            polar_temp, equatorial_temp, lapse_rate,
        )
        if ocean_temp_noise > 0.0:
            self._sim_temperature += compute_ocean_temp_noise(
                self._sim_lat, self._sim_lon, self._sim_ocean,
                seed=seed, amplitude=ocean_temp_noise,
            )
        self.sim_temperature: np.ndarray = self._upsample(self._sim_temperature)

        # Textures 4 & 5: moisture + precipitation [0, 1]
        # Not computed here — call run_moisture() explicitly (it's expensive).
        self.sim_moisture: np.ndarray | None = None
        self.sim_precipitation: np.ndarray | None = None

    def _upsample(self, arr: np.ndarray, ocean_value: float | None = None) -> np.ndarray:
        """Upsample a sim-resolution array to full heightmap resolution (bicubic).

        When *ocean_value* is given, land values are extrapolated into ocean
        cells before interpolation so the bicubic kernel never sees the
        ocean/land step edge (prevents ringing artifacts at coastlines).
        After upsampling, the full-resolution ocean mask is stamped back on.
        """
        if arr.shape == self._full_shape:
            return arr
        H, W = self._full_shape
        sh, sw = arr.shape
        src = arr
        if ocean_value is not None and self._sim_ocean.any() and not self._sim_ocean.all():
            # Replace ocean cells with nearest land value to remove the step
            src = arr.copy()
            idx = distance_transform_edt(self._sim_ocean, return_distances=False,
                                        return_indices=True)
            src[self._sim_ocean] = arr[idx[0][self._sim_ocean], idx[1][self._sim_ocean]]
        result = _zoom(src, (H / sh, W / sw), order=3)
        if ocean_value is not None:
            full_ocean = self.heightmap < self.sea_level
            result[full_ocean] = ocean_value
            result = np.clip(result, 0.0, None)
        return result

    def run_moisture(self) -> None:
        """Run the upwind advection moisture simulation."""
        m, p = compute_moisture_grid(
            self._sim_ocean, self._sim_lat, self._sim_lon,
            self._sim_elevation_km,
            self._sim_wind_u, self._sim_wind_v,
            self.radius_km,
            temperature=self._sim_temperature,
            dt=self.dt,
            reevaporation_factor=self.reevaporation_factor,
            precipitation_factor=self.precipitation_factor,
            orographic_scale_km=self.orographic_scale_km,
            n_steps=self.n_steps,
        )
        self.sim_moisture = self._upsample(m, ocean_value=1.0)
        self.sim_precipitation = self._upsample(p, ocean_value=1.0)
