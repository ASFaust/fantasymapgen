import numpy as np
import matplotlib

from sim.biomes import classify_biomes, BIOME_COLORS

SEA_LEVEL = 0.2


class LayerCompositor:
    """
    Pure-numpy layer compositor.  Holds references to data arrays and
    produces a final H×W×3 uint8 image for the selected base layer.

    Overlay drawing (wind arrows, etc.) is handled by the viewport widget
    via QPainter so that it scales correctly with the window size.
    """

    def __init__(self):
        self.heightmap: np.ndarray | None = None
        self.lat: np.ndarray | None = None
        self.lon: np.ndarray | None = None
        self.weather_sim = None  # WeatherSim instance or None

    def set_heightmap(self, heightmap: np.ndarray, lat: np.ndarray, lon: np.ndarray):
        self.heightmap = heightmap
        self.lat = lat
        self.lon = lon

    def set_weather_sim(self, ws):
        self.weather_sim = ws

    # ------------------------------------------------------------------
    # Public render entry point
    # ------------------------------------------------------------------

    def render_base(
        self,
        base_layer: str = "heightmap",
        cmap_name: str = "terrain",
        clip_sea: bool = False,
        precip_scale_mm: float = 2000.0,
        ice_overlay: bool = False,
        alpine_overlay: bool = False,
    ) -> np.ndarray:
        """Return H×W×3 uint8 for the selected base layer."""
        if self.heightmap is None:
            return np.zeros((64, 64, 3), dtype=np.uint8)

        if base_layer == "temperature":
            img = self._render_temperature()
        elif base_layer == "precipitation":
            img = self._render_precipitation(precip_scale_mm)
        elif base_layer == "biomes":
            img = self._render_biomes(precip_scale_mm)
        else:
            img = self._render_heightmap(cmap_name, clip_sea)

        if alpine_overlay and self.weather_sim is not None:
            img = self._apply_alpine_overlay(img)

        if ice_overlay and self.weather_sim is not None:
            img = self._apply_ice_overlay(img)

        return img

    # ------------------------------------------------------------------
    # Base layer renderers
    # ------------------------------------------------------------------

    def _render_heightmap(self, cmap_name: str, clip_sea: bool) -> np.ndarray:
        arr = self.heightmap.copy()

        if clip_sea:
            arr[arr <= SEA_LEVEL] = 0.0

        cmap = matplotlib.colormaps[cmap_name]
        return (cmap(arr)[:, :, :3] * 255).astype(np.uint8)

    def _render_temperature(self) -> np.ndarray:
        if self.weather_sim is None:
            return self._render_heightmap("Greys_r", False)

        t = self.weather_sim.sim_temperature
        # Normalise −30 … +40 °C to [0, 1]; blue=cold, red=hot
        t_norm = np.clip((t - (-30.0)) / 70.0, 0.0, 1.0)
        cmap = matplotlib.colormaps["RdBu_r"]
        return (cmap(t_norm)[:, :, :3] * 255).astype(np.uint8)

    def _render_precipitation(self, precip_scale_mm: float = 2000.0) -> np.ndarray:
        if self.weather_sim is None or self.weather_sim.sim_precipitation is None:
            return self._render_heightmap("Greys_r", False)

        m = self.weather_sim.sim_precipitation  # [0, 1], land max = 1.0
        # Log-scale the [0, 1] ratio to expand dry-to-moderate variation
        k = 10.0
        t = np.log1p(np.clip(m, 0.0, 1.0) * k) / np.log1p(k)

        land_mask = self.heightmap > SEA_LEVEL

        # Beige (dry) → blue (wet) interpolation
        dry = np.array([210, 180, 140], dtype=np.float32)   # beige
        wet = np.array([30,  80,  200], dtype=np.float32)   # blue

        rgb = np.zeros((*m.shape, 3), dtype=np.float32)
        rgb[land_mask] = dry + t[land_mask, np.newaxis] * (wet - dry)
        # ocean stays black (already zero)

        return np.clip(rgb, 0, 255).astype(np.uint8)

    def _render_biomes(self, precip_scale_mm: float = 2000.0) -> np.ndarray:
        ws = self.weather_sim
        if ws is None or ws.sim_precipitation is None:
            return self._render_heightmap("Greys_r", False)

        ocean_mask = self.heightmap <= SEA_LEVEL
        biome_ids = classify_biomes(
            temperature=ws.sim_temperature,
            precipitation_mm=ws.sim_precipitation * precip_scale_mm,
            elevation_km=ws.elevation_km,
            ocean_mask=ocean_mask,
        )
        return BIOME_COLORS[biome_ids]

    def _apply_alpine_overlay(self, img: np.ndarray) -> np.ndarray:
        """Solid grey tint where elevation is above 2.5 km."""
        mask = self.weather_sim.elevation_km >= 2.5
        result = img.copy()
        result[mask] = [0x9E, 0x9E, 0x9E]
        return result

    def _apply_ice_overlay(self, img: np.ndarray) -> np.ndarray:
        """Blend a pale blue-white tint over cold regions.

        Alpha ramp with a visible edge:
          T >= -2 °C  → no ice  (alpha = 0)
          T  = -2 °C  → alpha = 0.5  (sharp visible boundary)
          T  = -15 °C → alpha = 1.0  (fully opaque pack ice)
        """
        t = self.weather_sim.sim_temperature  # °C, (H, W)

        # Linear ramp from 0.5 at -2°C to 1.0 at -15°C
        alpha = np.where(
            t >= -2.0,
            0.0,
            np.clip(0.5 + 0.5 * (-2.0 - t) / 13.0, 0.5, 1.0),
        )

        ice_color = np.array([210, 235, 255], dtype=np.float32)
        result = img.astype(np.float32)
        a = alpha[:, :, np.newaxis]
        result = (1.0 - a) * result + a * ice_color
        return np.clip(result, 0, 255).astype(np.uint8)
