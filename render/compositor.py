import numpy as np
import matplotlib

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
        curve_editor=None,
        cmap_name: str = "terrain",
        clip_sea: bool = False,
        precip_scale_mm: float = 2000.0,
        ice_overlay: bool = False,
    ) -> np.ndarray:
        """Return H×W×3 uint8 for the selected base layer."""
        if self.heightmap is None:
            return np.zeros((64, 64, 3), dtype=np.uint8)

        if base_layer == "temperature":
            img = self._render_temperature()
        elif base_layer == "precipitation":
            img = self._render_precipitation(precip_scale_mm)
        else:
            img = self._render_heightmap(curve_editor, cmap_name, clip_sea)

        if ice_overlay and self.weather_sim is not None:
            img = self._apply_ice_overlay(img)

        return img

    # ------------------------------------------------------------------
    # Base layer renderers
    # ------------------------------------------------------------------

    def _render_heightmap(self, curve_editor, cmap_name: str, clip_sea: bool) -> np.ndarray:
        arr = self.heightmap.copy()

        # Curve remap on land only
        land_mask = arr > SEA_LEVEL
        if land_mask.any() and curve_editor is not None:
            land = arr[land_mask]
            land_norm = (land - SEA_LEVEL) / max(1.0 - SEA_LEVEL, 1e-6)
            land_remapped = curve_editor.apply(land_norm)
            arr[land_mask] = SEA_LEVEL + land_remapped * (1.0 - SEA_LEVEL)

        if clip_sea:
            arr[arr <= SEA_LEVEL] = 0.0

        cmap = matplotlib.colormaps[cmap_name]
        return (cmap(arr)[:, :, :3] * 255).astype(np.uint8)

    def _render_temperature(self) -> np.ndarray:
        if self.weather_sim is None:
            return self._render_heightmap(None, "Greys_r", False)

        t = self.weather_sim.sim_temperature
        # Normalise −30 … +40 °C to [0, 1]; blue=cold, red=hot
        t_norm = np.clip((t - (-30.0)) / 70.0, 0.0, 1.0)
        cmap = matplotlib.colormaps["RdBu_r"]
        return (cmap(t_norm)[:, :, :3] * 255).astype(np.uint8)

    def _render_precipitation(self, precip_scale_mm: float = 2000.0) -> np.ndarray:
        if self.weather_sim is None:
            return self._render_heightmap(None, "Greys_r", False)

        m = self.weather_sim.sim_moisture  # [0, 1]
        # Convert to mm/year; clamp to [0, scale] for colour mapping
        precip_mm = np.clip(m * precip_scale_mm, 0.0, precip_scale_mm)
        t = precip_mm / max(precip_scale_mm, 1.0)  # normalised [0, 1]

        land_mask = self.heightmap > SEA_LEVEL

        # Beige (dry) → blue (wet) interpolation
        dry = np.array([210, 180, 140], dtype=np.float32)   # beige
        wet = np.array([30,  80,  200], dtype=np.float32)   # blue

        rgb = np.zeros((*m.shape, 3), dtype=np.float32)
        rgb[land_mask] = dry + t[land_mask, np.newaxis] * (wet - dry)
        # ocean stays black (already zero)

        return np.clip(rgb, 0, 255).astype(np.uint8)

    def _apply_ice_overlay(self, img: np.ndarray) -> np.ndarray:
        """Blend a pale blue-white tint where temperature is at or below freezing.

        Alpha ramp: fully opaque at −30 °C and below, transparent at +2 °C and above.
        Covers both sea ice (ocean) and snow/glaciers (land).
        """
        t = self.weather_sim.sim_temperature  # °C, (H, W)

        ice_mask = t <= 0.0  # True where frozen

        result = img.copy()
        result[ice_mask] = [210, 235, 255]  # pale blue-white
        return result
