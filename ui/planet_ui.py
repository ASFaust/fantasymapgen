import numpy as np
import yaml
from pathlib import Path

_DEFAULT_PRESET = Path(__file__).parent.parent / "default_preset.yaml"

from PyQt6.QtWidgets import (
    QWidget, QLabel, QSlider, QVBoxLayout, QHBoxLayout,
    QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton, QProgressBar,
    QScrollArea, QSizePolicy, QFileDialog, QComboBox, QTabWidget,
    QStackedWidget, QButtonGroup,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from gen.iq_warp import create_planet_heightmap
from .utils import float_slider, slider_value
from .curve_editor import CurveEditor
from .gl_widget import PlanetGLWidget
from sim.weather_sim import WeatherSim
from sim.biomes import classify_biomes, BIOME_NAMES
from render.compositor import LayerCompositor
from .map_view_widget import MapViewWidget
from .layer_bar import LayerBar


class WeatherWorker(QThread):
    """Constructs WeatherSim and runs moisture advection in a background thread."""
    progress = pyqtSignal(int, int)   # (step, n_steps)
    finished = pyqtSignal(object)     # WeatherSim instance

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self._params = params

    def run(self):
        p = self._params
        ws = WeatherSim(
            p['heightmap'], p['lat'], p['lon'],
            earth_radius_factor=p['earth_radius_factor'],
            mountain_height_km=p['mountain_height_km'],
            sim_resolution=p['sim_resolution'],
            sea_level=0.2,
            polar_temp=p['polar_temp'],
            equatorial_temp=p['equatorial_temp'],
            lapse_rate=p['lapse_rate'],
            dt=p['dt'],
            sigma_diffusion=p['sigma_diffusion'],
            n_steps=p['n_steps'],
            max_transport_km=p['max_transport_km'],
            total_orog_loss_km=p['total_orog_loss_km'],
            precip_gamma=p['precip_gamma'],
            hum_gamma=p['hum_gamma'],
            precip_hum_ratio=p['precip_hum_ratio'],
            seed=p['seed'],
            ocean_temp_noise=p['ocean_temp_noise'],
        )
        ws.run_moisture()
        self.finished.emit(ws)


class PlanetUI(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Planet Heightmap Generator")
        self.resize(1500, 900)

        self._compositor = LayerCompositor()
        self._ice_overlay_visible = False
        self._alpine_overlay_visible = False
        self._contour_overlay_visible = False
        self._weather_running = False
        self._weather_rerun_pending = False

        main_layout = QHBoxLayout(self)

        # ── Left: scrollable control tabs ──────────────────────────
        control_tabs = QTabWidget()
        main_layout.addWidget(control_tabs, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        control_container = QWidget()
        self.controls_layout = QVBoxLayout(control_container)
        scroll.setWidget(control_container)
        control_tabs.addTab(scroll, "Terrain")

        weather_scroll = QScrollArea()
        weather_scroll.setWidgetResizable(True)
        weather_container = QWidget()
        self.weather_layout = QVBoxLayout(weather_container)
        weather_scroll.setWidget(weather_container)
        control_tabs.addTab(weather_scroll, "Weather Sim")

        # ── Right: viewport + layer bar + preset bar ────────────────
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)
        main_layout.addWidget(right_container, 3)

        # 2D / 3D toggle
        toggle_bar = QWidget()
        toggle_layout = QHBoxLayout(toggle_bar)
        toggle_layout.setContentsMargins(4, 4, 4, 0)
        self._btn_2d = QPushButton("2D")
        self._btn_3d = QPushButton("3D")
        for btn in (self._btn_2d, self._btn_3d):
            btn.setCheckable(True)
            btn.setMinimumWidth(60)
        self._btn_2d.setChecked(True)
        self._view_group = QButtonGroup(self)
        self._view_group.addButton(self._btn_2d, 0)
        self._view_group.addButton(self._btn_3d, 1)
        self._view_group.setExclusive(True)
        self._view_group.idClicked.connect(self._on_view_toggle)
        toggle_layout.addWidget(self._btn_2d)
        toggle_layout.addWidget(self._btn_3d)
        toggle_layout.addStretch()
        right_layout.addWidget(toggle_bar)

        # Stacked viewport
        self._view_stack = QStackedWidget()
        self.map_view = MapViewWidget()
        self.map_view.hover.connect(self._on_hover)
        self._view_stack.addWidget(self.map_view)      # index 0 = 2D
        self.gl_widget = PlanetGLWidget()
        self._view_stack.addWidget(self.gl_widget)     # index 1 = 3D
        right_layout.addWidget(self._view_stack, 1)

        # Layer bar
        self.layer_bar = LayerBar()
        self.layer_bar.base_changed.connect(self.render)
        self.layer_bar.overlay_changed.connect(self._on_overlay_changed)
        right_layout.addWidget(self.layer_bar)

        # Preset bar
        preset_bar = QWidget()
        preset_bar_layout = QHBoxLayout(preset_bar)
        preset_bar_layout.setContentsMargins(6, 4, 6, 4)
        save_preset_btn   = QPushButton("Save Preset")
        load_preset_btn   = QPushButton("Load Preset")
        set_default_btn   = QPushButton("Set as Default")
        reset_default_btn = QPushButton("Reset to Default")
        save_preset_btn.clicked.connect(self.save_preset)
        load_preset_btn.clicked.connect(self.load_preset)
        set_default_btn.clicked.connect(self.set_default_preset)
        reset_default_btn.clicked.connect(self.reset_to_default)
        preset_bar_layout.addWidget(set_default_btn)
        preset_bar_layout.addWidget(reset_default_btn)
        preset_bar_layout.addStretch()

        self._hover_label = QLabel("")
        self._hover_label.setStyleSheet(
            "color: #ccc; font-size: 11px; font-family: monospace;"
        )
        preset_bar_layout.addWidget(self._hover_label)

        preset_bar_layout.addStretch()
        preset_bar_layout.addWidget(load_preset_btn)
        preset_bar_layout.addWidget(save_preset_btn)
        right_layout.addWidget(preset_bar)

        self.build_controls()
        self.build_weather_controls()
        if _DEFAULT_PRESET.exists():
            with open(_DEFAULT_PRESET) as f:
                self._apply_preset(yaml.safe_load(f))
        self.generate()

    # ==========================================================
    # View toggle
    # ==========================================================

    def _on_view_toggle(self, idx: int):
        self._view_stack.setCurrentIndex(idx)

    def _on_overlay_changed(self, name: str, visible: bool):
        if name == "wind":
            self.map_view.set_wind_visible(visible)
            self.render()
        elif name == "ice":
            self._ice_overlay_visible = visible
            self.render()
        elif name == "alpine":
            self._alpine_overlay_visible = visible
            self.render()
        elif name == "contour":
            self._contour_overlay_visible = visible
            self.map_view.set_contour_visible(visible)

    def _on_hover(self, row_frac: float, col_frac: float):
        if row_frac < 0 or not hasattr(self, 'heightmap'):
            self._hover_label.setText("")
            return

        H, W = self.heightmap.shape
        r = min(int(row_frac * H), H - 1)
        c = min(int(col_frac * W), W - 1)

        # South is up (row 0 = south pole), so lat from _lat grid is inverted
        lat_deg = float(np.degrees(self._lat[r, c]))
        lon_deg = float(np.degrees(self._lon[r, c]))
        ns = "S" if lat_deg < 0 else "N"
        ew = "W" if lon_deg < 0 else "E"

        elev_raw = float(self.heightmap[r, c])
        sea_level = 0.2
        ws = self._compositor.weather_sim

        parts = [f"{abs(lat_deg):.1f}\u00b0{ns} {abs(lon_deg):.1f}\u00b0{ew}"]

        if ws is not None:
            elev_km = float(ws.elevation_km[r, c])
            if elev_raw < sea_level:
                parts.append(f"ocean")
            else:
                parts.append(f"elev {elev_km:.2f} km")

            temp = float(ws.sim_temperature[r, c])
            parts.append(f"{temp:.1f} \u00b0C")

            shore = float(ws.sim_shore_distance_km[r, c])
            parts.append(f"shore {shore:.0f} km")

            if ws.sim_precipitation is not None:
                precip_scale = slider_value(self.ws_precip_scale)
                precip_mm = float(ws.sim_precipitation[r, c]) * precip_scale
                parts.append(f"precip {precip_mm:.0f} mm/yr")

                biome_id = classify_biomes(
                    temperature=np.array([[temp]]),
                    precipitation_mm=np.array([[precip_mm]]),
                    elevation_km=np.array([[elev_km]]),
                    ocean_mask=np.array([[elev_raw < sea_level]]),
                )[0, 0]
                parts.append(BIOME_NAMES[biome_id])

        self._hover_label.setText("  \u2502  ".join(parts))

    # ==========================================================
    # Auto Regen Connector
    # ==========================================================

    def connect_auto_regen(self, widget):
        if isinstance(widget, QSlider):
            widget.sliderReleased.connect(self.generate)
        elif isinstance(widget, QSpinBox):
            widget.editingFinished.connect(self.generate)
        elif isinstance(widget, QCheckBox):
            widget.stateChanged.connect(lambda: self.generate())

    # ==========================================================
    # Control Builders
    # ==========================================================

    def add_slider(self, name, min_val, max_val, default, step=0.01):
        label = QLabel(f"{name}: {default}")
        slider = float_slider(min_val, max_val, step)
        slider.setValue(int((default - min_val) / step))
        slider.valueChanged.connect(
            lambda: label.setText(f"{name}: {slider_value(slider):.4f}")
        )
        self.controls_layout.addWidget(label)
        self.controls_layout.addWidget(slider)
        self.connect_auto_regen(slider)
        return slider

    def add_int(self, name, min_val, max_val, default):
        label = QLabel(name)
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        self.controls_layout.addWidget(label)
        self.controls_layout.addWidget(spin)
        self.connect_auto_regen(spin)
        return spin

    def add_bool(self, name, default):
        checkbox = QCheckBox(name)
        checkbox.setChecked(default)
        self.controls_layout.addWidget(checkbox)
        self.connect_auto_regen(checkbox)
        return checkbox

    # ==========================================================
    # Build Controls
    # ==========================================================

    def build_controls(self):
        self.size = self.add_int("size", 128, 2048, 512)
        self.seed = self.add_int("seed", 0, 99999, 42)

        rand_seed_btn = QPushButton("Random Seed")
        rand_seed_btn.clicked.connect(self.random_seed)
        self.controls_layout.addWidget(rand_seed_btn)

        self.octaves    = self.add_int("octaves", 1, 10, 5)
        self.base_freq  = self.add_slider("base_freq", 0.1, 2.5, 1.5)
        self.lacunarity = self.add_slider("lacunarity", 1.1, 4.0, 2.0)
        self.gain       = self.add_slider("gain", 0.1, 1.0, 0.5)

        self.warp1 = self.add_slider("warp1", 0.0, 3.0, 0.8)
        self.warp2 = self.add_slider("warp2", 0.0, 3.0, 0.8)

        self.ridge = self.add_bool("ridge", False)

        self.land_percentage = self.add_slider("land_percentage", 0.05, 0.95, 0.5)

        self.polar_strength  = self.add_slider("polar_strength", 0.0, 1.0, 0.0)
        self.polar_sharpness = self.add_slider("polar_sharpness", 0.1, 10.0, 1.0)

        self.backside_strength  = self.add_slider("backside_strength", 0.0, 1.0, 0.0)
        self.backside_sharpness = self.add_slider("backside_sharpness", 0.1, 10.0, 1.0)

        self.clip_sea = self.add_bool("Clip below sea level to 0", False)

        # ── Height Remap Curve ─────────────────────────────────
        curve_header = QLabel("Height Remap Curve")
        curve_header.setStyleSheet("font-weight: bold; margin-top: 8px;")
        curve_hint = QLabel("drag · right-click removes · click curve adds")
        curve_hint.setStyleSheet("color: #888; font-size: 10px;")
        self.controls_layout.addWidget(curve_header)
        self.controls_layout.addWidget(curve_hint)

        self.curve_editor = CurveEditor()
        self.curve_editor.curveChanged.connect(self._apply_curve)
        self.controls_layout.addWidget(self.curve_editor)

        reset_curve_btn = QPushButton("Reset Curve")
        reset_curve_btn.clicked.connect(self.curve_editor.reset)
        self.controls_layout.addWidget(reset_curve_btn)

        # ── Action buttons ─────────────────────────────────────
        gen_btn    = QPushButton("Generate")
        export_btn = QPushButton("Export PNG")
        gen_btn.clicked.connect(self.generate)
        export_btn.clicked.connect(self.export_image)
        self.controls_layout.addWidget(gen_btn)
        self.controls_layout.addWidget(export_btn)
        self.controls_layout.addStretch()

    def build_weather_controls(self):
        from .utils import slider_value as _sv

        self._weather_calculated = False

        calc_btn = QPushButton("Calculate Weather")
        calc_btn.clicked.connect(self._calculate_weather)
        self.weather_layout.addWidget(calc_btn)
        self._calc_btn = calc_btn

        self._weather_progress = QProgressBar()
        self._weather_progress.setRange(0, 100)
        self._weather_progress.setTextVisible(True)
        self._weather_progress.setVisible(False)
        self.weather_layout.addWidget(self._weather_progress)

        self._weather_status_label = QLabel("")
        self._weather_status_label.setStyleSheet("color: #aaa; font-size: 10px;")
        self.weather_layout.addWidget(self._weather_status_label)

        def _add(name, min_val, max_val, default, step):
            label = QLabel(f"{name}: {default}")
            slider = float_slider(min_val, max_val, step)
            slider.setValue(int((default - min_val) / step))
            slider.valueChanged.connect(
                lambda: label.setText(f"{name}: {_sv(slider):.4f}")
            )
            slider.sliderReleased.connect(self._update_weather_sim)
            self.weather_layout.addWidget(label)
            self.weather_layout.addWidget(slider)
            return slider

        self.earth_radius_factor  = _add("earth_radius_factor",  0.0,  2.0,   1.0,  0.01)
        self.mountain_height_km   = _add("mountain_height_km",   0.0, 20.0,   8.0,  0.1)
        sim_res_label = QLabel("Sim Resolution")
        self.sim_resolution_combo = QComboBox()
        for res in [128, 256, 512, 1024]:
            self.sim_resolution_combo.addItem(f"{res}\u00d7{res}", res)
        self.sim_resolution_combo.setCurrentIndex(1)  # default 256
        self.sim_resolution_combo.currentIndexChanged.connect(
            lambda: self._update_weather_sim()
        )
        self.weather_layout.addWidget(sim_res_label)
        self.weather_layout.addWidget(self.sim_resolution_combo)
        self.ws_polar_temp        = _add("polar_temp (°C)",     -60.0, 20.0, -25.0,  0.5)
        self.ws_equatorial_temp   = _add("equatorial_temp (°C)",  0.0, 60.0,  27.0,  0.5)
        self.ws_lapse_rate        = _add("lapse_rate (°C/km)",   0.0, 15.0,   6.5,  0.1)

        dt_label = QLabel("dt (hours)")
        self.ws_dt = QDoubleSpinBox()
        self.ws_dt.setRange(0.001, 1000.0)
        self.ws_dt.setDecimals(3)
        self.ws_dt.setSingleStep(0.5)
        self.ws_dt.setValue(1.0)
        self.ws_dt.valueChanged.connect(self._update_weather_sim)
        self.weather_layout.addWidget(dt_label)
        self.weather_layout.addWidget(self.ws_dt)
        self.ws_sigma_diffusion    = _add("sigma_diffusion (m/s)", 0.0, 20.0, 2.0, 0.1)

        max_transport_label = QLabel("max_transport_km: 2000")
        self.ws_max_transport_km = float_slider(1.0, 10000.0, 10.0)
        self.ws_max_transport_km.setValue(int((2000.0 - 1.0) / 10.0))
        self.ws_max_transport_km.valueChanged.connect(
            lambda: max_transport_label.setText(
                f"max_transport_km: {_sv(self.ws_max_transport_km):.0f}"
            )
        )
        self.ws_max_transport_km.sliderReleased.connect(self._update_weather_sim)
        self.weather_layout.addWidget(max_transport_label)
        self.weather_layout.addWidget(self.ws_max_transport_km)

        orog_loss_label = QLabel("total_orog_loss_km: 8.0")
        self.ws_total_orog_loss_km = float_slider(0.1, 20.0, 0.1)
        self.ws_total_orog_loss_km.setValue(int((8.0 - 0.1) / 0.1))
        self.ws_total_orog_loss_km.valueChanged.connect(
            lambda: orog_loss_label.setText(
                f"total_orog_loss_km: {_sv(self.ws_total_orog_loss_km):.1f}"
            )
        )
        self.ws_total_orog_loss_km.sliderReleased.connect(self._update_weather_sim)
        self.weather_layout.addWidget(orog_loss_label)
        self.weather_layout.addWidget(self.ws_total_orog_loss_km)

        self.ws_precip_gamma     = _add("precip_gamma",     0.0, 1.0, 0.5, 0.01)
        self.ws_hum_gamma        = _add("hum_gamma",        0.0, 1.0, 0.5, 0.01)
        self.ws_precip_hum_ratio = _add("precip_hum_ratio", 0.0, 1.0, 0.5, 0.01)

        n_steps_label = QLabel("n_steps: 100")
        self.ws_n_steps = QSpinBox()
        self.ws_n_steps.setRange(10, 20000)
        self.ws_n_steps.setValue(100)
        self.ws_n_steps.valueChanged.connect(
            lambda v: n_steps_label.setText(f"n_steps: {v}")
        )
        self.ws_n_steps.valueChanged.connect(self._update_weather_sim)
        self.weather_layout.addWidget(n_steps_label)
        self.weather_layout.addWidget(self.ws_n_steps)

        # Ocean temperature noise — breaks zonal symmetry of ice caps
        ocean_noise_label = QLabel("ocean_temp_noise (°C): 4.0")
        self.ws_ocean_temp_noise = float_slider(0.0, 10.0, 0.5)
        self.ws_ocean_temp_noise.setValue(int((4.0 - 0.0) / 0.5))
        self.ws_ocean_temp_noise.valueChanged.connect(
            lambda: ocean_noise_label.setText(
                f"ocean_temp_noise (°C): {_sv(self.ws_ocean_temp_noise):.1f}"
            )
        )
        self.ws_ocean_temp_noise.sliderReleased.connect(self._update_weather_sim)
        self.weather_layout.addWidget(ocean_noise_label)
        self.weather_layout.addWidget(self.ws_ocean_temp_noise)

        # Precipitation scale — render-only, no need to re-run sim
        precip_label = QLabel("precip_scale (mm/yr): 2000")
        self.ws_precip_scale = float_slider(0.0, 20000.0, 10.0)
        self.ws_precip_scale.setValue(int((2000.0 - 0.0) / 10.0))
        self.ws_precip_scale.valueChanged.connect(
            lambda: precip_label.setText(
                f"precip_scale (mm/yr): {_sv(self.ws_precip_scale):.0f}"
            )
        )
        self.ws_precip_scale.sliderReleased.connect(self.render)
        self.weather_layout.addWidget(precip_label)
        self.weather_layout.addWidget(self.ws_precip_scale)

        self.weather_layout.addStretch()

    # ==========================================================
    # Generation
    # ==========================================================

    def random_seed(self):
        self.seed.setValue(int(np.random.randint(0, 100000)))
        self.generate()

    def generate(self):
        self._raw_heightmap, self._lat, self._lon = create_planet_heightmap(
            size=self.size.value(),
            seed=self.seed.value(),
            octaves=self.octaves.value(),
            base_freq=slider_value(self.base_freq),
            lacunarity=slider_value(self.lacunarity),
            gain=slider_value(self.gain),
            warp1=slider_value(self.warp1),
            warp2=slider_value(self.warp2),
            ridge=self.ridge.isChecked(),
            land_percentage=slider_value(self.land_percentage),
            sea_level=0.2,
            polar_strength=slider_value(self.polar_strength),
            polar_sharpness=slider_value(self.polar_sharpness),
            backside_strength=slider_value(self.backside_strength),
            backside_sharpness=slider_value(self.backside_sharpness),
            seam_lon=-np.pi,  # seam at dateline (lon ±180°)
        )
        self._weather_calculated = False
        self._ice_overlay_visible = False
        self._apply_curve()

    def _apply_curve(self):
        """Apply the height remap curve to the raw heightmap and update downstream."""
        if not hasattr(self, '_raw_heightmap'):
            return
        sea_level = 0.2
        arr = self._raw_heightmap.copy()
        land_mask = arr > sea_level
        if land_mask.any():
            land = arr[land_mask]
            land_norm = (land - sea_level) / max(1.0 - sea_level, 1e-6)
            land_remapped = self.curve_editor.apply(land_norm)
            arr[land_mask] = sea_level + land_remapped * (1.0 - sea_level)
        self.heightmap = arr
        self.map_view.set_lat_lon(self._lat, self._lon)
        self.map_view.set_contour_data(self.heightmap)
        self.layer_bar.set_contour_enabled(True)
        self._compositor.set_heightmap(self.heightmap, self._lat, self._lon)
        self._update_weather_sim()

    def _calculate_weather(self):
        self._weather_calculated = True
        self._run_weather_worker()

    def _update_weather_sim(self):
        if not hasattr(self, 'heightmap'):
            return

        if self._weather_calculated:
            # Full path (includes moisture) — run in background thread.
            self._run_weather_worker()
            return

        # Fast path: base weather only (no moisture), stays on main thread.
        ws = WeatherSim(
            self.heightmap,
            self._lat,
            self._lon,
            earth_radius_factor=slider_value(self.earth_radius_factor),
            mountain_height_km=slider_value(self.mountain_height_km),
            sim_resolution=self.sim_resolution_combo.currentData(),
            sea_level=0.2,
            polar_temp=slider_value(self.ws_polar_temp),
            equatorial_temp=slider_value(self.ws_equatorial_temp),
            lapse_rate=slider_value(self.ws_lapse_rate),
            dt=self.ws_dt.value() * 3600.0,
            sigma_diffusion=slider_value(self.ws_sigma_diffusion),
            n_steps=self.ws_n_steps.value(),
            max_transport_km=slider_value(self.ws_max_transport_km),
            total_orog_loss_km=slider_value(self.ws_total_orog_loss_km),
            precip_gamma=slider_value(self.ws_precip_gamma),
            hum_gamma=slider_value(self.ws_hum_gamma),
            precip_hum_ratio=slider_value(self.ws_precip_hum_ratio),
            seed=self.seed.value(),
            ocean_temp_noise=slider_value(self.ws_ocean_temp_noise),
        )
        self._compositor.set_weather_sim(ws)
        self.map_view.set_wind_data(ws.sim_wind_u, ws.sim_wind_v, ws.sim_wind_speed)
        self.layer_bar.set_wind_enabled(True)
        self.layer_bar.set_alpine_enabled(True)
        self.layer_bar.set_ice_enabled(False)
        self.layer_bar.set_biomes_enabled(False)
        self.render()

    def _run_weather_worker(self):
        """Snapshot current UI params and launch WeatherWorker in a background thread."""
        if self._weather_running:
            self._weather_rerun_pending = True
            return

        self._weather_running = True
        self._weather_rerun_pending = False

        params = {
            'heightmap':          self.heightmap.copy(),
            'lat':                self._lat.copy(),
            'lon':                self._lon.copy(),
            'earth_radius_factor':  slider_value(self.earth_radius_factor),
            'mountain_height_km':   slider_value(self.mountain_height_km),
            'sim_resolution':       self.sim_resolution_combo.currentData(),
            'polar_temp':           slider_value(self.ws_polar_temp),
            'equatorial_temp':      slider_value(self.ws_equatorial_temp),
            'lapse_rate':           slider_value(self.ws_lapse_rate),
            'dt':                   self.ws_dt.value() * 3600.0,
            'sigma_diffusion':      slider_value(self.ws_sigma_diffusion),
            'n_steps':              self.ws_n_steps.value(),
            'max_transport_km':     slider_value(self.ws_max_transport_km),
            'total_orog_loss_km':   slider_value(self.ws_total_orog_loss_km),
            'precip_gamma':         slider_value(self.ws_precip_gamma),
            'hum_gamma':            slider_value(self.ws_hum_gamma),
            'precip_hum_ratio':     slider_value(self.ws_precip_hum_ratio),
            'seed':                 self.seed.value(),
            'ocean_temp_noise':     slider_value(self.ws_ocean_temp_noise),
        }

        self._weather_worker = WeatherWorker(params, self)
        self._weather_worker.progress.connect(self._on_weather_progress)
        self._weather_worker.finished.connect(self._on_weather_finished)
        self._calc_btn.setEnabled(False)
        self._weather_progress.setValue(0)
        self._weather_progress.setVisible(True)
        self._weather_status_label.setText("Calculating…")
        self._weather_worker.start()

    def _on_weather_progress(self, step: int, n_steps: int):
        pct = int(100 * step / max(n_steps, 1))
        self._weather_progress.setValue(pct)
        self._weather_status_label.setText(f"Step {step} / {n_steps}")

    def _on_weather_finished(self, ws):
        self._weather_running = False
        self._calc_btn.setEnabled(True)

        if not self._weather_calculated:
            # generate() was called while worker ran — discard stale result.
            self._weather_progress.setVisible(False)
            self._weather_status_label.setText("")
            return

        self._compositor.set_weather_sim(ws)
        self.map_view.set_wind_data(ws.sim_wind_u, ws.sim_wind_v, ws.sim_wind_speed)
        self.layer_bar.set_wind_enabled(True)
        self.layer_bar.set_alpine_enabled(True)
        self.layer_bar.set_ice_enabled(True)
        self.layer_bar.set_biomes_enabled(True)
        self._weather_progress.setValue(100)
        self._weather_status_label.setText("Done")
        self.render()

        if self._weather_rerun_pending:
            self._run_weather_worker()

    # ==========================================================
    # Rendering
    # ==========================================================

    def render(self):
        if not hasattr(self, 'heightmap'):
            return

        img = self._compositor.render_base(
            base_layer=self.layer_bar.base_layer(),
            cmap_name=self.layer_bar.cmap_name(),
            clip_sea=self.clip_sea.isChecked(),
            precip_scale_mm=slider_value(self.ws_precip_scale),
            ice_overlay=self._ice_overlay_visible,
            alpine_overlay=self._alpine_overlay_visible,
        )

        self.map_view.set_image(img)
        gl_img = self.map_view.bake_wind_overlay(img) if self.map_view._wind_visible else img
        self.gl_widget.upload_texture(gl_img)

    # ==========================================================
    # Presets
    # ==========================================================

    def _preset_data(self) -> dict:
        return {
            "terrain": {
                "size":               self.size.value(),
                "seed":               self.seed.value(),
                "octaves":            self.octaves.value(),
                "base_freq":          slider_value(self.base_freq),
                "lacunarity":         slider_value(self.lacunarity),
                "gain":               slider_value(self.gain),
                "warp1":              slider_value(self.warp1),
                "warp2":              slider_value(self.warp2),
                "ridge":              self.ridge.isChecked(),
                "land_percentage":    slider_value(self.land_percentage),
                "polar_strength":     slider_value(self.polar_strength),
                "polar_sharpness":    slider_value(self.polar_sharpness),
                "backside_strength":  slider_value(self.backside_strength),
                "backside_sharpness": slider_value(self.backside_sharpness),
                "clip_sea":           self.clip_sea.isChecked(),
                "cmap":               self.layer_bar.cmap_name(),
                "curve":              [list(p) for p in self.curve_editor._points],
            }
        }

    def _apply_preset(self, data: dict):
        t = data.get("terrain", {})

        def set_slider(widget, value):
            widget.setValue(int(round((value - widget._min) / widget._step)))

        if "size"               in t: self.size.setValue(t["size"])
        if "seed"               in t: self.seed.setValue(t["seed"])
        if "octaves"            in t: self.octaves.setValue(t["octaves"])
        if "base_freq"          in t: set_slider(self.base_freq,          t["base_freq"])
        if "lacunarity"         in t: set_slider(self.lacunarity,         t["lacunarity"])
        if "gain"               in t: set_slider(self.gain,               t["gain"])
        if "warp1"              in t: set_slider(self.warp1,              t["warp1"])
        if "warp2"              in t: set_slider(self.warp2,              t["warp2"])
        if "ridge"              in t: self.ridge.setChecked(t["ridge"])
        if "land_percentage"    in t: set_slider(self.land_percentage,    t["land_percentage"])
        if "polar_strength"     in t: set_slider(self.polar_strength,     t["polar_strength"])
        if "polar_sharpness"    in t: set_slider(self.polar_sharpness,    t["polar_sharpness"])
        if "backside_strength"  in t: set_slider(self.backside_strength,  t["backside_strength"])
        if "backside_sharpness" in t: set_slider(self.backside_sharpness, t["backside_sharpness"])
        if "clip_sea"           in t: self.clip_sea.setChecked(t["clip_sea"])
        if "cmap"               in t: self.layer_bar.set_cmap(t["cmap"])
        if "curve"              in t:
            self.curve_editor._points = [list(p) for p in t["curve"]]
            self.curve_editor.update()

    def set_default_preset(self):
        with open(_DEFAULT_PRESET, "w") as f:
            yaml.dump(self._preset_data(), f, default_flow_style=False, sort_keys=False)

    def reset_to_default(self):
        if not _DEFAULT_PRESET.exists():
            return
        with open(_DEFAULT_PRESET) as f:
            data = yaml.safe_load(f)
        self._apply_preset(data)
        self.generate()

    def save_preset(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Preset", "", "YAML Files (*.yaml *.yml)"
        )
        if not filename:
            return
        with open(filename, "w") as f:
            yaml.dump(self._preset_data(), f, default_flow_style=False, sort_keys=False)

    def load_preset(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Preset", "", "YAML Files (*.yaml *.yml)"
        )
        if not filename:
            return
        with open(filename, "r") as f:
            data = yaml.safe_load(f)
        self._apply_preset(data)
        self.generate()

    # ==========================================================
    # Export
    # ==========================================================

    def export_image(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "PNG Files (*.png)"
        )
        if not filename:
            return
        from PIL import Image
        img = self._compositor.render_base(
            base_layer=self.layer_bar.base_layer(),
            cmap_name=self.layer_bar.cmap_name(),
            clip_sea=self.clip_sea.isChecked(),
            precip_scale_mm=slider_value(self.ws_precip_scale),
            ice_overlay=self._ice_overlay_visible,
            alpine_overlay=self._alpine_overlay_visible,
        )
        Image.fromarray(img).save(filename)
