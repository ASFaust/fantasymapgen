import numpy as np
import yaml
from pathlib import Path

_DEFAULT_PRESET = Path(__file__).parent.parent / "default_preset.yaml"

from PyQt6.QtWidgets import (
    QWidget, QLabel, QSlider, QVBoxLayout, QHBoxLayout,
    QSpinBox, QCheckBox, QPushButton,
    QScrollArea, QSizePolicy, QFileDialog, QComboBox, QTabWidget,
    QStackedWidget, QButtonGroup,
)
from PyQt6.QtCore import Qt

from gen.iq_warp import create_planet_heightmap
from .utils import float_slider, slider_value
from .curve_editor import CurveEditor
from .gl_widget import PlanetGLWidget
from sim.weather_sim import WeatherSim
from render.compositor import LayerCompositor
from .map_view_widget import MapViewWidget
from .layer_bar import LayerBar


class PlanetUI(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Planet Heightmap Generator")
        self.resize(1500, 900)

        self._compositor = LayerCompositor()
        self._ice_overlay_visible = False

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
        self.curve_editor.curveChanged.connect(self.render)
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
        self.sim_resolution       = _add("sim_resolution",       0.1,  1.0,   1.0,  0.01)
        self.ws_polar_temp        = _add("polar_temp (°C)",     -60.0, 20.0, -25.0,  0.5)
        self.ws_equatorial_temp   = _add("equatorial_temp (°C)",  0.0, 60.0,  27.0,  0.5)
        self.ws_shore_decay_km    = _add("shore_decay_km",      50.0, 3000.0, 600.0, 10.0)
        self.ws_lapse_rate        = _add("lapse_rate (°C/km)",   0.0, 15.0,   6.5,  0.1)
        self.ws_onshore_strength  = _add("onshore_strength",     0.0,  1.0,   0.4,  0.01)
        self.ws_orographic_strength = _add("orographic_strength", 0.0, 1.0,  0.5,  0.01)

        n_steps_label = QLabel("orographic_n_steps: 20")
        self.ws_orographic_n_steps = float_slider(1.0, 100.0, 1.0)
        self.ws_orographic_n_steps.setValue(int((20.0 - 1.0) / 1.0))
        self.ws_orographic_n_steps.valueChanged.connect(
            lambda: n_steps_label.setText(
                f"orographic_n_steps: {int(_sv(self.ws_orographic_n_steps))}"
            )
        )
        self.ws_orographic_n_steps.sliderReleased.connect(self._update_weather_sim)
        self.weather_layout.addWidget(n_steps_label)
        self.weather_layout.addWidget(self.ws_orographic_n_steps)

        # Precipitation scale — render-only, no need to re-run sim
        precip_label = QLabel("precip_scale (mm/yr): 2000")
        self.ws_precip_scale = float_slider(0.0, 5000.0, 10.0)
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
        self.heightmap, self._lat, self._lon = create_planet_heightmap(
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
            seam_lon=0.0,
        )
        self._compositor.set_heightmap(self.heightmap, self._lat, self._lon)
        self._weather_calculated = False
        self._compositor.set_weather_sim(None)
        self.map_view.clear_wind()
        self.layer_bar.set_wind_enabled(False)
        self.layer_bar.set_ice_enabled(False)
        self._ice_overlay_visible = False
        self.render()

    def _calculate_weather(self):
        self._weather_calculated = True
        self._update_weather_sim()

    def _update_weather_sim(self):
        if not hasattr(self, 'heightmap') or not self._weather_calculated:
            return

        ws = WeatherSim(
            self.heightmap,
            self._lat,
            self._lon,
            earth_radius_factor=slider_value(self.earth_radius_factor),
            mountain_height_km=slider_value(self.mountain_height_km),
            sim_resolution=slider_value(self.sim_resolution),
            sea_level=0.2,
            polar_temp=slider_value(self.ws_polar_temp),
            equatorial_temp=slider_value(self.ws_equatorial_temp),
            shore_decay_km=slider_value(self.ws_shore_decay_km),
            lapse_rate=slider_value(self.ws_lapse_rate),
            onshore_strength=slider_value(self.ws_onshore_strength),
            orographic_strength=slider_value(self.ws_orographic_strength),
            orographic_n_steps=int(slider_value(self.ws_orographic_n_steps)),
        )
        self._compositor.set_weather_sim(ws)
        self.map_view.set_wind_data(ws.sim_wind_u, ws.sim_wind_v, ws.sim_wind_speed)
        self.layer_bar.set_wind_enabled(True)
        self.layer_bar.set_ice_enabled(True)
        self.render()

    # ==========================================================
    # Rendering
    # ==========================================================

    def render(self):
        if not hasattr(self, 'heightmap'):
            return

        img = self._compositor.render_base(
            base_layer=self.layer_bar.base_layer(),
            curve_editor=self.curve_editor,
            cmap_name=self.layer_bar.cmap_name(),
            clip_sea=self.clip_sea.isChecked(),
            precip_scale_mm=slider_value(self.ws_precip_scale),
            ice_overlay=self._ice_overlay_visible,
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
            curve_editor=self.curve_editor,
            cmap_name=self.layer_bar.cmap_name(),
            clip_sea=self.clip_sea.isChecked(),
            precip_scale_mm=slider_value(self.ws_precip_scale),
            ice_overlay=self._ice_overlay_visible,
        )
        Image.fromarray(img).save(filename)
