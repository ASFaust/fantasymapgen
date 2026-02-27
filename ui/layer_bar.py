from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QRadioButton, QCheckBox,
    QComboBox, QGroupBox, QButtonGroup,
)
from PyQt6.QtCore import pyqtSignal


class LayerBar(QWidget):
    """
    Horizontal strip below the viewport that controls:
      - which base layer is rendered (Heightmap / Temperature / Precipitation)
      - which overlays are drawn on top (Wind, …)

    Signals
    -------
    base_changed()          the base image needs to be re-rendered
    overlay_changed(name, visible)   an overlay was toggled
    """

    base_changed = pyqtSignal()
    overlay_changed = pyqtSignal(str, bool)

    CMAPS = ["terrain", "Greys_r", "gist_earth"]
    _BASE_IDS = ["heightmap", "temperature", "precipitation"]

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)

        # ── Base layer ─────────────────────────────────────────────
        base_group = QGroupBox("Base Layer")
        base_layout = QHBoxLayout(base_group)
        base_layout.setSpacing(6)

        self._btn_group = QButtonGroup(self)
        self._heightmap_btn = QRadioButton("Heightmap")
        self._heightmap_btn.setChecked(True)
        self._temp_btn = QRadioButton("Temperature")
        self._precip_btn = QRadioButton("Precipitation")

        for i, btn in enumerate([self._heightmap_btn, self._temp_btn, self._precip_btn]):
            self._btn_group.addButton(btn, i)
            base_layout.addWidget(btn)

        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(self.CMAPS)
        base_layout.addWidget(self._cmap_combo)

        layout.addWidget(base_group)

        # ── Overlays ───────────────────────────────────────────────
        overlay_group = QGroupBox("Overlays")
        overlay_layout = QHBoxLayout(overlay_group)
        overlay_layout.setSpacing(6)

        self._wind_check = QCheckBox("Wind")
        self._wind_check.setEnabled(False)  # enabled once weather sim runs
        overlay_layout.addWidget(self._wind_check)

        self._ice_check = QCheckBox("Ice")
        self._ice_check.setEnabled(False)  # enabled once weather sim runs
        overlay_layout.addWidget(self._ice_check)

        layout.addWidget(overlay_group)
        layout.addStretch()

        # ── Connections ────────────────────────────────────────────
        self._btn_group.idToggled.connect(self._on_base_toggled)
        self._cmap_combo.currentIndexChanged.connect(self._on_cmap_changed)
        self._wind_check.stateChanged.connect(
            lambda s: self.overlay_changed.emit("wind", bool(s))
        )
        self._ice_check.stateChanged.connect(
            lambda s: self.overlay_changed.emit("ice", bool(s))
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def base_layer(self) -> str:
        return self._BASE_IDS[self._btn_group.checkedId()]

    def cmap_name(self) -> str:
        return self._cmap_combo.currentText()

    def wind_visible(self) -> bool:
        return self._wind_check.isChecked()

    def set_cmap(self, name: str) -> None:
        self._cmap_combo.setCurrentText(name)

    def set_wind_enabled(self, enabled: bool) -> None:
        """Enable/disable the wind checkbox; unchecks it when disabling."""
        self._wind_check.setEnabled(enabled)
        if not enabled:
            self._wind_check.setChecked(False)

    def set_ice_enabled(self, enabled: bool) -> None:
        """Enable/disable the ice checkbox; unchecks it when disabling."""
        self._ice_check.setEnabled(enabled)
        if not enabled:
            self._ice_check.setChecked(False)

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    def _on_base_toggled(self, btn_id: int, checked: bool) -> None:
        if not checked:
            return
        # Cmap combo is only meaningful for the heightmap layer
        self._cmap_combo.setEnabled(btn_id == 0)
        self.base_changed.emit()

    def _on_cmap_changed(self) -> None:
        if self.base_layer() == "heightmap":
            self.base_changed.emit()
