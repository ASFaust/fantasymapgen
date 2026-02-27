import math

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QSizePolicy, QWidget


class MapViewWidget(QWidget):
    """
    2D flat-map viewport.

    Displays a base RGB image and optionally paints wind-arrow overlays
    directly via QPainter so arrows scale with the widget size.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bg: QPixmap | None = None
        self._wind_u: np.ndarray | None = None
        self._wind_v: np.ndarray | None = None
        self._wind_speed: np.ndarray | None = None
        self._wind_visible: bool = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # ------------------------------------------------------------------
    # Data setters
    # ------------------------------------------------------------------

    def set_image(self, rgb_img: np.ndarray) -> None:
        img_copy = np.ascontiguousarray(rgb_img)
        h, w, _ = img_copy.shape
        qimg = QImage(img_copy.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._bg = QPixmap.fromImage(qimg)
        self.update()

    def set_wind_data(
        self,
        wind_u: np.ndarray,
        wind_v: np.ndarray,
        wind_speed: np.ndarray,
    ) -> None:
        self._wind_u = wind_u
        self._wind_v = wind_v
        self._wind_speed = wind_speed
        self.update()

    def set_wind_visible(self, visible: bool) -> None:
        self._wind_visible = visible
        self.update()

    def clear_wind(self) -> None:
        self._wind_u = None
        self._wind_v = None
        self._wind_speed = None
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._bg is None:
            painter.setPen(QColor(180, 180, 180))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Generating…")
            return

        painter.drawPixmap(self.rect(), self._bg)

        if self._wind_visible and self._wind_u is not None:
            self._paint_wind(painter, self.width(), self.height())

    def bake_wind_overlay(self, img: np.ndarray) -> np.ndarray:
        """Return a copy of img with wind arrows painted at image resolution."""
        if self._wind_u is None:
            return img
        h, w = img.shape[:2]
        qimg = QImage(
            np.ascontiguousarray(img).data, w, h, 3 * w, QImage.Format.Format_RGB888
        ).copy()
        painter = QPainter(qimg)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._paint_wind(painter, w, h)
        painter.end()
        ptr = qimg.bits()
        ptr.setsize(h * w * 3)
        return np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 3).copy()

    def _paint_wind(self, painter: QPainter, ww: int, wh: int) -> None:
        H, W = self._wind_u.shape
        if ww < 1 or wh < 1:
            return

        max_speed = float(self._wind_speed.max())
        if max_speed < 1e-6:
            return

        # Target ~32 px between arrow centres in widget space
        grid_px = 32
        nx = max(4, ww // grid_px)
        ny = max(4, wh // grid_px)

        xs = np.linspace(0, W - 1, nx, dtype=int)
        ys = np.linspace(0, H - 1, ny, dtype=int)

        cell_w = ww / nx
        cell_h = wh / ny
        arrow_len_px = min(cell_w, cell_h) * 0.80

        pen = QPen()
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)

        for yi in ys:
            for xi in xs:
                u = float(self._wind_u[yi, xi])
                v = float(self._wind_v[yi, xi])
                speed = float(self._wind_speed[yi, xi])

                if speed < 0.04 * max_speed:
                    continue

                t = speed / max_speed  # 0 … 1

                # Centre of this grid cell in widget pixels
                cx = (float(xi) + 0.5) / W * ww
                cy = (float(yi) + 0.5) / H * wh

                # Wind direction: u→east (screen right), v→north.
                # South pole is row 0 (top of image) so north = +y on screen.
                dx = u / speed
                dy = v / speed

                scale = t * arrow_len_px

                # Colour ramp: blue (slow) → orange-red (fast)
                r = int(t * 220)
                g = int(50 + (1 - t) * 100)
                b = int((1 - t) * 255)
                alpha = 160 + int(t * 90)
                pen.setColor(QColor(r, g, b, alpha))
                pen.setWidthF(1.0 + t * 1.8)
                painter.setPen(pen)

                # Arrow centred on (cx, cy)
                sx = cx - dx * scale * 0.5
                sy = cy - dy * scale * 0.5
                tx = cx + dx * scale * 0.5
                ty = cy + dy * scale * 0.5

                painter.drawLine(int(sx), int(sy), int(tx), int(ty))

                # Arrowhead at tip
                head = max(3.0, scale * 0.42)
                angle = math.atan2(dy, dx)
                spread = math.pi / 5
                for sign in (-1, 1):
                    ax = tx - head * math.cos(angle + sign * spread)
                    ay = ty - head * math.sin(angle + sign * spread)
                    painter.drawLine(int(tx), int(ty), int(ax), int(ay))
