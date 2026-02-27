import math

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QSizePolicy, QWidget


class WindMapWidget(QWidget):
    """Terrain background with wind-field arrows overlaid."""

    def __init__(self):
        super().__init__()
        self._bg: QPixmap | None = None
        self._wind_u: np.ndarray | None = None
        self._wind_v: np.ndarray | None = None
        self._wind_speed: np.ndarray | None = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def update_data(
        self,
        rgb_img: np.ndarray,
        wind_u: np.ndarray,
        wind_v: np.ndarray,
        wind_speed: np.ndarray,
    ) -> None:
        h, w, _ = rgb_img.shape
        img_copy = np.ascontiguousarray(rgb_img)
        qimg = QImage(img_copy.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._bg = QPixmap.fromImage(qimg)
        self._wind_u = wind_u
        self._wind_v = wind_v
        self._wind_speed = wind_speed
        self.update()

    def clear(self) -> None:
        self._bg = None
        self._wind_u = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._bg is None or self._wind_u is None:
            painter.setPen(QColor(180, 180, 180))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Enable the weather sim in the Weather tab\nto see wind visualisation.",
            )
            return

        # Background image scaled to fill widget
        painter.drawPixmap(self.rect(), self._bg)

        H, W = self._wind_u.shape
        ww, wh = self.width(), self.height()
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

                # Wind direction: u→east (screen right), v→north
                # South pole is at row 0 (top of image), so north = +y on screen
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

                # Shaft
                painter.drawLine(int(sx), int(sy), int(tx), int(ty))

                # Arrowhead at tip
                head = max(3.0, scale * 0.42)
                angle = math.atan2(dy, dx)
                spread = math.pi / 5
                for sign in (-1, 1):
                    ax = tx - head * math.cos(angle + sign * spread)
                    ay = ty - head * math.sin(angle + sign * spread)
                    painter.drawLine(int(tx), int(ty), int(ax), int(ay))
