import math

import numpy as np
from PyQt6.QtCore import Qt, QLineF, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF, QTransform
from PyQt6.QtWidgets import QSizePolicy, QWidget


# Marching-squares segment lookup.
# Bit encoding: bit0=tl, bit1=tr, bit2=bl, bit3=br  (1 = above level)
# Edge indices:  0=top, 1=right, 2=bottom, 3=left
_MS_SEGS: dict[int, list[tuple[int, int]]] = {
    1:  [(0, 3)],
    2:  [(0, 1)],
    3:  [(3, 1)],
    4:  [(2, 3)],
    5:  [(0, 2)],
    6:  [(0, 1), (3, 2)],   # saddle
    7:  [(1, 2)],
    8:  [(1, 2)],
    9:  [(0, 3), (1, 2)],   # saddle
    10: [(0, 2)],
    11: [(3, 2)],
    12: [(3, 1)],
    13: [(0, 1)],
    14: [(0, 3)],
}


class MapViewWidget(QWidget):
    """
    2D flat-map viewport.

    Displays a base RGB image and optionally paints wind-arrow overlays
    directly via QPainter so arrows scale with the widget size.

    Supports scroll-wheel zoom and left-drag pan.  Double-click resets view.
    """

    # Emitted on mouse hover with (row_fraction, col_fraction) in [0,1].
    # (-1, -1) means the cursor left the widget.
    hover = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._bg: QPixmap | None = None
        self._wind_u: np.ndarray | None = None
        self._wind_v: np.ndarray | None = None
        self._wind_speed: np.ndarray | None = None
        self._wind_visible: bool = False
        self._heightmap: np.ndarray | None = None
        self._contour_visible: bool = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Zoom / pan state
        self._zoom: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0
        self._drag_start: QPointF | None = None
        self._drag_pan_start: tuple[float, float] = (0.0, 0.0)

        # Distance measurement (right-click drag)
        self._lat_grid: np.ndarray | None = None
        self._lon_grid: np.ndarray | None = None
        self._measure_start_frac: tuple[float, float] | None = None  # (col_frac, row_frac)
        self._measure_end_frac: tuple[float, float] | None = None

        self.setCursor(Qt.CursorShape.OpenHandCursor)

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

    def set_lat_lon(self, lat: np.ndarray, lon: np.ndarray) -> None:
        self._lat_grid = lat
        self._lon_grid = lon

    def set_contour_data(self, heightmap: np.ndarray) -> None:
        self._heightmap = heightmap
        self.update()

    def set_contour_visible(self, visible: bool) -> None:
        self._contour_visible = visible
        self.update()

    # ------------------------------------------------------------------
    # Interaction: zoom + pan
    # ------------------------------------------------------------------

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        pos = event.position()
        new_zoom = max(0.1, min(50.0, self._zoom * factor))
        ratio = new_zoom / self._zoom
        self._pan_x = pos.x() - ratio * (pos.x() - self._pan_x)
        self._pan_y = pos.y() - ratio * (pos.y() - self._pan_y)
        self._zoom = new_zoom
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.position()
            self._drag_pan_start = (self._pan_x, self._pan_y)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.RightButton:
            frac = self._widget_frac(event.position())
            if frac is not None:
                self._measure_start_frac = frac
                self._measure_end_frac = None
                self.update()

    def mouseMoveEvent(self, event):
        if self._drag_start is not None and event.buttons() & Qt.MouseButton.LeftButton:
            d = event.position() - self._drag_start
            self._pan_x = self._drag_pan_start[0] + d.x()
            self._pan_y = self._drag_pan_start[1] + d.y()
            self.update()
        if event.buttons() & Qt.MouseButton.RightButton and self._measure_start_frac is not None:
            frac = self._widget_frac(event.position(), clamp=True)
            if frac is not None:
                self._measure_end_frac = frac
                self.update()
        self._emit_hover(event.position())

    def leaveEvent(self, event):
        self.hover.emit(-1.0, -1.0)

    def _emit_hover(self, pos: QPointF):
        if self._bg is None:
            return
        inv, ok = self._view_transform().inverted()
        if not ok:
            return
        p = inv.map(pos)
        # p is in the coordinate space where (0,0)-(width,height) covers the image
        col_frac = p.x() / self.width()
        row_frac = p.y() / self.height()
        if 0 <= row_frac <= 1 and 0 <= col_frac <= 1:
            self.hover.emit(row_frac, col_frac)
        else:
            self.hover.emit(-1.0, -1.0)

    def _widget_frac(self, pos: QPointF, clamp: bool = False) -> tuple[float, float] | None:
        """Convert widget pixel pos to (col_frac, row_frac). Returns None if out of bounds."""
        if self._bg is None:
            return None
        inv, ok = self._view_transform().inverted()
        if not ok:
            return None
        p = inv.map(pos)
        col_frac = p.x() / self.width()
        row_frac = p.y() / self.height()
        if clamp:
            return max(0.0, min(1.0, col_frac)), max(0.0, min(1.0, row_frac))
        if 0 <= col_frac <= 1 and 0 <= row_frac <= 1:
            return col_frac, row_frac
        return None

    def _frac_to_widget_pos(self, col_frac: float, row_frac: float) -> QPointF:
        """Convert image-fraction coords to widget pixel coords."""
        p = QPointF(col_frac * self.width(), row_frac * self.height())
        return self._view_transform().map(p)

    def _haversine_km(self, row_frac1: float, col_frac1: float,
                      row_frac2: float, col_frac2: float) -> float:
        if self._lat_grid is None:
            return 0.0
        lat_min = float(self._lat_grid[0, 0])
        lat_range = float(self._lat_grid[-1, 0]) - lat_min
        lon_min = float(self._lon_grid[0, 0])
        lat1 = lat_min + row_frac1 * lat_range
        lon1 = lon_min + col_frac1 * 2.0 * math.pi
        lat2 = lat_min + row_frac2 * lat_range
        lon2 = lon_min + col_frac2 * 2.0 * math.pi
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 2.0 * math.asin(math.sqrt(min(a, 1.0))) * 6371.0

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def contextMenuEvent(self, event):
        event.accept()  # suppress OS context menu so right-drag works cleanly

    def mouseDoubleClickEvent(self, event):
        """Double-click resets zoom and pan."""
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def _view_transform(self) -> QTransform:
        t = QTransform()
        t.translate(self._pan_x, self._pan_y)
        t.scale(self._zoom, self._zoom)
        return t

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._bg is None:
            painter.setPen(QColor(180, 180, 180))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Generating…")
            return

        painter.setTransform(self._view_transform())
        painter.drawPixmap(
            QRectF(0, 0, self.width(), self.height()),
            self._bg,
            QRectF(self._bg.rect()),
        )

        if self._contour_visible and self._heightmap is not None:
            self._paint_contour(painter, self.width(), self.height())

        if self._wind_visible and self._wind_u is not None:
            self._paint_wind(painter, self.width(), self.height())

        if self._measure_start_frac is not None and self._measure_end_frac is not None:
            self._paint_ruler(painter)

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
        pen.setCosmetic(True)  # keep line width in screen pixels under zoom

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

    def _paint_contour(self, painter: QPainter, ww: int, wh: int) -> None:
        hm = self._heightmap
        if hm is None or ww < 1 or wh < 1:
            return

        H0, W0 = hm.shape
        # Downsample to ≤256 in each dimension for speed
        step = max(1, max(H0, W0) // 256)
        hm = hm[::step, ::step]
        H, W = hm.shape
        if H < 2 or W < 2:
            return

        sea = 0.2
        h_max = float(hm.max())
        if h_max <= sea:
            return

        n_lines = 10
        levels = np.linspace(sea, h_max, n_lines + 2)[1:-1]

        # Map grid index → screen pixel
        sc_x = ww / (W - 1)
        sc_y = wh / (H - 1)

        # Corner height arrays for every cell, shape (H-1, W-1)
        htl = hm[:-1, :-1].astype(np.float32)
        htr = hm[:-1, 1:].astype(np.float32)
        hbl = hm[1:, :-1].astype(np.float32)
        hbr = hm[1:, 1:].astype(np.float32)

        pen = QPen()
        pen.setCosmetic(True)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setColor(QColor(0, 0, 0, 110))
        pen.setWidthF(0.8)
        painter.setPen(pen)

        def lerp(va: np.ndarray, vb: np.ndarray, lv: float) -> np.ndarray:
            diff = vb - va
            return np.where(np.abs(diff) > 1e-8, (lv - va) / diff, np.float32(0.5))

        for level in levels:
            atl = (htl >= level).astype(np.uint8)
            atr = (htr >= level).astype(np.uint8)
            abl = (hbl >= level).astype(np.uint8)
            abr = (hbr >= level).astype(np.uint8)
            case = atl | (atr << 1) | (abl << 2) | (abr << 3)

            lines: list[QLineF] = []

            for cv, segs in _MS_SEGS.items():
                ri, ci = np.where(case == cv)
                if len(ri) == 0:
                    continue

                ha = htl[ri, ci]
                hb = htr[ri, ci]
                hc = hbl[ri, ci]
                hd = hbr[ri, ci]

                # Screen coordinates of each edge crossing (shape: N)
                ep_x = [
                    (ci + lerp(ha, hb, level)) * sc_x,   # 0: top
                    (ci + 1) * sc_x,                      # 1: right
                    (ci + lerp(hc, hd, level)) * sc_x,   # 2: bottom
                    ci * sc_x,                            # 3: left
                ]
                ep_y = [
                    ri * sc_y,                            # 0: top
                    (ri + lerp(hb, hd, level)) * sc_y,   # 1: right
                    (ri + 1) * sc_y,                      # 2: bottom
                    (ri + lerp(ha, hc, level)) * sc_y,   # 3: left
                ]

                for e1, e2 in segs:
                    x1s, y1s = ep_x[e1], ep_y[e1]
                    x2s, y2s = ep_x[e2], ep_y[e2]
                    lines.extend(
                        QLineF(float(x1s[k]), float(y1s[k]),
                               float(x2s[k]), float(y2s[k]))
                        for k in range(len(ri))
                    )

            if lines:
                painter.drawLines(lines)

    def _great_circle_fracs(self, col_frac1: float, row_frac1: float,
                             col_frac2: float, row_frac2: float,
                             n: int = 64) -> list[tuple[float, float]]:
        """Return (col_frac, row_frac) points along the great-circle arc."""
        if self._lat_grid is None:
            return [(col_frac1, row_frac1), (col_frac2, row_frac2)]

        # Equirectangular forward/inverse using grid corners — no integer snap.
        lat_min = float(self._lat_grid[0, 0])
        lat_range = float(self._lat_grid[-1, 0]) - lat_min   # ≈ π
        lon_min = float(self._lon_grid[0, 0])
        lon_range = 2.0 * math.pi                            # full planet

        def to_latlon(cf: float, rf: float) -> tuple[float, float]:
            return lat_min + rf * lat_range, lon_min + cf * lon_range

        def to_frac(lat: float, lon: float) -> tuple[float, float]:
            lon = lon_min + (lon - lon_min) % lon_range
            return (lon - lon_min) / lon_range, (lat - lat_min) / lat_range

        lat1, lon1 = to_latlon(col_frac1, row_frac1)
        lat2, lon2 = to_latlon(col_frac2, row_frac2)

        # Unit vectors on sphere
        v1 = np.array([math.cos(lat1) * math.cos(lon1),
                       math.cos(lat1) * math.sin(lon1),
                       math.sin(lat1)])
        v2 = np.array([math.cos(lat2) * math.cos(lon2),
                       math.cos(lat2) * math.sin(lon2),
                       math.sin(lat2)])

        omega = math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2)))))
        if omega < 1e-6:
            return [(col_frac1, row_frac1), (col_frac2, row_frac2)]
        sin_omega = math.sin(omega)

        pts = []
        for i in range(n + 1):
            t = i / n
            f1 = math.sin((1 - t) * omega) / sin_omega
            f2 = math.sin(t * omega) / sin_omega
            v = f1 * v1 + f2 * v2
            lat = math.asin(max(-1.0, min(1.0, float(v[2]))))
            lon = math.atan2(float(v[1]), float(v[0]))
            pts.append(to_frac(lat, lon))
        return pts

    def _paint_ruler(self, painter: QPainter) -> None:
        """Draw the distance-measurement geodesic in screen space."""
        painter.resetTransform()

        arc_fracs = self._great_circle_fracs(*self._measure_start_frac, *self._measure_end_frac)
        p1 = self._frac_to_widget_pos(*self._measure_start_frac)
        p2 = self._frac_to_widget_pos(*self._measure_end_frac)

        # Geodesic polyline — split at antimeridian crossings (col_frac jump > 0.5)
        pen = QPen(QColor(220, 30, 30))
        pen.setWidthF(2.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        segment: list[QPointF] = []
        prev_cf = None
        for cf, rf in arc_fracs:
            if prev_cf is not None and abs(cf - prev_cf) > 0.5:
                if len(segment) >= 2:
                    painter.drawPolyline(QPolygonF(segment))
                segment = []
            segment.append(self._frac_to_widget_pos(cf, rf))
            prev_cf = cf
        if len(segment) >= 2:
            painter.drawPolyline(QPolygonF(segment))

        # Endpoint dots
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(220, 30, 30))
        for pt in (p1, p2):
            painter.drawEllipse(pt, 4.0, 4.0)

        # Distance label near midpoint
        dist_km = self._haversine_km(
            self._measure_start_frac[1], self._measure_start_frac[0],
            self._measure_end_frac[1], self._measure_end_frac[0],
        )
        if dist_km >= 1000:
            label = f"{dist_km / 1000:.2f}k km"
        elif dist_km >= 10:
            label = f"{dist_km:.0f} km"
        else:
            label = f"{dist_km:.1f} km"

        mid = QPointF((p1.x() + p2.x()) / 2 + 6, (p1.y() + p2.y()) / 2 - 7)
        painter.setPen(QColor(0, 0, 0, 180))
        painter.drawText(mid + QPointF(1, 1), label)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(mid, label)
