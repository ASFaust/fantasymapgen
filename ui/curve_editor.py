import numpy as np

from PyQt6.QtWidgets import QWidget, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QPainterPath


class CurveEditor(QWidget):
    """
    A draggable spline curve editor for remapping height values [0,1] -> [0,1].

    - Left-click on the curve (not a point) to add a control point.
    - Left-drag an existing point to move it.
    - Right-click a non-endpoint point to remove it.

    Uses a monotone cubic (Fritsch-Carlson) spline so the curve stays
    smooth and oscillation-free.
    """

    curveChanged = pyqtSignal()

    _POINT_RADIUS = 6
    _HIT_RADIUS   = 10

    def __init__(self, parent=None):
        super().__init__(parent)
        # Control points as [x, y] in normalised [0,1]x[0,1].
        # Always kept sorted by x; endpoints locked at x=0 and x=1.
        self._points = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
        self._drag_idx = None

        self.setMinimumSize(180, 180)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(190)

    # ----------------------------------------------------------
    # Coordinate helpers
    # ----------------------------------------------------------

    def _pad(self):
        p = 14
        return p, p, self.width() - p, self.height() - p

    def _to_widget(self, nx, ny):
        l, t, r, b = self._pad()
        x = l + nx * (r - l)
        y = b - ny * (b - t)
        return int(x), int(y)

    def _to_norm(self, wx, wy):
        l, t, r, b = self._pad()
        nx = (wx - l) / max(r - l, 1)
        ny = (b - wy) / max(b - t, 1)
        return float(np.clip(nx, 0.0, 1.0)), float(np.clip(ny, 0.0, 1.0))

    def _hit_index(self, wx, wy):
        for i, (px, py) in enumerate(self._points):
            cx, cy = self._to_widget(px, py)
            if (wx - cx) ** 2 + (wy - cy) ** 2 <= self._HIT_RADIUS ** 2:
                return i
        return None

    # ----------------------------------------------------------
    # Monotone cubic spline (Fritsch-Carlson)
    # ----------------------------------------------------------

    def _build_lut(self, n=256):
        pts = sorted(self._points, key=lambda p: p[0])
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)

        if len(pts) < 2:
            return np.linspace(0.0, 1.0, n)

        npts = len(xs)
        d = np.diff(ys) / np.maximum(np.diff(xs), 1e-10)

        m = np.zeros(npts)
        m[0]  = d[0]
        m[-1] = d[-1]
        for i in range(1, npts - 1):
            if d[i - 1] * d[i] <= 0:
                m[i] = 0.0
            else:
                m[i] = (d[i - 1] + d[i]) / 2.0

        for i in range(npts - 1):
            if abs(d[i]) < 1e-10:
                m[i] = m[i + 1] = 0.0
            else:
                alpha = m[i]     / d[i]
                beta  = m[i + 1] / d[i]
                r = alpha ** 2 + beta ** 2
                if r > 9:
                    tau = 3.0 / np.sqrt(r)
                    m[i]     = tau * alpha * d[i]
                    m[i + 1] = tau * beta  * d[i]

        t_vals = np.linspace(0.0, 1.0, n)
        result = np.zeros(n)
        for j, tv in enumerate(t_vals):
            seg = int(np.searchsorted(xs, tv, side='right')) - 1
            seg = max(0, min(seg, npts - 2))
            h = xs[seg + 1] - xs[seg]
            if h < 1e-10:
                result[j] = ys[seg]
                continue
            tt   = (tv - xs[seg]) / h
            h00  =  2*tt**3 - 3*tt**2 + 1
            h10  =    tt**3 - 2*tt**2 + tt
            h01  = -2*tt**3 + 3*tt**2
            h11  =    tt**3 -   tt**2
            result[j] = (h00*ys[seg] + h10*h*m[seg] +
                         h01*ys[seg+1] + h11*h*m[seg+1])

        return np.clip(result, 0.0, 1.0)

    def apply(self, arr: np.ndarray) -> np.ndarray:
        """Remap a float32 array in [0,1] through the curve."""
        lut = self._build_lut(256)
        indices = np.clip((arr * 255).astype(int), 0, 255)
        return lut[indices].astype(np.float32)

    # ----------------------------------------------------------
    # Mouse events
    # ----------------------------------------------------------

    def mousePressEvent(self, event):
        wx = int(event.position().x())
        wy = int(event.position().y())

        if event.button() == Qt.MouseButton.RightButton:
            idx = self._hit_index(wx, wy)
            if idx is not None and idx not in (0, len(self._points) - 1):
                self._points.pop(idx)
                self.update()
                self.curveChanged.emit()
            return

        idx = self._hit_index(wx, wy)
        if idx is not None:
            self._drag_idx = idx
        else:
            nx, ny = self._to_norm(wx, wy)
            self._points.append([nx, ny])
            self._points.sort(key=lambda p: p[0])
            # Find the newly inserted point
            for i, pt in enumerate(self._points):
                if abs(pt[0] - nx) < 0.005 and abs(pt[1] - ny) < 0.005:
                    self._drag_idx = i
                    break
            self.update()
            self.curveChanged.emit()

    def mouseMoveEvent(self, event):
        if self._drag_idx is None:
            return
        wx = int(event.position().x())
        wy = int(event.position().y())
        nx, ny = self._to_norm(wx, wy)
        idx = self._drag_idx

        if idx == 0:
            nx = 0.0
        elif idx == len(self._points) - 1:
            nx = 1.0
        else:
            lo = self._points[idx - 1][0] + 0.01
            hi = self._points[idx + 1][0] - 0.01
            nx = float(np.clip(nx, lo, hi))

        self._points[idx] = [nx, ny]
        self.update()
        self.curveChanged.emit()

    def mouseReleaseEvent(self, event):
        self._drag_idx = None

    # ----------------------------------------------------------
    # Reset
    # ----------------------------------------------------------

    def reset(self):
        self._points = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
        self.update()
        self.curveChanged.emit()

    # ----------------------------------------------------------
    # Paint
    # ----------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        l, t, r, b = self._pad()

        # Background
        painter.fillRect(self.rect(), QColor(28, 28, 32))

        # Grid
        grid_pen = QPen(QColor(55, 55, 65), 1, Qt.PenStyle.DotLine)
        painter.setPen(grid_pen)
        for i in range(1, 4):
            frac = i / 4.0
            gx = int(l + frac * (r - l))
            gy = int(b - frac * (b - t))
            painter.drawLine(gx, t, gx, b)
            painter.drawLine(l, gy, r, gy)

        # Border
        painter.setPen(QPen(QColor(80, 80, 100), 1))
        painter.drawRect(l, t, r - l, b - t)

        # Identity diagonal
        painter.setPen(QPen(QColor(55, 55, 75), 1, Qt.PenStyle.DashLine))
        painter.drawLine(*self._to_widget(0, 0), *self._to_widget(1, 1))

        # Spline
        lut = self._build_lut(256)
        path = QPainterPath()
        cx0, cy0 = self._to_widget(0.0, float(lut[0]))
        path.moveTo(cx0, cy0)
        for i in range(1, 256):
            cx, cy = self._to_widget(i / 255.0, float(lut[i]))
            path.lineTo(cx, cy)

        painter.setPen(QPen(QColor(100, 200, 255), 2))
        painter.drawPath(path)

        # Fill under curve
        fill_path = QPainterPath(path)
        fill_path.lineTo(*self._to_widget(1.0, 0.0))
        fill_path.lineTo(*self._to_widget(0.0, 0.0))
        fill_path.closeSubpath()
        painter.fillPath(fill_path, QBrush(QColor(100, 200, 255, 28)))

        # Control points
        n = len(self._points)
        for i, (nx, ny) in enumerate(self._points):
            wx, wy = self._to_widget(nx, ny)
            is_end = (i == 0 or i == n - 1)
            painter.setPen(QPen(QColor(200, 220, 255), 1))
            painter.setBrush(QBrush(QColor(100, 180, 255) if is_end
                                    else QColor(60, 140, 220)))
            r2 = self._POINT_RADIUS
            painter.drawEllipse(wx - r2, wy - r2, r2 * 2, r2 * 2)

        painter.end()
