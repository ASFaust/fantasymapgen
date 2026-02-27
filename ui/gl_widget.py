import numpy as np

from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtCore import QPoint
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

from OpenGL.GL import *
from OpenGL.GLU import *


class PlanetGLWidget(QOpenGLWidget):
    """Interactive texture-mapped sphere using PyOpenGL."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.texture_id = None
        self.quadric = None
        self._rot_x = 20.0
        self._rot_y = 0.0
        self._last_pos = QPoint()
        self._zoom = 3.0

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(400, 400)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self._zoom -= delta * 0.005
        self._zoom = max(1.2, min(10.0, self._zoom))
        self.update()

    def mousePressEvent(self, event):
        self._last_pos = event.position().toPoint()

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()
        self._rot_y += dx * 0.5
        self._rot_x += dy * 0.5
        self._rot_x = max(-90, min(90, self._rot_x))
        self._last_pos = pos
        self.update()

    def upload_texture(self, rgb_img: np.ndarray):
        self._pending_texture = rgb_img
        self._upload_pending()

    def _upload_pending(self):
        if not hasattr(self, '_pending_texture') or self._pending_texture is None:
            return
        if not self.isValid():
            return

        self.makeCurrent()
        img = self._pending_texture
        img_flipped = np.flipud(img).copy()
        h, w, _ = img_flipped.shape

        if self.texture_id is None:
            self.texture_id = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, img_flipped)

        self._pending_texture = None
        self.doneCurrent()
        self.update()

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        glLightfv(GL_LIGHT0, GL_POSITION, [2.0, 2.0, 3.0, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [1.0, 1.0, 1.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.15, 0.15, 0.2, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.4, 0.4, 0.4, 1.0])

        glClearColor(0.05, 0.05, 0.1, 1.0)

        self.quadric = gluNewQuadric()
        gluQuadricTexture(self.quadric, GL_TRUE)
        gluQuadricNormals(self.quadric, GLU_SMOOTH)

        self._upload_pending()

    def resizeGL(self, w, h):
        glViewport(0, 0, w, max(h, 1))
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(40.0, w / max(h, 1), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(0, 0, self._zoom, 0, 0, 0, 0, 1, 0)

        glRotatef(self._rot_x, 1, 0, 0)
        glRotatef(self._rot_y, 0, 1, 0)

        if self.texture_id is not None:
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
        else:
            glBindTexture(GL_TEXTURE_2D, 0)

        glColor3f(1, 1, 1)
        glRotatef(-90, 0, 1, 0)
        glRotatef(-90, 1, 0, 0)

        gluSphere(self.quadric, 1.0, 128, 64)
