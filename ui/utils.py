from PyQt6.QtWidgets import QSlider
from PyQt6.QtCore import Qt


def float_slider(min_val, max_val, step=0.01):
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setMinimum(0)
    slider.setMaximum(int((max_val - min_val) / step))
    slider._min = min_val
    slider._step = step
    return slider


def slider_value(slider):
    return slider._min + slider.value() * slider._step
