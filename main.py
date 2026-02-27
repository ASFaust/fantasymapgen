#!/usr/bin/env python3

import sys
from PyQt6.QtWidgets import QApplication
from ui.planet_ui import PlanetUI


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlanetUI()
    window.show()
    sys.exit(app.exec())
