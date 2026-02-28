"""
main.py
-------
Application entry point.
Run:   python main.py
"""

import sys
import os

# Ensure the project root is on the path so 'src' and 'gui' are importable
sys.path.insert(0, os.path.dirname(__file__))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from gui.main_window import MainWindow


def main():
    # High-DPI support
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

    app = QApplication(sys.argv)
    app.setApplicationName("Gem Computers AI Planner")
    app.setOrganizationName("Gem Computers")

    # Base font
    font = QFont("Segoe UI", 9)
    app.setFont(font)

    window = MainWindow()
    window.show()

    # Auto-load default data if it exists
    from pathlib import Path
    default_data = Path(__file__).parent / "Data" / "AI_DATA.CSV"
    if default_data.exists():
        window._start_load(str(default_data))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
