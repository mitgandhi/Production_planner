"""
main.py
-------
Application entry point.
Run:   python main.py
"""

import sys
import os

# ── Windows CUDA DLL path fix ──────────────────────────────────────────────────
# Must happen BEFORE any torch / CUDA import to avoid:
#   "a dynamic dll initialization routine failed"
# Python 3.8+ on Windows uses os.add_dll_directory() to extend the DLL
# search path – the standard PATH is ignored for DLL resolution in newer Python.
if sys.platform == "win32":
    _cuda_candidates = []

    # 1) Use CUDA_PATH / CUDA_PATH_V* env vars that the CUDA installer sets
    for _key, _val in os.environ.items():
        if _key.startswith("CUDA_PATH") and os.path.isdir(_val):
            _cuda_candidates.append(os.path.join(_val, "bin"))
            _cuda_candidates.append(os.path.join(_val, "libnvvp"))

    # 2) Try common installation paths (CUDA 12.x)
    for _ver in ("12.9", "12.8", "12.7", "12.6", "12.5", "12.4", "12.3", "12.2", "12.1", "12.0", "11.8"):
        _p = rf"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{_ver}\bin"
        if os.path.isdir(_p):
            _cuda_candidates.append(_p)

    # 3) Also add the torch DLL directory (inside the venv / site-packages)
    try:
        import torch  # pre-import on main thread – critical for Windows DLL init
        _torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(_torch_lib):
            _cuda_candidates.insert(0, _torch_lib)
    except Exception:
        pass  # torch not installed yet – will fail gracefully later

    # Register all candidate directories
    for _d in _cuda_candidates:
        if os.path.isdir(_d):
            try:
                os.add_dll_directory(_d)
            except Exception:
                pass  # ignore duplicates / permission errors

    # Also trigger CUDA initialisation on the main thread so background
    # threads don't hit the "DLL init" race condition
    try:
        import torch as _torch
        _torch.cuda.is_available()   # forces libcuda.so / nvcuda.dll init
    except Exception:
        pass

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
