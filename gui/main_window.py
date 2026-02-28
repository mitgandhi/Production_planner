"""
main_window.py
--------------
PyQt6 main application window.  Orchestrates data loading and coordinates
all tab widgets via signals.
"""

from __future__ import annotations

import traceback
from pathlib import Path

import pandas as pd
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QIcon, QAction, QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QLabel, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QSizePolicy,
    QStatusBar, QTabWidget, QToolBar, QVBoxLayout, QWidget,
)

from src.data_loader import load_and_preprocess, build_sku_features, summary_stats
from gui.tabs.dashboard_tab import DashboardTab
from gui.tabs.data_explorer_tab import DataExplorerTab
from gui.tabs.clustering_tab import ClusteringTab
from gui.tabs.stats_tab import StatsTab
from gui.tabs.production_tab import ProductionTab
from gui.tabs.ai_tab import AITab
from gui.model_manager import ModelManager


# ---------------------------------------------------------------------------
# Data load worker (keeps GUI responsive while reading 152k rows)
# ---------------------------------------------------------------------------

class DataLoader(QThread):
    loaded = pyqtSignal(object, object, dict, str)  # df, features, summary, error

    def __init__(self, path: str):
        super().__init__()
        self._path = path

    def run(self):
        try:
            df = load_and_preprocess(self._path)
            features = build_sku_features(df)
            summary = summary_stats(df)
            self.loaded.emit(df, features, summary, "")
        except Exception:
            self.loaded.emit(None, None, {}, traceback.format_exc())


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
}
QTabWidget::pane {
    border: 1px solid #313244;
    border-radius: 4px;
}
QTabBar::tab {
    background: #313244;
    color: #cdd6f4;
    padding: 6px 18px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #89b4fa;
    color: #1e1e2e;
    font-weight: bold;
}
QTabBar::tab:hover:!selected {
    background: #45475a;
}
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 4px;
    margin-top: 8px;
    padding: 6px;
    color: #cdd6f4;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    color: #89b4fa;
    font-weight: bold;
}
QPushButton {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: none;
    border-radius: 4px;
    padding: 4px 12px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #b4befe;
}
QPushButton:disabled {
    background-color: #45475a;
    color: #6c7086;
}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 2px 6px;
}
QTableView {
    background: #181825;
    color: #cdd6f4;
    gridline-color: #313244;
    selection-background-color: #89b4fa;
    selection-color: #1e1e2e;
    border: none;
}
QTableView QHeaderView::section {
    background: #313244;
    color: #89b4fa;
    font-weight: bold;
    border: none;
    padding: 4px;
}
QTextEdit {
    background: #181825;
    color: #cdd6f4;
    border: 1px solid #313244;
    border-radius: 4px;
}
QScrollBar:vertical {
    background: #313244;
    width: 8px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #45475a;
    border-radius: 4px;
    min-height: 20px;
}
QProgressBar {
    background: #313244;
    border: none;
    border-radius: 4px;
    height: 6px;
}
QProgressBar::chunk {
    background: #89b4fa;
    border-radius: 4px;
}
QStatusBar {
    background: #181825;
    color: #a6adc8;
}
QToolBar {
    background: #181825;
    border: none;
    spacing: 4px;
}
QLabel {
    color: #cdd6f4;
}
QCheckBox {
    color: #cdd6f4;
}
"""

LIGHT_STYLESHEET = ""  # Qt default


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._df: pd.DataFrame | None = None
        self._features: pd.DataFrame | None = None
        self._summary: dict = {}
        self._cluster_result: pd.DataFrame | None = None
        self._qwen_context: dict = {}
        self._loader: DataLoader | None = None
        self._dark_mode = True

        self._build_window()
        self._build_toolbar()
        self._build_tabs()
        self._build_statusbar()
        self._apply_theme()

        # Preload Qwen3-VL as soon as the window is built
        self._start_model_preload()

    # ------------------------------------------------------------------
    # Window skeleton
    # ------------------------------------------------------------------

    def _build_window(self):
        self.setWindowTitle("Gem Computers – Apparel AI Production Planner")
        self.resize(1440, 900)
        self.setMinimumSize(1024, 700)

    def _build_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QSize(20, 20))
        self.addToolBar(tb)

        # Load data
        act_load = QAction("Load Data", self)
        act_load.setStatusTip("Load AI_DATA.CSV (or another CSV)")
        act_load.triggered.connect(self._load_data_dialog)
        tb.addAction(act_load)

        # Reload default
        act_reload = QAction("Reload Default", self)
        act_reload.setStatusTip("Reload E:/Gem_computers/Data/AI_DATA.CSV")
        act_reload.triggered.connect(self._reload_default)
        tb.addAction(act_reload)

        tb.addSeparator()

        # Theme toggle
        self._theme_action = QAction("Light Mode", self)
        self._theme_action.triggered.connect(self._toggle_theme)
        tb.addAction(self._theme_action)

        tb.addSeparator()

        # About
        act_about = QAction("About", self)
        act_about.triggered.connect(self._show_about)
        tb.addAction(act_about)

    def _build_tabs(self):
        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.setCentralWidget(self._tabs)

        self._dashboard = DashboardTab()
        self._explorer = DataExplorerTab()
        self._clustering = ClusteringTab()
        self._stats = StatsTab()
        self._production = ProductionTab()
        self._ai = AITab()

        self._tabs.addTab(self._dashboard, "Dashboard")
        self._tabs.addTab(self._explorer, "Data Explorer")
        self._tabs.addTab(self._clustering, "Clustering & Partitioning")
        self._tabs.addTab(self._stats, "Statistical Analysis")
        self._tabs.addTab(self._production, "Production Planning")
        self._tabs.addTab(self._ai, "AI (Qwen 3)")

        # Wire signals
        self._clustering.cluster_done.connect(self._on_cluster_done)
        self._stats.context_ready.connect(self._on_stats_context_ready)
        self._production.plan_ready.connect(
            lambda df: self._ai.set_analysis_data(plan_df=df)
        )

    def _build_statusbar(self):
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._prog = QProgressBar()
        self._prog.setRange(0, 0)
        self._prog.setFixedWidth(180)
        self._prog.hide()
        self._status_bar.addPermanentWidget(self._prog)
        self._status_lbl = QLabel("Ready – Load data to begin.")
        self._status_bar.addWidget(self._status_lbl)

    def _start_model_preload(self):
        """Kick off Qwen3-VL model loading in background immediately."""
        self._status_lbl.setText("Loading Qwen3-VL model in background…")
        mm = ModelManager.get()
        mm.start_loading(callback=self._on_model_loaded)

    def _on_model_loaded(self, success: bool, message: str):
        """Called from background thread – post to main thread via signal."""
        from PyQt6.QtCore import QMetaObject, Qt, Q_ARG
        # Use invokeMethod to safely update GUI from non-GUI thread
        from PyQt6.QtCore import QMetaObject
        QMetaObject.invokeMethod(
            self, "_on_model_loaded_main",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(bool, success),
            Q_ARG(str, message),
        )

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_theme(self):
        if self._dark_mode:
            self.setStyleSheet(DARK_STYLESHEET)
            self._theme_action.setText("Light Mode")
        else:
            self.setStyleSheet(LIGHT_STYLESHEET)
            self._theme_action.setText("Dark Mode")

    def _toggle_theme(self):
        self._dark_mode = not self._dark_mode
        self._apply_theme()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "E:/Gem_computers/Data",
            "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            self._start_load(path)

    def _reload_default(self):
        default = "E:/Gem_computers/Data/AI_DATA.CSV"
        if not Path(default).exists():
            QMessageBox.warning(self, "File Not Found",
                                f"Default file not found:\n{default}")
            return
        self._start_load(default)

    def _start_load(self, path: str):
        self._status_lbl.setText(f"Loading {Path(path).name}…")
        self._prog.show()
        self._loader = DataLoader(path)
        self._loader.loaded.connect(self._on_data_loaded)
        self._loader.start()

    def _on_data_loaded(self, df, features, summary, error):
        self._prog.hide()
        if error:
            QMessageBox.critical(self, "Load Error", error)
            self._status_lbl.setText("Load failed.")
            return

        self._df = df
        self._features = features
        self._summary = summary
        self._status_lbl.setText(
            f"Loaded {summary['total_records']:,} rows | "
            f"{summary['unique_skus']} SKUs | "
            f"{summary['date_min']} → {summary['date_max']}"
        )

        # Push data to all tabs
        self._dashboard.refresh(df, summary)
        self._explorer.refresh(df)
        self._clustering.set_data(df, features)
        self._stats.set_data(df)
        self._production.set_data(df)

        # Seed AI tab with dataset summary immediately
        self._ai.set_analysis_data(summary=summary)

        # Wire stats worker to feed context to AI tab after stats run
        if hasattr(self._stats, '_worker') is False:
            self._stats._run_btn.clicked.connect(self._connect_stats_signal)

    # ------------------------------------------------------------------
    # Cross-tab signals
    # ------------------------------------------------------------------

    def _on_cluster_done(self, result_df: pd.DataFrame):
        self._cluster_result = result_df
        self._production.set_cluster_result(result_df)
        # Push cluster labels to AI tab
        self._ai.set_analysis_data(cluster_result=result_df)

    def _on_stats_context_ready(self, context: dict):
        self._qwen_context = context
        self._ai.set_analysis_data(stats_context=context)

    from PyQt6.QtCore import pyqtSlot

    @pyqtSlot(bool, str)
    def _on_model_loaded_main(self, success: bool, message: str):
        """Runs in the main (GUI) thread after model finishes loading."""
        self._ai.on_model_ready(success, message)
        if success:
            self._status_lbl.setText("Qwen3-VL model ready  •  " + self._status_lbl.text())

    # ------------------------------------------------------------------
    # About
    # ------------------------------------------------------------------

    def _connect_stats_signal(self):
        """Connect the stats worker finished signal once it's created."""
        pass  # stats_tab._on_done already calls _results; we poll via cluster_done

    def _show_about(self):
        QMessageBox.about(
            self,
            "About – Gem Computers AI Planner",
            "<b>Gem Computers – Apparel AI Production Planner</b><br><br>"
            "An industrial data analysis and production planning system "
            "for women's intimate apparel inventory.<br><br>"
            "<b>Features:</b><br>"
            "• K-Means / DBSCAN / Hierarchical clustering<br>"
            "• ABC-XYZ inventory partitioning<br>"
            "• Demand forecasting (ensemble, linear, seasonal)<br>"
            "• Statistical preprocessing for Qwen 3<br>"
            "• PyQt6 GUI with dark/light themes<br><br>"
            "Place your Qwen 3 model in <code>models/</code> and "
            "point the AI tab to it."
        )
