"""
Dashboard Tab – KPI cards + quick overview charts.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel,
    QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from src import plots


class KPICard(QFrame):
    """A small card widget for a single KPI metric."""

    def __init__(self, title: str, value: str, colour: str = "#1f77b4", parent=None):
        super().__init__(parent)
        self.setFixedHeight(90)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(
            f"""
            QFrame {{
                background: {colour};
                border-radius: 8px;
                border: none;
            }}
            """
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        val_lbl = QLabel(value)
        val_lbl.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        val_lbl.setStyleSheet("color: white; background: transparent;")
        val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        ttl_lbl = QLabel(title)
        ttl_lbl.setFont(QFont("Segoe UI", 9))
        ttl_lbl.setStyleSheet("color: rgba(255,255,255,0.85); background: transparent;")
        ttl_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(val_lbl)
        layout.addWidget(ttl_lbl)


class DashboardTab(QWidget):
    """Tab 1 – Overview dashboard."""

    COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
               "#8c564b", "#e377c2", "#7f7f7f"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._df = None
        self._init_ui()

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # KPI row
        self._kpi_row = QHBoxLayout()
        self._kpi_row.setSpacing(10)
        root.addLayout(self._kpi_row)

        # Scrollable chart area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._chart_container = QWidget()
        self._chart_layout = QGridLayout(self._chart_container)
        self._chart_layout.setSpacing(8)
        scroll.setWidget(self._chart_container)
        root.addWidget(scroll)

        self._show_placeholder()

    def _show_placeholder(self):
        lbl = QLabel("Load data to view the dashboard.")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: grey; font-size: 14px;")
        self._chart_layout.addWidget(lbl, 0, 0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self, df, summary: dict):
        """Called by MainWindow after data is loaded / refreshed."""
        self._df = df
        self._rebuild_kpis(summary)
        self._rebuild_charts()

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _rebuild_kpis(self, summary: dict):
        # Clear old cards
        while self._kpi_row.count():
            item = self._kpi_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        kpis = [
            ("Total Records", f"{summary['total_records']:,}"),
            ("Unique SKUs", f"{summary['unique_skus']:,}"),
            ("Unique Sizes", f"{summary['unique_sizes']}"),
            ("Unique Colors", f"{summary['unique_colors']}"),
            ("Total Units", f"{summary['total_units']:,}"),
            ("Years Covered", f"{summary['years_covered']}"),
            ("Date From", summary["date_min"]),
            ("Date To", summary["date_max"]),
        ]
        for i, (title, val) in enumerate(kpis):
            card = KPICard(title, val, self.COLOURS[i % len(self.COLOURS)])
            self._kpi_row.addWidget(card)

    def _rebuild_charts(self):
        # Clear old widgets
        while self._chart_layout.count():
            item = self._chart_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        df = self._df

        chart_funcs = [
            (plots.plot_monthly_total, (df,), 0, 0, 1, 2),
            (plots.plot_annual_trend, (df,), 1, 0, 1, 1),
            (plots.plot_top_skus, (df,), 1, 1, 1, 1),
            (plots.plot_size_distribution, (df,), 2, 0, 1, 1),
            (plots.plot_color_distribution, (df,), 2, 1, 1, 1),
        ]

        for fn, args, row, col, rspan, cspan in chart_funcs:
            try:
                fig = fn(*args)
                canvas = FigureCanvas(fig)
                canvas.setMinimumHeight(280)
                self._chart_layout.addWidget(canvas, row, col, rspan, cspan)
            except Exception as exc:
                err = QLabel(f"Chart error: {exc}")
                err.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self._chart_layout.addWidget(err, row, col, rspan, cspan)
