"""
Statistical Analysis Tab – descriptive stats, seasonality, trend,
outlier detection, correlation, and Qwen 3 context export.
"""

from __future__ import annotations

import json
import traceback

import pandas as pd
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QGroupBox, QHBoxLayout,
    QLabel, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox, QSplitter,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from src import statistical_analysis as sa, plots
from gui.tabs.data_explorer_tab import PandasModel
from PyQt6.QtCore import QSortFilterProxyModel
from PyQt6.QtWidgets import QTableView, QAbstractItemView


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class StatsWorker(QThread):
    finished = pyqtSignal(dict, str)
    progress = pyqtSignal(str)

    def __init__(self, df: pd.DataFrame, top_n: int):
        super().__init__()
        self._df = df
        self._top_n = top_n

    def run(self):
        try:
            self.progress.emit("Descriptive statistics…")
            desc = sa.descriptive_stats(self._df)

            self.progress.emit("Seasonality profile…")
            season = sa.seasonality_profile(self._df)

            self.progress.emit("Trend analysis…")
            trend = sa.trend_analysis(self._df)

            self.progress.emit("Normality tests…")
            norm = sa.normality_tests(self._df, self._top_n)

            self.progress.emit("Outlier detection (IQR)…")
            outliers = sa.detect_outliers(self._df, method="iqr")

            self.progress.emit("Correlation matrix…")
            corr = sa.sku_correlation_matrix(self._df, top_n=min(self._top_n, 30))

            self.progress.emit("Building Qwen 3 context…")
            context = sa.build_qwen_context(self._df, desc, season, trend, self._top_n)

            self.finished.emit(
                {
                    "desc": desc,
                    "season": season,
                    "trend": trend,
                    "norm": norm,
                    "outliers": outliers,
                    "corr": corr,
                    "context": context,
                },
                "",
            )
        except Exception:
            self.finished.emit({}, traceback.format_exc())


# ---------------------------------------------------------------------------
# Tab
# ---------------------------------------------------------------------------

class StatsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._df: pd.DataFrame | None = None
        self._results: dict = {}
        self._worker: StatsWorker | None = None
        self._init_ui()

    # ------------------------------------------------------------------
    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ---- Controls ----
        ctrl = QGroupBox("Controls")
        ctrl.setMaximumHeight(90)
        cl = QHBoxLayout(ctrl)

        cl.addWidget(QLabel("Top-N SKUs:"))
        self._topn_spin = QSpinBox()
        self._topn_spin.setRange(5, 212)
        self._topn_spin.setValue(30)
        self._topn_spin.setFixedWidth(70)
        cl.addWidget(self._topn_spin)
        cl.addStretch()

        self._run_btn = QPushButton("Run Statistical Analysis")
        self._run_btn.setFixedHeight(34)
        font = self._run_btn.font()
        font.setBold(True)
        self._run_btn.setFont(font)
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run_analysis)
        cl.addWidget(self._run_btn)

        self._prog = QProgressBar()
        self._prog.setRange(0, 0)
        self._prog.hide()
        self._prog.setFixedWidth(120)
        cl.addWidget(self._prog)

        self._status = QLabel("")
        self._status.setFixedWidth(260)
        cl.addWidget(self._status)

        root.addWidget(ctrl)

        # ---- Inner tabs ----
        self._tabs = QTabWidget()

        # Descriptive stats table
        self._desc_view = QTableView()
        self._desc_view.setAlternatingRowColors(True)
        self._desc_view.setFont(QFont("Consolas", 8))
        self._desc_model = PandasModel(pd.DataFrame())
        dp = QSortFilterProxyModel()
        dp.setSourceModel(self._desc_model)
        self._desc_view.setModel(dp)
        self._tabs.addTab(self._desc_view, "Descriptive Stats")

        # Distribution chart
        dist_w = QWidget()
        distl = QVBoxLayout(dist_w)
        dist_ctrl = QHBoxLayout()
        dist_ctrl.addWidget(QLabel("SKU:"))
        self._dist_sku_combo = QComboBox()
        self._dist_sku_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        dist_ctrl.addWidget(self._dist_sku_combo)
        btn_dist = QPushButton("Plot")
        btn_dist.clicked.connect(self._plot_distribution)
        dist_ctrl.addWidget(btn_dist)
        distl.addLayout(dist_ctrl)
        self._dist_canvas_area = QVBoxLayout()
        distl.addLayout(self._dist_canvas_area)
        self._tabs.addTab(dist_w, "Distribution")

        # Seasonality heatmap
        self._season_w = QWidget()
        self._season_layout = QVBoxLayout(self._season_w)
        self._tabs.addTab(self._season_w, "Seasonality")

        # Trend chart
        self._trend_w = QWidget()
        self._trend_layout = QVBoxLayout(self._trend_w)
        self._tabs.addTab(self._trend_w, "Trend Analysis")

        # Correlation heatmap
        self._corr_w = QWidget()
        self._corr_layout = QVBoxLayout(self._corr_w)
        self._tabs.addTab(self._corr_w, "Correlation")

        # Outliers
        self._outlier_w = QWidget()
        outlier_l = QVBoxLayout(self._outlier_w)
        self._outlier_table = QTableView()
        self._outlier_table.setAlternatingRowColors(True)
        self._outlier_table.setFont(QFont("Consolas", 8))
        self._outlier_model = PandasModel(pd.DataFrame())
        self._outlier_table.setModel(self._outlier_model)
        outlier_l.addWidget(self._outlier_table)
        self._outlier_chart_area = QVBoxLayout()
        outlier_l.addLayout(self._outlier_chart_area)
        self._tabs.addTab(self._outlier_w, "Outliers")

        # Normality tests table
        self._norm_view = QTableView()
        self._norm_view.setAlternatingRowColors(True)
        self._norm_view.setFont(QFont("Consolas", 8))
        self._norm_model = PandasModel(pd.DataFrame())
        np_ = QSortFilterProxyModel()
        np_.setSourceModel(self._norm_model)
        self._norm_view.setModel(np_)
        self._tabs.addTab(self._norm_view, "Normality Tests")

        # Qwen 3 Context
        qwen_w = QWidget()
        qwl = QVBoxLayout(qwen_w)
        qwl.addWidget(QLabel("Structured context ready to pass to Qwen 3:"))
        self._qwen_text = QTextEdit()
        self._qwen_text.setReadOnly(True)
        self._qwen_text.setFont(QFont("Consolas", 8))
        qwl.addWidget(self._qwen_text)
        btn_row = QHBoxLayout()
        btn_save_json = QPushButton("Save as JSON")
        btn_save_json.clicked.connect(self._save_qwen_json)
        btn_copy = QPushButton("Copy to Clipboard")
        btn_copy.clicked.connect(lambda: self._qwen_text.selectAll() or self._qwen_text.copy())
        btn_row.addWidget(btn_save_json)
        btn_row.addWidget(btn_copy)
        btn_row.addStretch()
        qwl.addLayout(btn_row)
        self._tabs.addTab(qwen_w, "Qwen 3 Context")

        root.addWidget(self._tabs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_data(self, df: pd.DataFrame):
        self._df = df
        self._run_btn.setEnabled(True)
        # populate SKU combo
        self._dist_sku_combo.clear()
        top_skus = (
            df.groupby("c_sku")["c_qty"].sum()
            .nlargest(50).index.tolist()
        )
        self._dist_sku_combo.addItems(top_skus)

    # ------------------------------------------------------------------
    def _run_analysis(self):
        if self._df is None:
            return
        self._run_btn.setEnabled(False)
        self._prog.show()
        self._worker = StatsWorker(self._df, self._topn_spin.value())
        self._worker.progress.connect(self._status.setText)
        self._worker.finished.connect(self._on_done)
        self._worker.start()

    def _on_done(self, results: dict, error: str):
        self._prog.hide()
        self._run_btn.setEnabled(True)
        self._status.setText("Done." if not error else "Error!")
        if error:
            QMessageBox.critical(self, "Analysis Error", error)
            return

        self._results = results

        # Descriptive stats
        desc = results["desc"].reset_index()
        self._desc_model.update_data(desc)
        self._desc_view.resizeColumnsToContents()

        # Seasonality
        fig = plots.plot_seasonality_heatmap(results["season"])
        self._replace_canvas(self._season_layout, fig)

        # Trend
        trend_with_desc = results["trend"].join(
            results["desc"][["Total"]], how="left"
        )
        fig2 = plots.plot_trend_overview(trend_with_desc)
        self._replace_canvas(self._trend_layout, fig2)

        # Correlation
        fig3 = plots.plot_correlation_heatmap(results["corr"])
        self._replace_canvas(self._corr_layout, fig3)

        # Outliers
        self._outlier_model.update_data(
            results["outliers"][["date", "c_sku", "c_qty"]].reset_index(drop=True)
            if not results["outliers"].empty else pd.DataFrame()
        )
        fig4 = plots.plot_outlier_timeline(results["outliers"])
        self._replace_canvas(self._outlier_chart_area, fig4)

        # Normality
        self._norm_model.update_data(results["norm"].reset_index())
        self._norm_view.resizeColumnsToContents()

        # Qwen context
        ctx_str = json.dumps(results["context"], indent=2)
        self._qwen_text.setPlainText(ctx_str)

    def _plot_distribution(self):
        if self._df is None or self._dist_sku_combo.currentText() == "":
            return
        sku = self._dist_sku_combo.currentText()
        try:
            fig = plots.plot_distribution(self._df, sku)
            self._replace_canvas(self._dist_canvas_area, fig)
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def _replace_canvas(self, layout: QVBoxLayout, fig):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

    def _save_qwen_json(self):
        if "context" not in self._results:
            QMessageBox.warning(self, "No Data", "Run analysis first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Context", "", "JSON Files (*.json)")
        if path:
            sa.export_qwen_context_json(self._results["context"], path)
            QMessageBox.information(self, "Saved", f"Context saved to {path}")
