"""
Production Planning Tab – demand forecasting, safety stock,
reorder points, and production schedule.
"""

from __future__ import annotations

import traceback

import pandas as pd
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QGroupBox,
    QHBoxLayout, QLabel, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox, QSplitter,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from src import production_planning as pp, plots
from gui.tabs.data_explorer_tab import PandasModel
from PyQt6.QtCore import QSortFilterProxyModel
from PyQt6.QtWidgets import QTableView


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class PlanWorker(QThread):
    finished = pyqtSignal(object, list, str)
    progress = pyqtSignal(str)

    def __init__(self, df, horizon, lead_time, service_level, method, cluster_result):
        super().__init__()
        self._df = df
        self._horizon = horizon
        self._lead_time = lead_time
        self._service_level = service_level
        self._method = method
        self._cluster_result = cluster_result

    def run(self):
        try:
            self.progress.emit("Building production plan…")
            plan = pp.build_production_plan(
                self._df, self._horizon, self._lead_time,
                self._service_level, self._method
            )
            self.progress.emit("Generating recommendations…")
            recs = pp.generate_recommendations(plan, self._cluster_result)
            self.finished.emit(plan, recs, "")
        except Exception:
            self.finished.emit(None, [], traceback.format_exc())


# ---------------------------------------------------------------------------
# Tab
# ---------------------------------------------------------------------------

class ProductionTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._df: pd.DataFrame | None = None
        self._cluster_result: pd.DataFrame | None = None
        self._plan_df: pd.DataFrame | None = None
        self._worker: PlanWorker | None = None
        self._init_ui()

    # ------------------------------------------------------------------
    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ---- Controls ----
        ctrl = QGroupBox("Planning Parameters")
        ctrl.setMaximumHeight(130)
        cl = QHBoxLayout(ctrl)

        cl.addWidget(QLabel("Method:"))
        self._method_combo = QComboBox()
        self._method_combo.addItems(["ensemble", "moving_average", "linear", "seasonal"])
        self._method_combo.setFixedWidth(150)
        cl.addWidget(self._method_combo)

        cl.addWidget(QLabel("Horizon (months):"))
        self._horizon_spin = QSpinBox()
        self._horizon_spin.setRange(1, 24)
        self._horizon_spin.setValue(6)
        self._horizon_spin.setFixedWidth(60)
        cl.addWidget(self._horizon_spin)

        cl.addWidget(QLabel("Lead Time (months):"))
        self._lt_spin = QDoubleSpinBox()
        self._lt_spin.setRange(0.5, 6.0)
        self._lt_spin.setSingleStep(0.5)
        self._lt_spin.setValue(1.0)
        self._lt_spin.setFixedWidth(70)
        cl.addWidget(self._lt_spin)

        cl.addWidget(QLabel("Service Level:"))
        self._sl_combo = QComboBox()
        self._sl_combo.addItems(["0.90", "0.95", "0.97", "0.99"])
        self._sl_combo.setCurrentIndex(1)
        self._sl_combo.setFixedWidth(70)
        cl.addWidget(self._sl_combo)

        cl.addStretch()

        self._run_btn = QPushButton("Generate Production Plan")
        self._run_btn.setFixedHeight(34)
        font = self._run_btn.font()
        font.setBold(True)
        self._run_btn.setFont(font)
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run_plan)
        cl.addWidget(self._run_btn)

        self._prog = QProgressBar()
        self._prog.setRange(0, 0)
        self._prog.hide()
        self._prog.setFixedWidth(100)
        cl.addWidget(self._prog)

        self._status = QLabel("")
        cl.addWidget(self._status)

        root.addWidget(ctrl)

        # ---- Inner tabs ----
        self._tabs = QTabWidget()

        # Plan table
        plan_w = QWidget()
        plnl = QVBoxLayout(plan_w)
        self._plan_view = QTableView()
        self._plan_view.setAlternatingRowColors(True)
        self._plan_view.setFont(QFont("Consolas", 8))
        self._plan_model = PandasModel(pd.DataFrame())
        pp_ = QSortFilterProxyModel()
        pp_.setSourceModel(self._plan_model)
        self._plan_view.setModel(pp_)
        plnl.addWidget(self._plan_view)
        export_btn = QPushButton("Export Plan to Excel")
        export_btn.clicked.connect(self._export_plan)
        plnl.addWidget(export_btn)
        self._tabs.addTab(plan_w, "Production Plan Table")

        # Gantt-style chart
        self._gantt_w = QWidget()
        self._gantt_layout = QVBoxLayout(self._gantt_w)
        self._tabs.addTab(self._gantt_w, "Schedule Chart")

        # Safety stock comparison
        self._ss_w = QWidget()
        self._ss_layout = QVBoxLayout(self._ss_w)
        self._tabs.addTab(self._ss_w, "Safety Stock")

        # Forecast chart per SKU
        fc_w = QWidget()
        fcl = QVBoxLayout(fc_w)
        fc_ctrl = QHBoxLayout()
        fc_ctrl.addWidget(QLabel("SKU:"))
        self._fc_sku_combo = QComboBox()
        self._fc_sku_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        fc_ctrl.addWidget(self._fc_sku_combo)
        btn_plot = QPushButton("Plot Forecast")
        btn_plot.clicked.connect(self._plot_sku_forecast)
        fc_ctrl.addWidget(btn_plot)
        fcl.addLayout(fc_ctrl)
        self._fc_canvas_area = QVBoxLayout()
        fcl.addLayout(self._fc_canvas_area)
        self._tabs.addTab(fc_w, "SKU Forecast")

        # Recommendations
        rec_w = QWidget()
        recl = QVBoxLayout(rec_w)
        self._rec_text = QTextEdit()
        self._rec_text.setReadOnly(True)
        self._rec_text.setFont(QFont("Segoe UI", 9))
        recl.addWidget(self._rec_text)
        self._tabs.addTab(rec_w, "Recommendations")

        root.addWidget(self._tabs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_data(self, df: pd.DataFrame):
        self._df = df
        self._run_btn.setEnabled(True)
        skus = sorted(df["c_sku"].unique().tolist())
        self._fc_sku_combo.clear()
        self._fc_sku_combo.addItems(skus)

    def set_cluster_result(self, result_df: pd.DataFrame):
        self._cluster_result = result_df

    # ------------------------------------------------------------------
    def _run_plan(self):
        if self._df is None:
            return
        self._run_btn.setEnabled(False)
        self._prog.show()
        sl = float(self._sl_combo.currentText())
        self._worker = PlanWorker(
            self._df, self._horizon_spin.value(),
            self._lt_spin.value(), sl,
            self._method_combo.currentText(),
            self._cluster_result,
        )
        self._worker.progress.connect(self._status.setText)
        self._worker.finished.connect(self._on_plan_done)
        self._worker.start()

    def _on_plan_done(self, plan_df, recs, error):
        self._prog.hide()
        self._run_btn.setEnabled(True)
        if error:
            QMessageBox.critical(self, "Planning Error", error)
            return

        self._plan_df = plan_df
        self._status.setText("Done.")

        # Table
        display_cols = [c for c in plan_df.columns
                        if not c.startswith("forecast_date")]
        self._plan_model.update_data(plan_df[display_cols])
        self._plan_view.resizeColumnsToContents()

        # Gantt
        fig1 = plots.plot_production_plan_gantt(plan_df, self._horizon_spin.value())
        self._replace_canvas(self._gantt_layout, fig1)

        # Safety stock
        fig2 = plots.plot_safety_stock_comparison(plan_df)
        self._replace_canvas(self._ss_layout, fig2)

        # Recommendations text
        lines = []
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        sorted_recs = sorted(recs, key=lambda r: priority_order.get(r["priority"], 9))
        for r in sorted_recs:
            badge = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(r["priority"], "⚪")
            lines.append(
                f"{badge} [{r['priority']}] {r['sku']}\n"
                f"   Action : {r['action']}\n"
                f"   Reason : {r['reason']}\n"
                f"   Safety Stock : {r['safety_stock']:,.0f} units\n"
                f"   Reorder Point: {r['reorder_point']:,.0f} units\n"
                f"   6-Month Forecast: {r['next_6_months_forecast']:,.0f} units\n"
            )
        self._rec_text.setPlainText("\n".join(lines) if lines else "No recommendations generated.")

    def _plot_sku_forecast(self):
        if self._df is None or self._fc_sku_combo.currentText() == "":
            return
        sku = self._fc_sku_combo.currentText()
        horizon = self._horizon_spin.value()
        try:
            from src.production_planning import _monthly_series, weighted_ensemble_forecast
            series = _monthly_series(self._df, sku)
            if len(series) < 2:
                QMessageBox.warning(self, "Insufficient Data", f"Not enough data for {sku}.")
                return
            forecast, ci_lo, ci_hi, f_dates = weighted_ensemble_forecast(series, horizon)
            fig = plots.plot_sku_forecast(
                series.index, series.values,
                f_dates, forecast, sku, ci_lo, ci_hi
            )
            self._replace_canvas(self._fc_canvas_area, fig)
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def _replace_canvas(self, layout: QVBoxLayout, fig):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

    def _export_plan(self):
        if self._plan_df is None:
            QMessageBox.warning(self, "No Plan", "Generate a production plan first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Plan", "", "Excel Files (*.xlsx)")
        if path:
            try:
                self._plan_df.to_excel(path, index=False)
                QMessageBox.information(self, "Saved", f"Plan saved to {path}")
            except Exception as exc:
                QMessageBox.critical(self, "Error", str(exc))
