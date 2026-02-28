"""
Data Explorer Tab – filterable table view of the raw dataset.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PyQt6.QtCore import Qt, QSortFilterProxyModel, QAbstractTableModel, QModelIndex
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QAbstractItemView, QComboBox, QFileDialog, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QSizePolicy, QSplitter, QTableView, QTextEdit, QVBoxLayout, QWidget,
)


# ---------------------------------------------------------------------------
# Pandas ↔ Qt table model
# ---------------------------------------------------------------------------

class PandasModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self._df = df

    def rowCount(self, parent=QModelIndex()):
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return len(self._df.columns)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            val = self._df.iloc[index.row(), index.column()]
            if isinstance(val, float):
                return f"{val:,.2f}"
            return str(val)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._df.columns[section])
            return str(section + 1)
        return None

    def update_data(self, df: pd.DataFrame):
        self.beginResetModel()
        self._df = df
        self.endResetModel()


# ---------------------------------------------------------------------------
# Tab widget
# ---------------------------------------------------------------------------

class DataExplorerTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._df_full: pd.DataFrame | None = None
        self._df_view: pd.DataFrame | None = None
        self._init_ui()

    # ------------------------------------------------------------------
    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ---- Filter bar ----
        fbar = QGroupBox("Filters")
        fbar.setMaximumHeight(120)
        fgrid = QHBoxLayout(fbar)

        self._sku_combo = QComboBox()
        self._sku_combo.setEditable(True)
        self._sku_combo.setPlaceholderText("All SKUs")
        self._sku_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._color_combo = QComboBox()
        self._color_combo.setEditable(True)
        self._color_combo.setPlaceholderText("All Colors")
        self._color_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._size_combo = QComboBox()
        self._size_combo.setEditable(True)
        self._size_combo.setPlaceholderText("All Sizes")
        self._size_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._year_combo = QComboBox()
        self._year_combo.setEditable(True)
        self._year_combo.setPlaceholderText("All Years")
        self._year_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        btn_filter = QPushButton("Apply")
        btn_filter.setFixedWidth(80)
        btn_filter.clicked.connect(self._apply_filter)

        btn_reset = QPushButton("Reset")
        btn_reset.setFixedWidth(80)
        btn_reset.clicked.connect(self._reset_filter)

        for lbl_text, widget in [
            ("SKU:", self._sku_combo),
            ("Color:", self._color_combo),
            ("Size:", self._size_combo),
            ("Year:", self._year_combo),
        ]:
            fgrid.addWidget(QLabel(lbl_text))
            fgrid.addWidget(widget)

        fgrid.addWidget(btn_filter)
        fgrid.addWidget(btn_reset)
        root.addWidget(fbar)

        # ---- Splitter: table | stats ----
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Table
        self._table_view = QTableView()
        self._table_view.setSortingEnabled(True)
        self._table_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table_view.setAlternatingRowColors(True)
        self._table_view.setFont(QFont("Consolas", 8))
        self._model = PandasModel(pd.DataFrame())
        self._proxy = QSortFilterProxyModel()
        self._proxy.setSourceModel(self._model)
        self._table_view.setModel(self._proxy)
        splitter.addWidget(self._table_view)

        # Stats panel
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)
        rl.addWidget(QLabel("Quick Stats"))
        self._stats_box = QTextEdit()
        self._stats_box.setReadOnly(True)
        self._stats_box.setFont(QFont("Consolas", 8))
        self._stats_box.setMinimumWidth(240)
        rl.addWidget(self._stats_box)
        splitter.addWidget(right)
        splitter.setSizes([900, 300])

        root.addWidget(splitter)

        # ---- Bottom bar ----
        bbar = QHBoxLayout()
        self._row_count_lbl = QLabel("Rows: 0")
        bbar.addWidget(self._row_count_lbl)
        bbar.addStretch()

        btn_csv = QPushButton("Export CSV")
        btn_csv.clicked.connect(lambda: self._export("csv"))
        btn_excel = QPushButton("Export Excel")
        btn_excel.clicked.connect(lambda: self._export("xlsx"))
        bbar.addWidget(btn_csv)
        bbar.addWidget(btn_excel)
        root.addLayout(bbar)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self, df: pd.DataFrame):
        self._df_full = df
        self._df_view = df

        self._populate_combos()
        self._update_table(df)

    # ------------------------------------------------------------------
    def _populate_combos(self):
        df = self._df_full
        for combo, values in [
            (self._sku_combo, sorted(df["c_sku"].unique().tolist())),
            (self._color_combo, sorted(df["c_cl"].dropna().unique().tolist())),
            (self._size_combo, sorted(df["c_sz"].dropna().unique().tolist())),
            (self._year_combo, sorted(df["year"].unique().astype(str).tolist())),
        ]:
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("")
            combo.addItems(values)
            combo.blockSignals(False)

    def _apply_filter(self):
        df = self._df_full.copy()
        sku = self._sku_combo.currentText().strip()
        color = self._color_combo.currentText().strip()
        size = self._size_combo.currentText().strip()
        year = self._year_combo.currentText().strip()

        if sku:
            df = df[df["c_sku"] == sku]
        if color:
            df = df[df["c_cl"] == color]
        if size:
            df = df[df["c_sz"] == size]
        if year:
            df = df[df["year"].astype(str) == year]

        self._df_view = df
        self._update_table(df)

    def _reset_filter(self):
        for combo in [self._sku_combo, self._color_combo, self._size_combo, self._year_combo]:
            combo.setCurrentIndex(0)
        self._df_view = self._df_full
        self._update_table(self._df_full)

    def _update_table(self, df: pd.DataFrame):
        display_cols = [c for c in ["date", "c_sku", "c_sz", "c_cl", "c_qty", "year", "quarter"]
                        if c in df.columns]
        self._model.update_data(df[display_cols])
        self._row_count_lbl.setText(f"Rows: {len(df):,}")
        self._update_stats(df)

    def _update_stats(self, df: pd.DataFrame):
        lines = [
            f"Rows:          {len(df):,}",
            f"SKUs:          {df['c_sku'].nunique()}",
            f"Total Units:   {df['c_qty'].sum():,}",
            f"Avg Qty/Row:   {df['c_qty'].mean():.1f}",
            f"Max Qty:       {df['c_qty'].max():,}",
            f"Min Qty:       {df['c_qty'].min():,}",
        ]
        if "year" in df.columns:
            lines += [
                f"Year range:    {df['year'].min()} – {df['year'].max()}",
            ]
        self._stats_box.setPlainText("\n".join(lines))

    def _export(self, fmt: str):
        if self._df_view is None or self._df_view.empty:
            QMessageBox.warning(self, "Export", "No data to export.")
            return
        ext = f"*.{fmt}"
        path, _ = QFileDialog.getSaveFileName(self, "Export Data", "", f"{fmt.upper()} Files ({ext})")
        if not path:
            return
        try:
            if fmt == "csv":
                self._df_view.to_csv(path, index=False)
            else:
                self._df_view.to_excel(path, index=False)
            QMessageBox.information(self, "Export", f"Saved to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))
