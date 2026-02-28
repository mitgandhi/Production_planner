"""
Clustering Tab – run & visualise clustering algorithms + ABC-XYZ.
"""

from __future__ import annotations

import traceback

import pandas as pd
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QGroupBox,
    QHBoxLayout, QLabel, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox, QSplitter, QTabWidget,
    QTextEdit, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from src import clustering as cl, plots


# ---------------------------------------------------------------------------
# Worker thread for clustering (keeps GUI responsive)
# ---------------------------------------------------------------------------

class ClusterWorker(QThread):
    finished = pyqtSignal(object, object, str)  # result_df, coords_info, error_msg
    progress = pyqtSignal(str)

    def __init__(self, features: pd.DataFrame, df: pd.DataFrame,
                 algorithm: str, n_clusters: int,
                 eps: float, min_samples: int, linkage: str):
        super().__init__()
        self._features = features
        self._df = df
        self._algorithm = algorithm
        self._n_clusters = n_clusters
        self._eps = eps
        self._min_samples = min_samples
        self._linkage = linkage

    def run(self):
        try:
            feat = self._features
            self.progress.emit("Running ABC-XYZ analysis…")
            abc = cl.abc_analysis(feat)
            xyz = cl.xyz_analysis(feat)

            self.progress.emit(f"Running {self._algorithm}…")
            if self._algorithm == "K-Means":
                labels, score, _, scaler = cl.run_kmeans(feat, self._n_clusters)
            elif self._algorithm == "DBSCAN":
                labels, scaler = cl.run_dbscan(feat, self._eps, self._min_samples)
                score = 0.0
            else:
                labels, scaler = cl.run_hierarchical(feat, self._n_clusters, self._linkage)
                score = 0.0

            self.progress.emit("Computing PCA coordinates…")
            coords, var_ratio = cl.pca_coords(feat, scaler)
            label_map = cl.label_clusters(feat, labels)

            self.progress.emit("Building result table…")
            result_df = cl.build_cluster_result(feat, labels, self._algorithm, abc, xyz)
            result_df["silhouette"] = score

            self.finished.emit(
                result_df,
                {"coords": coords, "var_ratio": var_ratio,
                 "labels": labels, "label_map": label_map, "score": score},
                "",
            )
        except Exception:
            self.finished.emit(None, None, traceback.format_exc())


# ---------------------------------------------------------------------------
# Tab widget
# ---------------------------------------------------------------------------

class ClusteringTab(QWidget):
    cluster_done = pyqtSignal(object)   # emits result_df to other tabs

    def __init__(self, parent=None):
        super().__init__(parent)
        self._features: pd.DataFrame | None = None
        self._df: pd.DataFrame | None = None
        self._result_df: pd.DataFrame | None = None
        self._worker: ClusterWorker | None = None
        self._init_ui()

    # ------------------------------------------------------------------
    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ---- Left control panel ----
        left = QWidget()
        left.setFixedWidth(270)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(8)

        # Algorithm
        algo_box = QGroupBox("Algorithm")
        algl = QVBoxLayout(algo_box)
        self._algo_combo = QComboBox()
        self._algo_combo.addItems(["K-Means", "DBSCAN", "Hierarchical"])
        self._algo_combo.currentTextChanged.connect(self._on_algo_changed)
        algl.addWidget(self._algo_combo)
        ll.addWidget(algo_box)

        # K-Means params
        self._kmeans_box = QGroupBox("K-Means Parameters")
        kml = QVBoxLayout(self._kmeans_box)
        kml.addWidget(QLabel("Number of clusters (K):"))
        self._k_spin = QSpinBox()
        self._k_spin.setRange(2, 20)
        self._k_spin.setValue(5)
        kml.addWidget(self._k_spin)
        ll.addWidget(self._kmeans_box)

        # DBSCAN params
        self._dbscan_box = QGroupBox("DBSCAN Parameters")
        self._dbscan_box.hide()
        dbl = QVBoxLayout(self._dbscan_box)
        dbl.addWidget(QLabel("Epsilon (eps):"))
        self._eps_spin = QDoubleSpinBox()
        self._eps_spin.setRange(0.1, 10.0)
        self._eps_spin.setSingleStep(0.1)
        self._eps_spin.setValue(1.2)
        dbl.addWidget(self._eps_spin)
        dbl.addWidget(QLabel("Min Samples:"))
        self._min_samples_spin = QSpinBox()
        self._min_samples_spin.setRange(2, 20)
        self._min_samples_spin.setValue(3)
        dbl.addWidget(self._min_samples_spin)
        ll.addWidget(self._dbscan_box)

        # Hierarchical params
        self._hc_box = QGroupBox("Hierarchical Parameters")
        self._hc_box.hide()
        hcl = QVBoxLayout(self._hc_box)
        hcl.addWidget(QLabel("Number of clusters:"))
        self._hc_k_spin = QSpinBox()
        self._hc_k_spin.setRange(2, 20)
        self._hc_k_spin.setValue(5)
        hcl.addWidget(self._hc_k_spin)
        hcl.addWidget(QLabel("Linkage:"))
        self._linkage_combo = QComboBox()
        self._linkage_combo.addItems(["ward", "complete", "average", "single"])
        hcl.addWidget(self._linkage_combo)
        ll.addWidget(self._hc_box)

        # Elbow analysis
        btn_elbow = QPushButton("Elbow / Silhouette Analysis")
        btn_elbow.clicked.connect(self._run_elbow)
        ll.addWidget(btn_elbow)

        # Run button
        self._run_btn = QPushButton("Run Clustering")
        self._run_btn.setFixedHeight(38)
        font = self._run_btn.font()
        font.setBold(True)
        self._run_btn.setFont(font)
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run_clustering)
        ll.addWidget(self._run_btn)

        # Progress
        self._prog_bar = QProgressBar()
        self._prog_bar.setRange(0, 0)  # indeterminate
        self._prog_bar.hide()
        ll.addWidget(self._prog_bar)

        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        ll.addWidget(self._status_lbl)

        # Score display
        self._score_lbl = QLabel("")
        self._score_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        ll.addWidget(self._score_lbl)

        # Export
        btn_export = QPushButton("Export Cluster Results")
        btn_export.clicked.connect(self._export_results)
        ll.addWidget(btn_export)

        ll.addStretch()
        splitter.addWidget(left)

        # ---- Right: chart tabs ----
        self._inner_tabs = QTabWidget()
        self._scatter_widget = QWidget()
        self._scatter_layout = QVBoxLayout(self._scatter_widget)
        self._inner_tabs.addTab(self._scatter_widget, "Scatter (PCA)")

        self._elbow_widget = QWidget()
        self._elbow_layout = QVBoxLayout(self._elbow_widget)
        self._inner_tabs.addTab(self._elbow_widget, "Elbow / Silhouette")

        self._profile_widget = QWidget()
        self._profile_layout = QVBoxLayout(self._profile_widget)
        self._inner_tabs.addTab(self._profile_widget, "Cluster Profiles")

        self._pie_widget = QWidget()
        self._pie_layout = QVBoxLayout(self._pie_widget)
        self._inner_tabs.addTab(self._pie_widget, "Volume Share")

        self._abc_xyz_widget = QWidget()
        self._abc_xyz_layout = QVBoxLayout(self._abc_xyz_widget)
        self._inner_tabs.addTab(self._abc_xyz_widget, "ABC-XYZ Matrix")

        self._label_box = QTextEdit()
        self._label_box.setReadOnly(True)
        self._label_box.setFont(QFont("Consolas", 8))
        self._inner_tabs.addTab(self._label_box, "Cluster Labels")

        splitter.addWidget(self._inner_tabs)
        splitter.setSizes([270, 900])
        root.addWidget(splitter)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_data(self, df: pd.DataFrame, features: pd.DataFrame):
        self._df = df
        self._features = features
        self._run_btn.setEnabled(True)
        self._status_lbl.setText(
            f"Ready. {len(features)} SKUs loaded."
        )

    # ------------------------------------------------------------------
    def _on_algo_changed(self, algo: str):
        self._kmeans_box.setVisible(algo == "K-Means")
        self._dbscan_box.setVisible(algo == "DBSCAN")
        self._hc_box.setVisible(algo == "Hierarchical")

    def _run_elbow(self):
        if self._features is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return
        try:
            k_range, inertias, silhouettes = cl.find_optimal_k(self._features)
            fig = plots.plot_elbow(k_range, inertias, silhouettes)
            self._replace_canvas(self._elbow_layout, fig)
            self._inner_tabs.setCurrentIndex(1)
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def _run_clustering(self):
        if self._features is None:
            return
        algo = self._algo_combo.currentText()
        n_k = self._k_spin.value() if algo == "K-Means" else self._hc_k_spin.value()
        self._run_btn.setEnabled(False)
        self._prog_bar.show()
        self._status_lbl.setText("Running…")
        self._worker = ClusterWorker(
            self._features, self._df, algo, n_k,
            self._eps_spin.value(), self._min_samples_spin.value(),
            self._linkage_combo.currentText()
        )
        self._worker.progress.connect(self._status_lbl.setText)
        self._worker.finished.connect(self._on_cluster_done)
        self._worker.start()

    def _on_cluster_done(self, result_df, coords_info, error):
        self._prog_bar.hide()
        self._run_btn.setEnabled(True)
        if error:
            QMessageBox.critical(self, "Clustering Error", error)
            return

        self._result_df = result_df
        self._status_lbl.setText("Done.")
        score = coords_info["score"]
        self._score_lbl.setText(
            f"Silhouette Score: {score:.4f}" if score else "Silhouette: N/A (DBSCAN)"
        )

        # Scatter
        fig = plots.plot_cluster_scatter_2d(
            coords_info["coords"], coords_info["labels"],
            result_df["c_sku"].tolist() if "c_sku" in result_df.columns else [],
            coords_info["var_ratio"], coords_info["label_map"],
            title=f"{self._algo_combo.currentText()} Clusters (PCA)"
        )
        self._replace_canvas(self._scatter_layout, fig)

        # Profiles
        fig2 = plots.plot_cluster_profiles(result_df)
        self._replace_canvas(self._profile_layout, fig2)

        # Volume pie
        fig3 = plots.plot_cluster_volume_pie(result_df)
        self._replace_canvas(self._pie_layout, fig3)

        # ABC-XYZ
        fig4 = plots.plot_abc_xyz_matrix(result_df)
        self._replace_canvas(self._abc_xyz_layout, fig4)

        # Labels text
        lines = []
        for cid, lbl in coords_info["label_map"].items():
            cnt = (result_df["cluster_id"] == cid).sum() if "cluster_id" in result_df.columns else "?"
            lines.append(f"Cluster {cid:>2}  [{cnt:>3} SKUs]  →  {lbl}")
        if "abc" in result_df.columns:
            lines.append("\n── ABC Distribution ──")
            for cat, cnt in result_df["abc"].value_counts().items():
                lines.append(f"  {cat}: {cnt} SKUs")
        if "xyz" in result_df.columns:
            lines.append("\n── XYZ Distribution ──")
            for cat, cnt in result_df["xyz"].value_counts().items():
                lines.append(f"  {cat}: {cnt} SKUs")
        self._label_box.setPlainText("\n".join(lines))

        self._inner_tabs.setCurrentIndex(0)
        self.cluster_done.emit(result_df)

    def _replace_canvas(self, layout: QVBoxLayout, fig):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

    def _export_results(self):
        if self._result_df is None:
            QMessageBox.warning(self, "No Results", "Run clustering first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Cluster Results", "", "CSV Files (*.csv)")
        if path:
            self._result_df.to_csv(path, index=False)
            QMessageBox.information(self, "Saved", f"Results saved to {path}")
