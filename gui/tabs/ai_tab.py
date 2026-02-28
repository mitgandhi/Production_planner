"""
AI Integration Tab – load local Qwen 3 model and query it
with the preprocessed production context.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox, QFileDialog, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QProgressBar,
    QPushButton, QScrollArea, QSizePolicy, QSplitter,
    QTextEdit, QVBoxLayout, QWidget,
)


# ---------------------------------------------------------------------------
# Qwen 3 inference worker
# ---------------------------------------------------------------------------

class QwenWorker(QThread):
    token = pyqtSignal(str)          # streams tokens
    finished = pyqtSignal(str)       # full response or error

    def __init__(self, model_path: str, prompt: str,
                 context: dict, max_tokens: int = 512,
                 think: bool = False):
        super().__init__()
        self._model_path = model_path
        self._prompt = prompt
        self._context = context
        self._max_tokens = max_tokens
        self._think = think

    def run(self):
        try:
            # Dynamic import so the app loads without transformers installed
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            self.token.emit("[Loading tokenizer…]\n")
            tokenizer = AutoTokenizer.from_pretrained(
                self._model_path, trust_remote_code=True
            )
            self.token.emit("[Loading model…]\n")
            model = AutoModelForCausalLM.from_pretrained(
                self._model_path,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
            model.eval()

            system_prompt = (
                "You are an expert production planning AI for an apparel manufacturer. "
                "Use the following dataset context to answer questions accurately.\n\n"
                f"CONTEXT:\n{json.dumps(self._context, indent=2)}\n"
            )

            # Qwen 3 supports /think (extended reasoning) and /no_think flags
            user_content = self._prompt
            if self._think:
                user_content = "/think\n" + user_content

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer([text], return_tensors="pt").to(model.device)

            import torch
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self._max_tokens,
                    do_sample=False,
                )
            output = tokenizer.decode(
                output_ids[0][inputs.input_ids.shape[-1]:],
                skip_special_tokens=True,
            )
            self.finished.emit(output)

        except ImportError:
            self.finished.emit(
                "[ERROR] 'transformers' and 'torch' are not installed.\n"
                "Install them with:\n"
                "  pip install transformers torch\n"
                "Then place your Qwen 3 model files in the models/ folder."
            )
        except Exception:
            self.finished.emit(f"[ERROR]\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Tab widget
# ---------------------------------------------------------------------------

class AITab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._context: dict = {}
        self._model_path: str = ""
        self._history: list[dict] = []
        self._worker: QwenWorker | None = None
        self._init_ui()

    # ------------------------------------------------------------------
    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ---- Model path ----
        model_box = QGroupBox("Qwen 3 Model")
        model_box.setMaximumHeight(90)
        ml = QHBoxLayout(model_box)
        ml.addWidget(QLabel("Model Path:"))
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText(
            "E:/Gem_computers/models/qwen3  (folder containing config.json)"
        )
        self._path_edit.textChanged.connect(self._on_path_changed)
        ml.addWidget(self._path_edit)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_model)
        ml.addWidget(btn_browse)

        self._think_chk = QCheckBox("Extended Thinking (/think)")
        self._think_chk.setToolTip(
            "Enable Qwen 3 extended reasoning mode for more thorough answers."
        )
        ml.addWidget(self._think_chk)
        root.addWidget(model_box)

        # ---- Context status ----
        self._ctx_lbl = QLabel("No analysis context loaded yet. Run Statistical Analysis first.")
        self._ctx_lbl.setStyleSheet("color: #c0392b; font-style: italic;")
        root.addWidget(self._ctx_lbl)

        # ---- Chat area ----
        self._chat_display = QTextEdit()
        self._chat_display.setReadOnly(True)
        self._chat_display.setFont(QFont("Segoe UI", 9))
        self._chat_display.setStyleSheet(
            "background: #1e1e2e; color: #cdd6f4; border-radius: 4px;"
        )
        root.addWidget(self._chat_display)

        # ---- Input row ----
        input_row = QHBoxLayout()
        self._input_edit = QTextEdit()
        self._input_edit.setMaximumHeight(80)
        self._input_edit.setPlaceholderText(
            "Ask about production planning, demand trends, inventory levels…"
        )
        self._input_edit.setFont(QFont("Segoe UI", 9))
        input_row.addWidget(self._input_edit)

        btn_col = QVBoxLayout()
        self._send_btn = QPushButton("Send")
        self._send_btn.setFixedSize(80, 36)
        font = self._send_btn.font()
        font.setBold(True)
        self._send_btn.setFont(font)
        self._send_btn.setEnabled(False)
        self._send_btn.clicked.connect(self._send_query)
        btn_col.addWidget(self._send_btn)

        btn_clear = QPushButton("Clear")
        btn_clear.setFixedSize(80, 28)
        btn_clear.clicked.connect(self._clear_chat)
        btn_col.addWidget(btn_clear)
        btn_col.addStretch()
        input_row.addLayout(btn_col)
        root.addLayout(input_row)

        # Progress
        self._prog = QProgressBar()
        self._prog.setRange(0, 0)
        self._prog.hide()
        root.addWidget(self._prog)

        # ---- Quick prompts ----
        quick_box = QGroupBox("Quick Prompts")
        quick_box.setMaximumHeight(80)
        ql = QHBoxLayout(quick_box)
        quick_prompts = [
            ("Top Products", "Which are the top 5 SKUs by volume and what is their growth trend?"),
            ("Seasonal Demand", "Which SKUs have the strongest seasonal patterns and when are their peak months?"),
            ("Production Risk", "Which SKUs are declining or highly volatile and need production review?"),
            ("Inventory Plan", "Provide a 6-month production planning summary with safety stock recommendations."),
        ]
        for label, prompt in quick_prompts:
            btn = QPushButton(label)
            btn.setFixedHeight(28)
            btn.clicked.connect(lambda _, p=prompt: self._set_prompt(p))
            ql.addWidget(btn)
        root.addWidget(quick_box)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_context(self, context: dict):
        self._context = context
        self._ctx_lbl.setText(
            f"Context loaded: {len(context.get('top_skus_by_volume', []))} top SKUs, "
            f"{len(context.get('trend_summary', {}).get('uptrend_skus', []))} growing, "
            f"{len(context.get('trend_summary', {}).get('downtrend_skus', []))} declining."
        )
        self._ctx_lbl.setStyleSheet("color: #27ae60; font-style: italic;")
        self._check_ready()

    # ------------------------------------------------------------------
    def _browse_model(self):
        path = QFileDialog.getExistingDirectory(self, "Select Qwen 3 Model Folder",
                                                 "E:/Gem_computers/models")
        if path:
            self._path_edit.setText(path)

    def _on_path_changed(self, text: str):
        self._model_path = text.strip()
        self._check_ready()

    def _check_ready(self):
        ready = bool(self._model_path) and bool(self._context)
        self._send_btn.setEnabled(ready)

    def _set_prompt(self, text: str):
        self._input_edit.setPlainText(text)

    def _send_query(self):
        prompt = self._input_edit.toPlainText().strip()
        if not prompt:
            return

        self._append_chat("You", prompt, "#89b4fa")
        self._input_edit.clear()
        self._send_btn.setEnabled(False)
        self._prog.show()

        self._worker = QwenWorker(
            self._model_path, prompt, self._context,
            max_tokens=512, think=self._think_chk.isChecked()
        )
        self._worker.token.connect(lambda t: self._append_raw(t))
        self._worker.finished.connect(self._on_response)
        self._worker.start()

    def _on_response(self, response: str):
        self._prog.hide()
        self._send_btn.setEnabled(True)
        self._append_chat("Qwen 3", response, "#a6e3a1")

    def _append_chat(self, role: str, text: str, color: str):
        html = (
            f'<p><span style="color:{color}; font-weight:bold;">{role}:</span> '
            f'<span style="color:#cdd6f4;">{text.replace(chr(10), "<br>")}</span></p>'
        )
        self._chat_display.append(html)

    def _append_raw(self, text: str):
        self._chat_display.insertPlainText(text)

    def _clear_chat(self):
        self._chat_display.clear()
        self._history.clear()
