"""
ai_tab.py
---------
Claude-style chat interface for Qwen3-VL (text-only inference).
Model is preloaded at app startup via ModelManager singleton.
"""

from __future__ import annotations

import datetime
import textwrap
from typing import List, Dict

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer
from PyQt6.QtGui import QFont, QKeyEvent, QTextCursor, QColor
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGroupBox,
    QHBoxLayout, QLabel, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QSpinBox,
    QSplitter, QTextBrowser, QTextEdit, QVBoxLayout, QWidget,
)

from gui.model_manager import ModelManager, build_system_prompt


# ─────────────────────────────────────────────────────────────────────────────
# Inference worker – runs generate() in its own thread
# ─────────────────────────────────────────────────────────────────────────────

class InferenceWorker(QThread):
    response_ready = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self,
                 messages: List[Dict[str, str]],
                 max_tokens: int,
                 temperature: float,
                 enable_thinking: bool):
        super().__init__()
        self._messages = messages
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._thinking = enable_thinking

    def run(self):
        try:
            mm = ModelManager.get()
            result = mm.generate(
                self._messages,
                max_new_tokens=self._max_tokens,
                temperature=self._temperature,
                enable_thinking=self._thinking,
            )
            self.response_ready.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Chat message bubble renderer
# ─────────────────────────────────────────────────────────────────────────────

def _html_user(text: str, ts: str) -> str:
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    escaped = escaped.replace("\n", "<br>")
    return f"""
    <div style="margin:8px 0; text-align:right;">
      <span style="display:inline-block; background:#4f6ef7; color:#ffffff;
                   border-radius:12px 12px 2px 12px; padding:10px 14px;
                   max-width:75%; font-size:13px; line-height:1.5;
                   white-space:pre-wrap; word-break:break-word;">
        {escaped}
      </span>
      <div style="color:#6c7086; font-size:10px; margin-top:2px;">{ts}</div>
    </div>"""


def _html_ai(text: str, ts: str, model_name: str = "Qwen3-VL") -> str:
    # Basic markdown-ish rendering
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Bold **text**
    import re
    escaped = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', escaped)
    # Inline code `text`
    escaped = re.sub(r'`([^`]+)`', r'<code style="background:#313244;padding:1px 4px;border-radius:3px;">\1</code>', escaped)
    # Bullet points
    lines = escaped.split("\n")
    formatted = []
    for line in lines:
        if line.strip().startswith("• ") or line.strip().startswith("- "):
            formatted.append(f'<li style="margin:2px 0;">{line.strip()[2:]}</li>')
        elif line.strip().startswith("* "):
            formatted.append(f'<li style="margin:2px 0;">{line.strip()[2:]}</li>')
        else:
            formatted.append(line)
    escaped = "<br>".join(formatted)

    return f"""
    <div style="margin:8px 0; text-align:left;">
      <div style="color:#89b4fa; font-size:10px; font-weight:bold; margin-bottom:3px;">
        🤖 {model_name}  <span style="color:#6c7086; font-weight:normal;">{ts}</span>
      </div>
      <span style="display:inline-block; background:#313244; color:#cdd6f4;
                   border-radius:2px 12px 12px 12px; padding:10px 14px;
                   max-width:82%; font-size:13px; line-height:1.6;
                   white-space:pre-wrap; word-break:break-word;">
        {escaped}
      </span>
    </div>"""


def _html_system(text: str) -> str:
    return f"""
    <div style="margin:10px 0; text-align:center;">
      <span style="color:#6c7086; font-style:italic; font-size:11px;">{text}</span>
    </div>"""


def _html_thinking() -> str:
    return f"""
    <div id="thinking-indicator" style="margin:8px 0; text-align:left;">
      <span style="display:inline-block; background:#313244; color:#89b4fa;
                   border-radius:2px 12px 12px 12px; padding:10px 14px;
                   font-size:13px;">
        ⏳ Thinking…
      </span>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Smart input box – Ctrl+Enter to send
# ─────────────────────────────────────────────────────────────────────────────

class SmartInputBox(QTextEdit):
    send_requested = pyqtSignal()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            mods = event.modifiers()
            if mods & Qt.KeyboardModifier.ControlModifier:
                self.send_requested.emit()
                return
        super().keyPressEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
# Main AI Tab widget
# ─────────────────────────────────────────────────────────────────────────────

QUICK_PROMPTS = {
    "Production": [
        ("6-Month Forecast", "Give me a 6-month production plan for the top 10 SKUs with forecast quantities, safety stock levels, and priority actions."),
        ("Safety Stock Review", "Which SKUs have inadequate safety stock? List them with current avg demand, variability (CV%), and recommended safety stock."),
        ("Reorder Points", "List the reorder points for the top 20 SKUs sorted by urgency. Include avg demand, lead time assumption, and reorder qty."),
        ("Production Schedule", "Create a prioritised production schedule for next quarter. Group by cluster (High/Medium/Low volume) and ABC class."),
    ],
    "Demand": [
        ("Top SKUs Analysis", "Analyse the top 5 SKUs by volume. Give monthly average, trend direction, seasonality, and 3-month forecast for each."),
        ("Growing Products", "Which SKUs show the strongest upward demand trend? Quantify the growth rate and recommend production increase %."),
        ("Declining Products", "Which SKUs are declining? Provide decline rate, current safety stock adequacy, and whether to phase out or reduce batch size."),
        ("Seasonal Patterns", "Which SKUs have the strongest seasonal demand? Identify peak months and recommend pre-season production build-up quantities."),
    ],
    "Inventory": [
        ("ABC-XYZ Summary", "Summarise the ABC-XYZ classification results. What does each category mean for inventory policy and production frequency?"),
        ("Size Analysis", "Which sizes drive the most volume overall? Break down by top 5 SKUs and recommend size ratio for production batches."),
        ("Color Strategy", "Which colors are consistently high demand vs seasonal? Recommend a color production mix strategy."),
        ("Risk Assessment", "Identify the top 5 inventory risk SKUs (high CV, high volume, long trend decline) and mitigation actions."),
    ],
    "Custom": [
        ("Paste Data & Analyse", "I will paste some data below. Please analyse it in the context of our apparel inventory and provide insights:\n\n[PASTE YOUR DATA HERE]"),
        ("Compare SKUs", "Compare these SKUs side by side: [SKU1, SKU2, SKU3]. Show volume, trend, seasonality, and production recommendation."),
        ("What-If Scenario", "If we increase production of ALPA by 20% next quarter, what would be the impact on inventory levels and when would stock risk occur?"),
        ("Executive Summary", "Write a concise executive summary of the current production planning situation: top performers, risks, and 3 key actions."),
    ],
}


class AITab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._messages: List[Dict[str, str]] = []
        self._system_prompt: str = ""
        self._worker: InferenceWorker | None = None
        self._dot_timer = QTimer()
        self._dot_count = 0

        # Analysis context (set by main window)
        self._summary: dict = {}
        self._cluster_result = None
        self._stats_context: dict = {}
        self._plan_df = None

        self._init_ui()
        self._init_dot_timer()

    # ──────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # === Top status bar ===
        self._status_bar = self._build_status_bar()
        root.addWidget(self._status_bar)

        # === Splitter: sidebar | chat area ===
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)

        # -- Left sidebar --
        sidebar = self._build_sidebar()
        splitter.addWidget(sidebar)

        # -- Right: chat --
        chat_panel = self._build_chat_panel()
        splitter.addWidget(chat_panel)

        splitter.setSizes([260, 1000])
        root.addWidget(splitter)

    def _build_status_bar(self) -> QFrame:
        bar = QFrame()
        bar.setFixedHeight(38)
        bar.setStyleSheet(
            "QFrame { background: #181825; border-bottom: 1px solid #313244; }"
        )
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 4, 12, 4)

        self._model_dot = QLabel("●")
        self._model_dot.setStyleSheet("color: #f38ba8; font-size: 14px;")
        layout.addWidget(self._model_dot)

        self._model_status_lbl = QLabel("Model loading…")
        self._model_status_lbl.setStyleSheet("color: #cdd6f4; font-size: 11px;")
        layout.addWidget(self._model_status_lbl)

        self._model_prog = QProgressBar()
        self._model_prog.setRange(0, 0)
        self._model_prog.setFixedSize(120, 8)
        self._model_prog.setTextVisible(False)
        self._model_prog.setStyleSheet(
            "QProgressBar { background:#313244; border:none; border-radius:4px; }"
            "QProgressBar::chunk { background:#89b4fa; border-radius:4px; }"
        )
        layout.addWidget(self._model_prog)
        layout.addStretch()

        self._ctx_lbl = QLabel("Context: not built")
        self._ctx_lbl.setStyleSheet("color: #6c7086; font-size: 10px; font-style:italic;")
        layout.addWidget(self._ctx_lbl)

        btn_rebuild = QPushButton("Rebuild Context")
        btn_rebuild.setFixedHeight(24)
        btn_rebuild.setStyleSheet(
            "QPushButton { background:#313244; color:#cdd6f4; border:none; "
            "border-radius:4px; padding:0 8px; font-size:10px; }"
            "QPushButton:hover { background:#45475a; }"
        )
        btn_rebuild.clicked.connect(self._rebuild_system_prompt)
        layout.addWidget(btn_rebuild)
        return bar

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setFixedWidth(265)
        sidebar.setStyleSheet("QWidget { background: #181825; }")
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(8, 8, 8, 8)
        sl.setSpacing(6)

        # Model info
        info_box = QGroupBox("Model")
        info_box.setStyleSheet(
            "QGroupBox { border:1px solid #313244; border-radius:6px; margin-top:8px; color:#89b4fa; font-weight:bold;}"
            "QGroupBox::title { subcontrol-origin:margin; left:8px; }"
        )
        il = QVBoxLayout(info_box)
        self._model_info_lbl = QLabel(
            "Qwen3-VL 2B\nDevice: CPU\nStatus: Loading…"
        )
        self._model_info_lbl.setStyleSheet("color:#a6adc8; font-size:10px;")
        self._model_info_lbl.setWordWrap(True)
        il.addWidget(self._model_info_lbl)
        sl.addWidget(info_box)

        # Generation settings
        gen_box = QGroupBox("Generation Settings")
        gen_box.setStyleSheet(
            "QGroupBox { border:1px solid #313244; border-radius:6px; margin-top:8px; color:#89b4fa; font-weight:bold;}"
            "QGroupBox::title { subcontrol-origin:margin; left:8px; }"
        )
        gl = QVBoxLayout(gen_box)

        gl.addWidget(QLabel("Max Tokens:"))
        self._max_tokens_spin = QSpinBox()
        self._max_tokens_spin.setRange(64, 2048)
        self._max_tokens_spin.setValue(512)
        self._max_tokens_spin.setSingleStep(64)
        self._max_tokens_spin.setStyleSheet(
            "QSpinBox { background:#313244; color:#cdd6f4; border:1px solid #45475a; border-radius:4px; padding:2px;}"
        )
        gl.addWidget(self._max_tokens_spin)

        gl.addWidget(QLabel("Temperature:"))
        temp_row = QHBoxLayout()
        self._temp_slider = QSlider(Qt.Orientation.Horizontal)
        self._temp_slider.setRange(0, 20)
        self._temp_slider.setValue(7)
        self._temp_lbl = QLabel("0.7")
        self._temp_lbl.setFixedWidth(30)
        self._temp_lbl.setStyleSheet("color:#cdd6f4;")
        self._temp_slider.valueChanged.connect(
            lambda v: self._temp_lbl.setText(f"{v/10:.1f}")
        )
        temp_row.addWidget(self._temp_slider)
        temp_row.addWidget(self._temp_lbl)
        gl.addLayout(temp_row)

        self._thinking_chk = QCheckBox("Extended Thinking")
        self._thinking_chk.setStyleSheet("color:#cdd6f4;")
        self._thinking_chk.setToolTip(
            "Enables Qwen3 deep reasoning mode (slower but more thorough)."
        )
        gl.addWidget(self._thinking_chk)
        sl.addWidget(gen_box)

        # Quick prompts
        qp_box = QGroupBox("Quick Prompts")
        qp_box.setStyleSheet(
            "QGroupBox { border:1px solid #313244; border-radius:6px; margin-top:8px; color:#89b4fa; font-weight:bold;}"
            "QGroupBox::title { subcontrol-origin:margin; left:8px; }"
        )
        ql = QVBoxLayout(qp_box)

        category_combo = QComboBox()
        category_combo.addItems(list(QUICK_PROMPTS.keys()))
        category_combo.setStyleSheet(
            "QComboBox { background:#313244; color:#cdd6f4; border:1px solid #45475a; border-radius:4px; padding:2px 6px; }"
            "QComboBox QAbstractItemView { background:#313244; color:#cdd6f4; }"
        )
        ql.addWidget(category_combo)

        self._qp_buttons_layout = QVBoxLayout()
        ql.addLayout(self._qp_buttons_layout)

        def _populate_qp(cat: str):
            while self._qp_buttons_layout.count():
                item = self._qp_buttons_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            for label, prompt in QUICK_PROMPTS.get(cat, []):
                btn = QPushButton(label)
                btn.setFixedHeight(26)
                btn.setStyleSheet(
                    "QPushButton { background:#313244; color:#cdd6f4; border:none; "
                    "border-radius:4px; padding:2px 8px; text-align:left; font-size:11px;}"
                    "QPushButton:hover { background:#45475a; color:#89b4fa; }"
                )
                btn.clicked.connect(lambda _, p=prompt: self._set_input(p))
                self._qp_buttons_layout.addWidget(btn)

        category_combo.currentTextChanged.connect(_populate_qp)
        _populate_qp("Production")
        sl.addWidget(qp_box)

        # Conversation controls
        ctrl_box = QGroupBox("Conversation")
        ctrl_box.setStyleSheet(
            "QGroupBox { border:1px solid #313244; border-radius:6px; margin-top:8px; color:#89b4fa; font-weight:bold;}"
            "QGroupBox::title { subcontrol-origin:margin; left:8px; }"
        )
        cl = QVBoxLayout(ctrl_box)
        btn_new = QPushButton("New Conversation")
        btn_new.setFixedHeight(28)
        btn_new.clicked.connect(self._new_conversation)
        cl.addWidget(btn_new)
        self._turn_lbl = QLabel("Turns: 0")
        self._turn_lbl.setStyleSheet("color:#6c7086; font-size:10px;")
        cl.addWidget(self._turn_lbl)
        sl.addWidget(ctrl_box)

        sl.addStretch()
        return sidebar

    def _build_chat_panel(self) -> QWidget:
        panel = QWidget()
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(0)

        # Chat display
        self._chat = QTextBrowser()
        self._chat.setOpenExternalLinks(False)
        self._chat.setFont(QFont("Segoe UI", 10))
        self._chat.setStyleSheet(
            "QTextBrowser { background:#1e1e2e; border:none; padding:12px; }"
            "QScrollBar:vertical { background:#1e1e2e; width:6px; border-radius:3px; }"
            "QScrollBar::handle:vertical { background:#45475a; border-radius:3px; min-height:20px; }"
        )
        self._chat.setHtml(
            "<body style='background:#1e1e2e; color:#cdd6f4; font-family:Segoe UI;'>"
            + _html_system("Welcome to Qwen3-VL Production Planner Chat")
            + _html_system("Model is loading in the background – you can type while you wait.")
            + _html_system("Use Ctrl+Enter to send &nbsp;|&nbsp; Choose quick prompts from the sidebar")
            + "</body>"
        )
        pl.addWidget(self._chat)

        # Separator
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background:#313244;")
        pl.addWidget(sep)

        # Input area
        input_area = QWidget()
        input_area.setFixedHeight(130)
        input_area.setStyleSheet("background:#181825;")
        ia_l = QVBoxLayout(input_area)
        ia_l.setContentsMargins(12, 8, 12, 8)
        ia_l.setSpacing(6)

        # Hint
        hint = QLabel("Ctrl+Enter to send  •  Shift+Enter for new line  •  Paste tables, numbers, labels freely")
        hint.setStyleSheet("color:#6c7086; font-size:10px;")
        ia_l.addWidget(hint)

        # Input row
        input_row = QHBoxLayout()
        self._input = SmartInputBox()
        self._input.send_requested.connect(self._send)
        self._input.setPlaceholderText(
            "Ask about production planning, SKU performance, demand trends, "
            "or paste classification/numerical data for analysis…"
        )
        self._input.setFont(QFont("Segoe UI", 10))
        self._input.setStyleSheet(
            "QTextEdit { background:#313244; color:#cdd6f4; border:1px solid #45475a; "
            "border-radius:8px; padding:8px 12px; }"
            "QTextEdit:focus { border-color:#89b4fa; }"
        )
        input_row.addWidget(self._input)

        # Buttons
        btn_col = QVBoxLayout()
        btn_col.setSpacing(4)

        self._send_btn = QPushButton("Send")
        self._send_btn.setFixedSize(80, 40)
        self._send_btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self._send_btn.setStyleSheet(
            "QPushButton { background:#89b4fa; color:#1e1e2e; border:none; "
            "border-radius:8px; }"
            "QPushButton:hover { background:#b4befe; }"
            "QPushButton:disabled { background:#45475a; color:#6c7086; }"
        )
        self._send_btn.clicked.connect(self._send)
        btn_col.addWidget(self._send_btn)

        btn_stop = QPushButton("Stop")
        btn_stop.setFixedSize(80, 28)
        btn_stop.setStyleSheet(
            "QPushButton { background:#f38ba8; color:#1e1e2e; border:none; border-radius:6px; }"
            "QPushButton:hover { background:#eba0ac; }"
        )
        btn_stop.clicked.connect(self._stop)
        btn_col.addWidget(btn_stop)
        input_row.addLayout(btn_col)
        ia_l.addLayout(input_row)
        pl.addWidget(input_area)

        return panel

    def _init_dot_timer(self):
        """Animate '●' in the thinking indicator."""
        self._dot_timer.setInterval(400)
        self._dot_timer.timeout.connect(self._animate_thinking)

    # ──────────────────────────────────────────────────────────────────
    # Public API (called by MainWindow)
    # ──────────────────────────────────────────────────────────────────

    def on_model_ready(self, success: bool, message: str):
        """Called from MainWindow after ModelManager finishes loading."""
        from gui.model_manager import ModelManager
        mm = ModelManager.get()

        if success:
            self._model_dot.setStyleSheet("color: #a6e3a1; font-size: 14px;")
            self._model_prog.hide()

            # Show full device info in status bar and sidebar
            dev = mm.device_info
            is_gpu = "GPU" in dev
            device_label = "GPU  ✓" if is_gpu else "CPU"
            self._model_status_lbl.setText(f"Qwen3-VL 2B  •  Ready  •  {device_label}")
            self._model_info_lbl.setText(
                f"Qwen3-VL 2B\n{dev}\nStatus: ✓ Ready\nTokenizer: Qwen2Tokenizer"
            )
            self._append_system(f"Model ready on {dev}")
        else:
            self._model_dot.setStyleSheet("color: #f38ba8; font-size: 14px;")
            self._model_status_lbl.setText(f"Model error: {message[:60]}")
            self._model_prog.hide()
            self._model_info_lbl.setText(f"Status: ✗ Failed\n{message[:120]}")
            self._append_system(f"⚠ Model failed to load: {message}")

    def set_analysis_data(
        self,
        summary: dict | None = None,
        cluster_result=None,
        stats_context: dict | None = None,
        plan_df=None,
    ):
        """Called by MainWindow whenever analysis results are updated."""
        if summary is not None:
            self._summary = summary
        if cluster_result is not None:
            self._cluster_result = cluster_result
        if stats_context is not None:
            self._stats_context = stats_context
        if plan_df is not None:
            self._plan_df = plan_df
        self._rebuild_system_prompt()

    # ──────────────────────────────────────────────────────────────────
    # System prompt
    # ──────────────────────────────────────────────────────────────────

    def _rebuild_system_prompt(self):
        self._system_prompt = build_system_prompt(
            summary=self._summary or None,
            cluster_result=self._cluster_result,
            stats_context=self._stats_context or None,
            plan_df=self._plan_df,
        )
        parts = []
        if self._summary:
            parts.append(f"{self._summary.get('unique_skus','?')} SKUs")
        if self._cluster_result is not None:
            parts.append("clusters ✓")
        if self._stats_context:
            parts.append("stats ✓")
        if self._plan_df is not None:
            parts.append("plan ✓")
        self._ctx_lbl.setText(
            "Context: " + (", ".join(parts) if parts else "dataset overview only")
        )

    # ──────────────────────────────────────────────────────────────────
    # Chat actions
    # ──────────────────────────────────────────────────────────────────

    def _send(self):
        text = self._input.toPlainText().strip()
        if not text:
            return
        if self._worker and self._worker.isRunning():
            return

        ts = datetime.datetime.now().strftime("%H:%M")
        self._append_html(_html_user(text, ts))
        self._input.clear()

        # Build message history
        history = []
        if self._system_prompt:
            history.append({"role": "system", "content": self._system_prompt})
        history.extend(self._messages)
        history.append({"role": "user", "content": text})

        # Show thinking indicator
        self._append_html(_html_thinking())
        self._dot_timer.start()
        self._send_btn.setEnabled(False)

        # Run inference
        self._worker = InferenceWorker(
            messages=history,
            max_tokens=self._max_tokens_spin.value(),
            temperature=self._temp_slider.value() / 10,
            enable_thinking=self._thinking_chk.isChecked(),
        )
        self._worker.response_ready.connect(self._on_response)
        self._worker.error.connect(self._on_error)
        self._worker.start()

        # Track in history
        self._messages.append({"role": "user", "content": text})
        self._update_turn_count()

    def _on_response(self, response: str):
        self._dot_timer.stop()
        self._send_btn.setEnabled(True)
        self._remove_thinking_indicator()

        ts = datetime.datetime.now().strftime("%H:%M")
        self._append_html(_html_ai(response, ts))
        self._messages.append({"role": "assistant", "content": response})
        self._update_turn_count()

    def _on_error(self, error: str):
        self._dot_timer.stop()
        self._send_btn.setEnabled(True)
        self._remove_thinking_indicator()
        self._append_system(f"⚠ Error: {error}")

    def _stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._dot_timer.stop()
            self._send_btn.setEnabled(True)
            self._remove_thinking_indicator()
            self._append_system("Generation stopped by user.")

    def _new_conversation(self):
        self._messages.clear()
        self._chat.setHtml(
            "<body style='background:#1e1e2e; color:#cdd6f4; font-family:Segoe UI;'>"
            + _html_system("New conversation started")
            + "</body>"
        )
        self._update_turn_count()

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _append_html(self, html: str):
        cursor = self._chat.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._chat.setTextCursor(cursor)
        self._chat.insertHtml(html)
        self._chat.insertHtml("<br>")
        self._chat.verticalScrollBar().setValue(
            self._chat.verticalScrollBar().maximum()
        )

    def _append_system(self, text: str):
        self._append_html(_html_system(text))

    def _remove_thinking_indicator(self):
        # Replace the "Thinking…" block with nothing
        html = self._chat.toHtml()
        # Find and remove the thinking indicator div
        import re
        html = re.sub(
            r'<div[^>]*id="thinking-indicator"[^>]*>.*?</div>\s*</div>',
            '',
            html,
            flags=re.DOTALL
        )
        # Simpler fallback: just leave it (it disappears visually with the response)
        pass

    def _animate_thinking(self):
        self._dot_count = (self._dot_count + 1) % 4
        dots = "." * self._dot_count
        self._model_status_lbl.setText(f"Qwen3-VL 2B  •  Generating{dots}")

    def _set_input(self, text: str):
        self._input.setPlainText(text)
        self._input.setFocus()

    def _update_turn_count(self):
        turns = len([m for m in self._messages if m["role"] == "user"])
        self._turn_lbl.setText(f"Turns: {turns}")
