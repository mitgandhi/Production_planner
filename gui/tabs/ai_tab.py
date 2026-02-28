"""
ai_tab.py
---------
Claude-style chat interface for Qwen3-VL with PARALLEL STREAMING inference.

Threading model
---------------
  Send button clicked  (GUI thread)
      |
      v
  StreamingInferenceWorker.start()         -- QThread A
      |
      +--  ModelManager.generate_stream()
              |
              +-- Thread B (daemon)  -->  model.generate(..., streamer=...)  [GPU]
              |                                     |
              +-- TextIteratorStreamer  <------------+   (internal queue.Queue)
              |
              for token in streamer:
                  emit token_ready(token)   -->  GUI thread appends token to live bubble

Result: GPU generates tokens, decoding runs in Thread A, UI updates at ~30-60 tokens/s.
The chat bubble text appears token-by-token like ChatGPT / Claude.
"""

from __future__ import annotations

import datetime
import re
from typing import List, Dict

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QKeyEvent, QTextCursor
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QGroupBox,
    QHBoxLayout, QLabel, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QSpinBox,
    QSplitter, QTextBrowser, QTextEdit, QVBoxLayout, QWidget,
)

from gui.model_manager import ModelManager, build_system_prompt


# ─────────────────────────────────────────────────────────────────────────────
# Streaming inference worker
# Runs in its own QThread.  Emits one signal per token so the GUI thread can
# update the live bubble without blocking.
# ─────────────────────────────────────────────────────────────────────────────

class StreamingInferenceWorker(QThread):
    """
    Parallel streaming worker.

    Signals
    -------
    token_ready(str)   -- emitted for every decoded token
    stream_done(str)   -- emitted with the full text when generation ends
    error_occurred(str)-- emitted if an exception is raised
    """
    token_ready    = pyqtSignal(str)
    stream_done    = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        enable_thinking: bool,
    ):
        super().__init__()
        self._messages      = messages
        self._max_tokens    = max_tokens
        self._temperature   = temperature
        self._thinking      = enable_thinking
        self._stop_requested = False

    def request_stop(self):
        """Ask the worker to stop after the next token."""
        self._stop_requested = True

    def run(self):
        try:
            mm = ModelManager.get()
            full_text = ""
            for token in mm.generate_stream(
                self._messages,
                max_new_tokens=self._max_tokens,
                temperature=self._temperature,
                enable_thinking=self._thinking,
            ):
                if self._stop_requested:
                    break
                full_text += token
                self.token_ready.emit(token)

            self.stream_done.emit(full_text)

        except Exception as exc:
            self.error_occurred.emit(str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────────────────────────────────────

def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )


def _render_markdown(text: str) -> str:
    """Minimal markdown → HTML: bold, inline code, bullet points."""
    escaped = _escape(text)
    # Bold **...**
    escaped = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', escaped)
    # Inline code `...`
    escaped = re.sub(
        r'`([^`]+)`',
        r'<code style="background:#2a2a3d;padding:1px 5px;border-radius:3px;'
        r'font-family:Consolas,monospace;">\1</code>',
        escaped,
    )
    # Bullet lines  •  -  *
    lines = escaped.split("\n")
    out = []
    for line in lines:
        s = line.strip()
        if s.startswith(("- ", "* ", "+ ")):
            out.append(f'<li style="margin:2px 0;">{s[2:]}</li>')
        elif re.match(r'^[\u2022\u2023\u25E6\u2043] ', s):   # unicode bullets
            out.append(f'<li style="margin:2px 0;">{s[2:]}</li>')
        elif re.match(r'^\d+\.\s', s):   # numbered list
            item_text = re.sub(r'^\d+\.\s', '', s)
            out.append(f'<li style="margin:2px 0;">{item_text}</li>')
        else:
            out.append(line)
    return "<br>".join(out)


def _html_user(text: str, ts: str) -> str:
    escaped = _escape(text).replace("\n", "<br>")
    return (
        f'<div style="margin:10px 0; text-align:right;">'
        f'<span style="display:inline-block; background:#4f6ef7; color:#ffffff;'
        f'border-radius:14px 14px 3px 14px; padding:10px 16px;'
        f'max-width:75%; font-size:13px; line-height:1.55;'
        f'white-space:pre-wrap; word-break:break-word;">'
        f'{escaped}'
        f'</span>'
        f'<div style="color:#6c7086; font-size:10px; margin-top:3px;">{ts}</div>'
        f'</div>'
    )


def _html_ai(text: str, ts: str, model_name: str = "Qwen3", cursor: bool = False) -> str:
    rendered = _render_markdown(text)
    caret = '<span style="color:#89b4fa;">&#9646;</span>' if cursor else ""
    return (
        f'<div style="margin:10px 0; text-align:left;">'
        f'<div style="color:#89b4fa; font-size:10px; font-weight:bold; margin-bottom:3px;">'
        f'&#129302; {model_name}&nbsp;&nbsp;'
        f'<span style="color:#6c7086; font-weight:normal;">{ts}</span>'
        f'</div>'
        f'<span style="display:inline-block; background:#313244; color:#cdd6f4;'
        f'border-radius:3px 14px 14px 14px; padding:10px 16px;'
        f'max-width:84%; font-size:13px; line-height:1.6;'
        f'white-space:pre-wrap; word-break:break-word;">'
        f'{rendered}{caret}'
        f'</span>'
        f'</div>'
    )


def _html_system(text: str) -> str:
    return (
        f'<div style="margin:8px 0; text-align:center;">'
        f'<span style="color:#6c7086; font-style:italic; font-size:11px;">{text}</span>'
        f'</div>'
    )


_HTML_BODY_OPEN  = "<body style='background:#1e1e2e; color:#cdd6f4; font-family:Segoe UI;'>"
_HTML_BODY_CLOSE = "</body>"


# ─────────────────────────────────────────────────────────────────────────────
# Smart input: Ctrl+Enter sends, Enter inserts newline
# ─────────────────────────────────────────────────────────────────────────────

class SmartInputBox(QTextEdit):
    send_requested = pyqtSignal()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.send_requested.emit()
                return
        super().keyPressEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
# Quick prompt bank
# ─────────────────────────────────────────────────────────────────────────────

QUICK_PROMPTS: Dict[str, List[tuple]] = {
    "Production": [
        ("6-Month Forecast",
         "Give me a 6-month production plan for the top 10 SKUs with forecast "
         "quantities, safety stock levels, and priority actions."),
        ("Safety Stock Review",
         "Which SKUs have inadequate safety stock? List them with current avg "
         "demand, variability (CV%), and recommended safety stock."),
        ("Reorder Points",
         "List the reorder points for the top 20 SKUs sorted by urgency. "
         "Include avg demand, lead time assumption, and reorder qty."),
        ("Production Schedule",
         "Create a prioritised production schedule for next quarter. "
         "Group by cluster (High/Medium/Low volume) and ABC class."),
    ],
    "Demand": [
        ("Top SKUs Analysis",
         "Analyse the top 5 SKUs by volume. Give monthly average, trend "
         "direction, seasonality, and 3-month forecast for each."),
        ("Growing Products",
         "Which SKUs show the strongest upward demand trend? Quantify the "
         "growth rate and recommend production increase %."),
        ("Declining Products",
         "Which SKUs are declining? Provide decline rate, current safety stock "
         "adequacy, and whether to phase out or reduce batch size."),
        ("Seasonal Patterns",
         "Which SKUs have the strongest seasonal demand? Identify peak months "
         "and recommend pre-season production build-up quantities."),
    ],
    "Inventory": [
        ("ABC-XYZ Summary",
         "Summarise the ABC-XYZ classification results. What does each "
         "category mean for inventory policy and production frequency?"),
        ("Size Analysis",
         "Which sizes drive the most volume overall? Break down by top 5 SKUs "
         "and recommend size ratio for production batches."),
        ("Color Strategy",
         "Which colors are consistently high demand vs seasonal? Recommend a "
         "color production mix strategy."),
        ("Risk Assessment",
         "Identify the top 5 inventory risk SKUs (high CV, high volume, long "
         "trend decline) and mitigation actions."),
    ],
    "Custom": [
        ("Paste Data & Analyse",
         "I will paste some data below. Please analyse it in the context of "
         "our apparel inventory and provide insights:\n\n[PASTE YOUR DATA HERE]"),
        ("Compare SKUs",
         "Compare these SKUs side by side: [SKU1, SKU2, SKU3]. Show volume, "
         "trend, seasonality, and production recommendation."),
        ("What-If Scenario",
         "If we increase production of ALPA by 20% next quarter, what would "
         "be the impact on inventory levels and when would stock risk occur?"),
        ("Executive Summary",
         "Write a concise executive summary of the current production planning "
         "situation: top performers, risks, and 3 key actions."),
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Main AI Tab widget
# ─────────────────────────────────────────────────────────────────────────────

class AITab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Conversation state
        self._messages: List[Dict[str, str]] = []
        self._system_prompt: str = ""
        self._worker: StreamingInferenceWorker | None = None

        # Streaming state
        # _completed_html  – HTML for all finished messages (not rebuilt on every token)
        # _streaming_text  – text buffer for the message currently being streamed
        # _stream_ts       – timestamp of the current streaming message
        self._completed_html:  str = ""
        self._streaming_text:  str = ""
        self._stream_ts:       str = ""
        self._is_streaming:    bool = False

        # Blinking cursor timer (updates ▌ every 500ms when idle / between tokens)
        self._cursor_timer = QTimer()
        self._cursor_visible = True

        # Analysis context (set by MainWindow when tabs complete)
        self._summary: dict = {}
        self._cluster_result = None
        self._stats_context: dict = {}
        self._plan_df = None
        self._df = None          # raw DataFrame – injected into system prompt

        self._init_ui()
        self._init_cursor_timer()

        # Build initial chat HTML
        self._completed_html = (
            _html_system("Welcome to Qwen3 Production Planner Chat")
            + _html_system("Streaming inference active &mdash; responses appear token-by-token.")
            + _html_system("Ctrl+Enter to send &nbsp;|&nbsp; Choose quick prompts from the sidebar")
        )
        self._refresh_chat()

    # ──────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_status_bar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self._build_chat_panel())
        splitter.setSizes([265, 1100])
        root.addWidget(splitter)

    def _build_status_bar(self) -> QFrame:
        bar = QFrame()
        bar.setFixedHeight(38)
        bar.setStyleSheet(
            "QFrame { background:#181825; border-bottom:1px solid #313244; }"
        )
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(12, 4, 12, 4)

        self._model_dot = QLabel("&#9679;")
        self._model_dot.setStyleSheet("color:#f38ba8; font-size:14px;")
        lay.addWidget(self._model_dot)

        self._model_status_lbl = QLabel("Model loading…")
        self._model_status_lbl.setStyleSheet("color:#cdd6f4; font-size:11px;")
        lay.addWidget(self._model_status_lbl)

        self._model_prog = QProgressBar()
        self._model_prog.setRange(0, 0)
        self._model_prog.setFixedSize(120, 8)
        self._model_prog.setTextVisible(False)
        self._model_prog.setStyleSheet(
            "QProgressBar{background:#313244;border:none;border-radius:4px;}"
            "QProgressBar::chunk{background:#89b4fa;border-radius:4px;}"
        )
        lay.addWidget(self._model_prog)

        self._tps_lbl = QLabel("")  # tokens-per-second display
        self._tps_lbl.setStyleSheet("color:#a6e3a1; font-size:10px; margin-left:8px;")
        lay.addWidget(self._tps_lbl)

        lay.addStretch()

        self._ctx_lbl = QLabel("Context: not built")
        self._ctx_lbl.setStyleSheet("color:#6c7086; font-size:10px; font-style:italic;")
        lay.addWidget(self._ctx_lbl)

        btn_rebuild = QPushButton("Rebuild Context")
        btn_rebuild.setFixedHeight(24)
        btn_rebuild.setStyleSheet(
            "QPushButton{background:#313244;color:#cdd6f4;border:none;"
            "border-radius:4px;padding:0 8px;font-size:10px;}"
            "QPushButton:hover{background:#45475a;}"
        )
        btn_rebuild.clicked.connect(self._rebuild_system_prompt)
        lay.addWidget(btn_rebuild)
        return bar

    def _build_sidebar(self) -> QWidget:
        w = QWidget()
        w.setFixedWidth(268)
        w.setStyleSheet("QWidget{background:#181825;}")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # Model info
        info_box = QGroupBox("Model")
        info_box.setStyleSheet(
            "QGroupBox{border:1px solid #313244;border-radius:6px;margin-top:8px;color:#89b4fa;font-weight:bold;}"
            "QGroupBox::title{subcontrol-origin:margin;left:8px;}"
        )
        il = QVBoxLayout(info_box)
        self._model_info_lbl = QLabel("Qwen3-VL 2B\nDevice: --\nStatus: Loading…")
        self._model_info_lbl.setStyleSheet("color:#a6adc8; font-size:10px;")
        self._model_info_lbl.setWordWrap(True)
        il.addWidget(self._model_info_lbl)
        lay.addWidget(info_box)

        # Generation settings
        gen_box = QGroupBox("Generation Settings")
        gen_box.setStyleSheet(
            "QGroupBox{border:1px solid #313244;border-radius:6px;margin-top:8px;color:#89b4fa;font-weight:bold;}"
            "QGroupBox::title{subcontrol-origin:margin;left:8px;}"
        )
        gl = QVBoxLayout(gen_box)

        gl.addWidget(QLabel("Max Tokens:"))
        self._max_tokens_spin = QSpinBox()
        self._max_tokens_spin.setRange(64, 4096)
        self._max_tokens_spin.setValue(512)
        self._max_tokens_spin.setSingleStep(64)
        self._max_tokens_spin.setStyleSheet(
            "QSpinBox{background:#313244;color:#cdd6f4;border:1px solid #45475a;border-radius:4px;padding:2px;}"
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
        lay.addWidget(gen_box)

        # Quick prompts
        qp_box = QGroupBox("Quick Prompts")
        qp_box.setStyleSheet(
            "QGroupBox{border:1px solid #313244;border-radius:6px;margin-top:8px;color:#89b4fa;font-weight:bold;}"
            "QGroupBox::title{subcontrol-origin:margin;left:8px;}"
        )
        ql = QVBoxLayout(qp_box)

        cat_combo = QComboBox()
        cat_combo.addItems(list(QUICK_PROMPTS.keys()))
        cat_combo.setStyleSheet(
            "QComboBox{background:#313244;color:#cdd6f4;border:1px solid #45475a;border-radius:4px;padding:2px 6px;}"
            "QComboBox QAbstractItemView{background:#313244;color:#cdd6f4;}"
        )
        ql.addWidget(cat_combo)

        self._qp_lay = QVBoxLayout()
        ql.addLayout(self._qp_lay)

        def _populate(cat: str):
            while self._qp_lay.count():
                item = self._qp_lay.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            for label, prompt in QUICK_PROMPTS.get(cat, []):
                btn = QPushButton(label)
                btn.setFixedHeight(26)
                btn.setStyleSheet(
                    "QPushButton{background:#313244;color:#cdd6f4;border:none;"
                    "border-radius:4px;padding:2px 8px;text-align:left;font-size:11px;}"
                    "QPushButton:hover{background:#45475a;color:#89b4fa;}"
                )
                btn.clicked.connect(lambda _, p=prompt: self._set_input(p))
                self._qp_lay.addWidget(btn)

        cat_combo.currentTextChanged.connect(_populate)
        _populate("Production")
        lay.addWidget(qp_box)

        # Conversation controls
        conv_box = QGroupBox("Conversation")
        conv_box.setStyleSheet(
            "QGroupBox{border:1px solid #313244;border-radius:6px;margin-top:8px;color:#89b4fa;font-weight:bold;}"
            "QGroupBox::title{subcontrol-origin:margin;left:8px;}"
        )
        cl = QVBoxLayout(conv_box)
        btn_new = QPushButton("New Conversation")
        btn_new.setFixedHeight(28)
        btn_new.clicked.connect(self._new_conversation)
        cl.addWidget(btn_new)
        self._turn_lbl = QLabel("Turns: 0")
        self._turn_lbl.setStyleSheet("color:#6c7086; font-size:10px;")
        cl.addWidget(self._turn_lbl)
        lay.addWidget(conv_box)

        lay.addStretch()
        return w

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
            "QTextBrowser{background:#1e1e2e;border:none;padding:14px;}"
            "QScrollBar:vertical{background:#1e1e2e;width:6px;border-radius:3px;}"
            "QScrollBar::handle:vertical{background:#45475a;border-radius:3px;min-height:20px;}"
        )
        pl.addWidget(self._chat)

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background:#313244;")
        pl.addWidget(sep)

        # Input area
        input_area = QWidget()
        input_area.setFixedHeight(130)
        input_area.setStyleSheet("background:#181825;")
        ia = QVBoxLayout(input_area)
        ia.setContentsMargins(12, 8, 12, 8)
        ia.setSpacing(6)

        hint = QLabel(
            "Ctrl+Enter to send  \u2022  Shift+Enter for new line  \u2022  "
            "Paste tables, numbers, labels freely"
        )
        hint.setStyleSheet("color:#6c7086; font-size:10px;")
        ia.addWidget(hint)

        input_row = QHBoxLayout()

        self._input = SmartInputBox()
        self._input.send_requested.connect(self._send)
        self._input.setPlaceholderText(
            "Ask about production planning, SKU performance, demand trends, "
            "or paste classification/numerical data for analysis…"
        )
        self._input.setFont(QFont("Segoe UI", 10))
        self._input.setStyleSheet(
            "QTextEdit{background:#313244;color:#cdd6f4;"
            "border:1px solid #45475a;border-radius:8px;padding:8px 12px;}"
            "QTextEdit:focus{border-color:#89b4fa;}"
        )
        input_row.addWidget(self._input)

        btn_col = QVBoxLayout()
        btn_col.setSpacing(4)

        self._send_btn = QPushButton("Send")
        self._send_btn.setFixedSize(80, 40)
        self._send_btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self._send_btn.setStyleSheet(
            "QPushButton{background:#89b4fa;color:#1e1e2e;border:none;border-radius:8px;}"
            "QPushButton:hover{background:#b4befe;}"
            "QPushButton:disabled{background:#45475a;color:#6c7086;}"
        )
        self._send_btn.clicked.connect(self._send)
        btn_col.addWidget(self._send_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setFixedSize(80, 28)
        self._stop_btn.setStyleSheet(
            "QPushButton{background:#f38ba8;color:#1e1e2e;border:none;border-radius:6px;}"
            "QPushButton:hover{background:#eba0ac;}"
        )
        self._stop_btn.clicked.connect(self._stop)
        btn_col.addWidget(self._stop_btn)

        input_row.addLayout(btn_col)
        ia.addLayout(input_row)
        pl.addWidget(input_area)

        return panel

    def _init_cursor_timer(self):
        """Blink the ▌ cursor in the streaming bubble every 500 ms."""
        self._cursor_timer.setInterval(500)
        self._cursor_timer.timeout.connect(self._blink_cursor)

    # ──────────────────────────────────────────────────────────────────
    # Public API (called by MainWindow)
    # ──────────────────────────────────────────────────────────────────

    def on_model_ready(self, success: bool, message: str):
        mm = ModelManager.get()
        if success:
            self._model_dot.setStyleSheet("color:#a6e3a1; font-size:14px;")
            self._model_prog.hide()
            dev = mm.device_info
            is_gpu = "GPU" in dev
            device_label = "GPU" if is_gpu else "CPU"
            compiled = "compiled" in dev
            extra = " + torch.compile" if compiled else ""
            self._model_status_lbl.setText(
                f"Qwen3-VL 2B  \u2022  Ready  \u2022  {device_label}{extra}  \u2022  Streaming ON"
            )
            self._model_info_lbl.setText(
                f"Qwen3-VL 2B\n{dev}\nMode: streaming (parallel threads)\nStatus: Ready"
            )
            self._append_completed(_html_system(f"Model ready -- {dev}"))
        else:
            self._model_dot.setStyleSheet("color:#f38ba8; font-size:14px;")
            self._model_status_lbl.setText(f"Model error: {message[:60]}")
            self._model_prog.hide()
            self._model_info_lbl.setText(f"Status: Failed\n{message[:120]}")
            self._append_completed(_html_system(f"Model failed to load: {message}"))

    def set_analysis_data(
        self,
        summary: dict | None = None,
        cluster_result=None,
        stats_context: dict | None = None,
        plan_df=None,
        df=None,           # raw DataFrame – gives the LLM actual data access
    ):
        if summary is not None:
            self._summary = summary
        if cluster_result is not None:
            self._cluster_result = cluster_result
        if stats_context is not None:
            self._stats_context = stats_context
        if plan_df is not None:
            self._plan_df = plan_df
        if df is not None:
            self._df = df
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
            df=self._df,
        )
        parts = []
        if self._df is not None:
            rows = len(self._df)
            parts.append(f"DATA ({rows:,} rows)")
        elif self._summary:
            parts.append(f"{self._summary.get('unique_skus','?')} SKUs")
        if self._cluster_result is not None:
            parts.append("clusters")
        if self._stats_context:
            parts.append("stats")
        if self._plan_df is not None:
            parts.append("plan")
        self._ctx_lbl.setText(
            "Context: " + (", ".join(parts) if parts else "not loaded")
        )

    # ──────────────────────────────────────────────────────────────────
    # Send / receive
    # ──────────────────────────────────────────────────────────────────

    def _send(self):
        text = self._input.toPlainText().strip()
        if not text:
            return
        if self._worker and self._worker.isRunning():
            return

        ts = datetime.datetime.now().strftime("%H:%M")
        self._append_completed(_html_user(text, ts))
        self._input.clear()

        # Build message history
        history: List[Dict[str, str]] = []
        if self._system_prompt:
            history.append({"role": "system", "content": self._system_prompt})
        history.extend(self._messages)
        history.append({"role": "user", "content": text})

        # Begin streaming bubble
        self._stream_ts = datetime.datetime.now().strftime("%H:%M")
        self._streaming_text = ""
        self._is_streaming = True
        self._token_count = 0
        self._stream_start = datetime.datetime.now()
        self._cursor_visible = True
        self._cursor_timer.start()
        self._send_btn.setEnabled(False)
        self._refresh_chat()

        # Start parallel streaming worker
        self._worker = StreamingInferenceWorker(
            messages=history,
            max_tokens=self._max_tokens_spin.value(),
            temperature=self._temp_slider.value() / 10,
            enable_thinking=self._thinking_chk.isChecked(),
        )
        self._worker.token_ready.connect(self._on_token)
        self._worker.stream_done.connect(self._on_stream_done)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

        self._messages.append({"role": "user", "content": text})
        self._update_turn_count()

    def _on_token(self, token: str):
        """Called on GUI thread for every token emitted by the worker."""
        self._streaming_text += token
        self._token_count += 1
        # Update the live bubble (cursor stays visible while tokens arrive)
        self._cursor_visible = True
        self._refresh_chat()
        # Update tokens/s
        elapsed = (datetime.datetime.now() - self._stream_start).total_seconds()
        if elapsed > 0.5:
            tps = self._token_count / elapsed
            self._tps_lbl.setText(f"{tps:.1f} tok/s")

    def _on_stream_done(self, full_text: str):
        """Called on GUI thread when generation is complete."""
        self._cursor_timer.stop()
        self._is_streaming = False
        self._send_btn.setEnabled(True)

        # Move the finished message into _completed_html
        ts = self._stream_ts
        self._streaming_text = full_text   # use server-provided complete text
        self._append_completed(_html_ai(full_text, ts))
        self._streaming_text = ""
        self._refresh_chat()

        self._messages.append({"role": "assistant", "content": full_text})
        self._update_turn_count()

        elapsed = (datetime.datetime.now() - self._stream_start).total_seconds()
        tps = self._token_count / elapsed if elapsed > 0 else 0
        self._tps_lbl.setText(f"{tps:.1f} tok/s (done)")

    def _on_error(self, error: str):
        self._cursor_timer.stop()
        self._is_streaming = False
        self._send_btn.setEnabled(True)
        self._streaming_text = ""
        self._append_completed(_html_system(f"Error: {error}"))
        self._refresh_chat()

    def _stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.request_stop()
            self._cursor_timer.stop()
            self._is_streaming = False
            self._send_btn.setEnabled(True)
            # Finalise whatever text was received before stop
            partial = self._streaming_text
            self._streaming_text = ""
            if partial:
                self._append_completed(
                    _html_ai(partial + " [stopped]", self._stream_ts)
                )
                self._messages.append({"role": "assistant", "content": partial})
            self._append_completed(_html_system("Generation stopped by user."))
            self._refresh_chat()

    def _new_conversation(self):
        if self._worker and self._worker.isRunning():
            self._worker.request_stop()
        self._messages.clear()
        self._streaming_text = ""
        self._is_streaming = False
        self._cursor_timer.stop()
        self._send_btn.setEnabled(True)
        self._completed_html = _html_system("New conversation started")
        self._refresh_chat()
        self._update_turn_count()

    # ──────────────────────────────────────────────────────────────────
    # Chat HTML management
    # ──────────────────────────────────────────────────────────────────

    def _append_completed(self, html: str):
        """Add a finished HTML block to the permanent history."""
        self._completed_html += html

    def _refresh_chat(self):
        """
        Rebuild the QTextBrowser HTML from:
          _completed_html  +  (streaming bubble if active)

        This is called on every token.  The completed section is kept as a
        pre-built string so we only re-render the streaming part each time.
        """
        body = _HTML_BODY_OPEN + self._completed_html

        if self._is_streaming or self._streaming_text:
            caret = self._cursor_visible and self._is_streaming
            body += _html_ai(self._streaming_text, self._stream_ts, cursor=caret)

        body += _HTML_BODY_CLOSE

        # Preserve scroll position: stay at bottom if user hasn't scrolled up
        sb = self._chat.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 40

        self._chat.setHtml(body)

        if at_bottom:
            sb.setValue(sb.maximum())

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _blink_cursor(self):
        """Toggle the blinking cursor in the streaming bubble."""
        if self._is_streaming:
            self._cursor_visible = not self._cursor_visible
            self._refresh_chat()

    def _set_input(self, text: str):
        self._input.setPlainText(text)
        self._input.setFocus()

    def _update_turn_count(self):
        turns = len([m for m in self._messages if m["role"] == "user"])
        self._turn_lbl.setText(f"Turns: {turns}")
