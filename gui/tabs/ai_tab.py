"""
ai_tab.py
---------
Claude-style chat interface for Qwen3-VL with PARALLEL STREAMING inference.

How it works like a terminal session
--------------------------------------
1. When data is loaded, a ContextBuilderWorker builds the system prompt in a
   background thread (pandas aggregations on 152K rows) - GUI never freezes.

2. For every message the user sends, _build_message_context() queries the
   DataFrame directly using keyword detection (SKU names, sizes, colors,
   keywords like "forecast", "production") and injects the matching rows into
   the user message before it reaches the model.
   This is exactly how a terminal Claude session works: you paste data and ask
   questions - we do the same but automatically, behind the scenes.

3. Streaming via TextIteratorStreamer + two parallel threads (see model_manager).
"""

from __future__ import annotations

import datetime
import re
from typing import List, Dict

import pandas as pd

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QKeyEvent, QTextCursor
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QGroupBox,
    QHBoxLayout, QLabel, QProgressBar, QPushButton,
    QSlider, QSpinBox,
    QSplitter, QTextBrowser, QTextEdit, QVBoxLayout, QWidget,
)

from gui.model_manager import ModelManager, build_system_prompt, build_data_context


# ─────────────────────────────────────────────────────────────────────────────
# Background context builder – runs build_data_context() off the main thread
# ─────────────────────────────────────────────────────────────────────────────

class ContextBuilderWorker(QThread):
    """Builds the full system prompt in a background thread."""
    context_ready = pyqtSignal(str)   # emits the finished system-prompt string

    def __init__(self, summary, cluster_result, stats_context, plan_df, df):
        super().__init__()
        self._summary        = summary
        self._cluster_result = cluster_result
        self._stats_context  = stats_context
        self._plan_df        = plan_df
        self._df             = df

    def run(self):
        prompt = build_system_prompt(
            summary=self._summary or None,
            cluster_result=self._cluster_result,
            stats_context=self._stats_context or None,
            plan_df=self._plan_df,
            df=self._df,
        )
        self.context_ready.emit(prompt)


# ─────────────────────────────────────────────────────────────────────────────
# Streaming inference worker
# ─────────────────────────────────────────────────────────────────────────────

class StreamingInferenceWorker(QThread):
    token_ready    = pyqtSignal(str)
    stream_done    = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, messages, max_tokens, temperature, enable_thinking):
        super().__init__()
        self._messages       = messages
        self._max_tokens     = max_tokens
        self._temperature    = temperature
        self._thinking       = enable_thinking
        self._stop_requested = False

    def request_stop(self):
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
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _render_markdown(text: str) -> str:
    escaped = _escape(text)
    escaped = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', escaped)
    escaped = re.sub(
        r'`([^`]+)`',
        r'<code style="background:#2a2a3d;padding:1px 5px;border-radius:3px;'
        r'font-family:Consolas,monospace;">\1</code>',
        escaped,
    )
    lines = escaped.split("\n")
    out = []
    for line in lines:
        s = line.strip()
        if s.startswith(("- ", "* ", "+ ")):
            out.append(f'<li style="margin:2px 0;">{s[2:]}</li>')
        elif re.match(r'^[\u2022\u2023\u25E6\u2043] ', s):
            out.append(f'<li style="margin:2px 0;">{s[2:]}</li>')
        elif re.match(r'^\d+\.\s', s):
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
# Smart input: Ctrl+Enter sends
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
# Quick prompts
# ─────────────────────────────────────────────────────────────────────────────

QUICK_PROMPTS: Dict[str, List[tuple]] = {
    "Production": [
        ("6-Month Forecast",
         "Give me a 6-month production plan for the top 10 SKUs with forecast quantities, safety stock levels, and priority actions."),
        ("Safety Stock Review",
         "Which SKUs have inadequate safety stock? List them with current avg demand, variability (CV%), and recommended safety stock."),
        ("Reorder Points",
         "List the reorder points for the top 20 SKUs sorted by urgency. Include avg demand, lead time assumption, and reorder qty."),
        ("Production Schedule",
         "Create a prioritised production schedule for next quarter. Group by cluster (High/Medium/Low volume) and ABC class."),
    ],
    "Demand": [
        ("Top SKUs Analysis",
         "Analyse the top 5 SKUs by volume. Give monthly average, trend direction, seasonality, and 3-month forecast for each."),
        ("Growing Products",
         "Which SKUs show the strongest upward demand trend? Quantify the growth rate and recommend production increase %."),
        ("Declining Products",
         "Which SKUs are declining? Provide decline rate, current safety stock adequacy, and whether to phase out or reduce batch size."),
        ("Seasonal Patterns",
         "Which SKUs have the strongest seasonal demand? Identify peak months and recommend pre-season production build-up quantities."),
    ],
    "Inventory": [
        ("ABC-XYZ Summary",
         "Summarise the ABC-XYZ classification results. What does each category mean for inventory policy and production frequency?"),
        ("Size Analysis",
         "Which sizes drive the most volume overall? Break down by top 5 SKUs and recommend size ratio for production batches."),
        ("Color Strategy",
         "Which colors are consistently high demand vs seasonal? Recommend a color production mix strategy."),
        ("Risk Assessment",
         "Identify the top 5 inventory risk SKUs (high CV, high volume, long trend decline) and mitigation actions."),
    ],
    "Custom": [
        ("Paste Data & Analyse",
         "I will paste some data below. Please analyse it in the context of our apparel inventory and provide insights:\n\n[PASTE YOUR DATA HERE]"),
        ("Compare SKUs",
         "Compare these SKUs side by side: [SKU1, SKU2, SKU3]. Show volume, trend, seasonality, and production recommendation."),
        ("What-If Scenario",
         "If we increase production of [SKU] by 20% next quarter, what would be the impact on inventory levels and when would stock risk occur?"),
        ("Executive Summary",
         "Write a concise executive summary of the current production planning situation: top performers, risks, and 3 key actions."),
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic data injector
# ─────────────────────────────────────────────────────────────────────────────

def _build_message_context(user_text: str, df) -> str:
    """
    Like a terminal session: scan the user's question for SKU names, sizes,
    colors, and keywords, then fetch the matching rows from the DataFrame and
    append them as a DATA CONTEXT block to the message.

    This keeps the system prompt compact (no massive pre-dump) while still
    giving the model real numbers for whatever the user asks about.
    """
    if df is None or df.empty:
        return user_text

    upper = user_text.upper()
    snippets: List[str] = []

    # ── Detect SKU mentions ───────────────────────────────────────────────────
    all_skus = df["c_sku"].unique().tolist()
    mentioned_skus = [s for s in all_skus if s.upper() in upper]

    if mentioned_skus:
        for sku in mentioned_skus[:5]:
            sub = df[df["c_sku"] == sku]
            # Monthly totals
            monthly = sub.groupby("date")["c_qty"].sum().sort_index()
            lines = [f"SKU {sku} - monthly sales (last 18 months):"]
            for dt, qty in monthly.tail(18).items():
                lines.append(f"  {dt.strftime('%b-%Y')}: {int(qty):,}")
            # Top size×color for this SKU
            top_combo = (
                sub.groupby(["c_sz", "c_cl"])["c_qty"].sum()
                .sort_values(ascending=False).head(10)
            )
            lines.append(f"SKU {sku} - top size×color (last all-time):")
            for (sz, cl), qty in top_combo.items():
                lines.append(f"  {sz} / {cl}: {int(qty):,} units")
            snippets.append("\n".join(lines))

    # ── Detect size mentions ──────────────────────────────────────────────────
    all_sizes = df["c_sz"].astype(str).unique().tolist()
    mentioned_sizes = [s for s in all_sizes if s.upper() in upper]

    if mentioned_sizes and not mentioned_skus:
        for sz in mentioned_sizes[:3]:
            sub = df[df["c_sz"] == sz]
            total = int(sub["c_qty"].sum())
            top_skus = sub.groupby("c_sku")["c_qty"].sum().sort_values(ascending=False).head(10)
            lines = [f"Size {sz} - total {total:,} units. Top SKUs:"]
            for sku, qty in top_skus.items():
                lines.append(f"  {sku}: {int(qty):,}")
            snippets.append("\n".join(lines))

    # ── Detect color mentions ─────────────────────────────────────────────────
    all_colors = df["c_cl"].astype(str).unique().tolist()
    mentioned_colors = [c for c in all_colors if c.upper() in upper]

    if mentioned_colors and not mentioned_skus:
        for cl in mentioned_colors[:3]:
            sub = df[df["c_cl"] == cl]
            total = int(sub["c_qty"].sum())
            top_skus = sub.groupby("c_sku")["c_qty"].sum().sort_values(ascending=False).head(10)
            lines = [f"Color {cl} - total {total:,} units. Top SKUs:"]
            for sku, qty in top_skus.items():
                lines.append(f"  {sku}: {int(qty):,}")
            snippets.append("\n".join(lines))

    # ── Keyword: production / estimate ───────────────────────────────────────
    prod_keywords = ["production", "estimate", "manufacture", "produce", "requirement", "req qty"]
    if any(k in upper.lower() for k in prod_keywords) and not mentioned_skus:
        try:
            from src.production_planning import current_production_estimate
            est = current_production_estimate(df, sales_days=90, production_days=45)
            lines = ["PRODUCTION ESTIMATES (top 30, last-90-day basis):"]
            lines.append(f"{'SKU':<12} | {'Size':<6} | {'Color':<8} | {'Per-Day':>7} | {'45d Req':>10}")
            for _, r in est.head(30).iterrows():
                lines.append(
                    f"{str(r['c_sku']):<12} | {str(r['c_sz']):<6} | "
                    f"{str(r['c_cl']):<8} | {r['per_day_sales_qty']:>7.2f} | "
                    f"{int(r['production_req_qty']):>10,}"
                )
            snippets.append("\n".join(lines))
        except Exception:
            pass

    # ── Keyword: top / best / highest ────────────────────────────────────────
    top_keywords = ["top", "best", "highest", "most", "rank"]
    if any(k in upper.lower() for k in top_keywords) and not mentioned_skus:
        monthly_sku = df.groupby(["date", "c_sku"])["c_qty"].sum().reset_index()
        sku_totals = monthly_sku.groupby("c_sku")["c_qty"].sum().sort_values(ascending=False)
        lines = ["TOP 20 SKUs by total volume:"]
        for i, (sku, qty) in enumerate(sku_totals.head(20).items(), 1):
            lines.append(f"  {i:>2}. {sku}: {int(qty):,} units")
        snippets.append("\n".join(lines))

    # ── Keyword: trend / growth / decline ────────────────────────────────────
    trend_keywords = ["trend", "growth", "growing", "declining", "decline", "increase", "decrease"]
    if any(k in upper.lower() for k in trend_keywords) and not mentioned_skus:
        monthly_sku = df.groupby(["date", "c_sku"])["c_qty"].sum().reset_index()
        monthly_sku = monthly_sku.sort_values("date")
        recent = monthly_sku[monthly_sku["date"] >= monthly_sku["date"].max() - pd.DateOffset(months=12)]
        old    = monthly_sku[monthly_sku["date"] <= monthly_sku["date"].min() + pd.DateOffset(months=12)]
        recent_avg = recent.groupby("c_sku")["c_qty"].mean()
        old_avg    = old.groupby("c_sku")["c_qty"].mean()
        growth = ((recent_avg - old_avg) / (old_avg + 1) * 100).sort_values(ascending=False)
        lines = ["TREND (recent 12m avg vs first 12m avg):"]
        lines.append("Top growing SKUs:")
        for sku, pct in growth.head(10).items():
            lines.append(f"  {sku}: {pct:+.1f}%")
        lines.append("Most declining SKUs:")
        for sku, pct in growth.tail(10).items():
            lines.append(f"  {sku}: {pct:+.1f}%")
        snippets.append("\n".join(lines))

    if not snippets:
        return user_text

    data_block = "\n\n---\nDATA CONTEXT (fetched live from your dataset):\n" + "\n\n".join(snippets) + "\n---"
    return user_text + data_block


# ─────────────────────────────────────────────────────────────────────────────
# Main AI Tab widget
# ─────────────────────────────────────────────────────────────────────────────

class AITab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._messages: List[Dict[str, str]] = []
        self._system_prompt: str = ""
        self._worker: StreamingInferenceWorker | None = None
        self._ctx_worker: ContextBuilderWorker | None = None

        # Streaming state
        self._completed_html:  str = ""
        self._streaming_text:  str = ""
        self._stream_ts:       str = ""
        self._is_streaming:    bool = False
        self._token_count:     int  = 0
        self._stream_start     = None

        self._cursor_timer = QTimer()
        self._cursor_visible = True

        # Analysis context
        self._summary: dict = {}
        self._cluster_result = None
        self._stats_context: dict = {}
        self._plan_df = None
        self._df = None          # raw DataFrame – used for dynamic injection

        self._init_ui()
        self._init_cursor_timer()

        self._completed_html = (
            _html_system("Welcome to Qwen3 Production Planner Chat")
            + _html_system("Streaming active &mdash; responses appear token-by-token")
            + _html_system("Ctrl+Enter to send &nbsp;|&nbsp; Quick prompts in sidebar")
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
        splitter.setSizes([268, 1100])
        root.addWidget(splitter)

    def _build_status_bar(self) -> QFrame:
        bar = QFrame()
        bar.setFixedHeight(38)
        bar.setStyleSheet("QFrame{background:#181825;border-bottom:1px solid #313244;}")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(12, 4, 12, 4)

        self._model_dot = QLabel("&#9679;")
        self._model_dot.setStyleSheet("color:#f38ba8;font-size:14px;")
        lay.addWidget(self._model_dot)

        self._model_status_lbl = QLabel("Model loading…")
        self._model_status_lbl.setStyleSheet("color:#cdd6f4;font-size:11px;")
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

        self._tps_lbl = QLabel("")
        self._tps_lbl.setStyleSheet("color:#a6e3a1;font-size:10px;margin-left:8px;")
        lay.addWidget(self._tps_lbl)

        lay.addStretch()

        self._ctx_lbl = QLabel("Context: not loaded")
        self._ctx_lbl.setStyleSheet("color:#6c7086;font-size:10px;font-style:italic;")
        lay.addWidget(self._ctx_lbl)

        btn_rebuild = QPushButton("Rebuild Context")
        btn_rebuild.setFixedHeight(24)
        btn_rebuild.setStyleSheet(
            "QPushButton{background:#313244;color:#cdd6f4;border:none;"
            "border-radius:4px;padding:0 8px;font-size:10px;}"
            "QPushButton:hover{background:#45475a;}"
        )
        btn_rebuild.clicked.connect(self._trigger_context_build)
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
        self._model_info_lbl.setStyleSheet("color:#a6adc8;font-size:10px;")
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
        self._temp_slider.valueChanged.connect(lambda v: self._temp_lbl.setText(f"{v/10:.1f}"))
        temp_row.addWidget(self._temp_slider)
        temp_row.addWidget(self._temp_lbl)
        gl.addLayout(temp_row)

        self._thinking_chk = QCheckBox("Extended Thinking")
        self._thinking_chk.setStyleSheet("color:#cdd6f4;")
        self._thinking_chk.setToolTip("Enables Qwen3 deep reasoning mode.")
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
        self._turn_lbl.setStyleSheet("color:#6c7086;font-size:10px;")
        cl.addWidget(self._turn_lbl)
        lay.addWidget(conv_box)

        lay.addStretch()
        return w

    def _build_chat_panel(self) -> QWidget:
        panel = QWidget()
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(0)

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

        input_area = QWidget()
        input_area.setFixedHeight(130)
        input_area.setStyleSheet("background:#181825;")
        ia = QVBoxLayout(input_area)
        ia.setContentsMargins(12, 8, 12, 8)
        ia.setSpacing(6)

        hint = QLabel(
            "Ctrl+Enter to send  \u2022  Shift+Enter for newline  \u2022  "
            "Mention any SKU/size/color \u2014 data auto-fetched from your dataset"
        )
        hint.setStyleSheet("color:#6c7086;font-size:10px;")
        ia.addWidget(hint)

        input_row = QHBoxLayout()
        self._input = SmartInputBox()
        self._input.send_requested.connect(self._send)
        self._input.setPlaceholderText(
            "Ask anything: top SKUs, production requirements, trends, "
            "size analysis… mention a SKU name for live data lookup."
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
        self._cursor_timer.setInterval(500)
        self._cursor_timer.timeout.connect(self._blink_cursor)

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def on_model_ready(self, success: bool, message: str):
        mm = ModelManager.get()
        if success:
            self._model_dot.setStyleSheet("color:#a6e3a1;font-size:14px;")
            self._model_prog.hide()
            dev = mm.device_info
            compiled = "compiled" in dev
            extra = " + torch.compile" if compiled else ""
            device_label = "GPU" if "GPU" in dev else "CPU"
            self._model_status_lbl.setText(
                f"Qwen3-VL 2B  \u2022  Ready  \u2022  {device_label}{extra}  \u2022  Streaming ON"
            )
            self._model_info_lbl.setText(
                f"Qwen3-VL 2B\n{dev}\nMode: streaming + auto data lookup\nStatus: Ready"
            )
            self._append_completed(_html_system(f"Model ready -- {dev}"))
        else:
            self._model_dot.setStyleSheet("color:#f38ba8;font-size:14px;")
            self._model_status_lbl.setText(f"Model error: {message[:60]}")
            self._model_prog.hide()
            self._model_info_lbl.setText(f"Status: Failed\n{message[:120]}")
            self._append_completed(_html_system(f"Model failed: {message}"))

    def set_analysis_data(
        self,
        summary: dict | None = None,
        cluster_result=None,
        stats_context: dict | None = None,
        plan_df=None,
        df=None,
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
        self._trigger_context_build()

    # ──────────────────────────────────────────────────────────────────
    # Context building (background thread – never blocks the GUI)
    # ──────────────────────────────────────────────────────────────────

    def _trigger_context_build(self):
        """Launch context building in background so GUI stays responsive."""
        if self._ctx_worker and self._ctx_worker.isRunning():
            self._ctx_worker.quit()
            self._ctx_worker.wait(200)

        self._ctx_lbl.setText("Context: building…")
        self._ctx_worker = ContextBuilderWorker(
            summary=self._summary or None,
            cluster_result=self._cluster_result,
            stats_context=self._stats_context or None,
            plan_df=self._plan_df,
            df=self._df,
        )
        self._ctx_worker.context_ready.connect(self._on_context_ready)
        self._ctx_worker.start()

    def _on_context_ready(self, prompt: str):
        self._system_prompt = prompt
        parts = []
        if self._df is not None:
            parts.append(f"DATA ({len(self._df):,} rows + live lookup)")
        elif self._summary:
            parts.append(f"{self._summary.get('unique_skus','?')} SKUs")
        if self._cluster_result is not None:
            parts.append("clusters")
        if self._stats_context:
            parts.append("stats")
        if self._plan_df is not None:
            parts.append("plan")
        self._ctx_lbl.setText("Context: " + (", ".join(parts) if parts else "not loaded"))
        if self._df is not None:
            self._append_completed(
                _html_system(
                    f"Data context ready ({len(self._df):,} rows) -- "
                    f"mention any SKU, size or color for live lookup"
                )
            )
            self._refresh_chat()

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

        # ── Dynamic data injection: fetch relevant rows per message ───────────
        enriched_text = _build_message_context(text, self._df)

        # Build message history
        history: List[Dict[str, str]] = []
        if self._system_prompt:
            history.append({"role": "system", "content": self._system_prompt})
        history.extend(self._messages)
        history.append({"role": "user", "content": enriched_text})

        # Begin streaming bubble
        self._stream_ts     = datetime.datetime.now().strftime("%H:%M")
        self._streaming_text = ""
        self._is_streaming  = True
        self._token_count   = 0
        self._stream_start  = datetime.datetime.now()
        self._cursor_visible = True
        self._cursor_timer.start()
        self._send_btn.setEnabled(False)
        self._refresh_chat()

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

        # Store raw text (without injected data) in history for cleanliness
        self._messages.append({"role": "user", "content": text})
        self._update_turn_count()

    def _on_token(self, token: str):
        self._streaming_text += token
        self._token_count += 1
        self._cursor_visible = True
        self._refresh_chat()
        elapsed = (datetime.datetime.now() - self._stream_start).total_seconds()
        if elapsed > 0.5:
            self._tps_lbl.setText(f"{self._token_count / elapsed:.1f} tok/s")

    def _on_stream_done(self, full_text: str):
        self._cursor_timer.stop()
        self._is_streaming = False
        self._send_btn.setEnabled(True)
        self._append_completed(_html_ai(full_text, self._stream_ts))
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
            partial = self._streaming_text
            self._streaming_text = ""
            if partial:
                self._append_completed(_html_ai(partial + " [stopped]", self._stream_ts))
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
    # HTML management
    # ──────────────────────────────────────────────────────────────────

    def _append_completed(self, html: str):
        self._completed_html += html

    def _refresh_chat(self):
        body = _HTML_BODY_OPEN + self._completed_html
        if self._is_streaming or self._streaming_text:
            caret = self._cursor_visible and self._is_streaming
            body += _html_ai(self._streaming_text, self._stream_ts, cursor=caret)
        body += _HTML_BODY_CLOSE

        sb = self._chat.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 40
        self._chat.setHtml(body)
        if at_bottom:
            sb.setValue(sb.maximum())

    def _blink_cursor(self):
        if self._is_streaming:
            self._cursor_visible = not self._cursor_visible
            self._refresh_chat()

    def _set_input(self, text: str):
        self._input.setPlainText(text)
        self._input.setFocus()

    def _update_turn_count(self):
        turns = len([m for m in self._messages if m["role"] == "user"])
        self._turn_lbl.setText(f"Turns: {turns}")
