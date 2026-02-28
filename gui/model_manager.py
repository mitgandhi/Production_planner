"""
model_manager.py
----------------
Singleton that owns the Qwen3-VL model lifecycle:
  - Loads once in a background thread at app start
  - generate()        – blocking, returns full string  (legacy)
  - generate_stream() – yields one token at a time via TextIteratorStreamer
                        model.generate() runs in a second daemon thread so
                        the GPU and the token-decoding pipeline are parallel

Threading diagram for streaming inference
-----------------------------------------
  QThread (StreamingInferenceWorker)
      |
      | calls generate_stream()  ──▶  spawns threading.Thread  ──▶  model.generate() [GPU]
      |                                                                       |
      |  for token in TextIteratorStreamer  ◀──── internal Queue.put() ──────┘
      |      emit token_ready(token)
      |
      ▼
  GUI thread  ─── appends token to live bubble

Windows note:
  torch is imported at module level so CUDA DLLs are loaded on the main
  thread (where main.py has already registered CUDA DLL paths).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import List, Dict, Iterator, Optional

# ── Import torch at module level so CUDA DLLs are loaded on the main thread ──
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

MODEL_DIR = str(Path(__file__).parent.parent / "models" / "Qwen3")


class ModelManager:
    """Thread-safe singleton model manager for Qwen3-VL (text inference)."""

    _instance: "ModelManager | None" = None
    _lock = threading.Lock()

    # ------------------------------------------------------------------
    # Singleton accessor
    # ------------------------------------------------------------------

    @classmethod
    def get(cls) -> "ModelManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._ready = threading.Event()
        self._error: str = ""
        self._loading = False
        self.device_info: str = "unknown"
        self._compiled: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set() and self._model is not None

    @property
    def load_error(self) -> str:
        return self._error

    def start_loading(self, callback=None, compile_model: bool = True):
        """
        Begin loading the model in a daemon thread.
        callback(success: bool, message: str) is called when done.
        compile_model: if True, call torch.compile() after load for faster inference.
        """
        if self._loading or self.is_ready:
            return
        self._loading = True
        t = threading.Thread(
            target=self._load, args=(callback, compile_model), daemon=True
        )
        t.start()

    def _load(self, callback=None, compile_model: bool = True):
        try:
            from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer

            if not _TORCH_AVAILABLE:
                raise RuntimeError("torch is not installed.")

            # ── device / dtype selection ───────────────────────────────
            if torch.cuda.is_available():
                device_map = "cuda"
                dtype      = torch.bfloat16
                gpu_name   = torch.cuda.get_device_name(0)
                free_gb    = torch.cuda.mem_get_info(0)[0] / 1e9
                self.device_info = (
                    f"GPU  {gpu_name}  ({free_gb:.1f} GB free)  bfloat16"
                )
            else:
                device_map = "cpu"
                dtype      = torch.float32
                self.device_info = "CPU  float32  (no CUDA GPU detected)"

            self._tokenizer = AutoTokenizer.from_pretrained(
                MODEL_DIR, trust_remote_code=True
            )
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                MODEL_DIR,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
            self._model.eval()

            # ── Optional: torch.compile() for faster repeated inference ──
            # Compiles CUDA kernels; first call is slow (warm-up), later
            # calls are 20-40% faster.  Requires PyTorch >= 2.0.
            if compile_model and torch.cuda.is_available():
                try:
                    self._model = torch.compile(
                        self._model,
                        mode="reduce-overhead",   # best for inference loops
                        fullgraph=False,          # safer for HF models
                    )
                    self._compiled = True
                    self.device_info += "  compiled"
                except Exception as compile_err:
                    # torch.compile is optional – carry on without it
                    self.device_info += "  (compile skipped)"

            # ── Report VRAM after load ──────────────────────────────────
            if torch.cuda.is_available():
                total_b = torch.cuda.mem_get_info(0)[1]
                free_b  = torch.cuda.mem_get_info(0)[0]
                used_gb = (total_b - free_b) / 1e9
                self.device_info += f"  VRAM: {used_gb:.2f} GB"

            self._ready.set()
            if callback:
                callback(True, f"Qwen3-VL ready  --  {self.device_info}")

        except Exception as exc:
            self._error = str(exc)
            self._ready.set()
            if callback:
                callback(False, f"Model load failed: {exc}")

    # ------------------------------------------------------------------
    # Blocking generate (legacy – kept for compatibility)
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        enable_thinking: bool = False,
    ) -> str:
        """
        Run inference synchronously (call from a QThread, not the GUI thread).
        Returns the complete generated string.
        """
        self._ready.wait()
        if self._model is None:
            return f"[Model not loaded: {self._error}]"

        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        from transformers import GenerationConfig
        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, generation_config=gen_cfg)

        new_ids = output_ids[0][inputs.input_ids.shape[-1]:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # Streaming generate – token by token
    # ------------------------------------------------------------------

    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        enable_thinking: bool = False,
    ) -> Iterator[str]:
        """
        Stream inference one token at a time.

        Architecture (parallel threads):
            Thread A (caller / QThread)
                calls generate_stream()
                |
                spawns Thread B  ──▶  model.generate() on GPU
                |                         |
                |  TextIteratorStreamer  ◀─┘  (internal queue)
                |
                for token in streamer: yield token   ◀── tokens arrive as GPU generates

        Yields decoded token strings.  Call from a QThread (not GUI thread).
        """
        from transformers import TextIteratorStreamer, GenerationConfig

        # Wait for model to be ready (handles race if called before load finishes)
        self._ready.wait()
        if self._model is None:
            yield f"[Model not loaded: {self._error}]"
            return

        # Build prompt text
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        # Streamer: bridges generate() (Thread B) → token loop (Thread A) via queue
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,          # don't re-emit the input tokens
            skip_special_tokens=True,  # hide <eos>, <pad>, etc.
            timeout=60.0,              # max seconds to wait for next token
        )

        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        # Thread B: runs model.generate() on GPU; puts tokens into streamer's queue
        def _run_generate():
            with torch.no_grad():
                self._model.generate(
                    **inputs,
                    streamer=streamer,
                    generation_config=gen_cfg,
                )

        gen_thread = threading.Thread(target=_run_generate, daemon=True)
        gen_thread.start()

        # Thread A (this call): reads token strings from the queue as GPU produces them
        try:
            for token_text in streamer:
                yield token_text
        finally:
            gen_thread.join(timeout=90)   # always wait for the GPU thread to finish


# ------------------------------------------------------------------
# Data context builder  (injects actual DataFrame rows into the prompt)
# ------------------------------------------------------------------

def build_data_context(df) -> str:
    """
    Convert the loaded DataFrame into a comprehensive text report that the
    LLM can read as if it were given the file contents directly.

    Sections included
    -----------------
    1. Monthly total sales (last 24 months)
    2. All SKUs ranked by volume  (total, avg/month, CV%)
    3. Top 30 SKUs – last 12 months month-by-month breakdown
    4. Production estimates (last 90 days basis, 45-day horizon)
    5. Size distribution
    6. Color distribution (top 20)
    7. Top 50 SKU x Size x Color combinations
    """
    import pandas as pd
    import numpy as np

    if df is None or df.empty:
        return "(No data loaded yet.)"

    lines: list[str] = []

    # ── 1. Monthly totals (last 24 months) ──────────────────────────────────
    monthly_total = df.groupby("date")["c_qty"].sum().sort_index()
    lines += ["=== MONTHLY TOTAL SALES (last 24 months) ==="]
    for dt, qty in monthly_total.tail(24).items():
        lines.append(f"  {dt.strftime('%b-%Y')}: {int(qty):,}")

    # ── 2. All SKUs ranked by volume ─────────────────────────────────────────
    monthly_sku = df.groupby(["date", "c_sku"])["c_qty"].sum().reset_index()
    sku_agg = (
        monthly_sku.groupby("c_sku")["c_qty"]
        .agg(total="sum", mean="mean", std="std")
    )
    sku_agg["cv_pct"] = (
        (sku_agg["std"] / sku_agg["mean"] * 100)
        .fillna(0)
        .round(1)
    )
    sku_agg = sku_agg.sort_values("total", ascending=False)

    lines += [
        "",
        "=== ALL SKUs RANKED BY TOTAL VOLUME ===",
        f"{'SKU':<12} | {'Total Units':>11} | {'Avg/Month':>9} | {'CV%':>6}",
        "-" * 48,
    ]
    for sku, row in sku_agg.iterrows():
        lines.append(
            f"{str(sku):<12} | {int(row['total']):>11,} | "
            f"{row['mean']:>9,.0f} | {row['cv_pct']:>5.1f}%"
        )

    # ── 3. Top 30 SKUs – last 12 months breakdown ────────────────────────────
    top30 = sku_agg.head(30).index.tolist()
    last_12_cutoff = monthly_sku["date"].max() - pd.DateOffset(months=11)
    last_12 = monthly_sku[
        (monthly_sku["date"] >= last_12_cutoff) &
        (monthly_sku["c_sku"].isin(top30))
    ]
    pivot = last_12.pivot_table(
        index="c_sku", columns="date", values="c_qty", fill_value=0
    )
    pivot.columns = [c.strftime("%b%y") for c in pivot.columns]

    col_hdr = " | ".join(f"{c:>6}" for c in pivot.columns)
    lines += [
        "",
        "=== TOP 30 SKUs - LAST 12 MONTHS (monthly qty) ===",
        f"{'SKU':<12} | {col_hdr}",
        "-" * (14 + len(pivot.columns) * 9),
    ]
    for sku, row in pivot.iterrows():
        vals = " | ".join(f"{int(v):>6,}" for v in row.values)
        lines.append(f"{str(sku):<12} | {vals}")

    # ── 4. Production estimates ───────────────────────────────────────────────
    try:
        from src.production_planning import current_production_estimate
        prod_est = current_production_estimate(df, sales_days=90, production_days=45)
        lines += [
            "",
            "=== PRODUCTION ESTIMATES (basis: last 90 days sales, 45-day production horizon) ===",
            f"{'SKU':<12} | {'Size':<6} | {'Color':<8} | {'90d Sales':>9} | {'Per-Day':>7} | {'Prod Req (45d)':>13}",
            "-" * 68,
        ]
        for _, r in prod_est.head(120).iterrows():
            lines.append(
                f"{str(r['c_sku']):<12} | {str(r['c_sz']):<6} | "
                f"{str(r['c_cl']):<8} | {int(r['total_sales_90d']):>9,} | "
                f"{r['per_day_sales_qty']:>7.2f} | {int(r['production_req_qty']):>13,}"
            )
    except Exception as _e:
        lines.append(f"(Production estimates unavailable: {_e})")

    # ── 5. Size distribution ──────────────────────────────────────────────────
    size_agg = df.groupby("c_sz")["c_qty"].sum().sort_values(ascending=False)
    total_qty = int(size_agg.sum())
    lines += ["", "=== SIZE DISTRIBUTION ==="]
    for sz, qty in size_agg.items():
        lines.append(
            f"  Size {str(sz):<8}: {int(qty):>10,} units  ({qty/total_qty*100:.1f}%)"
        )

    # ── 6. Color distribution (top 20) ────────────────────────────────────────
    color_agg = df.groupby("c_cl")["c_qty"].sum().sort_values(ascending=False)
    lines += ["", "=== COLOR DISTRIBUTION (top 20) ==="]
    for cl, qty in color_agg.head(20).items():
        lines.append(
            f"  Color {str(cl):<10}: {int(qty):>10,} units  ({qty/total_qty*100:.1f}%)"
        )

    # ── 7. Top 50 SKU × Size × Color combos ──────────────────────────────────
    combo = (
        df.groupby(["c_sku", "c_sz", "c_cl"])["c_qty"]
        .sum().reset_index()
        .sort_values("c_qty", ascending=False)
    )
    lines += [
        "",
        "=== TOP 50 SKU x SIZE x COLOR COMBINATIONS ===",
        f"{'SKU':<12} | {'Size':<6} | {'Color':<10} | {'Total Units':>11}",
        "-" * 48,
    ]
    for _, r in combo.head(50).iterrows():
        lines.append(
            f"{str(r['c_sku']):<12} | {str(r['c_sz']):<6} | "
            f"{str(r['c_cl']):<10} | {int(r['c_qty']):>11,}"
        )

    return "\n".join(lines)


# ------------------------------------------------------------------
# System prompt builder
# ------------------------------------------------------------------

def build_system_prompt(
    summary: dict | None = None,
    cluster_result=None,
    stats_context: dict | None = None,
    plan_df=None,
    df=None,           # <-- actual DataFrame for rich data injection
) -> str:
    """
    Build a rich, structured system prompt from all available analysis artefacts.
    The model uses this as background knowledge for every user query.
    """
    lines = [
        "You are an expert AI Production Planning Assistant for Gem Computers, "
        "an apparel manufacturing company specialising in women's intimate apparel (bras).",
        "",
        "IMPORTANT: The complete inventory dataset has been loaded and is embedded below.",
        "You have full access to every SKU, size, color, monthly sales figure, and production",
        "estimate. Treat this data as if you have read the file yourself.",
        "Always answer with specific numbers from the data. Never say you cannot access the file.",
        "",
        "=== DATASET OVERVIEW ===",
    ]

    if summary:
        lines += [
            f"Total records  : {summary.get('total_records', 'N/A'):,}",
            f"Unique SKUs    : {summary.get('unique_skus', 'N/A')}",
            f"Unique sizes   : {summary.get('unique_sizes', 'N/A')}",
            f"Unique colors  : {summary.get('unique_colors', 'N/A')}",
            f"Total units    : {summary.get('total_units', 'N/A'):,}",
            f"Date range     : {summary.get('date_min', '?')} to {summary.get('date_max', '?')}",
            f"Years covered  : {summary.get('years_covered', 'N/A')}",
        ]

    # ── Inject actual data tables from the DataFrame ─────────────────────────
    if df is not None:
        lines += ["", build_data_context(df)]

    # ── Cluster results ───────────────────────────────────────────────────────
    if cluster_result is not None and not cluster_result.empty:
        lines += ["", "=== CLUSTER / PARTITION ANALYSIS ==="]
        if "cluster_label" in cluster_result.columns:
            for lbl, grp in cluster_result.groupby("cluster_label"):
                skus = grp["c_sku"].tolist() if "c_sku" in grp.columns else []
                vol = int(grp["total_qty"].sum()) if "total_qty" in grp.columns else 0
                lines.append(f"[{lbl}] - {len(skus)} SKUs, total vol {vol:,}")
                lines.append(f"  SKUs: {', '.join(skus[:8])}{'...' if len(skus) > 8 else ''}")
        if "abc" in cluster_result.columns:
            abc_cnt = cluster_result["abc"].value_counts().to_dict()
            lines.append(f"ABC - A:{abc_cnt.get('A',0)} B:{abc_cnt.get('B',0)} C:{abc_cnt.get('C',0)} SKUs")
        if "xyz" in cluster_result.columns:
            xyz_cnt = cluster_result["xyz"].value_counts().to_dict()
            lines.append(f"XYZ - X:{xyz_cnt.get('X',0)} Y:{xyz_cnt.get('Y',0)} Z:{xyz_cnt.get('Z',0)} SKUs")

    # ── Statistical insights ──────────────────────────────────────────────────
    if stats_context:
        lines += ["", "=== STATISTICAL INSIGHTS (from Stats tab) ==="]
        top_skus = stats_context.get("top_skus_by_volume", [])[:10]
        lines.append(f"Top SKUs by volume: {', '.join(top_skus)}")
        ts = stats_context.get("trend_summary", {})
        up = ts.get("uptrend_skus", [])[:6]
        dn = ts.get("downtrend_skus", [])[:6]
        if up:
            lines.append(f"Growing (uptrend) SKUs: {', '.join(up)}")
        if dn:
            lines.append(f"Declining SKUs: {', '.join(dn)}")
        sku_stats = stats_context.get("sku_stats", {})
        if sku_stats:
            lines.append("Key SKU details:")
            for sku, st in list(sku_stats.items())[:8]:
                lines.append(
                    f"  {sku}: avg {st.get('avg_monthly', 0):,.0f}/mo, "
                    f"CV {st.get('cv_pct', 0):.1f}%, "
                    f"trend={st.get('trend_direction','?')}, "
                    f"peak_month=M{st.get('peak_month', '?')}"
                )

    # ── Production plan ───────────────────────────────────────────────────────
    if plan_df is not None and not plan_df.empty:
        lines += ["", "=== PRODUCTION PLAN (latest run from Production tab) ==="]
        top5 = (
            plan_df.nlargest(5, "avg_monthly_qty")
            if "avg_monthly_qty" in plan_df.columns
            else plan_df.head(5)
        )
        for _, r in top5.iterrows():
            lines.append(
                f"{r.get('c_sku','?')}: avg {r.get('avg_monthly_qty',0):,.0f}/mo, "
                f"SS={r.get('safety_stock',0):,.0f}, "
                f"6m forecast={r.get('total_planned',0):,.0f}"
            )

    lines += [
        "",
        "=== INSTRUCTIONS ===",
        "You have READ the dataset above. Answer every question using the actual numbers.",
        "Never say 'I cannot access the file' - the data IS already here in this prompt.",
        "When asked about a SKU, size, or color - look it up in the tables above and cite the number.",
        "When you give recommendations, state which table/section the data came from.",
        "If the user pastes additional data, analyse it together with the embedded dataset.",
    ]

    return "\n".join(lines)
