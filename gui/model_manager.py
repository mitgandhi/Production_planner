"""
model_manager.py
----------------
Singleton that owns the Qwen3-VL model lifecycle:
  - Loads once in a background thread at app start
  - Exposes a blocking generate() call (run inside a QThread from the UI)
  - Safe to call generate() before load completes – it blocks until ready
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import List, Dict, Optional

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
        self.device_info: str = "unknown"   # set after load

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set() and self._model is not None

    @property
    def load_error(self) -> str:
        return self._error

    def start_loading(self, callback=None):
        """
        Begin loading the model in a daemon thread.
        callback(success: bool, message: str) is called when done.
        """
        if self._loading or self.is_ready:
            return
        self._loading = True
        t = threading.Thread(target=self._load, args=(callback,), daemon=True)
        t.start()

    def _load(self, callback=None):
        try:
            from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer
            import torch

            # ── device / dtype selection ──────────────────────────────
            if torch.cuda.is_available():
                device_map = "cuda"
                dtype      = torch.bfloat16          # RTX 50xx natively supports BF16
                gpu_name   = torch.cuda.get_device_name(0)
                free_gb    = torch.cuda.mem_get_info(0)[0] / 1e9
                self.device_info = f"GPU  {gpu_name}  ({free_gb:.1f} GB free)  •  bfloat16"
            else:
                device_map = "cpu"
                dtype      = torch.float32
                self.device_info = "CPU  •  float32  (no CUDA GPU detected)"

            self._tokenizer = AutoTokenizer.from_pretrained(
                MODEL_DIR, trust_remote_code=True
            )
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                MODEL_DIR,
                dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
            self._model.eval()

            # report VRAM used after load
            if torch.cuda.is_available():
                used_gb = (torch.cuda.mem_get_info(0)[1] - torch.cuda.mem_get_info(0)[0]) / 1e9
                self.device_info += f"  •  VRAM used: {used_gb:.2f} GB"

            self._ready.set()
            if callback:
                callback(True, f"Qwen3-VL ready on {self.device_info}")
        except Exception as exc:
            self._error = str(exc)
            self._ready.set()
            if callback:
                callback(False, f"Model load failed: {exc}")

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        enable_thinking: bool = False,
        progress_callback=None,
    ) -> str:
        """
        Run inference synchronously (call from a QThread).
        messages: list of {"role": "system"|"user"|"assistant", "content": str}
        Returns the generated text string.
        """
        # Block until model is ready (handles race between load & first query)
        self._ready.wait()

        if self._model is None:
            return f"[Model not loaded: {self._error}]"

        import torch

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
        response = self._tokenizer.decode(new_ids, skip_special_tokens=True)
        return response.strip()


# ------------------------------------------------------------------
# System prompt builder
# ------------------------------------------------------------------

def build_system_prompt(
    summary: dict | None = None,
    cluster_result=None,
    stats_context: dict | None = None,
    plan_df=None,
) -> str:
    """
    Build a rich, structured system prompt from all available analysis artefacts.
    The model uses this as background knowledge for every user query.
    """
    import pandas as pd

    lines = [
        "You are an expert AI Production Planning Assistant for Gem Computers, "
        "an apparel manufacturing company specialising in women's intimate apparel (bras).",
        "Answer questions analytically. Use the data context below as your knowledge base.",
        "Be concise, precise, and always support recommendations with numbers from the data.",
        "",
        "═══ DATASET OVERVIEW ═══",
    ]

    # Dataset summary
    if summary:
        lines += [
            f"• Total records  : {summary.get('total_records', 'N/A'):,}",
            f"• Unique SKUs    : {summary.get('unique_skus', 'N/A')}",
            f"• Unique sizes   : {summary.get('unique_sizes', 'N/A')}",
            f"• Unique colors  : {summary.get('unique_colors', 'N/A')}",
            f"• Total units    : {summary.get('total_units', 'N/A'):,}",
            f"• Date range     : {summary.get('date_min', '?')} to {summary.get('date_max', '?')}",
            f"• Years covered  : {summary.get('years_covered', 'N/A')}",
        ]

    # Cluster results
    if cluster_result is not None and not cluster_result.empty:
        lines += ["", "═══ CLUSTER / PARTITION ANALYSIS ═══"]
        if "cluster_label" in cluster_result.columns:
            for lbl, grp in cluster_result.groupby("cluster_label"):
                skus = grp["c_sku"].tolist() if "c_sku" in grp.columns else []
                vol = int(grp["total_qty"].sum()) if "total_qty" in grp.columns else 0
                lines.append(f"• [{lbl}] – {len(skus)} SKUs, total vol {vol:,}")
                lines.append(f"  SKUs: {', '.join(skus[:8])}{'...' if len(skus) > 8 else ''}")
        if "abc" in cluster_result.columns:
            abc_cnt = cluster_result["abc"].value_counts().to_dict()
            lines.append(f"• ABC – A:{abc_cnt.get('A',0)} B:{abc_cnt.get('B',0)} C:{abc_cnt.get('C',0)} SKUs")
        if "xyz" in cluster_result.columns:
            xyz_cnt = cluster_result["xyz"].value_counts().to_dict()
            lines.append(f"• XYZ – X:{xyz_cnt.get('X',0)} Y:{xyz_cnt.get('Y',0)} Z:{xyz_cnt.get('Z',0)} SKUs")

    # Statistical context (top SKUs)
    if stats_context:
        lines += ["", "═══ STATISTICAL INSIGHTS ═══"]
        top_skus = stats_context.get("top_skus_by_volume", [])[:10]
        lines.append(f"• Top SKUs by volume: {', '.join(top_skus)}")
        ts = stats_context.get("trend_summary", {})
        up = ts.get("uptrend_skus", [])[:6]
        dn = ts.get("downtrend_skus", [])[:6]
        if up:
            lines.append(f"• Growing (uptrend) SKUs: {', '.join(up)}")
        if dn:
            lines.append(f"• Declining SKUs: {', '.join(dn)}")

        sku_stats = stats_context.get("sku_stats", {})
        if sku_stats:
            lines.append("• Key SKU details:")
            for sku, st in list(sku_stats.items())[:8]:
                lines.append(
                    f"  {sku}: avg {st.get('avg_monthly', 0):,.0f}/mo, "
                    f"CV {st.get('cv_pct', 0):.1f}%, "
                    f"trend={st.get('trend_direction','?')}, "
                    f"peak_month=M{st.get('peak_month', '?')}"
                )

    # Production plan
    if plan_df is not None and not plan_df.empty:
        lines += ["", "═══ PRODUCTION PLAN (latest run) ═══"]
        top5 = plan_df.nlargest(5, "avg_monthly_qty") if "avg_monthly_qty" in plan_df.columns else plan_df.head(5)
        for _, r in top5.iterrows():
            lines.append(
                f"• {r.get('c_sku','?')}: avg {r.get('avg_monthly_qty',0):,.0f}/mo, "
                f"SS={r.get('safety_stock',0):,.0f}, "
                f"6m forecast={r.get('total_planned',0):,.0f}"
            )

    lines += [
        "",
        "═══ INSTRUCTIONS ═══",
        "When the user asks about a specific SKU, size, or color – refer to the data above.",
        "When you give numerical recommendations, state the source (cluster / trend / forecast).",
        "If the user provides additional data (classification labels, numbers), analyse them directly.",
    ]

    return "\n".join(lines)
