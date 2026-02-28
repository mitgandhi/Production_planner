"""
agent.py
--------
Terminal AI agent powered by your local Qwen3 model.
Works exactly like Claude Code:  continuous conversation, tool use,
file access, data queries -- all from the command line.

Usage
-----
    python agent.py
    python agent.py --data "E:\\Gem_computers\\Data\\AI_DATA.CSV"
    python agent.py --data path\\to\\file.csv --max-tokens 1024

How it works
------------
Uses a ReAct (Reason + Act) loop:
  1. User sends a message
  2. Model thinks and outputs one or more <tool> calls
  3. Agent executes each tool and feeds the result back
  4. Model reads the result and continues until it gives a final answer
  5. Repeat

Available tools
---------------
  query_data        run any pandas/Python code on the loaded DataFrame
  read_file         read any CSV / Excel / JSON / text file from disk
  write_file        write content to any file on disk
  production_est    compute per-SKU×Size×Color production requirements
  run_command       run a shell / cmd command and return output
  load_data         load a new CSV file as the active DataFrame
  summarise_data    print a quick data summary (shape, columns, top SKUs)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Windows CUDA DLL fix (mirrors main.py) ────────────────────────────────────
if sys.platform == "win32":
    _cuda_candidates: list[str] = []
    for _k, _v in os.environ.items():
        if _k.startswith("CUDA_PATH") and os.path.isdir(_v):
            _cuda_candidates.append(os.path.join(_v, "bin"))
    for _ver in ("12.9", "12.8", "12.7", "12.6", "12.5", "12.4", "12.3", "12.2", "12.1", "12.0"):
        _p = rf"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{_ver}\bin"
        if os.path.isdir(_p):
            _cuda_candidates.append(_p)
    try:
        import torch as _t
        _tlib = os.path.join(os.path.dirname(_t.__file__), "lib")
        if os.path.isdir(_tlib):
            _cuda_candidates.insert(0, _tlib)
        _t.cuda.is_available()   # force CUDA init on main thread
    except Exception:
        pass
    for _d in _cuda_candidates:
        if os.path.isdir(_d):
            try:
                os.add_dll_directory(_d)
            except Exception:
                pass

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# ANSI colour helpers (work on Windows 10+ with ANSI support)
# ─────────────────────────────────────────────────────────────────────────────

RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
MAGENTA = "\033[95m"
BLUE    = "\033[94m"
WHITE   = "\033[97m"

def _c(text: str, *codes: str) -> str:
    return "".join(codes) + text + RESET

def _banner():
    print(_c("=" * 62, CYAN, BOLD))
    print(_c("  Gem Computers  AI Agent  —  Qwen3 (local)", CYAN, BOLD))
    print(_c("  Type your request.  'help' for tools.  'exit' to quit.", DIM))
    print(_c("=" * 62, CYAN, BOLD))

def _tool_block(name: str, inp: str):
    print(_c(f"\n[Tool] {name}", YELLOW, BOLD))
    if inp.strip():
        for line in inp.strip().splitlines()[:6]:
            print(_c(f"  {line}", DIM))

def _result_block(result: str, truncate: int = 60):
    lines = result.strip().splitlines()
    shown = lines[:truncate]
    print(_c("[Result]", GREEN, BOLD))
    for line in shown:
        print(_c(f"  {line}", GREEN))
    if len(lines) > truncate:
        print(_c(f"  ... ({len(lines) - truncate} more lines)", DIM))

def _error_block(msg: str):
    print(_c(f"[Error] {msg}", RED))

def _thinking(msg: str = "Thinking…"):
    print(_c(f"\n{msg}", MAGENTA), end="", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tool executor
# ─────────────────────────────────────────────────────────────────────────────

# Global mutable state for the active DataFrame
_df: Optional[pd.DataFrame] = None


def _get_df() -> Optional[pd.DataFrame]:
    return _df


def _set_df(df: pd.DataFrame):
    global _df
    _df = df


def tool_query_data(code: str) -> str:
    """Execute pandas / Python code.  `df` refers to the active DataFrame."""
    df = _get_df()
    if df is None:
        return "No DataFrame loaded.  Use load_data first."
    local_ns: dict = {"df": df, "pd": pd}
    try:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = eval(compile(code, "<agent>", "eval"), local_ns)   # try expression first
        output = buf.getvalue()
        if result is not None:
            output += str(result)
        return output.strip() or "(no output)"
    except SyntaxError:
        pass
    try:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(code, local_ns)                                         # fall back to exec
        output = buf.getvalue()
        return output.strip() or "(executed, no output)"
    except Exception as exc:
        return f"Error: {exc}"


def tool_read_file(path: str) -> str:
    """Read CSV / Excel / JSON / text from disk."""
    path = path.strip().strip('"\'')
    if not os.path.isfile(path):
        return f"File not found: {path}"
    ext = os.path.splitext(path)[1].lower()
    size_kb = os.path.getsize(path) / 1024
    header = f"File: {path}  ({size_kb:.1f} KB)"
    try:
        if ext == ".csv":
            df_file = pd.read_csv(path, low_memory=False)
        elif ext in (".xlsx", ".xls"):
            df_file = pd.read_excel(path)
        else:
            df_file = None

        if df_file is not None:
            out = [header,
                   f"Rows: {len(df_file):,}   Cols: {len(df_file.columns)}",
                   f"Columns: {', '.join(df_file.columns)}",
                   "", "First 20 rows:",
                   df_file.head(20).to_string(index=False)]
            try:
                out += ["", "Statistics:", df_file.describe(include="all").round(2).to_string()]
            except Exception:
                pass
            return "\n".join(out)

        if ext == ".json":
            import json
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            text = json.dumps(data, indent=2, ensure_ascii=False)
            return header + "\n" + (text[:5000] + "\n...(truncated)" if len(text) > 5000 else text)

        with open(path, encoding="utf-8", errors="ignore") as f:
            content = f.read(6000)
        return header + "\n" + content + ("\n...(truncated)" if len(content) == 6000 else "")
    except Exception as exc:
        return f"Error reading {path}: {exc}"


def tool_write_file(path: str, content: str) -> str:
    """Write content to a file."""
    path = path.strip().strip('"\'')
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Written {len(content):,} chars to {path}"
    except Exception as exc:
        return f"Error writing {path}: {exc}"


def tool_production_est(args_str: str) -> str:
    """Compute per-SKU×Size×Color production requirements."""
    df = _get_df()
    if df is None:
        return "No DataFrame loaded."
    parts = [p.strip() for p in args_str.split(",")]
    sales_days = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 90
    prod_days  = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 45
    try:
        from src.production_planning import current_production_estimate
        est = current_production_estimate(df, sales_days=sales_days, production_days=prod_days)
        lines = [
            f"Production estimates (last {sales_days}d sales, {prod_days}d production horizon)",
            f"{'SKU':<12} | {'Size':<6} | {'Color':<8} | {'90d Sales':>9} | {'Per-Day':>7} | {'Prod Req':>10}",
            "-" * 60,
        ]
        for _, r in est.head(60).iterrows():
            lines.append(
                f"{str(r['c_sku']):<12} | {str(r['c_sz']):<6} | "
                f"{str(r['c_cl']):<8} | {int(r['total_sales_90d']):>9,} | "
                f"{r['per_day_sales_qty']:>7.2f} | {int(r['production_req_qty']):>10,}"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"Error: {exc}"


def tool_run_command(cmd: str) -> str:
    """Run a shell command and return stdout + stderr."""
    import subprocess
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        out = result.stdout + result.stderr
        return out.strip()[:4000] or "(no output)"
    except subprocess.TimeoutExpired:
        return "Command timed out after 30s."
    except Exception as exc:
        return f"Error: {exc}"


def tool_load_data(path: str) -> str:
    """Load a CSV file as the active DataFrame."""
    path = path.strip().strip('"\'')
    if not os.path.isfile(path):
        return f"File not found: {path}"
    try:
        df = pd.read_csv(path, dtype=str, low_memory=False)
        # Try to coerce c_qty to numeric if present
        if "c_qty" in df.columns:
            df["c_qty"] = pd.to_numeric(df["c_qty"], errors="coerce").fillna(0)
        _set_df(df)
        return (f"Loaded {path}\n"
                f"Rows: {len(df):,}   Columns: {len(df.columns)}\n"
                f"Columns: {', '.join(df.columns.tolist())}")
    except Exception as exc:
        return f"Error loading {path}: {exc}"


def tool_summarise_data(_: str) -> str:
    """Print a quick summary of the active DataFrame."""
    df = _get_df()
    if df is None:
        return "No DataFrame loaded."
    lines = [
        f"Rows: {len(df):,}   Columns: {len(df.columns)}",
        f"Columns: {', '.join(df.columns)}",
    ]
    if "c_qty" in df.columns:
        lines.append(f"Total qty: {df['c_qty'].sum():,.0f}")
    if "c_sku" in df.columns:
        top = df.groupby("c_sku")["c_qty"].sum().sort_values(ascending=False).head(10)
        lines.append("Top 10 SKUs: " + ", ".join(f"{s}={int(q):,}" for s, q in top.items()))
    if "date" in df.columns:
        lines.append(f"Date range: {df['date'].min()} → {df['date'].max()}")
    return "\n".join(lines)


TOOL_REGISTRY: Dict[str, callable] = {
    "query_data":      tool_query_data,
    "read_file":       tool_read_file,
    "write_file":      tool_write_file,
    "production_est":  tool_production_est,
    "run_command":     tool_run_command,
    "load_data":       tool_load_data,
    "summarise_data":  tool_summarise_data,
}


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

def _build_system_prompt(data_summary: str) -> str:
    return f"""You are an AI Production Planning Agent for Gem Computers (women's intimate apparel).
You work like Claude Code: you reason step-by-step and use tools to get real data before answering.

TOOLS — call a tool by outputting EXACTLY this format (no extra text around it):
<tool>tool_name</tool>
<input>tool input here</input>

Available tools:
  query_data      — run pandas/Python code on the active DataFrame (variable: `df`)
                    Example: <tool>query_data</tool><input>df.groupby('c_sku')['c_qty'].sum().sort_values(ascending=False).head(10)</input>
  read_file       — read any file (CSV/Excel/JSON/TXT) from a path
                    Example: <tool>read_file</tool><input>E:\\data\\sales.csv</input>
  write_file      — write text content to a file
                    Example: <tool>write_file</tool><input>E:\\output\\plan.txt\n---content---\nSKU ALPA: 5000 units</input>
  production_est  — compute per-SKU×Size×Color production estimates
                    Example: <tool>production_est</tool><input>90,45</input>
  run_command     — run a shell command
                    Example: <tool>run_command</tool><input>dir E:\\Gem_computers\\Data</input>
  load_data       — load a CSV as the active DataFrame
                    Example: <tool>load_data</tool><input>E:\\data\\newfile.csv</input>
  summarise_data  — print a quick summary of the current DataFrame
                    Example: <tool>summarise_data</tool><input></input>

RULES:
- Always use a tool to get real data before making a numerical claim.
- You may call multiple tools in sequence — reason between each result.
- After all tools, give your final answer clearly.
- When writing files or plans, be specific with numbers from the data.
- If the user gives a file path, use read_file to read it automatically.

CURRENT DATASET SUMMARY:
{data_summary}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Tool call parser
# ─────────────────────────────────────────────────────────────────────────────

_TOOL_RE = re.compile(
    r'<tool>\s*(\w+)\s*</tool>\s*<input>(.*?)</input>',
    re.DOTALL | re.IGNORECASE,
)


def _parse_tool_calls(text: str) -> List[Tuple[str, str]]:
    """Extract all <tool>name</tool><input>...</input> pairs from model output."""
    return [(m.group(1).strip(), m.group(2).strip()) for m in _TOOL_RE.finditer(text)]


def _strip_tool_calls(text: str) -> str:
    """Remove tool call tags from text for display."""
    return _TOOL_RE.sub("", text).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_model():
    """Load Qwen3 model + tokenizer.  Reuses CUDA init done at module top."""
    model_dir = str(Path(__file__).parent / "models" / "Qwen3")
    print(_c(f"Loading model from {model_dir} …", DIM))

    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer

    if torch.cuda.is_available():
        device_map = "cuda"
        dtype = torch.bfloat16
        gpu = torch.cuda.get_device_name(0)
        free_gb = torch.cuda.mem_get_info(0)[0] / 1e9
        print(_c(f"  GPU: {gpu}  ({free_gb:.1f} GB free)  bfloat16", GREEN))
    else:
        device_map = "cpu"
        dtype = torch.float32
        print(_c("  CPU mode  float32", YELLOW))

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=dtype, device_map=device_map, trust_remote_code=True
    )
    model.eval()
    print(_c("  Model loaded.", GREEN))
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Inference  (streaming to terminal)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_stream(model, tokenizer, messages: list, max_new_tokens: int = 512):
    """Stream tokens to stdout; return the full generated text."""
    import torch
    from transformers import TextIteratorStreamer, GenerationConfig
    import threading

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=60.0
    )
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.6,
        top_p=0.8,
        top_k=20,
        pad_token_id=tokenizer.eos_token_id,
    )

    def _run():
        with torch.no_grad():
            model.generate(**inputs, streamer=streamer, generation_config=gen_cfg)

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    full = ""
    print(_c("\nQwen3: ", BLUE, BOLD), end="", flush=True)
    for token in streamer:
        print(token, end="", flush=True)
        full += token
    print()   # newline after streaming
    t.join(timeout=90)
    return full


# ─────────────────────────────────────────────────────────────────────────────
# Context window management
# ─────────────────────────────────────────────────────────────────────────────

_MAX_HISTORY_TURNS = 12   # keep last N user+assistant pairs


def _trim_history(messages: list, system_msg: dict) -> list:
    """Keep system message + last _MAX_HISTORY_TURNS turns."""
    non_system = [m for m in messages if m["role"] != "system"]
    if len(non_system) > _MAX_HISTORY_TURNS * 2:
        non_system = non_system[-(  _MAX_HISTORY_TURNS * 2):]
        print(_c("  (older messages trimmed to fit context window)", DIM))
    return [system_msg] + non_system


# ─────────────────────────────────────────────────────────────────────────────
# ReAct agent loop
# ─────────────────────────────────────────────────────────────────────────────

_MAX_TOOL_ROUNDS = 8   # max consecutive tool calls per user turn


def _agent_turn(model, tokenizer, messages: list, max_tokens: int) -> str:
    """
    Run one full ReAct loop for the current conversation state.
    Returns the final text response (tool calls stripped).
    """
    for _round in range(_MAX_TOOL_ROUNDS):
        # Generate next model output
        response = _generate_stream(model, tokenizer, messages, max_new_tokens=max_tokens)

        # Parse tool calls from response
        tool_calls = _parse_tool_calls(response)

        if not tool_calls:
            # No tools — this is the final answer
            return _strip_tool_calls(response)

        # Execute each tool and collect results
        tool_results: list[str] = []
        for tool_name, tool_input in tool_calls:
            _tool_block(tool_name, tool_input)

            # Handle write_file: split on first \n--- to get path + content
            if tool_name == "write_file" and "\n" in tool_input:
                first_line, rest = tool_input.split("\n", 1)
                # Strip possible separator line
                content = rest.lstrip("-").strip()
                result = tool_write_file(first_line.strip(), content)
            elif tool_name in TOOL_REGISTRY:
                try:
                    result = TOOL_REGISTRY[tool_name](tool_input)
                except Exception as exc:
                    result = f"Tool error: {exc}"
            else:
                result = f"Unknown tool: {tool_name}"

            _result_block(result)
            tool_results.append(f"[Tool: {tool_name}]\n{result}")

        # Append assistant message (with tool calls) + tool results to history
        messages.append({"role": "assistant", "content": response})
        combined_results = "\n\n".join(tool_results)
        messages.append({
            "role": "user",
            "content": f"[TOOL RESULTS]\n{combined_results}\n[/TOOL RESULTS]\nContinue."
        })

    # Fallback: generate final answer after max rounds
    final = _generate_stream(model, tokenizer, messages, max_new_tokens=max_tokens)
    return _strip_tool_calls(final)


# ─────────────────────────────────────────────────────────────────────────────
# Data bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_data(path: Optional[str]) -> str:
    """Load data and return a compact summary for the system prompt."""
    global _df
    default = Path(__file__).parent / "Data" / "AI_DATA.CSV"

    # Try provided path first, then default
    for candidate in [path, str(default)]:
        if candidate and Path(candidate).is_file():
            try:
                from src.data_loader import load_and_preprocess
                df = load_and_preprocess(candidate)
            except Exception:
                df = pd.read_csv(candidate, low_memory=False)
            _set_df(df)
            print(_c(f"  Data loaded: {candidate}  ({len(df):,} rows)", GREEN))

            # Quick summary for the system prompt
            lines = [f"File: {candidate}", f"Rows: {len(df):,}"]
            if "c_sku" in df.columns:
                lines.append(f"Unique SKUs: {df['c_sku'].nunique()}")
            if "c_qty" in df.columns:
                lines.append(f"Total units: {df['c_qty'].sum():,.0f}")
                top5 = df.groupby("c_sku")["c_qty"].sum().sort_values(ascending=False).head(5)
                lines.append("Top 5 SKUs: " + ", ".join(f"{s}={int(q):,}" for s, q in top5.items()))
            if "date" in df.columns:
                lines.append(f"Date range: {df['date'].min()} to {df['date'].max()}")
            if "c_sz" in df.columns:
                lines.append(f"Sizes: {', '.join(df['c_sz'].unique()[:10])}")
            if "c_cl" in df.columns:
                lines.append(f"Colors: {', '.join(df['c_cl'].unique()[:10])}")
            return "\n".join(lines)

    print(_c("  No data file found — use load_data tool to load one.", YELLOW))
    return "No data loaded. Use load_data tool to load a CSV file."


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

_HELP_TEXT = f"""
{_c('Tools you can ask the agent to use:', CYAN, BOLD)}
  query_data      — run pandas code on the loaded data
  read_file       — read any CSV/Excel/JSON/TXT file
  write_file      — write content to a file
  production_est  — compute production requirements
  run_command     — run a shell/cmd command
  load_data       — load a new CSV as active data
  summarise_data  — quick overview of the current data

{_c('Example prompts:', CYAN, BOLD)}
  "What are the top 10 SKUs by volume?"
  "Show me the monthly trend for SKU ALPA over the last year"
  "Compute production estimates for 60 days sales / 30 days production"
  "Read the file E:\\exports\\new_orders.csv and compare with our data"
  "Write a production plan for the top 5 SKUs to E:\\output\\plan.txt"
  "Run: dir E:\\Gem_computers\\Data"
  "Load E:\\data\\custom.csv and tell me what it contains"

{_c('Commands:', CYAN, BOLD)}
  help   — show this help
  clear  — clear conversation history
  exit   — quit the agent
"""


def main():
    # Enable ANSI on Windows
    if sys.platform == "win32":
        os.system("")   # activates VT100 in Windows console

    parser = argparse.ArgumentParser(description="Gem Computers AI Terminal Agent")
    parser.add_argument("--data", default=None, help="Path to CSV data file")
    parser.add_argument("--max-tokens", type=int, default=768, help="Max tokens per response")
    args = parser.parse_args()

    _banner()

    # Load data
    print(_c("\nLoading data…", DIM))
    data_summary = _bootstrap_data(args.data)

    # Load model
    print(_c("Loading Qwen3 model…", DIM))
    try:
        model, tokenizer = _load_model()
    except Exception as exc:
        print(_c(f"Failed to load model: {exc}", RED))
        print(_c("Run  python cuda_fix.py  to diagnose GPU/CUDA issues.", YELLOW))
        sys.exit(1)

    # Build system message (stays constant across turns)
    system_msg = {"role": "system", "content": _build_system_prompt(data_summary)}
    messages: list[dict] = [system_msg]

    print(_c("\nAgent ready. Type your request below.\n", GREEN, BOLD))

    # ── Main conversation loop ────────────────────────────────────────────────
    while True:
        try:
            user_input = input(_c("You: ", BLUE, BOLD)).strip()
        except (EOFError, KeyboardInterrupt):
            print(_c("\nExiting.", DIM))
            break

        if not user_input:
            continue

        # Built-in commands
        if user_input.lower() in ("exit", "quit", "q"):
            print(_c("Goodbye.", DIM))
            break

        if user_input.lower() == "help":
            print(_HELP_TEXT)
            continue

        if user_input.lower() == "clear":
            messages = [system_msg]
            print(_c("Conversation cleared.", DIM))
            continue

        # Add user message
        messages.append({"role": "user", "content": user_input})

        # Trim context if needed
        messages = _trim_history(messages, system_msg)

        # Run ReAct agent turn (may call tools multiple times before answering)
        try:
            final_answer = _agent_turn(model, tokenizer, messages, args.max_tokens)
        except Exception as exc:
            _error_block(f"Agent error: {exc}")
            traceback.print_exc()
            messages.pop()   # remove the failed user message
            continue

        # Store final assistant answer in history (tool calls already added internally)
        # Replace any intermediate messages added during tool rounds with the clean answer
        # (messages list was mutated inside _agent_turn — keep it as-is)
        # Just ensure the final clean answer is the last assistant entry
        if messages and messages[-1]["role"] == "user" and "TOOL RESULTS" in messages[-1]["content"]:
            # Remove the "Continue." injection we added after tools
            pass  # history already has tool rounds; final answer printed to terminal

        print()   # blank line between turns


if __name__ == "__main__":
    main()
