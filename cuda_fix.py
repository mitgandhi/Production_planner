"""
cuda_fix.py
-----------
Run this standalone to diagnose / fix CUDA DLL issues on Windows.

Usage:
    python cuda_fix.py

It will:
  1. Print CUDA env-vars and discovered CUDA bin paths
  2. Register DLL directories and test torch CUDA availability
  3. Print a clear PASS / FAIL result with remediation hints
"""

import sys
import os

print("=" * 65)
print("  Gem Computers  –  CUDA DLL Diagnostic")
print("=" * 65)

# ── 1. Platform check ────────────────────────────────────────────────────────
if sys.platform != "win32":
    print("This script is for Windows only.")
    sys.exit(0)

print(f"\nPython  : {sys.version}")
print(f"Exec    : {sys.executable}")

# ── 2. CUDA environment variables ────────────────────────────────────────────
print("\n[1] CUDA environment variables:")
cuda_env_vars = {k: v for k, v in os.environ.items() if "CUDA" in k.upper()}
if cuda_env_vars:
    for k, v in cuda_env_vars.items():
        exists = "(exists)" if os.path.isdir(v) else "(NOT FOUND)"
        print(f"    {k} = {v}  {exists}")
else:
    print("    (none found)  <-- CUDA installer may not have run, or env not reloaded")

# ── 3. Scan common CUDA installation directories ─────────────────────────────
print("\n[2] Common CUDA installation paths:")
toolkit_root = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
found_cuda_bins: list[str] = []

if os.path.isdir(toolkit_root):
    for entry in sorted(os.listdir(toolkit_root), reverse=True):
        bin_dir = os.path.join(toolkit_root, entry, "bin")
        if os.path.isdir(bin_dir):
            found_cuda_bins.append(bin_dir)
            print(f"    FOUND  {bin_dir}")
else:
    print(f"    Toolkit root not found: {toolkit_root}")

# ── 4. Register DLL directories ──────────────────────────────────────────────
print("\n[3] Registering DLL directories with os.add_dll_directory():")
registered: list[str] = []
for d in found_cuda_bins:
    try:
        os.add_dll_directory(d)
        print(f"    OK   {d}")
        registered.append(d)
    except Exception as e:
        print(f"    FAIL {d}  ({e})")

# Also add from env vars
for k, v in cuda_env_vars.items():
    b = os.path.join(v, "bin")
    if os.path.isdir(b) and b not in registered:
        try:
            os.add_dll_directory(b)
            print(f"    OK   {b}  (from {k})")
            registered.append(b)
        except Exception as e:
            print(f"    FAIL {b}  ({e})")

if not registered:
    print("    WARNING: No CUDA bin directories could be registered!")

# ── 5. torch import + CUDA availability ─────────────────────────────────────
print("\n[4] Importing torch:")
try:
    import torch
    print(f"    torch version    : {torch.__version__}")
    print(f"    torch location   : {torch.__file__}")

    # Also add torch/lib to DLL search
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(torch_lib):
        try:
            os.add_dll_directory(torch_lib)
            print(f"    torch/lib added  : {torch_lib}")
        except Exception:
            pass

    cuda_avail = torch.cuda.is_available()
    print(f"\n[5] torch.cuda.is_available() : {cuda_avail}")

    if cuda_avail:
        print(f"    CUDA version     : {torch.version.cuda}")
        count = torch.cuda.device_count()
        for i in range(count):
            name = torch.cuda.get_device_name(i)
            free, total = torch.cuda.mem_get_info(i)
            print(f"    GPU [{i}]          : {name}  |  {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
        print("\n  [PASS] CUDA is working correctly.")
    else:
        print("    No CUDA GPU detected via torch.")
        print("    Possible causes:")
        print("      - torch was installed as CPU-only (torch+cpu)")
        print("        Fix: pip uninstall torch -y && pip install torch --index-url https://download.pytorch.org/whl/cu128")
        print("      - NVIDIA driver is outdated (need >= 525 for CUDA 12)")
        print("        Fix: install latest Game Ready / Studio driver from nvidia.com")
        print("      - CUDA toolkit not installed")
        print("        Fix: install from https://developer.nvidia.com/cuda-downloads")
        print("\n  [WARN] App will run on CPU (slower model inference).")

except ImportError as e:
    print(f"    FAIL – could not import torch: {e}")
    print("    Fix: pip install torch --index-url https://download.pytorch.org/whl/cu128")

except Exception as e:
    print(f"    FAIL – unexpected error: {e}")
    if "dll" in str(e).lower() or "DLL" in str(e):
        print("\n  [FAIL] DLL initialization error detected!")
        print("  Remediation steps:")
        print("   1) Install/repair CUDA Toolkit 12.x from:")
        print("      https://developer.nvidia.com/cuda-downloads")
        print("   2) Ensure CUDA_PATH env var points to the toolkit (e.g. C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8)")
        print("   3) Add %CUDA_PATH%\\bin to your System PATH")
        print("   4) Reinstall torch with cu128 build:")
        print("      pip uninstall torch -y")
        print("      pip install torch --index-url https://download.pytorch.org/whl/cu128")
        print("   5) Reboot and re-run this script")

# ── 6. Quick transformer import test ────────────────────────────────────────
print("\n[6] Testing transformers import:")
try:
    from transformers import AutoTokenizer
    import transformers
    print(f"    transformers version : {transformers.__version__}  OK")
except ImportError as e:
    print(f"    FAIL: {e}")
    print("    Fix: pip install transformers>=5.0.0")

print("\n" + "=" * 65)
print("  Diagnostic complete.  Run  python main.py  to start the app.")
print("=" * 65)
