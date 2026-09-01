#!/usr/bin/env python3
"""
build_mamba2.py — Build libmamba2.so from mamba2.cu

Run this on bbqbaddie (where nvcc lives):

    python build_mamba2.py            # auto-detect arch
    python build_mamba2.py --arch sm_86   # RTX 3080
    python build_mamba2.py --arch sm_89   # bbqbaddie RTX 5000 (Ada)
    python build_mamba2.py --arch sm_80   # A100

Output: kernels/libmamba2.so
Then scp to dev machine or bundle with the package.

The .so exposes:
    mamba2_step_fp8()
    mamba2_forward_fp8()
    mamba2_get_version()

Haskell links via:
    ghc -L<kernels_dir> -lmamba2 -rpath <kernels_dir> BOB/Mamba2FFI.hs
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

KERNELS_DIR = Path(__file__).parent.resolve()
CUDA_SRC    = KERNELS_DIR / "mamba2.cu"
OUT_SO      = KERNELS_DIR / "libmamba2.so"
OUT_OBJ     = KERNELS_DIR / "mamba2.o"


def detect_arch() -> str:
    """Detect GPU compute capability via torch."""
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            arch = f"sm_{cap[0]*10 + cap[1]}"
            print(f"[build_mamba2] detected GPU arch: {arch}")
            return arch
    except ImportError:
        pass
    print("[build_mamba2] WARNING: torch not available, defaulting to sm_86")
    return "sm_86"


def find_nvcc() -> str:
    """Return path to nvcc binary."""
    # 1. On PATH
    r = subprocess.run(["which", "nvcc"], capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        return r.stdout.strip()

    # 2. Via torch CUDA_HOME
    try:
        from torch.utils.cpp_extension import CUDA_HOME
        if CUDA_HOME:
            candidate = Path(CUDA_HOME) / "bin" / "nvcc"
            if candidate.exists():
                return str(candidate)
    except ImportError:
        pass

    # 3. Common Linux paths
    for p in ["/usr/local/cuda/bin/nvcc", "/usr/bin/nvcc"]:
        if Path(p).exists():
            return p

    raise FileNotFoundError(
        "nvcc not found. Run this script on bbqbaddie where CUDA toolkit is installed.\n"
        "On dev machine (no nvcc) use the pure-PyTorch fallback in mamba2_torch.py."
    )


def build(arch: str, debug: bool = False):
    nvcc = find_nvcc()
    print(f"[build_mamba2] nvcc:  {nvcc}")
    print(f"[build_mamba2] arch:  {arch}")
    print(f"[build_mamba2] src:   {CUDA_SRC}")
    print(f"[build_mamba2] out:   {OUT_SO}")

    if not CUDA_SRC.exists():
        raise FileNotFoundError(f"Source not found: {CUDA_SRC}")

    opt_flags = ["-G", "-g"] if debug else ["-O3", "--use_fast_math"]

    # Step 1: compile to relocatable device code object
    compile_cmd = [
        nvcc,
        str(CUDA_SRC),
        f"-arch={arch}",
        "--compiler-options", "-fPIC",
        "-dc",            # device code compilation (relocatable)
        "-o", str(OUT_OBJ),
        *opt_flags,
        "-I", str(KERNELS_DIR),
    ]

    # Step 2: link into shared library
    link_cmd = [
        nvcc,
        str(OUT_OBJ),
        f"-arch={arch}",
        "--shared",
        "-o", str(OUT_SO),
        *opt_flags,
    ]

    print("\n[build_mamba2] Compiling...")
    print(" ".join(compile_cmd))
    r = subprocess.run(compile_cmd, capture_output=False)
    if r.returncode != 0:
        print("[build_mamba2] COMPILE FAILED")
        sys.exit(r.returncode)

    print("\n[build_mamba2] Linking...")
    print(" ".join(link_cmd))
    r = subprocess.run(link_cmd, capture_output=False)
    if r.returncode != 0:
        print("[build_mamba2] LINK FAILED")
        sys.exit(r.returncode)

    # Verify symbols
    nm_r = subprocess.run(["nm", "-D", str(OUT_SO)], capture_output=True, text=True)
    required_syms = ["mamba2_step_fp8", "mamba2_forward_fp8", "mamba2_get_version"]
    missing = [s for s in required_syms if s not in nm_r.stdout]
    if missing:
        print(f"[build_mamba2] WARNING: missing symbols in .so: {missing}")
    else:
        print("[build_mamba2] All required symbols present.")

    so_size = OUT_SO.stat().st_size
    print(f"\n[build_mamba2] SUCCESS: {OUT_SO} ({so_size // 1024} KB)")
    print("\nTo use from Python:")
    print(f"  import ctypes")
    print(f"  lib = ctypes.CDLL('{OUT_SO}')")
    print(f"  print(lib.mamba2_get_version().decode())")
    print("\nTo link from Haskell:")
    print(f"  ghc -L{KERNELS_DIR} -lmamba2 -rpath {KERNELS_DIR} BOB/Mamba2FFI.hs")


def verify_so():
    """Quick sanity check: load the .so and call mamba2_get_version."""
    if not OUT_SO.exists():
        print(f"[verify] {OUT_SO} not found — run build first")
        return False
    import ctypes, ctypes.util
    try:
        lib = ctypes.CDLL(str(OUT_SO))
        lib.mamba2_get_version.restype = ctypes.c_char_p
        version = lib.mamba2_get_version().decode()
        print(f"[verify] mamba2_get_version() = '{version}'")
        return True
    except OSError as e:
        print(f"[verify] Failed to load {OUT_SO}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Build libmamba2.so from mamba2.cu")
    parser.add_argument("--arch",  default=None, help="CUDA arch (e.g. sm_86, sm_89)")
    parser.add_argument("--debug", action="store_true", help="Debug build (-G -g)")
    parser.add_argument("--verify", action="store_true", help="Verify existing .so only")
    args = parser.parse_args()

    if args.verify:
        ok = verify_so()
        sys.exit(0 if ok else 1)

    arch = args.arch or detect_arch()
    build(arch, debug=args.debug)
    verify_so()


if __name__ == "__main__":
    main()
