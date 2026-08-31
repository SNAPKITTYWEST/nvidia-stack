# NVIDIA Stack — Reverse-Engineered GPU Compute Stack

[![License: BSL-1.1](https://img.shields.io/badge/License-BSL--1.1-ff6b35.svg)](https://github.com/SNAPKITTYWEST/nvidia-stack/blob/main/LICENSE)
[![License: AGPL--3.0](https://img.shields.io/badge/License-AGPL--3.0-red.svg)](https://github.com/SNAPKITTYWEST/nvidia-stack/blob/main/LICENSE-AGPL)
[![Rust](https://img.shields.io/badge/Rust-2021-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![AMDGPU](https://img.shields.io/badge/AMDGPU-gfx942-red.svg)](https://rocm.docs.amd.com/)
[![Sovereign](https://img.shields.io/badge/Sovereign-Node%20Key%20Only-black.svg)](https://github.com/SNAPKITTYWEST)

**⚠️ NOT OPEN SOURCE** — Sovereign corporate product. Commercial use requires a Sovereign Node Key.

---

## What This Is

A complete reverse-engineered GPU compute stack covering the full chain from high-level tensor operations down to hardware cycles:

```
PyTorch/CuTe Layouts → PTX/SASS ISA → Tensor Core/MFMA Microarchitecture → Hardware Signals
```

### Coverage

| Layer | NVIDIA | AMD |
|-------|--------|-----|
| Tensor Layout | CuTe layouts (Rust) | A/B row-major / column-major (Python) |
| Instruction Set | SASS HMMA/LDG/STG (Rust) | AMDGPU MFMA ISA (asm) |
| Microarchitecture | Tensor Core MAC simulation (Rust) | Matrix Core wave simulation |
| Memory | Global/L1/L2 cache model | LDS bank conflict avoidance + XOR swizzle |
| Validation | — | Fragment map validator + structural checks |
| Layout Search | — | Padding + XOR swizzle optimizer |
| Assembly | — | gfx942 MFMA GEMM kernels |

---

## Repository Structure

```
nvidia-stack/
├── src/
│   └── main.rs                          Rust NVIDIA stack simulator
│       ├── CuTe Layouts                 Tensor-to-memory coordinate mapping
│       ├── SASS ISA                     HMMA/LDG/STG instruction model
│       ├── Tensor Core Hardware          MAC units, pipeline, clock simulation
│       └── Stack Orchestrator            Full chain execution + timing
├── asm/
│   ├── mfma_f16_16x16x16.s             AMDGPU MFMA basic tile (gfx90a)
│   ├── mfma_lds_staging.s              gfx942 MFMA with LDS ping-pong staging
│   └── mfma_lds_xor_swizzle.s          gfx942 MFMA with XOR swizzle bank conflict avoidance
├── python/
│   ├── fragment_map.py                  Opcode-accurate fragment map + layout search
│   ├── structural_validator.py          Bijectivity, per-lane, VGPR, C/D checks
│   └── lds_padding.py                   ds_read_b128 padding calculator
├── LICENSE                              Business Source License 1.1
├── LICENSE-AGPL                         GNU AGPL v3.0
└── README.md                            This file
```

---

## Quick Start

### Rust (NVIDIA Stack Simulator)

```bash
cd nvidia-stack
cargo run
```

Output:
```
--- Starting Stack Execution ---
[Stack] Layouts Generated: A([16, 16], [16, 1]), B([16, 16], [16, 1])
[HW] Memory Load (L1/L2 Cache Hit)
[HW] Memory Load (L1/L2 Cache Hit)
[HW] Executing HMMA 16x16x16 | Cycles: 1.00 | Latency: 6.19ns
[HW] Memory Store
--- Stack Execution Complete ---
Total Wall-Clock Time (Simulated): 36.1905 ns
```

### Python (Fragment Map + Layout Optimizer)

```bash
cd python
python fragment_map.py
```

Output:
```
Fragment map validation passed.

=== Operand A (row-major) ===
Layout: padded
 Padding: 0 FP16 elements
 Row stride: 16 FP16 elements
 = 32 bytes

=== Operand B (column-major) ===
Layout: padded
 Padding: 0 FP16 elements
 Column stride: 16 FP16 elements
 = 32 bytes

=== Layout Certificate ===
{
  "target": "gfx942",
  "opcode": "v_mfma_f32_16x16x16f16",
  "wavefront_size": 64,
  "mfma_tile": {"M": 16, "N": 16, "K": 16},
  "operand_A": {
    "load": "ds_read_b64",
    "conflicts": []
  },
  "operand_B": {
    "load": "ds_read_b64",
    "conflicts": []
  }
}
```

### Structural Validator

```bash
cd python
python structural_validator.py
```

Validates:
- Element count (256 A, 256 B, 256 C, 256 D)
- Coordinate bijectivity (no duplicates, no missing)
- Per-lane occupancy (4 FP16 A, 4 FP16 B, 4 FP32 C per lane)
- Packed FP16 register pairs (one low, one high per VGPR)
- C/D accumulator correspondence

### AMDGPU Assembly

```bash
# Assemble for gfx942
llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=obj asm/mfma_lds_xor_swizzle.s -o mfma.o

# Assemble for gfx90a
llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx90a -filetype=obj asm/mfma_f16_16x16x16.s -o mfma_basic.o
```

---

## Fragment Map (v_mfma_f32_16x16x16f16)

The canonical lane-to-fragment mapping for gfx942:

### A Operand (M×K = 16×16 FP16)
- `m = lane >> 2` (row, 0..15)
- `k0 = (lane & 0x3) << 2` (column start, step 4)
- 4 FP16 elements per lane → 2 packed VGPRs (v4, v5)

### B Operand (K×N = 16×16 FP16)
- `k0 = (lane >> 4) << 2` (row start, step 4)
- `n = lane & 0xF` (column, 0..15)
- 4 FP16 elements per lane → 2 packed VGPRs (v8, v9)

### C/D Operand (M×N = 16×16 FP32)
- `n = lane & 0xF` (column, 0..15)
- `m0 = lane >> 4` (row start, step 4)
- 4 FP32 elements per lane → 4 accumulator VGPRs (v0, v1, v2, v3)

---

## LDS Bank Conflict Avoidance

### ds_read_b128 Lane Groups (gfx942)
```
G0: lanes 0-3 + 20-23    G4: lanes 32-35 + 52-55
G1: lanes 4-7 + 16-19    G5: lanes 36-39 + 48-51
G2: lanes 8-11 + 28-31   G6: lanes 40-43 + 60-63
G3: lanes 12-15 + 24-27  G7: lanes 44-47 + 56-59
```

### XOR Swizzle Formula
```
physical_col_word = logical_col_word XOR (row >> row_shift) << xor_shift
```

Eliminates bank conflicts without increasing LDS consumption.

---

## Protected Inventions

  1. REVERSE-ENGINEERED NVIDIA TENSOR CORE STACK
     Complete CuTe → SASS → Hardware chain simulation with MAC unit
     counting, pipeline depth modeling, and cycle-accurate timing.

  2. AMD MFMA FRAGMENT MAP VALIDATOR
     Structural validation proving bijection, per-lane occupancy,
     packed FP16 register pairs, and C/D accumulator correspondence
     for v_mfma_f32_16x16x16f16.

  3. LDS BANK CONFLICT PADDING OPTIMIZER
     Automated search over row-major padding and XOR swizzle
     parameters to eliminate ds_read_b128 bank conflicts.

  4. CROSS-VENDOR GPU COMPUTE MODEL
     Unified abstraction covering NVIDIA HMMA and AMD MFMA with
     hardware-specific lane-to-fragment mappings.

---

## License

**⚠️ THIS IS NOT OPEN SOURCE**

This project is a **sovereign corporate product** licensed under **Business Source License 1.1 (BSL-1.1)** with **GNU AGPL v3.0 copyleft** for network services.

| Component | License | File | Scope |
|-----------|---------|------|-------|
| **Core Stack & Simulators** | BSL-1.1 | `LICENSE` | Rust simulator, Python validators |
| **API/Network** | GNU AGPL v3.0 | `LICENSE-AGPL` | Any network service exposure |

---

## Citation

```bibtex
@misc{nvidiastack2026,
  title={NVIDIA Stack: Reverse-Engineered GPU Compute Stack},
  author={Ahmad Ali Parr and Jessica Westerhoff},
  year={2026},
  note={CuTe/SASS/MFMA simulator with fragment map validation},
  publisher={SNAPKITTYWEST},
  howpublished={\url{https://github.com/SNAPKITTYWEST/nvidia-stack}},
  license={BSL-1.1}
}
```

---

## Contact

**Ahmad Ali Parr** - ahmedparr93@gmail.com
**Jessica Westerhoff** - jessicalw34@gmail.com

Bel Esprit d'Accord Trust — 50/50 equal sovereigns