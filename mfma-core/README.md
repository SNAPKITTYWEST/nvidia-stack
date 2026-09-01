# MFMA Core — OCaml → C → HLS → RTL → FPGA/ASIC Pipeline

**v1.0 Release** — Sovereign corporate product. Commercial use requires a Sovereign Node Key.

---

## Overview

Complete hardware design flow for the MFMA (Matrix Fused Multiply-Add) core computation, covering:

```
Algorithm (OCaml) → C Wrapper → HLS → RTL → FPGA → ASIC → GDSII → Silicon
```

Implements 16x16x16 FP16 → FP32 matrix tile multiplication matching AMD gfx942 `v_mfma_f32_16x16x16f16` semantics with IEEE-754 compliant NaN propagation.

## Repository Structure

```
mfma-core/
├── src/
│   ├── mfma_core.ml                  OCaml algorithm specification
│   ├── mfma_hls_wrapper.c            HLS-compatible C wrapper
│   ├── mfma_core.h                   Public C interface
│   ├── mfma_core_hip.cpp             AMD gfx942 HIP kernel
│   └── mfma_core.cu                  NVIDIA RTX 3080 CUDA kernel
├── rtl/
│   └── fpga_mfma_accelerator.sv      SystemVerilog FPGA implementation
├── analog/
│   └── mfma_power_supply_droop.vams  Verilog-A power/droop model
├── formal/
│   └── mfma_nan.why                  Why3 NaN propagation proof
├── fpga/
│   └── scripts/
│       ├── run_synth.tcl             Vivado synthesis
│       ├── run_impl.tcl              Vivado place & route
│       └── generate_bitstream.tcl    Vivado bitstream
├── asic/
│   └── scripts/
│       ├── synthesize_asic.tcl       Synopsys DC synthesis
│       ├── signoff_sta.tcl           PrimeTime STA
│       ├── run_lec.tcl               Logic equivalence checking
│       ├── run_drc_lvs.py            KLayout DRC/LVS
│       └── mfma_core_layout.py       GDSFactory layout
├── Makefile                          Master build pipeline
└── README.md                         This file
```

## Quick Start

### Build HLS Library (OCaml → C → .so)

```bash
make all
```

Produces `libmfmacore.so` with zero OCaml runtime in the HLS region (verified via `objdump`).

### Build HIP Kernel (AMD gfx942)

```bash
make hip
```

### Build CUDA Kernel (NVIDIA RTX 3080)

```bash
make cuda
```

### FPGA Synthesis (AMD Vivado)

```bash
make fpga
```

Generates bitstream for AMD Alveo U55C / U250.

### ASIC Synthesis (Synopsys DC + PrimeTime)

```bash
make asic
```

Targets TSMC N6 at 300 MHz.

## Features

- **OCaml → C**: `ocamlopt -output-obj` with `-noautolink -runtime-variant _nolithic` strips Caml runtime
- **HLS Pragmas**: `#pragma HLS PIPELINE II=1`, `UNROLL`, `m_axi` interface binding
- **NaN Propagation**: IEEE-754 compliant, verified in Why3 with zero sorries
- **gfx942 Match**: HIP kernel maps to `v_mfma_f32_16x16x16f16` instruction
- **RTX 3080 Match**: CUDA kernel uses `wmma::mma_sync` on SM_86 Tensor Cores
- **FPGA/ASIC**: SystemVerilog RTL, Vivado + Synopsys DC flow, GDSII tape-out ready

## Verification

```bash
# Verify NO OCaml runtime in HLS region
objdump -T libmfmacore.so | grep -E "caml_alloc|caml_callback"
# Expected: NO OUTPUT

# Verify RTL is SystemVerilog (NOT Verilog-A)
grep -r "analog\|branch\|electrical" rtl/
# Expected: NO OUTPUT (only in analog/ directory)
```

## Formal Verification

Why3 proof (`formal/mfma_nan.why`) verifies:

- `mfma_tile_nan_safety`: Single-element NaN propagation
- `mfma_full_tile_nan_safety`: Full tile NaN propagation

Run with: `why3 ide formal/mfma_nan.why`

---

## Sovereign Source License v1.0

Copyright 2026 Ahmad Ali Parr and Jessica Westerhoff

This is a sovereign corporate product. No public access. Commercial use requires a Sovereign Node Key.
