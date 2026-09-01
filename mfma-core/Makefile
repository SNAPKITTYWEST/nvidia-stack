# Master Makefile — MFMA Core (OCaml → C → HLS → RTL → FPGA/ASIC)
# v1.0 Release

.PHONY: all hls fpga asic hip cuda clean

# ============================================================
# Toolchain Configuration
# ============================================================

OCAMLOPT = ocamlopt
CC = clang
CFLAGS = -O3 -Wall -Wextra -fPIC -noautolink -std=c11
HIPCC = hipcc
NVCC = nvcc

# ============================================================
# HLS Library Build (OCaml → C → .so)
# ============================================================

TARGET_LIB = libmfmacore.so
OBJS = src/mfma_core.o src/mfma_hls_wrapper.o

all: $(TARGET_LIB)

$(TARGET_LIB): $(OBJS)
	$(CC) -shared -o $@ $^ -lm

src/mfma_core.o: src/mfma_core.ml
	$(OCAMLOPT) -output-obj -noautolink -runtime-variant _nolithic $< -o $@

src/mfma_hls_wrapper.o: src/mfma_hls_wrapper.c src/mfma_core.h
	$(CC) $(CFLAGS) -c $< -o $@

# ============================================================
# HIP Build (AMD gfx942)
# ============================================================

hip:
	$(HIPCC) --offload-arch=gfx942 -O3 src/mfma_core_hip.cpp -o mfma_hip

# ============================================================
# CUDA Build (NVIDIA RTX 3080)
# ============================================================

cuda:
	$(NVCC) -O3 -arch=sm_86 -Xcompiler -fPIC -shared src/mfma_core.cu -o libmfmacore_cuda.so

# ============================================================
# FPGA Synthesis (AMD Vivado)
# ============================================================

fpga:
	mkdir -p fpga/reports fpga/checkpoints fpga/bitstream
	cd fpga && vivado -mode batch -source scripts/run_synth.tcl
	cd fpga && vivado -mode batch -source scripts/run_impl.tcl
	cd fpga && vivado -mode batch -source scripts/generate_bitstream.tcl

# ============================================================
# ASIC Synthesis (Synopsys DC + PrimeTime)
# ============================================================

asic:
	mkdir -p asic/reports
	cd asic && dc_shell -f scripts/synthesize_asic.tcl
	cd asic && pt_shell -f scripts/signoff_sta.tcl

# ============================================================
# GDSII Layout (GDSFactory)
# ============================================================

layout:
	cd asic/scripts && python3 mfma_core_layout.py

# ============================================================
# DRC/LVS (KLayout)
# ============================================================

drc:
	cd asic/scripts && python3 run_drc_lvs.py mfma_core.gdsii ../reports/drc_report.txt tsmc_n6

# ============================================================
# Clean
# ============================================================

clean:
	rm -f $(OBJS) $(TARGET_LIB) *.o *.cmi *.cmx *.annot
	rm -f mfma_hip libmfmacore_cuda.so
	rm -rf fpga/reports fpga/checkpoints fpga/bitstream
	rm -rf asic/reports
