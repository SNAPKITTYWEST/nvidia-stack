# run_synth.tcl — Vivado RTL Synthesis Script
# Target: AMD Alveo U55C / U250 (gfx942 equivalent prototyping)

read_verilog [glob ../rtl/*.sv]
read_xdc constraints.xdc

synth_design -top fpga_mfma_accelerator -part xcu55c-fsvh2892-2L-e
report_timing_summary -file reports/post_synth_timing_summary.rpt
report_utilization -file reports/post_synth_utilization.rpt

write_checkpoint -force checkpoints/post_synth.dcp
