# synthesize_asic.tcl — Synopsys DC Compiler ASIC Synthesis
# Target: TSMC N6 / gfx942-class performance (300 MHz)

set search_path ". /opt/foundry/tsmc/n6/lib/typ /opt/foundry/tsmc/n6/lib/bc /opt/foundry/tsmc/n6/lib/wc"
set link_path "* tsmc_n6_typ.db tsmc_n6_wc.db tsmc_n6_bc.db"

read_verilog ../rtl/fpga_mfma_accelerator.sv
set_top fpga_mfma_accelerator

create_clock -name clk -period 3.33 [get_ports clk]
set_clock_uncertainty 0.15 [get_clocks clk]
set_clock_transition 0.08 [get_clocks clk]

set_operating_conditions -max WC_TYP -min BC_TYP
set_wire_load_mode enclosed

compile_ultra -gate_clock -no_auto_ungroup -no_ecc

write_verilog -hierarchy -output mfma_core_gated.v
write_sdc mfma_core.sdc

puts "ASIC synthesis complete: mfma_core_gated.v"
