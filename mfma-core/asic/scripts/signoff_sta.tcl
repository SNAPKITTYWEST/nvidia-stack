# signoff_sta.tcl — PrimeTime Static Timing Analysis Sign-Off
# Target: 300 MHz (3.33 ns clock period) on TSMC N6

set search_path ". /opt/foundry/tsmc/n6/lib/typ /opt/foundry/tsmc/n6/lib/bc /opt/foundry/tsmc/n6/lib/wc"
set link_path "* tsmc_n6_typ.db tsmc_n6_wc.db tsmc_n6_bc.db"

read_verilog mfma_core_gated.v
current_design fpga_mfma_accelerator
link_design

read_parasitics -format spef mfma_core_post_route.spef

create_clock -name clk -period 3.33 [get_ports clk]
set_clock_uncertainty 0.15 [get_clocks clk]
set_clock_transition 0.08 [get_clocks clk]

set_operating_conditions -max WC_TYP -min BC_TYP
set_wire_load_mode enclosed

check_timing
update_timing -full

redirect -file reports/setup_violations.rpt { report_timing -delay_type max -max_paths 50 -path_type full_clock_expanded }
redirect -file reports/hold_violations.rpt { report_timing -delay_type min -max_paths 50 -path_type full_clock_expanded }
redirect -file reports/summary_qor.rpt { report_qor }

set worst_slack [get_attribute [get_timing_paths -delay_type max] slack]
if {$worst_slack < 0.0} {
    puts "ERROR: Timing violation detected! Worst negative slack: $worst_slack ns"
    exit 1
} else {
    puts "SUCCESS: Timing closure achieved. Worst slack: $worst_slack ns"
}
