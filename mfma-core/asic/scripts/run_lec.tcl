# run_lec.tcl — Logic Equivalence Checking (LEC) via Synopsys Formality

set_svf mfma_core.svf

read_sverilog -libname WORK -work_library WORK ../rtl/fpga_mfma_accelerator.sv
set_top fpga_mfma_accelerator

read_verilog -container rev -libname WORK mfma_core_gated.v
set_top -container rev fpga_mfma_accelerator

match
verify

report_passing_points > reports/lec_passing.rpt
report_failing_points > reports/lec_failing.rpt
report_uncompared_points > reports/lec_uncompared.rpt

set unmatched [get_uncompared_points -count]
set failing [get_failing_points -count]
if {$failing > 0 || $unmatched > 0} {
    puts "ERROR: LEC Verification Failed! Failing: $failing, Unmatched: $unmatched"
    exit 1
} else {
    puts "SUCCESS: Post-route netlist is provably equivalent to golden HLS RTL."
}
