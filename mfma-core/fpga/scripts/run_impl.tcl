# run_impl.tcl — Vivado Place & Route Script

read_checkpoint checkpoints/post_synth.dcp

opt_design
place_design
phys_opt_design
route_design

report_timing_summary -file reports/post_route_timing_summary.rpt
report_utilization -file reports/post_route_utilization.rpt
report_drc -file reports/post_route_drc.rpt

write_checkpoint -force checkpoints/post_route.dcp
