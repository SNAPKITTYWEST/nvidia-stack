# generate_bitstream.tcl — Vivado Bitstream Generation

read_checkpoint checkpoints/post_route.dcp

write_bitstream -force bitstream/mfma_core.bit
write_cfgmem -format BIN -interface SPIx4 -size 256 -loadbit "up 0x0 bitstream/mfma_core.bit" -force bitstream/mfma_core.bin

puts "Bitstream generation complete: bitstream/mfma_core.bit"
