// fpga_mfma_accelerator.sv — FPGA Digital RTL Implementation
// Synthesizable SystemVerilog for AMD Alveo U55C / U250

module fpga_mfma_accelerator (
    input  logic        clk,
    input  logic        rst_n,
    input  logic [15:0] a_tile [0:255],
    input  logic [15:0] b_tile [0:255],
    input  logic [31:0] c_tile [0:255],
    output logic [31:0] out_tile [0:255],
    output logic        activity_pulse
);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < 256; i++) begin
                out_tile[i] <= '0;
            end
            activity_pulse <= 1'b0;
        end else begin
            activity_pulse <= 1'b1;
            for (int m = 0; m < 16; m++) begin
                for (int n = 0; n < 16; n++) begin
                    automatic logic [31:0] acc = c_tile[m * 16 + n];
                    for (int k = 0; k < 16; k++) begin
                        acc += (32'(a_tile[m * 16 + k]) * 32'(b_tile[k * 16 + n]));
                    end
                    out_tile[m * 16 + n] <= acc;
                end
            end
        end
    end

endmodule
