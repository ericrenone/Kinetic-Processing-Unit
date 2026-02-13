// ============================================================================
// PROJECT: FPGA NOVELTY DETECTION ACCELERATOR
// CHIP: GOWIN TANG NANO 9K
// LOGIC: LEAKY INTEGRATOR NEURON + UART E2E BENCHMARK ECHO
// ============================================================================

module top_novelty (
    input clk,            // Pin 52 (27MHz)
    input reset_n,        // Pin 4 (Active Low)
    input rx_pin,         // Pin 18 (UART RX)
    output tx_pin,        // Pin 17 (UART TX Echo)
    output [5:0] leds     // Onboard LEDs (Novelty Visualization)
);

    wire [7:0] rx_data;
    wire rx_tick;
    wire tx_busy;
    reg tx_start;
    
    wire [7:0] weight;
    reg [3:0] addr = 4'd0;
    reg [15:0] energy_acc = 16'd0;

    // --- 1. UART RECEIVER (INPUT GATE) ---
    uart_rx #(.CLK_FREQ(27000000), .BAUD(115200)) rx_inst (
        .clk(clk), .rx(rx_pin), .data(rx_data), .tick(rx_tick)
    );

    // --- 2. UART TRANSMITTER (BENCHMARK ECHO) ---
    uart_tx #(.CLK_FREQ(27000000), .BAUD(115200)) tx_inst (
        .clk(clk), .start(tx_start), .data(rx_data), .tx(tx_pin), .busy(tx_busy)
    );

    // --- 3. WEIGHT MEMORY (QUANTIZED ROM) ---
    weight_bram mem_inst (.clka(clk), .addra(addr), .douta(weight));

    // --- 4. NOVELTY THRESHOLDING (VISUAL OUTPUT) ---
    // LEDs turn ON (Low) if energy density exceeds weight
    assign leds = (energy_acc[11:4] > weight) ? 6'b000000 : 6'b111111;

    // --- 5. MAIN INFERENCE ENGINE ---
    always @(posedge clk) begin
        tx_start <= 1'b0;
        if (!reset_n) begin
            addr <= 4'd0;
            energy_acc <= 16'd0;
        end else if (rx_tick) begin
            addr <= addr + 4'd1;
            // Leaky Integrator: E = (E/2) + New_Data
            energy_acc <= (energy_acc >> 1) + {8'd0, rx_data};
            // Trigger Echo for Python E2E Timing
            if (!tx_busy) tx_start <= 1'b1;
        end
    end
endmodule

// --- UART RX MODULE ---
module uart_rx #(parameter CLK_FREQ = 27000000, parameter BAUD = 115200) (
    input clk, input rx, output reg [7:0] data, output reg tick
);
    localparam WAIT_COUNT = CLK_FREQ / BAUD;
    localparam HALF_COUNT = WAIT_COUNT / 2;
    reg [31:0] count = 32'd0;
    reg [3:0] state = 4'd0;
    reg receiving = 1'b0;
    always @(posedge clk) begin
        tick <= 1'b0;
        if (!receiving) begin
            if (rx == 1'b0) begin
                if (count < HALF_COUNT) count <= count + 32'd1;
                else begin count <= 32'd0; receiving <= 1'b1; state <= 4'd0; end
            end else count <= 32'd0;
        end else begin
            if (count < WAIT_COUNT - 1) count <= count + 32'd1;
            else begin
                count <= 32'd0;
                if (state < 4'd8) begin data[state[2:0]] <= rx; state <= state + 4'd1; end
                else begin receiving <= 1'b0; tick <= 1'b1; end
            end
        end
    end
endmodule

// --- UART TX MODULE ---
module uart_tx #(parameter CLK_FREQ = 27000000, parameter BAUD = 115200) (
    input clk, input start, input [7:0] data, output reg tx, output reg busy
);
    localparam WAIT_COUNT = CLK_FREQ / BAUD;
    reg [31:0] count = 32'd0;
    reg [3:0] state = 4'd0;
    reg [7:0] d_reg;
    initial tx = 1'b1;
    initial busy = 1'b0;
    always @(posedge clk) begin
        if (!busy) begin
            if (start) begin d_reg <= data; busy <= 1'b1; state <= 4'd0; count <= 32'd0; tx <= 1'b0; end
        end else begin
            if (count < WAIT_COUNT - 1) count <= count + 32'd1;
            else begin
                count <= 32'd0;
                if (state < 4'd8) begin tx <= d_reg[state[2:0]]; state <= state + 4'd1; end
                else if (state == 4'd8) begin tx <= 1'b1; state <= state + 4'd1; end
                else busy <= 1'b0;
            end
        end
    end
endmodule

// --- WEIGHT ROM MODULE ---
module weight_bram (input clka, input [3:0] addra, output reg [7:0] douta);
    reg [7:0] rom [0:15];
    initial begin
        rom[0]=8'h07; rom[1]=8'h15; rom[2]=8'h08; rom[3]=8'h12;
        rom[4]=8'h1e; rom[5]=8'h21; rom[6]=8'h08; rom[7]=8'h22;
        rom[8]=8'h09; rom[9]=8'h22; rom[10]=8'h0a; rom[11]=8'h1a;
        rom[12]=8'h1e; rom[13]=8'h0c; rom[14]=8'h1e; rom[15]=8'h06;
    end
    always @(posedge clka) douta <= rom[addra];
endmodule
