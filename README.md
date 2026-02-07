# FPGA Novelty Detection Accelerator

**Target FPGA:** Gowin Tang Nano 9K

**Logic:** Leaky Integrator Neuron + UART E2E Benchmark Echo

---

## Project Overview
This project implements a **novelty detection accelerator** on the Gowin Tang Nano 9K FPGA, designed with a focus on **extremely low resource usage**—a core principle in my research on federated learning (FL) and decentralized systems. The goal is to **dismantle the "compute wall" paradigm** by demonstrating how lightweight, deterministic hardware can achieve real-time performance with minimal resources.

The accelerator receives streaming input via UART, accumulates signal energy using a **leaky integrator**, compares it to pre-set thresholds in a **weight ROM**, and visualizes novelty events on **onboard LEDs**. It also echoes input back to a host for **end-to-end (E2E) latency benchmarking**, enabling direct comparison with CPU-based implementations.

This design serves as a **hardware-software co-design benchmark**, aligning with my broader research goals of advancing **large-scale device networks** (1M–100M devices) and exploring novel gated mechanisms in FL.

---

## Key Features
- **Leaky Integrator Neuron**: Smooths input and accumulates energy with minimal computational overhead.
- **Weight ROM Thresholding**: Triggers LED feedback when energy exceeds a threshold, ensuring deterministic behavior.
- **UART RX/TX Echo**: Enables real-time benchmarking of FPGA vs CPU, validating hardware efficiency.
- **LED Visualization**: Provides immediate feedback on novelty events, useful for debugging and demonstration.
- **Deterministic Behavior**: Achieves low jitter (<0.5 µs) for reproducible results, critical for large-scale deployments.

---

