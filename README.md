# FPGA Novelty Detection Accelerator

**Target FPGA:** Gowin Tang Nano 9K  
**Logic:** Leaky Integrator Neuron + UART E2E Benchmark Echo

---

## Project Overview

This project implements a **novelty detection accelerator** on the Gowin Tang Nano 9K FPGA, designed with a focus on **extremely low resource usage**—a core principle in research on federated learning (FL) and decentralized systems. The goal is to **dismantle the "compute wall" paradigm** by demonstrating how lightweight, deterministic hardware can achieve real-time performance with minimal resources.

The accelerator receives streaming input via UART, accumulates signal energy using a **leaky integrator**, compares it to pre-set thresholds in a **weight ROM**, and visualizes novelty events on **onboard LEDs**. It also echoes input back to a host for **end-to-end (E2E) latency benchmarking**, enabling direct comparison with CPU-based implementations.

This design serves as a **hardware-software co-design benchmark**, aligning with broader research goals of advancing **large-scale device networks** (1M–100M devices) and exploring novel gated mechanisms in FL.

---

## 1. Lambda Calculus Formalization

The **state transition** of the neuron can be expressed functionally:

- `energy_acc` — current energy accumulator state.  
- `rx_data` — input stimulus value.  
- `>>` — bitwise right-shift operator implementing temporal decay.  
- Integration step — combines new data into the existing potential.  

The **novelty predicate** determines system output:

- Synaptic weights retrieved from BRAM ROM.  
- Hardware comparator logic triggers LED visualization.

---

## Key Features

- **Leaky Integrator Neuron:** Smooths input and accumulates energy with minimal computational overhead.  
- **Weight ROM Thresholding:** Triggers LED feedback when energy exceeds a threshold, ensuring deterministic behavior.  
- **UART RX/TX Echo:** Enables real-time benchmarking of FPGA vs CPU, validating hardware efficiency.  
- **LED Visualization:** Provides immediate feedback on novelty events, useful for debugging and demonstration.  
- **Deterministic Behavior:** Achieves low jitter (<0.5 µs) for reproducible results, critical for large-scale deployments.  

---

# Conclusions

## 1. Architectural Determinism

- **Hardware Superiority:** The FPGA implementation provides deterministic latency, ensuring consistent processing times.  
- **Jitter Reduction:** Hardware execution eliminates the unpredictable timing variability found in software, maintaining jitter levels below the measurable threshold.  
- **Silicon Efficiency:** Compute time on the silicon is effectively negligible when compared to data transport overhead.  

## 2. Algorithmic Performance

- **Functional Parity:** The "Software Mirror" accumulator validates that the Verilog "Silicon" logic is mathematically accurate.  
- **Neuromorphic Success:** The **Leaky Integrator Neuron** model accurately accumulates input "energy" and decays over time, maintaining temporal relevance.  
- **Novelty Discrimination:** The system effectively distinguishes between normal patterns (alphanumeric) and novel events (symbols), as evidenced by energy spikes and subsequent gating behavior.  

## 3. System Constraints

- **I/O-Bound Limitation:** The primary system bottleneck is UART transport delay, not the processing capabilities of the FPGA or CPU.  
- **Overflow Stability:** Bit-shifting (`>> 1`) for the "leak" and software-side safety caps maintain accumulator stability over 1,000+ testing rounds.  

## 4. Experimental Validity

- **Reproducibility:** Fixed seeds (`SEED = 42`) and deterministic anomaly windows ensure experiments are scientifically repeatable.  
- **Physical Feedback:** Onboard LEDs successfully triggered based on energy thresholds, confirming the novelty gating logic operates as intended in hardware.
