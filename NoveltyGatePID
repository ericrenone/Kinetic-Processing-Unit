#!/usr/bin/env python3
"""
Production-Ready Novelty Gate Simulation
- 10 clients, 2D global parameter
- Adversarial outliers
- Realistic Novelty Gate with adaptive threshold
- Energy based on aggregated gradient norm (no hidden target)
- 6 publication-style charts in one popup
- Automatic summary printed
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. CONFIGURATION
# -----------------------------
NUM_CLIENTS = 10
DIM = 2
STEPS = 200
MOVING_AVG_WINDOW = 10  # for adaptive gate threshold

# Dynamic seed
SEED = np.random.randint(0, 1_000_000)
np.random.seed(SEED)

# Global target (hidden from gate, used only for evaluation)
THETA_STAR = np.array([10.0, 10.0])
# Increase diversity to test gate robustness
LOCAL_TARGETS = THETA_STAR + np.random.normal(0, 1.0, (NUM_CLIENTS, DIM))

# -----------------------------
# 2. NOVELTY GATE CONTROLLER
# -----------------------------
class NoveltyGateController:
    def __init__(self, use_gate=True):
        self.use_gate = use_gate
        self.past_spreads = []

    def compute_learning_rate(self, aggregated_grad):
        # Energy based on norm of aggregated gradient
        energy = 0.5 * np.linalg.norm(aggregated_grad)**2
        gamma_base = 0.5 * np.exp(-0.01 * energy)

        if self.use_gate:
            # Compute spread of current aggregated gradient
            grad_norm = np.linalg.norm(aggregated_grad)
            self.past_spreads.append(grad_norm)
            # Moving average for adaptive threshold
            window = self.past_spreads[-MOVING_AVG_WINDOW:]
            adaptive_threshold = np.mean(window)
            # Sigmoid gate
            gate = 1 / (1 + np.exp(5 * (grad_norm - adaptive_threshold)))
            return gamma_base * gate, energy
        return gamma_base, energy

# -----------------------------
# 3. SIMULATION FUNCTION
# -----------------------------
def run_simulation(use_gate=True):
    controller = NoveltyGateController(use_gate=use_gate)
    x_glob = np.zeros(DIM)
    dist_history = []
    gamma_history = []
    energy_history = []

    for t in range(STEPS):
        grads = []
        for i in range(NUM_CLIENTS):
            grad = 0.2 * (LOCAL_TARGETS[i] - x_glob) + np.random.normal(0, 0.1, DIM)
            grads.append(grad)
        # Adversarial injection
        if t % 15 == 0:
            grads[-1] += np.random.uniform(-50, -40, DIM)
        grads = np.array(grads)

        aggregated_grad = np.mean(grads, axis=0)
        gamma, energy = controller.compute_learning_rate(aggregated_grad)

        gamma_history.append(gamma)
        energy_history.append(energy)
        x_glob += gamma * aggregated_grad
        dist_history.append(np.linalg.norm(x_glob - THETA_STAR))

    return np.array(dist_history), np.array(gamma_history), np.array(energy_history)

# -----------------------------
# 4. RUN SIMULATIONS
# -----------------------------
dist_sota, gamma_sota, energy_sota = run_simulation(use_gate=False)
dist_gate, gamma_gate, energy_gate = run_simulation(use_gate=True)

# -----------------------------
# 5. METRICS
# -----------------------------
def calculate_metrics(distances):
    resilience = 20.0 / (np.max(distances) + 1e-9)
    precision = 1.0 / (np.mean(distances[-10:]) + 1e-9)
    stability = np.mean(np.diff(distances) <= 0.1) * 100
    idx = np.where(distances < 1.5)[0]
    efficiency = (STEPS - idx[0]) if len(idx) > 0 else 0
    return resilience, precision, stability, efficiency

metrics_sota = calculate_metrics(dist_sota)
metrics_gate = calculate_metrics(dist_gate)
metric_names = ['Resilience', 'Precision', 'Stability (%)', 'Efficiency']

# -----------------------------
# 6. PLOT 6 CHARTS IN SINGLE POPUP
# -----------------------------
plt.ion()
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 6a. Convergence curves
ax1 = axes[0, 0]
ax1.plot(dist_sota, label='SOTA', color='#eb4d4b', linewidth=2)
ax1.plot(dist_gate, label='Novelty Gate', color='#6ab04c', linewidth=2)
ax1.set_title('Convergence to Target', fontsize=14, fontweight='bold')
ax1.set_xlabel('Steps')
ax1.set_ylabel('Distance')
ax1.grid(True)
ax1.legend()

# 6b. Learning rate curves
ax2 = axes[0, 1]
ax2.plot(gamma_sota, label='SOTA', color='#eb4d4b', linewidth=2)
ax2.plot(gamma_gate, label='Novelty Gate', color='#6ab04c', linewidth=2)
ax2.set_title('Adaptive Learning Rates', fontsize=14, fontweight='bold')
ax2.set_xlabel('Steps')
ax2.set_ylabel('Gamma')
ax2.grid(True)
ax2.legend()

# 6c. Energy curves
ax3 = axes[0, 2]
ax3.plot(energy_sota, label='SOTA', color='#eb4d4b', linewidth=2)
ax3.plot(energy_gate, label='Novelty Gate', color='#6ab04c', linewidth=2)
ax3.set_title('Aggregated Gradient Energy', fontsize=14, fontweight='bold')
ax3.set_xlabel('Steps')
ax3.set_ylabel('Energy')
ax3.grid(True)
ax3.legend()

# 6d-f. Metric bar charts
colors = ['#eb4d4b', '#6ab04c']
for i, ax in enumerate(axes[1, :]):
    ax.bar(['SOTA', 'Novelty Gate'], [metrics_sota[i], metrics_gate[i]], color=colors)
    ax.set_title(metric_names[i], fontsize=14, fontweight='bold')
    ax.set_ylabel('Higher = Better')
    gain = ((metrics_gate[i] - metrics_sota[i]) / (metrics_sota[i] + 1e-9)) * 100
    ax.annotate(f"+{gain:.1f}%", xy=(1, metrics_gate[i]), xytext=(0, 5),
                textcoords="offset points", ha='center', fontweight='bold', fontsize=12)

plt.suptitle(f"Novelty Gate Simulation (Seed={SEED})", fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show(block=True)
plt.ioff()

# -----------------------------
# 7. SUMMARY PRINT
# -----------------------------
print("\n=== Simulation Summary ===")
print(f"Random Seed: {SEED}")
print("Metric Comparison (Higher = Better):")
for i, name in enumerate(metric_names):
    print(f"{name}: SOTA = {metrics_sota[i]:.4f}, Novelty Gate = {metrics_gate[i]:.4f}, Gain = {(metrics_gate[i]-metrics_sota[i])/metrics_sota[i]*100:.1f}%")
print("\nFinal Convergence Distances:")
print(f"SOTA final distance: {dist_sota[-1]:.4f}")
print(f"Novelty Gate final distance: {dist_gate[-1]:.4f}")
print("============================\n")
