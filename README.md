# Semantic Novelty Engine: Information-Geometric Text Analysis

## Core

Traditional novelty detection relies on simple statistical outliers. This engine operates in the **Geometric Domain**:
1.  **Surprise (KL Divergence):** Measures how "unlikely" a sequence is compared to a base uniform distribution.
2.  **Sensitivity (Fisher Information):** Measures the "strain" or "work" required by the model's parameters to process the input.
3.  **The Novelty Functional ($\Phi$):** A unified metric that triggers alerts when an input provides high-value, novel information that the model hasn't "mastered."

---

## 🛠 Technical Architecture

### 1. The Novelty Functional ($\Phi$)
The engine calculates a novelty score by balancing distributional divergence against parameter sensitivity:

$$\Phi = \frac{D_{KL} \cdot Trace(\mathcal{I})}{\frac{N_{tokens}}{Normalizer} + \epsilon}$$

### 2. Information Metrics
* **KL Divergence ($D_{KL}$):** Calculated between the model's output `log_softmax` and a log-uniform distribution. This captures the specificity of the prediction.
    
* **Fisher Information Trace ($\mathcal{I}$):** Derived from the squared norm of gradients in the `lm_head` (Semantic Bottleneck) during a backward pass. This represents the "Learning Pressure" exerted by the text.
* **Attention Normalization:** Adjusts the score based on sequence length to prevent long, repetitive strings from inflating novelty.

### 3. Semantic Bottleneck Targeting
Instead of analyzing all model weights ($O(n)$), the engine targets the `lm_head`. This focus allows for real-time analysis while capturing the most critical semantic gradients where internal representations project into vocabulary space.

---

## 📊 Live Simulation & Visualization

The engine includes an interactive suite that tracks three distinct data streams in real-time:

| Metric | Visualization | Purpose |
| :--- | :--- | :--- |
| **Novelty ($\Phi$)** | **Teal Line** | Final decision metric. Alerts trigger when $\Phi > \tau$ (Threshold). |
| **KL Divergence** | **Orange Line** | Tracks "predictive surprise" relative to vocabulary distribution. |
| **Fisher Trace** | **Purple Line** | Tracks "parameter stress" or new knowledge acquisition. |



