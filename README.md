# Identity-Geometry

**Fisher-Rao information geometry + rational inattention novelty scoring for LLM outputs.**

Lightweight, research-grade Python library for computing **novelty and information density** in language models. Provides reproducible **diagonal Fisher trace**, **KL divergence**, and a **novelty functional**.

---

# LLM Informational Novelty

A minimal, production-ready framework for measuring **informational novelty** in LLM outputs. 
Provides a deterministic, model-relative score by analyzing internal signals.

## Overview
This metric answers: *How much new information does this text contain from the model’s own perspective?*

Unlike traditional similarity metrics, this approach is:
* **Internal:** Uses model-specific signals (not vector distance).
* **Deterministic:** Produces an auditable, repeatable scalar value.
* **Non-Stylistic:** Focuses on information density over creativity or prose.

---

## How It Works
The novelty score distills three internal signals into a single scalar:

* **Prediction Confidence:** Measures the certainty of token predictions.
* **Parameter Sensitivity:** Tracks how meaningfully the input activates model weights.
* **Length Normalization:** Prevents score inflation from verbosity or repetition.

---

## Interpreting the Score

| Score | Impact | Content Characteristics |
| :--- | :--- | :--- |
| **High** | Significant | Informative, non-generic, and engages specific model weights. |
| **Medium** | Moderate | Predictable structures with some unique signal. |
| **Low** | Minimal | Generic boilerplate, rote memorization, or high redundancy. |

---

## Key Use Cases
* **Prompt Engineering:** Evaluate which prompts elicit the most substantive responses.
* **Data Curation:** Deduplicate datasets and filter for high-information density.
* **Memorization Audits:** Detect if a model is "parroting" training data.
* **Agent Efficiency:** Enable agents to self-filter redundant or circular reasoning.
