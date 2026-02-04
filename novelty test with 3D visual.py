#!/usr/bin/env python3
"""
Novelty Functional - Real-Time 3D Popout Visualization
======================================================

Version: 1.2.0

This script computes novelty functional for a toy LLM dataset
and dynamically visualizes KL, Fisher trace, and novelty scores
in a live 3D popout simulation.

Dependencies:
- torch
- transformers
- matplotlib
"""

import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

# -----------------------------
# Configurations
# -----------------------------
class FisherMethod(Enum):
    DIAGONAL = "diagonal"

@dataclass
class FisherConfig:
    method: FisherMethod = FisherMethod.DIAGONAL
    device: Optional[str] = None
    target_layers: Optional[List[str]] = field(default_factory=lambda: ["lm_head"])

@dataclass
class KLConfig:
    epsilon: float = 1e-8

@dataclass
class NoveltyConfig:
    fisher: FisherConfig = field(default_factory=FisherConfig)
    kl: KLConfig = field(default_factory=KLConfig)
    attention_normalizer: float = 512.0
    eps: float = 1e-6

# -----------------------------
# Fisher Information
# -----------------------------
class FisherInfo:
    def __init__(self, config: Optional[FisherConfig] = None):
        self.config = config or FisherConfig()

    def get_trace(self, model: PreTrainedModel, inputs: Dict[str, torch.Tensor]) -> float:
        model.eval()
        with torch.enable_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            model.zero_grad(set_to_none=True)
            loss.backward()
            trace = 0.0
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                if self.config.target_layers and not any(t in name for t in self.config.target_layers):
                    continue
                trace += p.grad.pow(2).sum().item()
            model.zero_grad(set_to_none=True)
        return trace

# -----------------------------
# KL Divergence
# -----------------------------
class KLDivergence:
    def __init__(self, config: Optional[KLConfig] = None):
        self.config = config or KLConfig()

    def vs_uniform(self, logits: torch.Tensor) -> float:
        vocab_size = logits.shape[-1]
        log_uniform = -torch.log(torch.tensor(vocab_size, device=logits.device, dtype=logits.dtype))
        log_p = F.log_softmax(logits, dim=-1)
        kl = F.kl_div(
            input=log_uniform.expand_as(log_p),
            target=log_p,
            log_target=True,
            reduction="batchmean"
        )
        return float(kl)

# -----------------------------
# Novelty Functional
# -----------------------------
class NoveltyFunctional:
    def __init__(self, config: Optional[NoveltyConfig] = None):
        self.config = config or NoveltyConfig()
        self.fisher = FisherInfo(self.config.fisher)
        self.kl_div = KLDivergence(self.config.kl)

    @torch.no_grad()
    def compute(self, text: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Dict[str, float]:
        device = next(model.parameters()).device
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)

        logits = model(**inputs).logits[:, -1, :]
        kl_score = self.kl_div.vs_uniform(logits)
        fisher_trace = self.fisher.get_trace(model, inputs)

        token_count = inputs.input_ids.shape[1]
        length_penalty = (token_count / self.config.attention_normalizer) + self.config.eps
        novelty_score = (kl_score * fisher_trace) / length_penalty

        return {
            "novelty_score": novelty_score,
            "kl_divergence": kl_score,
            "fisher_trace": fisher_trace,
            "token_count": token_count,
        }

# -----------------------------
# Real-Time 3D Popout Simulation
# -----------------------------
def run_real_time_simulation_3d(texts: List[str], model_name="gpt2", seed_base=123):
    torch.manual_seed(seed_base)
    random.seed(seed_base)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    nf = NoveltyFunctional()

    kl_scores = [0]*len(texts)
    fisher_scores = [0]*len(texts)
    novelty_scores = [0]*len(texts)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    xpos = range(len(texts))
    ypos = [0]*len(texts)
    zpos = [0]*len(texts)
    dx = [0.3]*len(texts)
    dy = [0.3]*len(texts)
    dz = novelty_scores

    bars = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='teal', alpha=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels([t[:20]+"..." if len(t)>20 else t for t in texts], rotation=45, ha='right')
    ax.set_ylabel('Score Type')
    ax.set_zlabel('Value')
    ax.set_title("Real-Time 3D Novelty Functional Simulation")

    def update(frame):
        seed = seed_base + frame
        torch.manual_seed(seed)
        random.seed(seed)

        result = nf.compute(texts[frame], model, tokenizer)
        kl_scores[frame] = result['kl_divergence']
        fisher_scores[frame] = result['fisher_trace']
        novelty_scores[frame] = result['novelty_score']

        ax.cla()
        dz_novelty = novelty_scores
        dz_kl = kl_scores
        dz_fisher = fisher_scores

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz_novelty, color='teal', label='Novelty', alpha=0.8)
        ax.bar3d(xpos, [y+0.35 for y in ypos], zpos, dx, dy, dz_kl, color='orange', label='KL', alpha=0.6)
        ax.bar3d(xpos, [y+0.7 for y in ypos], zpos, dx, dy, dz_fisher, color='purple', label='Fisher', alpha=0.6)

        ax.set_xticks(xpos)
        ax.set_xticklabels([t[:20]+"..." if len(t)>20 else t for t in texts], rotation=45, ha='right')
        ax.set_ylabel('Score Type Offset')
        ax.set_zlabel('Value')
        ax.set_title("Real-Time 3D Novelty Functional Simulation")
        ax.set_zlim(0, max(max(novelty_scores)*1.2, 1))
        ax.legend()

        print(f"[{frame+1}/{len(texts)}] Text: {texts[frame][:50]}... | Novelty: {result['novelty_score']:.4f}")

    ani = FuncAnimation(fig, update, frames=len(texts), repeat=False, interval=1000)
    plt.show()

    print("\n--- Final Summary ---")
    for i, t in enumerate(texts):
        print(f"Text: {t[:50]}...")
        print(f"  KL Divergence: {kl_scores[i]:.4f}")
        print(f"  Fisher Trace:  {fisher_scores[i]:.4f}")
        print(f"  Novelty:       {novelty_scores[i]:.4f}\n")

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    toy_dataset = [
        "The quick brown fox jumps over the lazy dog.",
        "In a galaxy far, far away, there was a star.",
        "Hello world! This is a test sentence.",
        "Once upon a time, magic filled the air.",
        "Data science is transforming the world rapidly."
    ]
    run_real_time_simulation_3d(toy_dataset)
