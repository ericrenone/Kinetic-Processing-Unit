#!/usr/bin/env python3
"""
Novelty Functional - Real-Time Simulation
=========================================

Version: 1.0.0

Computes a novelty functional for LLM outputs and dynamically
visualizes novelty metrics in real-time.

Dependencies:
- torch
- transformers
- matplotlib
"""

import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# Configuration Classes
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
                if self.config.target_layers:
                    if not any(t in name for t in self.config.target_layers):
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
# Real-Time Simulation & Visualization
# -----------------------------
def simulate_dynamic_novelty(texts: List[str], model_name="gpt2", seed_base=42):
    # Seed everything
    torch.manual_seed(seed_base)
    random.seed(seed_base)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    nf = NoveltyFunctional()

    # Setup live plotting
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(texts)), [0.0]*len(texts), color='teal')
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(texts)))
    ax.set_xticklabels([t[:15] + "..." if len(t)>15 else t for t in texts], rotation=45)
    ax.set_ylabel("Novelty Score")
    ax.set_title("Real-Time Novelty Functional Simulation")

    scores = [0.0]*len(texts)
    for i, text in enumerate(texts):
        # Dynamic seed per input
        dynamic_seed = seed_base + i
        torch.manual_seed(dynamic_seed)
        random.seed(dynamic_seed)

        result = nf.compute(text, model, tokenizer)
        scores[i] = result["novelty_score"]

        # Update bars
        for bar, score in zip(bars, scores):
            bar.set_height(score)
        ax.set_ylim(0, max(scores)*1.2 if max(scores) > 0 else 1)
        plt.pause(0.5)  # small delay for dynamic effect
        print(f"[{i+1}/{len(texts)}] Text: {text[:50]}... Novelty: {result['novelty_score']:.4f}")

    plt.ioff()
    plt.show()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Once upon a time in a distant galaxy far away...",
        "Hello world!",
        "In the beginning, there was only darkness.",
        "To be or not to be, that is the question."
    ]
    simulate_dynamic_novelty(sample_texts)
