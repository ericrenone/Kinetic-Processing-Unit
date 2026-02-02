"""
Novelty Functional for LLM Outputs
=================================

Version: 1.0.0

This module implements a **minimal, production-ready novelty functional**
for Large Language Model (LLM) outputs. It combines:

1. **KL Divergence vs. Uniform**
   - Measures distributional certainty / peakedness of the model’s prediction.
2. **Diagonal Fisher Information Trace**
   - Measures parameter sensitivity of the model to the given input.
3. **Context-Length Normalization**
   - Penalizes long or repetitive inputs that would otherwise inflate scores.

The resulting scalar is designed to answer:

    “How *informationally novel* is this text to the model?”

---

Formal Definition
-----------------

Given input text x and model parameters θ:

- Let pθ be the next-token distribution.
- Let U be the uniform distribution over the vocabulary.
- Let Fθ(x) be the diagonal Fisher trace:

    Fθ(x) = Σ_i (∂ log pθ(x) / ∂θ_i)²

We define the novelty functional:

    Novelty(x) = ( KL(pθ || U) · Fθ(x) ) / L(x)

where:

    L(x) = (token_count / attention_normalizer) + ε

---

Interpretation
--------------

High novelty implies:
- The model is **confident** (non-uniform prediction),
- The input is **parameter-sensitive** (high Fisher trace),
- The signal is **not trivially long or repetitive**.

Low novelty implies:
- Generic phrasing,
- Memorized or low-information content,
- Or weak interaction with model parameters.

---

Design Constraints
------------------

- Diagonal Fisher only (tractable, stable, auditable)
- No sampling, no stochastic baselines
- Explicit layer filtering for production scalability
- Deterministic and reproducible

---

Dependencies
------------

- torch
- transformers

---

Example
-------

>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> from novelty_functional import NoveltyFunctional
>>>
>>> model = AutoModelForCausalLM.from_pretrained("gpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
>>> nf = NoveltyFunctional()
>>> nf.compute("The quick brown fox jumps over the lazy dog", model, tokenizer)

Returns a dict with:
- novelty_score
- kl_divergence
- fisher_trace
- token_count
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer


# =============================================================================
# Configuration
# =============================================================================

class FisherMethod(Enum):
    """Supported Fisher estimation methods."""
    DIAGONAL = "diagonal"


@dataclass
class FisherConfig:
    """
    Configuration for Fisher Information computation.

    target_layers:
        If None → use all parameters.
        If set → only parameters whose names contain these substrings
                 (e.g. ['lm_head'] for large speedups).
    """
    method: FisherMethod = FisherMethod.DIAGONAL
    device: Optional[str] = None
    target_layers: Optional[List[str]] = field(
        default_factory=lambda: ["lm_head"]
    )


@dataclass
class KLConfig:
    """Configuration for KL divergence."""
    epsilon: float = 1e-8


@dataclass
class NoveltyConfig:
    """Top-level configuration for the novelty functional."""
    fisher: FisherConfig = field(default_factory=FisherConfig)
    kl: KLConfig = field(default_factory=KLConfig)
    attention_normalizer: float = 512.0
    eps: float = 1e-6


# =============================================================================
# Fisher Information
# =============================================================================

class FisherInfo:
    """
    Empirical diagonal Fisher Information estimator.

    Computes:
        trace(F) = Σ_i (∂L / ∂θ_i)²
    """

    def __init__(self, config: Optional[FisherConfig] = None):
        self.config = config or FisherConfig()

    def get_trace(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor]
    ) -> float:
        """
        Compute diagonal Fisher trace for a single input.

        Args:
            model: HuggingFace causal LM
            inputs: Tokenized inputs with input_ids

        Returns:
            Scalar Fisher trace
        """
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


# =============================================================================
# KL Divergence
# =============================================================================

class KLDivergence:
    """
    KL divergence against a uniform prior.

    Measures how concentrated the model’s prediction is.
    """

    def __init__(self, config: Optional[KLConfig] = None):
        self.config = config or KLConfig()

    def vs_uniform(self, logits: torch.Tensor) -> float:
        """
        Compute KL(Uniform || Model).

        Args:
            logits: Tensor of shape [batch, vocab]

        Returns:
            Scalar KL divergence
        """
        vocab_size = logits.shape[-1]

        log_uniform = -torch.log(
            torch.tensor(
                vocab_size,
                device=logits.device,
                dtype=logits.dtype
            )
        )

        log_p = F.log_softmax(logits, dim=-1)

        kl = F.kl_div(
            input=log_uniform.expand_as(log_p),
            target=log_p,
            log_target=True,
            reduction="batchmean"
        )

        return float(kl)


# =============================================================================
# Novelty Functional
# =============================================================================

class NoveltyFunctional:
    """
    Main novelty functional.

    Final score:
        (KL Divergence × Fisher Trace) / Length Penalty
    """

    def __init__(self, config: Optional[NoveltyConfig] = None):
        self.config = config or NoveltyConfig()
        self.fisher = FisherInfo(self.config.fisher)
        self.kl_div = KLDivergence(self.config.kl)

    @torch.no_grad()
    def compute(
        self,
        text: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer
    ) -> Dict[str, float]:
        """
        Compute novelty metrics for a given input string.

        Returns:
            Dictionary with:
                - novelty_score
                - kl_divergence
                - fisher_trace
                - token_count
        """
        device = next(model.parameters()).device

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(device)

        # 1. Distributional certainty
        logits = model(**inputs).logits[:, -1, :]
        kl_score = self.kl_div.vs_uniform(logits)

        # 2. Parameter sensitivity
        fisher_trace = self.fisher.get_trace(model, inputs)

        # 3. Length normalization
        token_count = inputs.input_ids.shape[1]
        length_penalty = (
            token_count / self.config.attention_normalizer
        ) + self.config.eps

        novelty_score = (kl_score * fisher_trace) / length_penalty

        return {
            "novelty_score": novelty_score,
            "kl_divergence": kl_score,
            "fisher_trace": fisher_trace,
            "token_count": token_count,
        }


# =============================================================================
# Metadata
# =============================================================================

__version__ = "1.0.0"

__all__ = [
    "FisherMethod",
    "FisherConfig",
    "KLConfig",
    "NoveltyConfig",
    "FisherInfo",
    "KLDivergence",
    "NoveltyFunctional",
]
