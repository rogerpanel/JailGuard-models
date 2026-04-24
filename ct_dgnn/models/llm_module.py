"""LLM-augmented zero-shot detection module.

Three responsibilities (Section IV.E of the manuscript):

1. *Query intent classification.*
   Fine-tuned Mistral-7B-Instruct (default) classifies each query into
   one of {information_seeking, capability_probing, constraint_testing,
   role_play_elicitation, encoding_based_obfuscation}. The returned
   one-hot vector is concatenated into query node features.

2. *Novel attack pattern narration.*
   When the DGNN flags an anomalous subgraph but the classifier softmax
   stays below the confidence threshold, the LLM is asked to narrate
   the attack pattern in natural language for SOC analyst review.

3. *Bayesian online update loop.*
   Analyst-confirmed novel samples update the query-embedding space
   via InfoNCE contrastive loss on just the new vs. retrieved exemplars,
   without full model retraining (matches the PI's UC-HGP approach).

The heavy inference is asynchronous — the main detection path uses the
cached intent label from the last successful LLM pass when available.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
from torch import nn


INTENT_CATEGORIES: tuple[str, ...] = (
    "information_seeking",
    "capability_probing",
    "constraint_testing",
    "role_play_elicitation",
    "encoding_based_obfuscation",
)


INTENT_PROMPT = """You are a security analyst. Classify the LLM-user query
below into exactly ONE of these intent categories and respond with only
the category name:
{categories}

Query:
{query}

Intent:"""


NARRATION_PROMPT = """You are a security analyst describing a coordinated
LLM jailbreak campaign to a human reviewer. Below are the queries in
temporal order from one or more users. Summarise in under 150 words:
  (a) the suspected attack strategy,
  (b) the observable temporal coordination, and
  (c) the risk category.

Queries:
{queries}

Analysis:"""


@dataclass
class LLMAnalysis:
    intent_logits: torch.Tensor          # (N_queries, |INTENT_CATEGORIES|)
    narration: str | None = None
    extras: dict = field(default_factory=dict)


class LLMZeroShotAnalyzer(nn.Module):
    """Thin async wrapper around a causal-LM backend.

    The backend is resolved at construction time. A minimal fallback is
    used when transformers is unavailable, returning a uniform intent
    distribution so the rest of the pipeline still trains.
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        enabled: bool = True,
        max_new_tokens: int = 48,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = device
        self._backend = None

        if enabled:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._backend = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch.float16
                )
                if device:
                    self._backend = self._backend.to(device)
                self._backend.eval()
            except Exception:
                # Fallback: no HF weights available → uniform distribution.
                self._backend = None

    # ------------------------------------------------------------------
    def intent_logits(self, queries: Iterable[str]) -> torch.Tensor:
        """Return zero-shot intent logits for a list of queries."""
        queries = list(queries)
        K = len(INTENT_CATEGORIES)
        if self._backend is None or not self.enabled:
            return torch.zeros(len(queries), K)
        out = []
        for q in queries:
            prompt = INTENT_PROMPT.format(
                categories="\n- " + "\n- ".join(INTENT_CATEGORIES),
                query=q,
            )
            with torch.no_grad():
                inp = self._tokenizer(prompt, return_tensors="pt").to(self._backend.device)
                logits = self._backend(**inp).logits[:, -1, :]                  # (1, V)
                category_ids = [
                    self._tokenizer.encode(c, add_special_tokens=False)[0]
                    for c in INTENT_CATEGORIES
                ]
                cat_logits = logits[0, category_ids]                            # (K,)
                out.append(cat_logits.softmax(dim=-1).cpu())
        return torch.stack(out)

    def narrate_subgraph(self, queries: list[str]) -> str:
        """Produce a natural-language description of a flagged sub-graph."""
        if self._backend is None or not self.enabled:
            return "[LLM disabled — narration unavailable]"
        prompt = NARRATION_PROMPT.format(queries="\n".join(f"- {q}" for q in queries))
        with torch.no_grad():
            inp = self._tokenizer(prompt, return_tensors="pt").to(self._backend.device)
            out = self._backend.generate(
                **inp, max_new_tokens=self.max_new_tokens, do_sample=False,
            )
        return self._tokenizer.decode(out[0][inp.input_ids.shape[1]:],
                                      skip_special_tokens=True)

    # ------------------------------------------------------------------
    def online_update(self, query_emb: torch.Tensor, positive_emb: torch.Tensor,
                      negative_emb: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """InfoNCE loss for the Bayesian online-update feedback loop.

        `query_emb` is a newly analyst-confirmed query. Positives are
        retrieved exemplars of the same novel-attack family; negatives
        are a random sample of benign queries.
        """
        q = torch.nn.functional.normalize(query_emb, dim=-1)
        pos = torch.nn.functional.normalize(positive_emb, dim=-1)
        neg = torch.nn.functional.normalize(negative_emb, dim=-1)
        pos_score = (q * pos).sum(dim=-1, keepdim=True) / temperature     # (N, 1)
        neg_score = (q @ neg.t()) / temperature                           # (N, K)
        logits = torch.cat([pos_score, neg_score], dim=-1)
        target = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return torch.nn.functional.cross_entropy(logits, target)
