"""Smoke test for the API interaction graph builder.

We bypass the heavy sentence-transformer by monkey-patching the
embedder with a deterministic stub so the test is fast and offline.
"""
from __future__ import annotations

import types

import torch

from ct_dgnn.data.graph_builder import APIInteractionGraphBuilder, Event


class _FakeEmbedder:
    dim = 384

    def encode(self, texts):
        # Hash-based deterministic 384-dim "embeddings" — no network.
        out = []
        for t in texts:
            rng = torch.Generator().manual_seed(abs(hash(t)) % (2**31))
            out.append(torch.randn(384, generator=rng))
        return torch.stack(out) if out else torch.zeros(0, 384)


def test_builder_produces_interaction_batch():
    builder = APIInteractionGraphBuilder(similarity_threshold=0.99)
    builder.embedder = _FakeEmbedder()                     # type: ignore

    events = [
        Event("u1", "s1", "hello", "gpt4", "hi", 0.0, 0, None, None),
        Event("u1", "s1", "hello again", "gpt4", "hi", 1.0, 0, None, None),
        Event("u2", "s2", "malicious prompt", "gpt4", "...", 2.0, 1,
              "camp1", "pair"),
    ]
    batch = builder.ingest(events)
    assert batch.num_nodes > 0
    assert batch.edge_index.shape[0] == 2
    assert batch.query_labels.numel() == 3
    # node type ids: 0 user, 1 session, 2 query, 3 model
    assert (batch.node_type_ids == 2).sum().item() == 3
    assert (batch.node_type_ids == 1).sum().item() == 2
