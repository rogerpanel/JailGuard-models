"""Adapter that exposes CT-DGNN-JailGuard as an LLM-API security
module for the RobustIDPS.ai platform (Section VI-B of the manuscript).

The adapter mirrors the 60+ Flask endpoints already exposed by
``robustidps_web_app``, adding one additional WebSocket channel
``/ws/llm_jailguard`` that emits campaign decisions in real time.

Start with:
    python -m deployment.robustidps_integration --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import torch

try:
    from fastapi import FastAPI, WebSocket
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:  # runtime-optional
    FastAPI = WebSocket = CORSMiddleware = uvicorn = None

from ct_dgnn.data.graph_builder import APIInteractionGraphBuilder, Event
from ct_dgnn.models.ct_dgnn import CTDGNNJailGuard
from ct_dgnn.utils.config import load_config
from ct_dgnn.utils.logging import get_logger


def build_app(ckpt: str | None, config_path: str) -> "FastAPI":
    if FastAPI is None:
        raise RuntimeError("fastapi + uvicorn are required for the adapter")

    cfg = load_config(config_path)
    log = get_logger("robustidps.adapter")
    device = cfg.device if torch.cuda.is_available() else "cpu"

    model = CTDGNNJailGuard(
        node_dims=cfg.graph.node_dims,
        embed_dim=cfg.graph.embed_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_heads=cfg.model.num_heads,
        use_adjoint=cfg.model.adjoint,
        dropout=cfg.model.dropout,
    ).to(device).eval()
    if ckpt and Path(ckpt).exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
        log.info(f"loaded ckpt {ckpt}")

    builder = APIInteractionGraphBuilder(
        node_dims=cfg.graph.node_dims,
        similarity_threshold=cfg.data.similarity_threshold,
        sentence_model=cfg.data.sentence_transformer,
    )

    app = FastAPI(title="CT-DGNN-JailGuard (RobustIDPS.ai module)")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "ct-dgnn-jailguard", "version": "1.0"}

    @app.post("/detect")
    def detect(payload: dict):
        events = [Event(**e) for e in payload["events"]]
        batch = builder.ingest(events)
        with torch.no_grad():
            batch.x = batch.x.to(device)
            batch.node_type_ids = batch.node_type_ids.to(device)
            batch.edge_index = batch.edge_index.to(device)
            batch.edge_delta_t = batch.edge_delta_t.to(device)
            batch.event_times = batch.event_times.to(device)
            batch.session_component_index = batch.session_component_index.to(device)
            t0 = time.perf_counter()
            out = model(batch)
            dt = (time.perf_counter() - t0) * 1000.0
        return {
            "latency_ms": dt,
            "campaign_probs": out.campaign_logits.softmax(-1).tolist(),
            "query_probs":    out.query_logits.softmax(-1).tolist(),
            "lipschitz":      {k: float(v) for k, v in out.lipschitz.items()},
        }

    @app.websocket("/ws/llm_jailguard")
    async def stream(ws: WebSocket):
        await ws.accept()
        while True:
            try:
                payload = await ws.receive_json()
            except Exception:
                break
            events = [Event(**e) for e in payload["events"]]
            batch = builder.ingest(events)
            with torch.no_grad():
                out = model(batch)
            await ws.send_json({
                "campaign_probs": out.campaign_logits.softmax(-1).tolist(),
            })

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.environ.get(
        "CTDGNN_CONFIG", "configs/default.yaml"))
    ap.add_argument("--ckpt", default=os.environ.get("CTDGNN_CKPT"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    app = build_app(args.ckpt, args.config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
