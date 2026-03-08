"""
HuggingFace Spaces entry point for Hypothesis Engine.

Deploy with:
    openenv-core serve --app app:app

Or run locally:
    uvicorn app:app --host 0.0.0.0 --port 7860
"""

import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from hypothesis_engine.openenv_wrapper import create_hypothesis_app


# ── Root-page middleware (openenv-core doesn't expose "/") ──
class RootPageMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/":
            return JSONResponse({
                "name": "Hypothesis Engine",
                "version": "2.0.0",
                "status": "running",
                "description": (
                    "A procedurally-generated RL environment for training LLMs on "
                    "scientific reasoning through causal discovery, physics simulation, "
                    "state machine reverse-engineering, and adversarial self-play."
                ),
                "endpoints": {
                    "GET  /": "This page",
                    "GET  /health": "Health check",
                    "GET  /metadata": "Environment metadata",
                    "GET  /schema": "Action/Observation JSON schemas",
                    "POST /reset": "Start a new episode",
                    "POST /step": "Take an action (experiment, hypothesize, predict)",
                    "GET  /state": "Current environment state",
                    "WS   /ws": "WebSocket for real-time stateful sessions",
                },
                "example_usage": {
                    "1_reset": "POST /reset  body: {}",
                    "2_experiment": 'POST /step  body: {"action": {"action": "experiment", "inputs": {"x": 3.0}}}',
                    "3_hypothesize": 'POST /step  body: {"action": {"action": "hypothesize", "expression": "2*x + 3"}}',
                    "4_predict": 'POST /step  body: {"action": {"action": "predict", "predictions": [9.0, -5.0]}}',
                },
                "github": "https://github.com/AbhinavDubey30/OpenMax",
            })
        return await call_next(request)


app = create_hypothesis_app(
    difficulty=1,
    experiment_budget=30,
    auto_curriculum=True,
    use_self_play=False,
    max_concurrent_envs=10,
)

app.add_middleware(RootPageMiddleware)
