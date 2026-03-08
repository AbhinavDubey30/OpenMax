"""
HuggingFace Spaces entry point for Hypothesis Engine.

Deploy with:
    openenv-core serve --app app:app

Or run locally:
    uvicorn app:app --host 0.0.0.0 --port 7860
"""

from hypothesis_engine.openenv_wrapper import create_hypothesis_app

app = create_hypothesis_app(
    difficulty=1,
    experiment_budget=30,
    auto_curriculum=True,
    use_self_play=False,
    max_concurrent_envs=10,
)


# ── Landing page so visitors see something useful at the root URL ──
@app.get("/")
def root():
    return {
        "name": "Hypothesis Engine",
        "version": "2.0.0",
        "description": (
            "A procedurally-generated RL environment for training LLMs on "
            "scientific reasoning through causal discovery, physics simulation, "
            "state machine reverse-engineering, and adversarial self-play."
        ),
        "status": "running",
        "endpoints": {
            "GET  /":         "This page",
            "GET  /health":   "Health check",
            "GET  /metadata": "Environment metadata",
            "GET  /schema":   "Action/Observation JSON schemas",
            "POST /reset":    "Start a new episode",
            "POST /step":     "Take an action (experiment, hypothesize, predict)",
            "GET  /state":    "Current environment state",
            "WS   /ws":       "WebSocket for real-time stateful sessions",
        },
        "example_usage": {
            "1_reset": "POST /reset  body: {}",
            "2_experiment": 'POST /step  body: {"action": {"action": "experiment", "inputs": {"x": 3.0}}}',
            "3_hypothesize": 'POST /step  body: {"action": {"action": "hypothesize", "expression": "2*x + 3"}}',
            "4_predict": 'POST /step  body: {"action": {"action": "predict", "predictions": [9.0, -5.0, ...]}}',
        },
        "github": "https://github.com/AbhinavDubey30/OpenMax",
        "world_categories": [
            "Function Discovery (Levels 1-3): Linear, polynomial, multi-variable",
            "Causal Reasoning (Levels 4-6): Observe vs intervene, confounders",
            "Physics Simulation (Levels 7-8): Springs, projectiles",
            "State Machine Discovery (Level 9): Hidden finite automata",
            "Stochastic Systems (Level 10): Signal in noise",
        ],
    }