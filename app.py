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
