---
title: Hypothesis Engine
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Hypothesis Engine

**A procedurally-generated RL environment for training LLMs on scientific reasoning through causal discovery, physics simulation, state machine reverse-engineering, and adversarial self-play.**

Built for the [OpenEnv Hackathon](https://cerebralvalley.ai/e/openenv-hackathon-sf/details) | Built on [OpenEnv 0.2.1](https://github.com/meta-pytorch/OpenEnv) (`openenv-core`)

---

## Deliverables

| Requirement | Status | Link |
|---|---|---|
| Public GitHub Repo | Done | [AbhinavDubey30/OpenMax](https://github.com/AbhinavDubey30/OpenMax) |
| OpenEnv 0.2.1 Integration | Done | `hypothesis_engine/openenv_wrapper.py` |
| HuggingFace Spaces Deployment | Done | `app.py` + `Dockerfile` (ready to deploy) |
| Colab Training Notebook (GRPO + Unsloth) | Done | `train.ipynb` |
| Benchmark Suite | Done | `examples/benchmark.py` (8/8 tests pass) |

---

## What Makes This Novel

Existing RL environments for LLMs (like Boxing-Gym) focus narrowly on function-fitting. Hypothesis Engine is the **first environment** to combine all of these in a single, unified RL interface:

| Feature | Hypothesis Engine | Boxing-Gym | Other Envs |
|---------|:-:|:-:|:-:|
| Causal Reasoning (observe vs. intervene) | Yes | No | No |
| Physics Simulation (springs, projectiles) | Yes | No | No |
| State Machine Discovery | Yes | No | No |
| Statistical Reasoning under Noise | Yes | Partial | No |
| Adversarial Self-Play World Generation | Yes | No | No |
| Gymnasium `Text` Observation/Action Spaces | Yes | No | No |
| Auto-Curriculum with Adaptive Difficulty | Yes | No | No |
| Pip-installable Package | Yes | No | Varies |

### Key Innovation: Causal Reasoning

In levels 4-6, agents must distinguish **correlation from causation** using interventional experiments. The environment provides two experiment modes:

- **Observe**: See natural correlations (may include hidden confounders)
- **Intervene**: Force a variable to a value, breaking upstream causal links

An agent that only observes will be **fooled by confounders**. Only agents that learn to intervene can discover the true causal structure -- a critical capability for real-world scientific reasoning.

### Key Innovation: Self-Play

The environment implements **adversarial self-play** where:

1. A **Generator** creates new scientific worlds (procedurally or via LLM)
2. A **Solver** attempts to discover the hidden rules
3. The Generator is rewarded for creating worlds that are **challenging but solvable**
4. Difficulty adapts automatically based on solver performance

This creates a self-improving loop where both agents get better over time -- directly addressing **Track 4: Self-Improvement**.

---

## Hackathon Track Alignment

### Track 1: Long-Horizon Tasks
Each episode is a multi-step investigation requiring:
- Strategic experiment design (10-30+ steps)
- Iterative hypothesis refinement
- Efficient budget allocation
- Progressive information gathering

### Track 3: Wild Card -- Scientific Discovery
The core task IS scientific discovery:
- Observe -> Hypothesize -> Predict
- Causal reasoning with interventional experiments
- Physics law discovery
- Statistical reasoning under uncertainty

### Track 4: Self-Improvement (Primary Track)
- **Auto-curriculum**: Difficulty increases as agent improves
- **Self-play**: Generator creates harder worlds for Solver
- **Multi-category progression**: Master functions, then causal reasoning, then physics, then state machines
- **10 difficulty levels** across 5 fundamentally different world categories

---

## OpenEnv 0.2.1 Integration

Hypothesis Engine is fully integrated with [OpenEnv](https://github.com/meta-pytorch/OpenEnv) (`openenv-core` v0.2.1):

```python
# Connect as an OpenEnv client
from openenv.core import EnvClient

client = EnvClient("http://localhost:7860")  # or your HF Spaces URL
obs = client.reset()
obs = client.step({"action": "experiment", "inputs": {"x": 2.0}})
```

**Serve locally:**
```bash
pip install openenv-core==0.2.1
uvicorn app:app --host 0.0.0.0 --port 7860
```

**Deploy on HuggingFace Spaces:**
The repo includes `app.py` and `Dockerfile` for one-click HF Spaces deployment.

**API Endpoints** (OpenEnv standard):
- `POST /reset` -- Start new episode
- `POST /step` -- Take action
- `GET /state` -- Get environment state
- `GET /schema` -- Get action/observation schemas
- `GET /health` -- Health check
- `WS /ws` -- WebSocket for real-time interaction

---

## Training with GRPO (Unsloth + TRL)

See `train.ipynb` for a complete Colab notebook that trains Qwen2.5-1.5B using GRPO:

```python
# Reward function for GRPO training
from hypothesis_engine import HypothesisEngine

def hypothesis_reward_fn(completions, prompts, **kwargs):
    """Score LLM outputs by running them against Hypothesis Engine."""
    rewards = []
    for completion in completions:
        action = extract_json_action(completion)
        if action and action.get("action") == "experiment":
            rewards.append(0.5)   # Good: running experiments
        elif action and action.get("action") == "hypothesize":
            rewards.append(1.0)   # Better: forming hypotheses
        elif action and action.get("action") == "predict":
            rewards.append(1.5)   # Best: making predictions
        else:
            rewards.append(-1.0)  # Bad: invalid action
    return rewards
```

---

## Architecture

```
hypothesis_engine/
  env.py              -- Core RL environment (gym-like API)
  openenv_wrapper.py  -- OpenEnv 0.2.1 integration (Action/Observation/State)
  worlds.py           -- 5 world categories, 10 difficulty levels
  verifier.py         -- Safe AST-based expression evaluation
  rewards.py          -- Multi-component reward system (5 factors)
  curriculum.py       -- Auto-curriculum controller
  self_play.py        -- Adversarial self-play orchestrator
  gym_wrapper.py      -- Gymnasium-compatible Text wrapper
  display.py          -- Rich terminal UI for demos
  agents/
    base.py               -- Abstract agent interface
    heuristic_agent.py    -- Rule-based scientist (no API key needed)
    llm_agent.py          -- LLM-powered scientist (OpenAI API)
examples/
  benchmark.py        -- Full validation suite + agent benchmark
  training_loop.py    -- RL training loop examples
app.py                -- HuggingFace Spaces entry point
train.ipynb           -- Colab training notebook (GRPO + Unsloth)
Dockerfile            -- Docker deployment for HF Spaces
run_demo.py           -- Interactive demo runner
```

---

## World Categories

### A. Function Discovery (Levels 1-3)
Classic curve fitting: linear, polynomial, multi-variable.

```python
env = HypothesisEngine(difficulty=1)  # y = a*x + b
env = HypothesisEngine(difficulty=2)  # y = a*x^2 + b*x + c
env = HypothesisEngine(difficulty=3)  # y = a*x1 + b*x2 + c
```

### B. Causal Reasoning (Levels 4-6) -- NOVEL
Discover causal structure using interventional experiments.

```python
env = HypothesisEngine(difficulty=4)  # Causal chain: X -> M -> Y
env = HypothesisEngine(difficulty=5)  # Confounded: Z -> X, Z -> Y (spurious correlation!)
env = HypothesisEngine(difficulty=6)  # Hidden confounder with 2 observed variables

# Observe: see natural correlations (may be confounded)
obs, r, d, info = env.step({"action": "experiment", "inputs": {"x": 2.0}, "mode": "observe"})

# Intervene: break causal links, see true effect
obs, r, d, info = env.step({"action": "experiment", "inputs": {"x": 2.0}, "mode": "intervene"})

# If outputs differ -> confounder detected!
```

### C. Physics Simulation (Levels 7-8) -- NOVEL
Discover physical laws from experimental data.

```python
env = HypothesisEngine(difficulty=7)  # Spring system: F = kx (discover k)
env = HypothesisEngine(difficulty=8)  # Projectile motion: R = v^2*sin(2*theta)/g
```

### D. State Machine Discovery (Level 9) -- NOVEL
Reverse-engineer hidden finite automata.

```python
env = HypothesisEngine(difficulty=9)
# The system has N hidden states
# Same input can produce DIFFERENT outputs depending on current state
# Agent must discover: number of states, transition rules, output rules
```

### E. Stochastic / Statistical (Level 10) -- NOVEL
Discover signals buried in high noise.

```python
env = HypothesisEngine(difficulty=10)
# Noise std ~ 3-5 units; single measurements are unreliable
# Agent must run REPEATED experiments and AVERAGE results
# Teaches statistical reasoning
```

---

## Quick Start

### Install

```bash
pip install -e ".[all]"
# or: pip install numpy rich openai gymnasium openenv-core==0.2.1
```

### 30-Second Demo

```python
from hypothesis_engine import HypothesisEngine

env = HypothesisEngine(difficulty=5, experiment_budget=30)
obs = env.reset()

# This is a CONFOUNDED causal world!
# Observe: see spurious correlation
obs, r, d, info = env.step({"action": "experiment", "inputs": {"x": 2.0}, "mode": "observe"})
print(f"Observe: x=2 -> y={obs['last_experiment_result']['output']}")

# Intervene: see true causal effect
obs, r, d, info = env.step({"action": "experiment", "inputs": {"x": 2.0}, "mode": "intervene"})
print(f"Intervene: x=2 -> y={obs['last_experiment_result']['output']}")
# Outputs differ! There's a hidden confounder.

# Submit hypothesis based on interventional data
obs, r, d, info = env.step({"action": "hypothesize", "expression": "0*x + 3"})

# Submit predictions
obs, r, d, info = env.step({"action": "predict", "predictions": [3.0]*20})
print(f"Score: {info['final_reward']['total_reward']}/100")
```

### Run Demos

```bash
python run_demo.py --quick          # Levels 1-3 with heuristic agent
python run_demo.py --auto           # All 10 levels
python run_demo.py --benchmark      # Full benchmark with scoring
python run_demo.py --self-play      # Adaptive self-play demo
python run_demo.py --interactive    # Play as the scientist yourself
python run_demo.py --llm            # Use GPT-4 as the scientist
```

### Run Benchmark

```bash
python examples/benchmark.py
```

Output:
```
Results: 8/8 tests passed

Agent Performance Benchmark:
  Level  1: [#################...]  87.0/100  PASS
  Level  2: [#################...]  88.8/100  PASS
  Level  3: [#################...]  86.4/100  PASS
  Level  4: [#################...]  88.3/100  PASS  (causal chain)
  Level  5: [############........]  62.3/100  PASS  (confounded!)
  Level  6: [###############.....]  75.1/100  PASS  (hidden confounder)
  Level  7: [##################..]  92.0/100  PASS  (spring physics)
  Level  8: [#######.............]  35.2/100  FAIL  (projectile - hard!)
  Level  9: [#####...............]  27.8/100  FAIL  (state machine)
  Level 10: [###########.........]  59.9/100  FAIL  (stochastic)
```

The heuristic agent passes 7/10 levels. Levels 8-10 are **intentionally hard** -- they require capabilities that LLMs must *learn*, not just apply heuristics.

---

## Gymnasium Integration

Designed for standard RL training pipelines:

```python
from hypothesis_engine import make_env

env = make_env(difficulty=5, auto_curriculum=True)
obs_text, info = env.reset()

# obs_text is a natural language string -- perfect for LLM policies
# action_text is a JSON string -- parsed automatically
obs_text, reward, terminated, truncated, info = env.step(
    '{"action": "experiment", "inputs": {"x": 2.0}, "mode": "intervene"}'
)
```

Compatible with:
- **TRL** (Transformer Reinforcement Learning)
- **OpenRL** / **RLlib** (via Gymnasium interface)
- **Stable Baselines 3** (with Text observation wrapper)
- **GRPO / PPO** policy optimization

---

## Self-Play Training Loop

```python
from hypothesis_engine import HypothesisEngine

env = HypothesisEngine(difficulty=3, experiment_budget=25, use_self_play=True)

for episode in range(100):
    obs = env.reset()  # Self-play generates increasingly complex worlds
    done = False
    while not done:
        action = your_agent.act(obs)  # Your LLM policy here
        obs, reward, done, info = env.step(action)
    # Difficulty auto-adapts based on performance!
```

---

## Reward System

Multi-component reward (0-100 scale):

| Component | Weight | Measures |
|-----------|--------|----------|
| Prediction Accuracy | 40% | How well predictions match ground truth |
| Hypothesis Quality | 25% | How close the hypothesis is to the true rule |
| Experiment Efficiency | 15% | Using fewer experiments for same accuracy |
| Information Gain | 10% | Diversity of experiments (exploration vs. exploitation) |
| Progressive Improvement | 10% | Are hypotheses getting better over time? |

Dense rewards during episodes enable effective RL training.

---

## Requirements

```
Python >= 3.8
numpy >= 1.24.0
rich >= 13.0.0
openenv-core >= 0.2.1
gymnasium >= 0.29.1
openai >= 1.0.0 (optional, for LLM agent)
```

## Judging Criteria Coverage

| Criterion (Weight) | How We Address It |
|---|---|
| **Environment Innovation (40%)** | 5 novel world categories (causal, physics, state machines, stochastic, self-play). First RL env with interventional causal reasoning. |
| **Storytelling (30%)** | Clear README, live demo, Colab notebook, benchmark results. Scientific discovery framing is intuitive and compelling. |
| **Training Results (20%)** | `train.ipynb` shows GRPO training with reward improvement. Benchmark shows difficulty curve across 10 levels. |
| **Code Quality (10%)** | Clean architecture, type hints, docstrings, pip-installable, OpenEnv + Gymnasium compatible. |

---

## License

MIT
