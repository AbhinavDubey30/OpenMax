# HYPOTHESIS ENGINE

### A Procedurally-Generated Scientific Discovery RL Environment for Training LLM Reasoning

> *"The scientist is not a person who gives the right answers; they're the one who asks the right questions."*

**OpenEnv Hackathon | Track 4: Self-Improvement + Track 5: Wild Card**

---

## The Problem

Current LLMs are trained to retrieve and regurgitate knowledge -- but they're terrible at **generating** new knowledge. The most powerful reasoning framework ever invented is the **scientific method**: observe, hypothesize, experiment, revise. Yet no RL environment exists to train LLMs on this meta-skill.

**Hypothesis Engine** changes that.

It's an RL environment where an LLM agent must act as a **scientist** discovering the hidden rules of procedurally-generated black-box worlds -- purely through designing experiments, observing results, forming hypotheses, and making testable predictions.

Think of it as: **"Can an LLM rediscover Newton's Laws from scratch, with only 30 experiments?"**

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# OR install as a package
pip install -e .

# Run the demo -- watch the AI scientist discover hidden patterns
python run_demo.py --quick

# Full benchmark -- all 10 difficulty levels
python run_demo.py --benchmark

# Interactive mode -- YOU are the scientist
python run_demo.py --interactive

# Use GPT-4 as the scientist (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
python run_demo.py --llm

# Run the environment validation benchmark
python examples/benchmark.py
```

---

## Hackathon Track Alignment

### Track 4: Self-Improvement (Primary)

The hackathon asks for environments where agents **learn to generate new challenges, escalate difficulty, and improve through adaptive curricula**.

Hypothesis Engine delivers exactly this:

| Requirement | How We Deliver |
|---|---|
| **Auto-generated challenges** | Procedural world generator creates infinite unique scientific mysteries with random coefficients -- no memorization possible |
| **Escalating difficulty** | 10 difficulty levels from `y = 2x + 3` to compositional nested functions with hidden variables, noise, and temporal state |
| **Adaptive curriculum** | Built-in `CurriculumController` automatically advances difficulty when the agent consistently scores above threshold |
| **Self-play improvement** | The environment's reward signal directly measures scientific reasoning quality, enabling self-improving training loops |
| **Expected outcome: environment for improving self-play of an LLM** | Agents interact via text observations and JSON actions -- direct drop-in for LLM RL training (RLHF, PPO, GRPO) |

### Track 5: Wild Card -- Impress Us

No one has built a **scientific discovery RL environment** before. This is novel, creative, and deeply useful:

- Trains the single most valuable AI capability: **scientific reasoning**
- Every major AI lab (DeepMind, OpenAI, Anthropic) is pursuing scientific AI
- The environment is immediately useful for LLM training research

### Track 2: Long-Horizon Planning (Secondary)

Episodes span 30-50 steps of interconnected experimentation, hypothesis formation, and prediction. The agent must maintain state, plan strategically, and recover from wrong hypotheses -- exactly the deep multi-step reasoning the track demands.

---

## How It Works

### Episode Flow

```
+--------------------------------------------------------------+
|  1. WORLD GENERATION                                         |
|     A new black-box system is procedurally created.          |
|     The agent receives only a vague description.             |
|                         |                                    |
|  2. EXPLORATION PHASE                                        |
|     Agent designs & runs experiments (budget-limited).       |
|     -> Chooses inputs strategically                          |
|     -> Observes outputs                                      |
|     -> Builds mental model                                   |
|                         |                                    |
|  3. HYPOTHESIS PHASE                                         |
|     Agent states a formal mathematical hypothesis.           |
|     -> Gets qualitative feedback (not the answer)            |
|     -> Can refine and resubmit                               |
|                         |                                    |
|  4. PREDICTION PHASE                                         |
|     Agent predicts outputs for 20 unseen test cases.         |
|     -> Predictions verified against ground truth             |
|                         |                                    |
|  5. SCORING                                                  |
|     Multi-component reward computed.                         |
|     -> Episode ends, reward returned for RL training         |
+--------------------------------------------------------------+
```

### The Standard API

Two interfaces are provided:

**1. Core API (Gym-like)**

```python
from hypothesis_engine import HypothesisEngine

env = HypothesisEngine(difficulty=3, experiment_budget=30, seed=42)
obs = env.reset()

# Run an experiment
obs, reward, done, info = env.step({
    "action": "experiment",
    "inputs": {"x1": 3.0, "x2": -1.0}
})

# Submit a hypothesis
obs, reward, done, info = env.step({
    "action": "hypothesize",
    "expression": "2*x1 + x2**2 - 3"
})

# Submit predictions (ends episode)
obs, reward, done, info = env.step({
    "action": "predict",
    "predictions": [9.0, -5.0, 7.0, ...]
})

# Final reward in info["final_reward"]["total_reward"] (0-100)
```

**2. Gymnasium-Compatible Wrapper (for RL frameworks)**

```python
from hypothesis_engine.gym_wrapper import make_env

env = make_env(difficulty=3, experiment_budget=30)

# Standard gymnasium interface
obs_text, info = env.reset()

# Action is a JSON string (natural for LLMs)
obs_text, reward, terminated, truncated, info = env.step(
    '{"action": "experiment", "inputs": {"x": 3.0}}'
)
```

**The Gymnasium wrapper uses `Text` observation and action spaces** -- ideal for LLM training where inputs and outputs are natural language / JSON strings.

---

## RL Training Integration

The environment is designed to plug directly into LLM RL training pipelines:

### Basic Training Loop

```python
from hypothesis_engine import HypothesisEngine

env = HypothesisEngine(difficulty=1, auto_curriculum=True)

for episode in range(1000):
    obs = env.reset()
    done = False
    trajectory = []

    while not done:
        # LLM generates action from observation
        action = your_llm_policy(obs)
        obs, reward, done, info = env.step(action)
        trajectory.append((obs, action, reward))

    # Use trajectory for policy gradient update
    update_policy(trajectory)
```

### TRL / GRPO Integration

```python
from hypothesis_engine.gym_wrapper import make_env

env = make_env(difficulty=1, auto_curriculum=True)

# Text-based observations and actions work natively with LLMs
obs_text, info = env.reset()

# LLM generates JSON action as text
action_text = llm.generate(obs_text)
obs_text, reward, terminated, truncated, info = env.step(action_text)
```

### Stable-Baselines3 / RLlib

```python
from hypothesis_engine.gym_wrapper import HypothesisEngineGymEnv

# Compatible with any framework that supports gymnasium.Env
env = HypothesisEngineGymEnv(difficulty=3, experiment_budget=30)
```

See `examples/training_loop.py` for complete runnable examples.

---

## 10 Difficulty Levels -- Automatic Curriculum

| Level | Name | System Type | Example | Challenge |
|:---:|---|---|---|---|
| **1** | The Straight Line | Linear (1 var) | `y = 2x - 3` | Basic pattern recognition |
| **2** | The Parabola's Secret | Polynomial (1 var) | `y = x^2 - 4x + 7` | Detecting curvature |
| **3** | The Two-Body Problem | Multi-var linear | `y = 5x1 + x2 + 5` | Isolating variable effects |
| **4** | The Fork in the Road | Conditional | `y = 4x+1 if x>0 else -2x+3` | Detecting regime changes |
| **5** | The Entangled Variables | Interaction effects | `y = -3*x1*x2 + 2*x1 + 4` | Discovering variable interactions |
| **6** | The Oscillator | Trigonometric | `y = 3*sin(2x) - 1` | Recognizing periodic patterns |
| **7** | Through the Fog | Stochastic (noisy) | `y = x^2 + 3x + N(0, s^2)` | Separating signal from noise |
| **8** | The Invisible Hand | Hidden variables | `y = 2x + 3h + 1` (h cycles) | Discovering unobservable factors |
| **9** | The Time Machine | Dynamic / stateful | `y_t = 2*x_t + 0.5*y_{t-1}` | Temporal dependencies |
| **10** | The Nested Puzzle | Compositional | `y = -1*(2*x1+1)^2 + x2` | Decomposing complex systems |

Each level is **procedurally generated** with random coefficients -- no memorization possible. The auto-curriculum advances difficulty when the agent consistently succeeds.

---

## Reward Signal Design

The reward is a weighted combination of 5 fully-automated, verifiable components:

| Component | Weight | What It Measures | Why It Matters for RL |
|---|:---:|---|---|
| **Prediction Accuracy** | 40% | MSE & exact match on held-out test cases | Core performance metric -- did the agent actually discover the rule? |
| **Hypothesis Quality** | 25% | R^2 score of hypothesis vs. ground truth | Tests whether the agent can articulate its understanding formally |
| **Experiment Efficiency** | 15% | Fewer experiments used = higher reward | Incentivizes strategic experimentation over brute-force |
| **Information Gain** | 10% | Did experiments explore diverse regions? | Rewards scientific exploration strategy |
| **Progressive Improvement** | 10% | Did hypotheses get better over time? | Rewards the iterative refinement loop |

### Why This Reward Design Works for RL

- **Dense + Sparse signals**: Small rewards per experiment + large final reward
- **Fully automated**: No human judgment needed -- predictions are right or wrong
- **Verifiable**: Ground truth is known, so rewards are exact
- **Multi-objective**: Balances accuracy, efficiency, and scientific process
- **Composable**: Weights are configurable for different training objectives
- **Scalable**: Compute reward for millions of episodes with zero human involvement

---

## Architecture

```
hypothesis_engine/
|-- __init__.py          # Package exports
|-- env.py               # Main RL environment (gym-like interface)
|-- gym_wrapper.py       # Gymnasium-compatible wrapper (Text spaces)
|-- worlds.py            # Procedural world generator (10 difficulty levels)
|-- verifier.py          # Safe expression evaluator & hypothesis scorer
|-- rewards.py           # Multi-component reward calculator
|-- curriculum.py        # Auto-curriculum difficulty controller
|-- display.py           # Rich terminal visualization
+-- agents/
    |-- base.py              # Abstract agent interface
    |-- heuristic_agent.py   # Rule-based scientist (no API needed)
    +-- llm_agent.py         # LLM-powered scientist (OpenAI API)

examples/
|-- training_loop.py     # RL training integration examples
+-- benchmark.py         # Environment validation benchmark

run_demo.py              # Interactive demo runner
pyproject.toml           # Package configuration (pip installable)
requirements.txt         # Dependencies
```

### Key Design Decisions

1. **Text-Based Observation/Action Spaces**: The Gymnasium wrapper uses `Text` spaces, making the environment natively compatible with LLM training (no embedding conversion needed).

2. **Safe Expression Evaluation**: Uses Python's AST module for hypothesis parsing -- no `eval()`, no code injection. Supports arithmetic, trig functions, conditionals, and comparisons safely.

3. **Procedural Generation**: Each world is created from random seeds with controlled parameters. Infinite unique environments for training -- no memorization possible.

4. **Stateful Worlds**: Levels 8-9 maintain internal state across experiments, requiring the agent to reason about temporal patterns and hidden variables.

5. **Gym-Compatible Interface**: Standard `reset()` -> `step(action)` -> `(obs, reward, done, info)` loop, compatible with any RL training framework.

6. **Installable Package**: Can be installed via `pip install -e .` with proper `pyproject.toml`.

---

## Agents

### Heuristic Scientist (No API Required)

A rule-based agent that follows the scientific method:
1. **Probe**: Strategic baseline experiments
2. **Analyze**: Pattern detection (linearity, quadratics, interactions)
3. **Hypothesize**: Polynomial fitting and coefficient extraction
4. **Test**: Targeted validation experiments
5. **Refine**: Update hypothesis based on evidence
6. **Predict**: Apply hypothesis to test cases

**Benchmark Results** (Heuristic Agent):
| Levels 1-3 | Levels 4-5 | Levels 6-7 | Levels 8-10 |
|:---:|:---:|:---:|:---:|
| 87+ avg | 62-88 | 27-67 | 28-61 |

The heuristic agent nails the easy/medium levels but struggles with trigonometric, noisy, and compositional patterns -- **proving the environment presents a genuine challenge that benefits from RL training**.

### LLM Scientist (GPT-4 / Any OpenAI-Compatible Model)

Uses an LLM with carefully designed prompts to perform scientific reasoning:
- Receives observations as structured text
- Outputs actions as JSON
- Maintains conversation history for multi-step reasoning

```bash
export OPENAI_API_KEY=sk-...
python run_demo.py --llm --model gpt-4o
```

---

## Environment Validation

Run the included benchmark to verify all environment properties:

```bash
python examples/benchmark.py
```

This validates:

| Test | What It Checks |
|---|---|
| **Determinism** | Same seed produces identical worlds and results |
| **Reward Signal** | Dense rewards during exploration + sparse final reward |
| **Difficulty Levels** | All 10 levels generate valid, solvable worlds |
| **Curriculum** | Auto-curriculum advances difficulty after wins |
| **Observation Richness** | Observations contain all info an LLM needs to reason |
| **Gymnasium Wrapper** | Standard gymnasium interface works correctly |

---

## Why This Environment Wins

### 1. Trains the Most Valuable Capability
Scientific reasoning -- forming hypotheses, designing experiments, updating beliefs -- is THE frontier capability for AI. Every major AI lab is pursuing this. This environment directly trains it.

### 2. Perfect Reward Signal
Unlike environments with fuzzy rewards, predictions are **right or wrong**. Hypotheses are **verifiable against ground truth**. This is RL gold -- clean, automated, scalable.

### 3. Infinite Scalability
Procedural generation means infinite unique worlds. No dataset to overfit. No memorization possible. Automatic curriculum scales difficulty with agent capability.

### 4. Directly Addresses Hackathon Tracks

| Track | How Hypothesis Engine Addresses It |
|---|---|
| **Track 2: Long-Horizon** | Episodes span 30-50 steps of experimentation, hypothesis, and prediction |
| **Track 4: Self-Improvement** | Auto-curriculum escalates difficulty; procedural generation creates infinite training scenarios; the environment evolves with the agent |
| **Track 5: Wild Card** | Entirely novel framing -- no one has built a scientific discovery RL environment |

### 5. Killer Demo

Watch the AI discover `y = 2x - 2` from scratch in 14 experiments:

```
Experiment #1  |  x=0.0   -> y = -2.0     (Baseline)
Experiment #2  |  x=1.0   -> y = 0.0      (Testing unit change)
Experiment #3  |  x=-1.0  -> y = -4.0     (Testing negative)
Experiment #4  |  x=2.0   -> y = 2.0      (Confirming pattern)
...
Hypothesis: y = 2*x - 2  [PASS] Excellent match!
SCORE: 87.0/100 -- PASSED!
```

### 6. Production-Ready

- Pip-installable package with `pyproject.toml`
- Gymnasium-compatible wrapper with `Text` spaces
- Safe expression evaluation (no code injection)
- Comprehensive error handling
- Works on Windows, macOS, Linux
- Zero external API dependencies for core functionality

### 7. Massively Applicable

| Domain | Application |
|---|---|
| Drug Discovery | Train models to design experiments for molecular properties |
| Materials Science | Discover material relationships through simulated testing |
| Debugging | Systematically isolate root causes through targeted probing |
| Economic Modeling | Discover market dynamics through policy experiments |
| Scientific Research | Accelerate hypothesis-driven research pipelines |

---

## Advanced Usage

### Custom Reward Weights

```python
from hypothesis_engine import HypothesisEngine, RewardWeights

# Emphasize prediction accuracy, ignore efficiency
weights = RewardWeights(
    prediction_accuracy=0.60,
    hypothesis_quality=0.20,
    experiment_efficiency=0.05,
    information_gain=0.10,
    progressive_improvement=0.05,
)

env = HypothesisEngine(difficulty=5, experiment_budget=50)
env.reward_calc.weights = weights
```

### Programmatic World Inspection

```python
from hypothesis_engine import WorldGenerator

# Generate and inspect a world
world = WorldGenerator.generate(difficulty=4, seed=42)
print(f"Name: {world.name}")
print(f"Ground Truth: {world.ground_truth_expr}")
print(f"Variables: {world.variables}")
print(f"Test Cases: {len(world.test_cases)}")

# Run experiments directly
result = world.run_experiment({"x": 3.0})
print(f"f(3.0) = {result['output']}")
```

---

## Demo Modes

| Command | Description |
|---|---|
| `python run_demo.py` | Interactive menu -- choose your mode |
| `python run_demo.py --quick` | Quick demo: levels 1-3 with heuristic agent |
| `python run_demo.py --auto` | Full auto: all 10 levels with heuristic agent |
| `python run_demo.py --interactive` | You play as the scientist |
| `python run_demo.py --llm` | GPT-4 plays as the scientist |
| `python run_demo.py --benchmark` | Full benchmark with detailed scoring |
| `python run_demo.py --level 5` | Start from a specific level |

---

## Future Extensions

- **Multi-Agent Discovery**: Multiple agents collaborate to discover complex systems faster (Track 1 compatibility)
- **Real-World Data Integration**: Use real scientific datasets instead of procedural generation
- **Symbolic Regression Reward**: Bonus for discovering the exact symbolic form
- **Experiment Cost Model**: Different experiments have different costs (like real labs)
- **Active Learning Integration**: Combine with active learning for optimal experiment design
- **Adversarial World Generation**: A second agent generates worlds designed to challenge the first (self-play Track 4)
- **Long-Context Episodes**: 300+ step episodes with tool use for Track 2 compatibility

---

## Requirements

- Python 3.8+
- `numpy` -- Numerical computation
- `rich` -- Terminal visualization
- `gymnasium` -- Standard RL environment interface
- `openai` (optional) -- LLM agent support

```bash
pip install -r requirements.txt

# Or install as a package
pip install -e .

# Or install with all extras
pip install -e ".[all]"
```

---

<p align="center">
  <b>Built for the OpenEnv Hackathon</b><br>
  <i>Track 4: Self-Improvement + Track 5: Wild Card</i><br><br>
  <b>Hypothesis Engine</b> -- Teaching AI to think like a scientist.<br>
  Because the models that can <i>discover</i> knowledge will always surpass<br>
  the ones that can only <i>retrieve</i> it.
</p>
