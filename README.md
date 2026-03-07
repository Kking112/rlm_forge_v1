# RLM-Forge

**Recursive Language Model training environment for AI coding agents.**

RLM-Forge is an [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment that trains language models to solve coding tasks on real Python repositories using Recursive Language Model (RLM) patterns.

## How It Works

1. **Clone** a real Python repo (e.g., python-slugify, humanize)
2. **Extract** a source file and replace it with a broken stub (correct signatures, wrong implementations)
3. **Agent** explores the repo via a sandboxed multi-step REPL with built-in tools
4. **Reward** = test pass rate (55%) + structural validity (15%) + efficiency (30%)
5. **Train** with GRPO to improve the agent's coding ability over time

### The REPL Tools

The agent has access to these functions in the sandbox:

| Function | Description |
|----------|-------------|
| `read_file(path)` | Read a file from the repo |
| `list_dir(path='.')` | List directory contents |
| `search(pattern, path='.')` | Grep for a pattern |
| `write_file(path, content)` | Write/create a file |
| `run_tests(test_path=None)` | Run pytest |
| `spawn_agent(scope, mission)` | Explore a directory scope |
| `FINAL()` | Signal implementation is complete |

## Project Structure

```
rlm_forge/
├── __init__.py              # Package exports
├── models.py                # Pydantic models (Action, Observation, State)
├── client.py                # EnvClient for remote connections
└── server/
    ├── app.py               # FastAPI server (create_app)
    ├── environment.py       # Core Environment (reset/step)
    ├── sandbox.py           # Sandboxed Python REPL
    ├── repo_manager.py      # Repo cloning & dependency management
    ├── feature_extractor.py # Source file extraction & stub generation
    └── reward.py            # Composite reward computation
```

## Quick Start

### Install

```bash
uv sync
```

### Run the Server

```bash
uv run uvicorn rlm_forge.server.app:app --host 0.0.0.0 --port 8000
```

### Use the Environment Directly

```python
from rlm_forge.server.environment import RLMForgeEnvironment
from rlm_forge.models import RLMForgeAction

env = RLMForgeEnvironment()
obs = env.reset(seed=1)
print(obs.task_description)

# Agent takes actions
obs = env.step(RLMForgeAction(code="print(read_file('test.py'))"))
obs = env.step(RLMForgeAction(code="write_file('slugify/slugify.py', '...')"))
obs = env.step(RLMForgeAction(code="FINAL()"))
print(f"Reward: {obs.reward}")
```

### Connect via Client

```python
from rlm_forge.client import RLMForgeClient
from rlm_forge.models import RLMForgeAction

client = RLMForgeClient(base_url="http://localhost:8000")
client.connect()

result = client.reset(seed=1)
result = client.step(RLMForgeAction(code="print(list_dir())"))
result = client.step(RLMForgeAction(code="FINAL()"))
print(f"Reward: {result.reward}")
```

## Training

See `rlm_forge_training.ipynb` for the full GRPO training notebook. Designed for Google Colab with an H100 GPU.

Key training approach:
- **Multi-step trajectory concatenation**: Full episode (all code actions) treated as one GRPO "completion"
- **Group Relative Policy Optimization**: Multiple completions per task, advantages computed relative to group mean
- **LoRA fine-tuning**: 4-bit quantized Qwen2.5-Coder-32B with LoRA adapter

## Reward Breakdown

| Component | Weight | Description |
|-----------|--------|-------------|
| Test Pass Rate | 55% | Fraction of tests passing |
| Structural Validity | 15% | AST parse check + import check |
| Efficiency | 30% | Tiered by iteration budget used |

## Curated Repos

| Repo | Source File | Tests | Difficulty |
|------|-----------|-------|------------|
| python-slugify | `slugify/slugify.py` | 82 | Easy |
| humanize (number) | `src/humanize/number.py` | 219 | Medium |
| humanize (time) | `src/humanize/time.py` | varies | Medium |

## Docker

```bash
docker build -t rlm-forge .
docker run -p 8000:8000 rlm-forge
```

The Dockerfile pre-clones curated repos to avoid network I/O on each `reset()`.

## Deploy to HF Spaces

```bash
openenv push -r your-username/rlm-forge
```
