# RLM-Forge: A Recursive Language Model Training Environment for AI Coding Agents

## Project Overview

RLM-Forge is an OpenEnv environment designed to train small language models to utilize the Recursive Language Model (RLM) framework for solving complex coding tasks on large repositories. It is inspired by the research paper "Recursive Language Models" (Zhang, Kraska, & Khattab, MIT CSAIL, December 2025), which demonstrated that LLMs can process inputs orders of magnitude beyond their context windows by treating prompts as external environment variables and interacting with them through code execution in a REPL.

The core innovation of RLM-Forge is combining the RLM paradigm with depth-limited sub-agents for repository exploration, creating an environment where a root agent can orchestrate multiple sub-agents — each with their own scoped REPL and file-system tools — to understand and modify codebases far too large for any single model's context window.

The environment is self-supervised: it clones open-source repositories, programmatically removes a file or module that has associated test coverage, and tasks the agent with rebuilding that feature using only the surrounding codebase. The removed feature's test suite serves as an automatic, objective reward signal.

---

## Motivation & Research Background

### The Problem

Modern AI coding agents (Claude Code, Cursor, Codex CLI) struggle with very large repositories because a single agent must somehow fit enough context to understand the entire system. Context windows are finite, and even within those limits, model quality degrades as context grows longer — a phenomenon known as "context rot."

### The RLM Insight

The Recursive Language Models paper (arXiv:2512.24601) proposes a paradigm shift: instead of feeding long prompts directly into the neural network, treat the prompt as part of an external environment. The model interacts with the context through code — slicing, searching, chunking — and only pulls small pieces into its context window at a time. Crucially, the model can programmatically invoke sub-LM calls on constructed snippets, enabling recursive decomposition.

Key findings from the paper:
- RLMs handle inputs up to 10M+ tokens (two orders of magnitude beyond context windows)
- On information-dense tasks, RLMs outperform base models by 28-58% absolute
- The approach is model-agnostic and works with both closed and open-source models
- Costs remain comparable to base model calls at the median
- Emergent strategies appear without explicit training: regex filtering, intelligent chunking, answer verification, variable-based output stitching

### The Gap We Fill

The paper's "Future Work" section explicitly identifies the opportunity we are pursuing:

> "Explicitly training models to be used as RLMs (e.g. as root or sub-LMs) could provide additional performance improvements... We hypothesize that RLM trajectories can be viewed as a form of reasoning, which can be trained by bootstrapping existing frontier models."

We plan to allow a recursion depth of 1 (or 2?), so that a root agent can spawn sub-agents, and those sub-agents have access to their own REPL and file system tools, but the sub-agents cannot spawn their own sub-agents.

This will allow the model to be trained as both a root agent and a sub-agent, which is key to the success of the RLM-Forge environment.

### Why Coding Tasks?

Coding is the ideal domain for RLM training because:
1. **Natural structure**: Repositories have files, modules, imports, and tests — providing clear decomposition targets
2. **Objective evaluation**: Test suites provide automatic, binary reward signals
3. **Unlimited data**: Every well-tested open-source repository is a potential training example
4. **Real-world impact**: Improved coding agents have immediate practical value
5. **Complexity scaling**: Repositories naturally range from simple (100 LOC) to enormous (1M+ LOC), providing a natural curriculum

---

## Architecture Design

### Environment Type

RLM-Forge is an **OpenEnv environment** built on the OpenEnv 0.2.1 framework. It follows the standard OpenEnv pattern:

```
rlm_forge/
├── __init__.py
├── README.md
├── models.py              # Action, Observation, State (Pydantic models)
├── client.py              # HTTPEnvClient subclass
├── openenv.yaml           # Environment manifest
├── pyproject.toml
├── uv.lock
└── server/
    ├── __init__.py
    ├── app.py             # FastAPI server using create_app()
    ├── environment.py     # Core Environment implementation
    ├── repo_manager.py    # Repository cloning, feature extraction, test discovery
    ├── sandbox.py         # Sandboxed code execution (REPL)
    ├── sub_agent.py       # Sub-agent lifecycle management
    ├── reward.py          # Composite reward computation
    ├── feature_extractor.py  # Module/file removal and test mapping
    └── Dockerfile
```

### Core Concepts

#### The Root Agent

The root agent operates in an iterative REPL loop. It receives a task description and a high-level manifest of the repository (directory tree, file sizes, README excerpt). It does NOT see the actual source code in its context window. Instead, it writes Python code to:
- Explore the repository structure
- Read specific files
- Search for patterns (grep, regex, AST parsing)
- Spawn sub-agents to explore specific directories or modules
- Write implementation code
- Save files to rebuild the removed feature

#### Sub-Agents (Depth = 1)

Sub-agents are scoped explorers. When the root agent spawns a sub-agent, it specifies:
- A target scope (directory path or set of files)
- A mission (what to look for, what to report back)
- A budget (maximum iterations)

The sub-agent gets its own sandboxed REPL with:
- Read-only access to its scoped portion of the repository
- The ability to execute Python code (read files, parse ASTs, search, analyze)
- An `llm_query()` function for semantic understanding of code snippets
- NO ability to spawn further sub-agents (depth limit = 1)

The sub-agent runs its own iteration loop and returns a structured report to the root agent's REPL environment as a variable.

**Important distinction from the RLM paper**: In the paper, sub-calls are stateless LM calls — simple prompt-in, text-out. In RLM-Forge, sub-agents have their own REPL state, their own iteration loop, and their own tool access. They are mini-RLMs, not plain LM calls. This is the "depth-1 recursive RLM with tools" architecture. Sub-agents CANNOT spawn their own sub-agents.

#### The REPL Environment

Both root and sub-agents operate within sandboxed Python REPL environments. Key properties:
- **Persistent state**: Variables persist across iterations within an episode
- **Sandboxed execution**: Code runs in an isolated environment with controlled file system access
- **Truncated output**: stdout/stderr is truncated to prevent context overflow (configurable limit)
- **Iteration tracking**: The environment tracks iteration count against a configurable maximum
- **Built-in functions**:
  - `llm_query(prompt: str) -> str` — Invoke a sub-LM for semantic understanding
  - `spawn_agent(scope: str, mission: str, budget: int) -> dict` — Spawn a sub-agent (root only)
  - `read_file(path: str) -> str` — Read a file from the repository
  - `list_dir(path: str) -> list` — List directory contents
  - `search(pattern: str, path: str) -> list` — Grep/regex search
  - `write_file(path: str, content: str)` — Write implementation files (root only)
  - `run_tests(test_path: str) -> dict` — Run specific test files and get results
  - `FINAL()` — Signal episode completion

---

## Episode Lifecycle

### Phase 1: Environment Setup (on `reset()`)

1. **Repository selection**: The environment selects a repository from its configured dataset (a list of Git repository URLs or local paths)
2. **Clone and baseline**: Clone the repository. Run the full test suite to establish a baseline (all tests should pass)
3. **Feature extraction**: Select a target file or module for removal:
   - Identify files/modules that have dedicated test files with clear mappings (e.g., `src/auth.py` → `tests/test_auth.py`)
   - Prefer modules with moderate complexity (configurable LOC range)
   - Record which tests are associated with the target
   - Record the original content of the target (this is the ground truth, never shown to the agent)
4. **Feature removal**: Delete the target file(s) from the repository working copy
5. **Manifest generation**: Create a high-level manifest for the agent:
   - Directory tree structure
   - File sizes and languages
   - README excerpt (first N characters)
   - List of failing tests (names and file paths)
   - Task description: "The following module has been removed: `[path]`. N tests in `[test_path]` are now failing. Your task is to implement the missing module so that all tests pass."
6. **REPL initialization**: Set up the root agent's REPL environment with the repository loaded and built-in functions available
7. **Return initial observation**: The observation includes the manifest, the task description, the failing test list, and REPL environment metadata (available variables, available functions)

### Phase 2: Agent Interaction (the `step()` loop)

Each step, the agent submits an action containing Python code to execute. The environment:

1. **Extracts code blocks** from the agent's response
2. **Executes each code block** in the sandboxed REPL
3. **Captures output** (stdout, stderr, success/failure, any variables set)
4. **Checks for sub-agent spawns**: If the code calls `spawn_agent()`, the environment:
   - Creates a new scoped REPL for the sub-agent
   - Runs the sub-agent's iteration loop (the sub-agent is driven by an `llm_query()` call internally, or by a policy if training the sub-agent)
   - Returns the sub-agent's report as a variable in the root agent's REPL
5. **Checks for termination**: Episode ends if:
   - Agent calls `FINAL()` — voluntary completion
   - Maximum iterations reached — forced termination
   - Maximum wall-clock time exceeded — timeout
6. **Returns observation**: stdout/stderr (truncated), success boolean, iteration count, list of available variables, any sub-agent reports

### Phase 3: Evaluation (on episode completion)

When the episode ends (either through `FINAL()` or iteration limit):

1. **Collect implementation**: Gather all files the agent wrote via `write_file()`
2. **Run target tests**: Execute the test files associated with the removed feature
3. **Run regression tests**: Execute the full test suite to check for regressions
4. **Compute composite reward** (see Reward Function below)
5. **Return final observation** with done=True, reward, and detailed test results

---

## Reward Function

The reward is a weighted composite of three components. Weights are configurable via environment parameters, with these defaults:

### Test Pass Rate (Default: 55% of total reward)

```
test_pass_reward = (num_target_tests_passed / num_target_tests_total)
```

This is the primary signal. The agent is rewarded proportionally to how many of the removed feature's tests it gets passing. Partial credit is given — passing 7 out of 10 tests yields 0.70 on this component.

### Structural Validity (Default: 15% of total reward)

```
structural_reward = weighted_average(
    parse_success,        # Does the code parse without syntax errors? (weight: 0.3)
    import_success,       # Do imports resolve correctly? (weight: 0.3)
    no_regressions,       # Do previously-passing tests still pass? (weight: 0.4)
)
```

This penalizes agents that produce invalid code or hack solutions that break the rest of the codebase. The regression check is particularly important — it prevents the agent from modifying shared utilities in ways that pass target tests but break everything else.

### Efficiency Bonus (Default: 30% of total reward)

```
if iterations_used <= budget * 0.5:
    efficiency_reward = 1.0   # Full bonus for fast solutions
elif iterations_used <= budget * 0.75:
    efficiency_reward = 0.75  # Reduced bonus
elif iterations_used <= budget:
    efficiency_reward = 0.5   # Minimal bonus for using full budget
else:
    efficiency_reward = 0.0   # No bonus if forced termination

# Sub-agent efficiency modifier
sub_agent_penalty = max(0, 1.0 - (num_sub_agents_spawned / max_reasonable_sub_agents))
efficiency_reward *= (0.7 + 0.3 * sub_agent_penalty)
```

This encourages the agent to learn efficient exploration and decomposition strategies. It rewards agents that solve problems quickly and use sub-agents judiciously rather than spawning one for every directory.

### Total Reward Computation

```
total_reward = (
    test_weight * test_pass_reward +
    structural_weight * structural_reward +
    efficiency_weight * efficiency_reward
)
```

Where `test_weight`, `structural_weight`, and `efficiency_weight` are configurable and default to 0.55, 0.15, and 0.30 respectively.

---

## Data Models (Pydantic Schemas)

### Action

```python
class RLMForgeAction(Action):
    """Agent's action: Python code to execute in the REPL."""
    code: str = Field(..., description="Python code to execute in the REPL environment")
    action_type: str = Field(
        default="execute",
        description="Type of action: 'execute' for code, 'final' to submit solution"
    )
```

### Observation

```python
class RLMForgeObservation(Observation):
    """What the agent sees after each step."""
    # REPL execution results
    stdout: str = Field(default="", description="Truncated stdout from code execution")
    stderr: str = Field(default="", description="Truncated stderr from code execution")
    success: bool = Field(default=True, description="Whether code executed without errors")

    # Episode tracking
    iteration: int = Field(default=0, description="Current iteration number")
    max_iterations: int = Field(default=50, description="Maximum allowed iterations")

    # Repository context (provided on reset, may be refreshed)
    repo_manifest: Optional[dict] = Field(default=None, description="Repository structure manifest")
    task_description: Optional[str] = Field(default=None, description="The coding task to complete")
    failing_tests: Optional[list[str]] = Field(default=None, description="List of currently failing test names")

    # REPL state
    available_variables: list[str] = Field(default_factory=list, description="Variables currently in REPL scope")
    available_functions: list[str] = Field(default_factory=list, description="Built-in functions available")

    # Sub-agent reports (populated when sub-agents complete)
    sub_agent_reports: list[dict] = Field(default_factory=list, description="Reports from completed sub-agents")

    # Test results (populated on final evaluation)
    test_results: Optional[dict] = Field(default=None, description="Detailed test results on completion")
```

### State

```python
class RLMForgeState(State):
    """Internal environment state, not directly sent to agent."""
    episode_id: Optional[str] = None
    step_count: int = 0

    # Repository info
    repo_url: str = ""
    repo_local_path: str = ""
    removed_feature_path: str = ""
    removed_feature_content: dict[str, str] = {}  # filename -> original content
    target_test_files: list[str] = []
    baseline_test_count: int = 0

    # Agent progress
    files_written: dict[str, str] = {}  # filename -> content written by agent
    sub_agents_spawned: int = 0
    total_llm_queries: int = 0

    # Evaluation
    final_reward: Optional[float] = None
    test_pass_rate: Optional[float] = None
    has_regressions: Optional[bool] = None
```

---

## Feature Extraction Pipeline

The feature extraction pipeline is responsible for selecting what to remove from a repository and mapping it to tests. This is a critical component that must work reliably.

### Strategy: File and Module Level Extraction

The pipeline operates in two modes:

#### Single-File Mode
1. Scan the repository for Python/Rust/TS/Julia source files
2. For each source file, look for a corresponding test file using common patterns:
   - `src/foo.py` → `tests/test_foo.py`
   - `src/foo.py` → `tests/foo_test.py`
   - `src/foo/bar.py` → `tests/test_bar.py`
   - `lib/foo.rs` → `tests/foo.rs` or `tests/test_foo.rs`
   - `src/foo.ts` → `__tests__/foo.test.ts` or `tests/foo.spec.ts`
3. Verify the test file actually imports from / tests the source file
4. Run the test file in isolation to confirm it passes
5. Score candidates by:
   - Number of tests (prefer 5-30 tests; too few = trivial, too many = too complex)
   - Source file LOC (prefer 50-500 lines for hackathon scope)
   - Import complexity (prefer files that are imported by few other files, to minimize cascade)

#### Module Mode
1. Scan for directories that represent modules (contain `__init__.py` or are listed in package config)
2. Find test directories or files that correspond to the module
3. Same scoring criteria but at the module (directory) level
4. Prefer small, self-contained modules (2-8 files)

### Output of Feature Extraction

```python
@dataclass
class ExtractedFeature:
    """Represents a feature to be removed for training."""
    source_paths: list[str]          # Files to remove
    test_paths: list[str]            # Test files that exercise this feature
    original_content: dict[str, str] # Map of path -> original file content
    num_tests: int                   # Number of individual test cases
    estimated_complexity: str        # "easy", "medium", "hard"
    import_dependents: list[str]     # Files that import from the removed feature
    task_description: str            # Auto-generated task description for the agent
```

---

## Sub-Agent Mechanism

### Spawning a Sub-Agent

From the root agent's REPL:

```python
report = spawn_agent(
    scope="/src/database/",
    mission="Explore the database module. Report: 1) What ORM or database library is used, 2) What models/tables exist, 3) What patterns are used for queries, 4) The public API of this module",
    budget=10  # max iterations for the sub-agent
)
# `report` is now a dict variable in the root agent's REPL
print(report["summary"])
print(report["files_examined"])
```

### Sub-Agent Lifecycle

1. **Initialization**: A new sandboxed REPL is created with read-only access to the specified scope
2. **Mission prompt**: The sub-agent receives a system prompt with:
   - Its scoped directory listing
   - The mission description from the root agent
   - Available built-in functions (read_file, list_dir, search, llm_query)
   - Its iteration budget
3. **Iteration loop**: The sub-agent iterates (driven by `llm_query` internally):
   - Writes code to explore its scope
   - Executes code, observes results
   - Refines its understanding
   - Calls `FINAL(report)` when done or budget exhausted
4. **Report return**: The sub-agent's final report (a structured dict) is injected as a variable into the root agent's REPL

### Sub-Agent Constraints

- **Read-only file access**: Sub-agents can read files within their scope but cannot write files
- **No sub-agent spawning**: Sub-agents cannot spawn their own sub-agents (depth = 1)
- **Scoped access**: Sub-agents can only access files within their assigned directory scope
- **Budget limited**: Each sub-agent has a maximum iteration count
- **Concurrent limit**: The root agent can have at most N sub-agents per episode (configurable, default 10)

---

## Repository Dataset

### Requirements for Training Repositories

Each repository used as a training dataset must have:
1. **Strong test coverage** with test files that clearly map to source modules
2. **Modular architecture** where individual files/modules can be removed without collapsing the entire project
3. **Medium-large size** (10,000 - 150,000 LOC)
4. **Active maintenance** (commits within last 3 months)
5. **Permissive license** (MIT, Apache 2.0, BSD)
6. **80%+ in one of**: Python, Rust, TypeScript, or Julia

### Repository Configuration

```yaml
# repos.yaml - Dataset configuration
repositories:
  - url: "https://github.com/org/repo1"
    language: "python"
    difficulty: "medium"
    test_command: "pytest"
    source_dir: "src/"
    test_dir: "tests/"

  - url: "https://github.com/org/repo2"
    language: "rust"
    difficulty: "hard"
    test_command: "cargo test"
    source_dir: "src/"
    test_dir: "tests/"

  # ... more repositories

settings:
  max_file_loc: 500           # Max LOC for single-file extraction
  max_module_files: 8          # Max files for module extraction
  min_tests: 3                 # Minimum tests for a valid feature
  max_tests: 50                # Maximum tests (avoid overly complex features)
  preferred_test_range: [5, 30] # Sweet spot for test count
```

---

## Hackathon Problem Statement Alignment

RLM-Forge addresses multiple hackathon problem statements:

### Primary: Statement 2 — Long-Horizon Planning & Instruction Following

The environment requires deep, multi-step reasoning with delayed rewards. The agent must:
- Decompose the goal of rebuilding a feature into exploration sub-tasks
- Track state across an extended REPL trajectory (potentially dozens of iterations)
- Recover from wrong turns (exploring irrelevant code, writing buggy implementations)
- Plan sub-agent deployments strategically

### Secondary: Statement 3.1 — World Modeling (Professional Tasks)

The environment involves real interaction with tools and dynamic systems:
- File system exploration with real code
- Test execution with real pass/fail results
- Code execution in a sandboxed REPL
- Multi-step workflows: explore → understand → plan → implement → verify

### Partner Sub-Theme: Mercor (Statement 2)

"Make an environment with capped/uncapped rewards where frontier model rewards scale with token output." — RLM-Forge naturally fits this: longer, more sophisticated RLM trajectories that correctly process more of the codebase should earn higher rewards, as they'll pass more tests.

---

## Implementation Plan

### Phase 1: Core Environment Scaffold

1. Set up the OpenEnv project structure using `openenv init`
2. Define all Pydantic models (Action, Observation, State)
3. Implement the basic `Environment` class with `reset()` and `step()` stubs
4. Implement the sandboxed REPL (code execution with safety restrictions)
5. Implement the `app.py` FastAPI server and `client.py`
6. Verify the environment scaffold works with `openenv validate`

### Phase 2: Repository & Feature Pipeline 

1. Implement `repo_manager.py` — repository cloning, caching, test suite discovery
2. Implement `feature_extractor.py` — file/module selection, test mapping, feature removal
3. Build the manifest generator (directory tree, file metadata, task description)
4. Test the pipeline end-to-end on 2-3 repositories
5. Handle multi-language support (Python pytest, Rust cargo test, TS jest/vitest)

### Phase 3: Sub-Agent System 

1. Implement `sub_agent.py` — sub-agent REPL creation, scoping, lifecycle
2. Implement `spawn_agent()` as a built-in REPL function
3. Implement the sub-agent iteration loop with `llm_query()` integration
4. Implement sub-agent report format and injection into root REPL
5. Add sub-agent budget tracking and concurrent limits
6. Test sub-agent spawning and report aggregation

### Phase 4: Reward & Evaluation 

1. Implement `reward.py` — test execution, pass rate calculation, regression detection
2. Implement structural validity checks (parsing, import resolution)
3. Implement efficiency scoring
4. Implement the composite reward computation with configurable weights
5. Test reward computation on sample episodes

### Phase 5: Integration, Docker & HF Spaces 

1. Full integration testing — run complete episodes end-to-end
2. Build the Dockerfile with all dependencies (git, language runtimes, test frameworks)
3. Configure the Gradio web UI for the HF Space
4. Deploy to HF Spaces using `openenv push`
5. Verify the deployed environment works remotely

### Phase 6: Minimal Training Demo 

1. Create a Google Colab notebook
2. Set up Unsloth + a small model (Qwen2.5-1.5B or similar)
3. Connect to the deployed environment
4. Implement GRPO training loop with the environment's reward function
5. Run a few training steps to demonstrate the pipeline works
6. Save results and training curves

### Phase 7: Demo Video & Submission 

1. Record 1-minute YouTube demo video
2. Final testing and bug fixes
3. Submit to hackathon

---

## Key Technical Resources

### OpenEnv Framework
- OpenEnv GitHub: `https://github.com/meta-pytorch/OpenEnv`
- OpenEnv 0.2.1 stable release
- Environment builder guide: `docs/source/getting_started/environment-builder.md`
- Existing REPL environment: `src/envs/repl_env/` (study this closely as a reference)
- Existing coding environment: `src/envs/coding_env/` (another key reference)
- 2048 RL training tutorial: `docs/source/tutorials/rl-training-2048.md`

### RLM Paper
- arXiv: `https://arxiv.org/abs/2512.24601`
- Key sections: §2 (methods), §3.1 (emergent patterns), §5 (limitations/future work)
- System prompts: Appendix D (pages 24-28)
- Example trajectories: Appendix B (pages 13-20)

### Training Stack
- Unsloth: Memory-efficient fine-tuning with LoRA
- HuggingFace TRL: GRPO (Group Relative Policy Optimization)
- Google Colab: Free T4 GPU for the training demo

### Sandboxing
- Docker isolation (primary — OpenEnv already uses this)
- RestrictedPython or similar for additional code execution safety
- Filesystem scoping via chroot or bind mounts

---

## Configuration & Defaults

All key parameters should be configurable through the environment's reset kwargs or openenv.yaml:

```yaml
# openenv.yaml
name: rlm_forge
version: "0.1.0"
description: "RLM training environment for AI coding agents"

defaults:
  # Episode parameters
  max_iterations: 50
  max_wall_clock_seconds: 600
  max_sub_agents: 10
  sub_agent_budget: 15
  output_truncation_chars: 5000

  # Reward weights
  test_pass_weight: 0.55
  structural_validity_weight: 0.15
  efficiency_weight: 0.30

  # Feature extraction
  extraction_mode: "mixed"  # "file", "module", or "mixed"
  min_source_loc: 50
  max_source_loc: 500
  min_tests: 3
  max_tests: 50

  # Sub-agent configuration
  sub_agent_max_iterations: 15
  sub_agent_output_truncation: 3000
  sub_agent_read_only: true
  sub_agent_depth_limit: 1
```

---

## Success Criteria

### For the Hackathon

1. **Working environment** deployed on HF Spaces that accepts reset/step/state API calls
2. **Feature extraction** working on at least 2-3 demonstration repositories
3. **Sub-agent spawning** functional with scoped REPL access
4. **Reward computation** returning meaningful composite scores
5. **Minimal training notebook** in Colab showing GRPO training loop connecting to the environment
6. **1-minute demo video** explaining the concept and showing the environment in action

### For Long-Term Value

1. Environment generalizes across programming languages and repository structures
2. Reward signal is informative enough for models to learn meaningful exploration strategies
3. Sub-agent reports genuinely improve root agent performance vs. no sub-agents
4. Trained models show transfer to unseen repositories
5. Environment can serve as a benchmark for comparing coding agent architectures
