# Architecture

## Overview

MathAgent Bench evaluates LLM agents on math problems by:

1. Loading standardized benchmarks (GSM8K, MATH)
2. Running agents with tool access (Python REPL, symbolic math)
3. Scoring answers and logging detailed traces
4. Aggregating metrics for comparison

## Components

### Core Evaluation Engine

**Problem**: Dataclass representing a math problem with metadata (difficulty, tags, answer type).

**Evaluator**: Orchestrates the evaluation loop:

- Loads benchmark problems
- Runs agent on each problem
- Scores answers based on type (integer, float, symbolic)
- Collects traces, costs, tokens
- Aggregates metrics

**Metrics**: Computes pass@1, average cost, tool usage statistics, breakdowns by difficulty/tags.

**Storage**: Dual backend (JSONL for simplicity, SQLite for queryability).

### Agents

Both agents follow the `MathAgent` protocol:

**ReActAgent (one-shot)**:

```
loop until answer or max_steps:
  - Thought: reason about next step
  - Action: call tool or provide answer
  - Observation: see tool result
```

**ReflexionAgent (self-correct)**:

```
attempt = solve with ReAct loop
for retry in range(max_retries):
  critique = LLM evaluates attempt quality
  if critique.confident:
    return attempt
  attempt = solve with critique context
```

### Tools

**DockerREPL**:

- Spins up `ghcr.io/astral-sh/uv:python3.13-alpine` container per execution
- Pre-installs sympy, numpy, scipy at runtime
- Enforces memory limit (512MB), timeout (30s), no network
- Returns stdout/stderr/success

**Symbolic Math**:

- Wrapper over sympy operations: solve, simplify, diff, integrate
- Handles parsing and error recovery
- Returns string representations for LLM consumption

### LLM Client

**OpenRouterClient**:

- Wraps OpenRouter API for multi-model access
- Tracks token usage and cost per request
- Handles retries and error recovery

### Interfaces

**CLI** (`mathagent`):

- `run`: Execute evaluation on benchmark
- `compare`: Detect drift between two runs
- `report`: Print detailed breakdown
- `export`: Convert JSONL → SQLite

**Gradio UI**:

- Tab 1: Run evaluation (upload benchmark, select model/agent)
- Tab 2: View traces (expandable per-problem logs)
- Tab 3: Compare models (side-by-side metrics + charts)

## Data Flow

```
Benchmark JSON
    ↓
load_benchmark() → list[Problem]
    ↓
Evaluator.run() ← Agent + Tools
    ↓ (for each problem)
agent.solve(problem, tools)
    ↓
Agent: thought → action → observation loop
    ↓ (tool calls)
DockerREPL.execute(code) or symbolic_math_handler(...)
    ↓
AgentResult(answer, trace, cost, tokens)
    ↓
Evaluator._score(problem, result) → bool
    ↓
collect all results
    ↓
compute_metrics(results) → aggregated stats
    ↓
Storage.save() → JSONL (+ optional SQLite)
```

## Extensibility

**Adding new agents**:

1. Subclass `MathAgent`
2. Implement `solve()` method
3. Register in `agents/__init__.py`

**Adding new tools**:

1. Create handler function
2. Add to `TOOLS` registry with description
3. Agents automatically see tool in context

**Adding new benchmarks**:

1. Format as JSON per schema
2. Place in `benchmarks/`
3. Run with `mathagent run`

**Adding new storage backends**:

1. Subclass `ResultsStorage`
2. Implement `save()` and `load()`
3. Wire into CLI options
