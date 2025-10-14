# MathAgent Bench

Evaluation framework for LLM agents solving math problems with tool use.

## Features

- **Standardized Benchmarks**: GSM8K and MATH dataset subsets
- **Multiple Agent Types**: ReAct (one-shot) and Reflexion (self-correct)
- **Sandboxed Execution**: Docker-isolated Python REPL with sympy/numpy/scipy
- **Model Comparison**: Test GPT-4, Claude, Llama via OpenRouter
- **Drift Detection**: Compare runs to catch regressions
- **Dual Interface**: CLI for automation + Gradio UI for exploration

## Quick Start

### Installation

```bash
# Clone repo
git clone https://github.com/rlarson20/MathAgentBench.git
cd MathAgentBench

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .

# Copy environment template
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

### Run Evaluation

```bash
# CLI
mathagent run benchmarks/gsm8k_subset.json --model openai/gpt-4o-mini --agent react

# Gradio UI
python -m mathagent_bench.ui.app
```

### Compare Results

```bash
mathagent compare results/run1.json results/run2.json
```

## Architecture

```
Problem → Agent (ReAct/Reflexion) → Tools (Python REPL, Sympy)
                ↓
         Score Answer → Log Trace → Aggregate Metrics
```

See [docs/architecture.md](docs/architecture.md) for details.

## Project Structure

```
mathagent-bench/
├── src/mathagent_bench/
│   ├── core/          # Evaluation engine
│   ├── agents/        # ReAct and Reflexion agents
│   ├── tools/         # Docker REPL, symbolic math
│   ├── llm/           # OpenRouter client
│   └── ui/            # Gradio interface
├── benchmarks/        # GSM8K and MATH subsets
├── tests/             # Pytest suite
└── results/           # Evaluation outputs
```

## Development

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Format code
ruff check src/ tests/

# Type check
mypy src/
```

## License

MIT
