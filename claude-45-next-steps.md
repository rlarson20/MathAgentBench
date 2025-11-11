## Phase 2: Agent Implementation (Week 2)

### 2.2 Reflexion Agent

**Goal**: Add self-correction capability

**Tasks**:

1. **Implement critique mechanism**:

   ```python
   def _critique(self, problem: Problem, answer: str, trace: list) -> dict[str, Any]:
       critique_prompt = f"""
   Problem: {problem.question}
   Proposed Answer: {answer}

   Review this solution. Rate confidence (0-1) and suggest improvements.

   Response format:
   Confidence: [0.0-1.0]
   Analysis: [your analysis]
   Suggestions: [improvements if confidence < 0.8]
   """

       response = self.llm.complete([{
           "role": "user",
           "content": critique_prompt
       }])

       # Parse confidence score
       import re
       conf_match = re.search(r"Confidence: (0\.\d+|1\.0)", response["content"])
       confidence = float(conf_match.group(1)) if conf_match else 0.5

       return {
           "confidence": confidence,
           "analysis": response["content"],
           "should_retry": confidence < 0.7
       }
   ```

2. **Implement retry loop**:

   ```python
   def solve(self, problem: Problem, tools: dict[str, Any]) -> AgentResult:
       attempts = []

       for retry in range(self.max_retries + 1):
           # Solve with ReAct
           result = self._react_solve(problem, tools, previous_attempts=attempts)
           attempts.append(result)

           # Critique
           critique = self._critique(problem, result.answer, result.trace)

           if not critique["should_retry"] or retry == self.max_retries:
               result.metadata["critique"] = critique
               result.metadata["attempts"] = len(attempts)
               return result
   ```

**Milestone**: Agent retries and improves answer after low-confidence critique.

---

## Phase 3: Evaluation Engine (Week 3)

### 3.1 Scoring Logic

**Goal**: Accurately judge correctness

**Tasks**:

1. **Implement answer matching**:

   ```python
   # src/mathagentbench/core/evaluator.py
   def _score(self, problem: Problem, result: AgentResult) -> bool:
       expected = problem.answer.strip()
       actual = result.answer.strip()

       if problem.answer_type == "integer":
           return int(expected) == int(actual)

       elif problem.answer_type == "float":
           tolerance = problem.tolerance or 1e-3
           return abs(float(expected) - float(actual)) < tolerance

       elif problem.answer_type == "symbolic":
           import sympy as sp
           try:
               diff = sp.simplify(sp.sympify(expected) - sp.sympify(actual))
               return diff == 0
           except:
               return False

       else:  # string
           return expected.lower() == actual.lower()
   ```

2. **Write comprehensive scoring tests**:

   ```python
   # tests/test_scoring.py
   def test_score_integer_exact():
       # Test implementation
       pass

   def test_score_symbolic_equivalence():
       # Test "x^2 - 1" == "(x-1)(x+1)"
       pass
   ```

**Milestone**: 100% test coverage on scoring logic.

---

### 3.2 Metrics & Aggregation

**Goal**: Compute meaningful statistics

**Tasks**:

1. **Implement `compute_metrics()`**:

   ```python
   # src/mathagentbench/core/metrics.py
   def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
       total = len(results)
       correct = sum(1 for r in results if r["correct"])

       metrics = {
           "pass@1": correct / total if total > 0 else 0,
           "total_problems": total,
           "correct": correct,
           "avg_tokens": sum(r["tokens"] for r in results) / total,
           "avg_cost": sum(r["cost"] for r in results) / total,
           "avg_steps": sum(len(r["trace"]) for r in results) / total,
       }

       # By difficulty
       for diff in ["easy", "medium", "hard"]:
           subset = [r for r in results if r.get("difficulty") == diff]
           if subset:
               metrics[f"pass@1_{diff}"] = sum(1 for r in subset if r["correct"]) / len(subset)

       return metrics
   ```

2. **Implement comparison**:

   ```python
   def compare_results(results1: dict[str, Any], results2: dict[str, Any]) -> dict[str, Any]:
       m1, m2 = results1["metrics"], results2["metrics"]

       delta_pass = m2["pass@1"] - m1["pass@1"]
       delta_cost = (m2["avg_cost"] - m1["avg_cost"]) / m1["avg_cost"] * 100

       return {
           "pass@1_delta": delta_pass,
           "cost_delta_pct": delta_cost,
           "regression": delta_pass < -0.05,  # 5% drop
           "warnings": {
               "cost_increase": delta_cost > 20,
               "performance_drop": delta_pass < -0.02
           }
       }
   ```

**Milestone**: Generate full metrics report from evaluation run.

---

### 3.3 Storage Backends

**Goal**: Persist results

**Tasks**:

1. **Implement JSONL storage**:

   ```python
   # src/mathagentbench/core/storage.py
   import json
   from datetime import datetime

   class JSONLStorage(ResultsStorage):
       def save(self, results: dict[str, Any]) -> None:
           run_id = results["run_id"]
           timestamp = datetime.now().isoformat()

           # Save per-problem results
           problems_file = self.output_dir / f"{run_id}_{timestamp}_problems.jsonl"
           with problems_file.open("w") as f:
               for prob in results["problems"]:
                   f.write(json.dumps(prob) + "\n")

           # Save summary
           summary_file = self.output_dir / f"{run_id}_{timestamp}_summary.json"
           summary = {k: v for k, v in results.items() if k != "problems"}
           summary_file.write_text(json.dumps(summary, indent=2))
   ```

2. **Implement SQLite storage** (optional but impressive):

   ```python
   class SQLiteStorage(ResultsStorage):
       def __init__(self, db_path: str | Path):
           import sqlite3
           self.db_path = Path(db_path)
           self.conn = sqlite3.connect(db_path)
           self._init_schema()

       def _init_schema(self):
           self.conn.executescript("""
               CREATE TABLE IF NOT EXISTS runs (
                   run_id TEXT PRIMARY KEY,
                   model TEXT,
                   agent_type TEXT,
                   benchmark_name TEXT,
                   timestamp TEXT,
                   metrics JSON
               );

               CREATE TABLE IF NOT EXISTS problems (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   run_id TEXT,
                   problem_id TEXT,
                   correct BOOLEAN,
                   answer TEXT,
                   tokens INTEGER,
                   cost REAL,
                   trace JSON,
                   FOREIGN KEY (run_id) REFERENCES runs(run_id)
               );
           """)
   ```

**Milestone**: Save and load results from both JSONL and SQLite.

---

## Phase 4: CLI & UI (Week 4)

### 4.1 CLI Commands

**Goal**: Make it usable from terminal

**Tasks**:

1. **Implement `run` command**:

   ```python
   # src/mathagentbench/cli.py
   @cli.command()
   def run(benchmark: str, model: str, agent: str, output: str, db: bool):
       from pathlib import Path
       from src.mathagentbench.core.problem import load_benchmark
       from src.mathagentbench.core.evaluator import Evaluator
       from src.mathagentbench.llm.client import OpenRouterClient
       from src.mathagentbench.agents import ReActAgent, ReflexionAgent
       from src.mathagentbench.tools import TOOLS

       # Load
       problems = load_benchmark(benchmark)
       click.echo(f"Loaded {len(problems)} problems")

       # Initialize
       llm = OpenRouterClient(model=model)
       agent_cls = ReActAgent if agent == "react" else ReflexionAgent
       agent_instance = agent_cls(llm)

       # Run
       evaluator = Evaluator(agent_instance, TOOLS)
       results = evaluator.run(problems, benchmark_name=Path(benchmark).stem)

       # Save
       from src.mathagentbench.core.storage import JSONLStorage
       storage = JSONLStorage(output)
       storage.save(results.__dict__)

       click.echo(f"Results saved to {output}")
       click.echo(f"Pass@1: {results.metrics['pass@1']:.2%}")
   ```

2. **Implement other commands** (compare, report, export)

3. **Add rich output**:

   ```python
   # Use rich library for nice tables
   from rich.console import Console
   from rich.table import Table

   console = Console()
   table = Table(title="Evaluation Results")
   table.add_column("Metric", style="cyan")
   table.add_column("Value", style="magenta")
   # ... add rows ...
   console.print(table)
   ```

**Milestone**: Run full evaluation from command line.

---

### 4.2 Gradio UI

**Goal**: Interactive exploration

**Tasks**:

1. **Implement evaluation tab**:

   ```python
   # src/mathagentbench/ui/app.py
   def run_eval(benchmark_file, model: str, agent_type: str):
       if benchmark_file is None:
           return {"error": "No file uploaded"}

       # Save uploaded file
       temp_path = Path("/tmp") / benchmark_file.name
       temp_path.write_bytes(benchmark_file.read())

       # Run evaluation (same as CLI)
       # ... implementation ...

       return {
           "status": "complete",
           "metrics": results.metrics,
           "run_id": results.run_id
       }
   ```

2. **Add trace viewer**:

   ```python
   with gr.Tab("View Traces"):
       results_dropdown = gr.Dropdown(label="Select Run")
       trace_display = gr.JSON(label="Trace")

       def load_traces(run_id):
           # Load from storage
           # Format for display
           pass
   ```

3. **Add comparison charts**:

   ```python
   import plotly.graph_objects as go

   def create_comparison_chart(runs):
       fig = go.Figure()
       fig.add_trace(go.Bar(
           x=[r["name"] for r in runs],
           y=[r["metrics"]["pass@1"] for r in runs]
       ))
       return fig
   ```

**Milestone**: Launch UI and run evaluation through browser.

---

## Phase 5: Polish & Documentation (Week 5)

### 5.1 Testing

**Goal**: >80% coverage, all critical paths tested

**Tasks**:

1. **Write all remaining tests** (see stubs)
2. **Add integration tests**:

   ```python
   # tests/test_integration.py
   def test_end_to_end_evaluation(tmp_path):
       """Full pipeline test with mock LLM."""
       # Create benchmark
       # Mock OpenRouter responses
       # Run evaluation
       # Verify results
   ```

3. **Run coverage**:
   ```bash
   pytest --cov=src --cov-report=html
   open htmlcov/index.html
   ```

**Milestone**: `pytest --cov=src` shows >80% coverage.

---

### 5.2 Documentation

**Goal**: Professional README and docs

**Tasks**:

1. **Expand README.md**:

   - Add architecture diagram (use Mermaid or ASCII)
   - Add example outputs
   - Add troubleshooting section

2. **Create `docs/` folder**:

   ```
   docs/
   ├── architecture.md      # System design
   ├── benchmarks.md        # Dataset details
   ├── agents.md            # Agent algorithms
   └── extending.md         # How to add tools/agents
   ```

3. **Add docstrings everywhere**:

   ```python
   def solve(self, problem: Problem, tools: dict[str, Any]) -> AgentResult:
       """Solve a math problem using ReAct reasoning.

       Args:
           problem: Problem instance with question and metadata
           tools: Dictionary mapping tool names to handlers

       Returns:
           AgentResult containing answer, trace, and metrics

       Example:
           >>> agent = ReActAgent(llm_client)
           >>> result = agent.solve(problem, TOOLS)
           >>> print(result.answer)
           '42'
       """
   ```

**Milestone**: Documentation is clear enough for someone to use without asking questions.

---

### 5.3 Production Ready

**Goal**: Deploy-ready code

**Tasks**:

1. **Add logging**:

   ```python
   import logging

   logger = logging.getLogger(__name__)

   def solve(self, problem, tools):
       logger.info(f"Solving problem {problem.id}")
       # ...
       logger.debug(f"Generated {len(trace)} steps")
   ```

2. **Add error handling**:

   ```python
   try:
       result = self.llm.complete(messages)
   except httpx.HTTPError as e:
       logger.error(f"LLM API error: {e}")
       return AgentResult(
           answer="ERROR",
           trace=[],
           cost=0,
           tokens=0,
           metadata={"error": str(e)}
       )
   ```

3. **Add CI/CD** (GitHub Actions):

   ```yaml
   # .github/workflows/test.yml
   name: Tests
   on: [push, pull_request]
   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
         - uses: actions/setup-python@v4
         - run: pip install -e ".[dev]"
         - run: pytest --cov=src
   ```

4. **Setup pre-commit hooks** (already configured):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

**Milestone**: Push to GitHub, see green checkmarks on CI.

---

## Phase 6: Portfolio Presentation

### 6.1 Demo Materials

**Tasks**:

1. **Create demo video** (2-3 minutes):

   - Show CLI usage
   - Show Gradio UI
   - Show comparison of models
   - Highlight key differentiators

2. **Add screenshots to README**:

   ```markdown
   ## Demo

   ![CLI Output](docs/images/cli.png)
   ![Gradio UI](docs/images/ui.png)
   ![Comparison](docs/images/compare.png)
   ```

3. **Create example notebook**:
   ```jupyter
   # examples/demo.ipynb
   # Shows programmatic API usage
   from mathagentbench import Evaluator, ReActAgent
   # ...
   ```

### 6.2 Results & Analysis

**Tasks**:

1. **Run real benchmarks** (with actual API):

   - GSM8K subset (50 problems)
   - Compare GPT-4, Claude, Llama
   - Generate comparison report

2. **Write blog post** (or docs/analysis.md):

   - "Evaluating LLM Math Agents: ReAct vs Reflexion"
   - Include charts, insights, failure analysis

3. **Add results to repo**:
   ```
   results/
   ├── gpt4_react_gsm8k.json
   ├── claude_reflexion_gsm8k.json
   └── comparison_report.md
   ```

### 6.3 GitHub Profile

**Tasks**:

1. **Polish GitHub repo**:

   - Add topics: `llm`, `evaluation`, `agents`, `math`, `benchmarking`
   - Add nice banner image
   - Add shields.io badges (coverage, license, version)

2. **Write standout README intro**:

   ```markdown
   # 🧮 MathAgent Bench

   > **Rigorous evaluation framework for LLM agents solving math problems**

   Built to answer: _Do self-correcting agents outperform one-shot reasoners?_

   🎯 **Key Features:**

   - Docker-isolated code execution (security first)
   - Multi-model comparison via OpenRouter
   - ReAct + Reflexion agent implementations
   - Comprehensive trace analysis
   - Drift detection for production monitoring
   ```

3. **Add to portfolio**:
   - LinkedIn post with demo video
   - Add to resume projects section
   - Prepare to discuss architecture decisions in interviews

---

## Quick Win Checklist (If Time Constrained)

If you need to prioritize for fastest portfolio impact:

**Week 1**: Core (Phases 1-2)

- ✅ Problem loading
- ✅ LLM client
- ✅ Docker REPL
- ✅ Basic ReAct agent

**Week 2**: Make it work (Phase 3)

- ✅ Evaluation loop
- ✅ Scoring logic
- ✅ JSONL storage
- ✅ Basic metrics

**Week 3**: Make it usable (Phase 4)

- ✅ CLI `run` command
- ✅ Gradio UI basics
- ✅ Run 1 real benchmark

**Week 4**: Make it impressive (Phases 5-6)

- ✅ Tests (>60% coverage minimum)
- ✅ Documentation
- ✅ Demo video
- ✅ Polish README

---

## Key Differentiators for Interviews

When presenting this project, emphasize:

1. **Production-grade decisions**:

   - "I used Docker for isolation because code execution from LLMs is a security risk"
   - "I implemented both JSONL and SQLite to show I understand different storage trade-offs"

2. **Testing rigor**:

   - "I wrote tests first using TDD to ensure correctness"
   - "I mocked external APIs to keep tests fast and reliable"

3. **Evaluation expertise**:

   - "I implemented multiple answer matching strategies because math has unique challenges"
   - "I added drift detection because production LLM apps need regression monitoring"

4. **Agent understanding**:
   - "I chose ReAct and Reflexion to compare one-shot vs self-correcting approaches"
   - "The trace structure enables detailed failure analysis"

---

## Common Pitfalls to Avoid

1. **Don't over-engineer early** - Get something working first
2. **Don't skip tests** - They're how you prove it works
3. **Don't hardcode** - Use environment variables, config files
4. **Don't neglect error handling** - LLM APIs fail, Docker can hang
5. **Don't forget documentation** - Your future self will thank you

---

## Success Metrics

You'll know you're done when:

- ✅ `pytest` passes with >80% coverage
- ✅ CLI runs full evaluation without errors
- ✅ Gradio UI launches and works
- ✅ README explains project clearly
- ✅ Can demo in <5 minutes
- ✅ Results are reproducible
- ✅ Code passes pre-commit hooks

---

This is ambitious but achievable. Start with Phase 1, validate each piece works, then move forward. Good luck! 🚀
