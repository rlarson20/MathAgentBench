"""Main evaluation orchestration."""

from dataclasses import dataclass
from typing import Any

from .problem import Problem
from ..agents.base import MathAgent, AgentResult


@dataclass
class EvalResults:
    """Results from running evaluation on a benchmark."""

    run_id: str
    model: str
    agent_type: str
    benchmark_name: str
    problems: list[dict[str, Any]]
    metrics: dict[str, Any]
    metadata: dict[str, Any]


class Evaluator:
    """Orchestrates agent evaluation on math problems."""

    def __init__(self, agent: MathAgent, tools: dict[str, Any]):
        self.agent = agent
        self.tools = tools

    def run(
        self, problems: list[Problem], benchmark_name: str = "unnamed"
    ) -> EvalResults:
        """Run agent on all problems.

        Args:
            problems: List of problems to evaluate
            benchmark_name: Name of benchmark for reporting

        Returns:
            Structured evaluation results
        """
        # TODO: Implement evaluation loop
        # - For each problem: agent.solve()
        # - Score answer with _score()
        # - Collect traces, costs, tokens
        # - Aggregate metrics
        raise NotImplementedError

    def _score(self, problem: Problem, result: AgentResult) -> bool:
        """Score agent result against expected answer.

        Args:
            problem: Problem being solved
            result: Agent's result

        Returns:
            True if answer is correct
        """
        # TODO: Implement answer matching logic
        # - symbolic: sympy simplification check
        expected = problem.answer.strip()
        actual = result.answer.strip()

        if problem.answer_type == "integer":
            return int(expected) == int(actual)
        elif problem.answer_type == "float":
            tol = problem.tolerance or 1e-6
            return abs(float(expected) - float(actual)) < tol
        # TODO: improve symbolic implementation
        # elif problem.answer_type == "symbolic":
        #     # Use sympy simplify for equivalence
        #     from sympy import sympify, simplify
        #
        #     return simplify(sympify(expected) - sympify(actual)) == 0
        else:  # string
            return expected.lower() == actual.lower()
