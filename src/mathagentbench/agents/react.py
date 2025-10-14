"""ReAct agent implementation (one-shot)."""

from typing import Any

from .base import MathAgent, AgentResult
from ..core.problem import Problem


class ReActAgent(MathAgent):
    """ReAct agent: Thought → Action → Observation loop."""

    category = "one-shot"

    def solve(self, problem: Problem, tools: dict[str, Any]) -> AgentResult:
        """Solve problem using ReAct loop.

        Process:
        1. Generate thought about next step
        2. Choose action (tool call or answer)
        3. Observe result
        4. Repeat until answer or max_steps
        """
        # TODO: Implement ReAct loop
        # - Build prompt with problem + tool descriptions
        # - Parse LLM response for thought/action
        # - Execute tool calls
        # - Track tokens/cost
        # - Return AgentResult
        raise NotImplementedError
