"""Base agent protocol."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import re
import json

from ..core.problem import Problem


@dataclass
class AgentResult:
    """Result from agent solving a problem."""

    answer: str
    trace: list[dict[str, Any]]  # [{step, thought, action, observation}, ...]
    cost: float
    tokens: int
    metadata: dict[str, Any]


class MathAgent(ABC):
    """Abstract base class for math-solving agents."""

    category: str  # "one-shot" or "self-correct"

    def __init__(self, llm_client: Any, max_steps: int = 10):
        self.llm = llm_client
        self.max_steps = max_steps

    def _parse_action(self, content: str):
        action_match = re.search(r"Action: (\w+)", content)
        input_match = re.search(r"Action Input: ({.*})", content, re.DOTALL)

        if not action_match:
            return "python", {"code": content}

        action = action_match.group(1)
        action_input = json.loads(input_match.group(1)) if input_match else {}

        return action, action_input

    @abstractmethod
    def solve(self, problem: Problem, tools: dict[str, Any]) -> AgentResult:
        """Solve a math problem using available tools.

        Args:
            problem: Problem to solve
            tools: Dictionary of available tools

        Returns:
            AgentResult with answer and trace
        """
        pass
