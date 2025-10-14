"""Problem representation and benchmark loading."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import json


@dataclass
class Problem:
    """Represents a single math problem."""

    id: str
    question: str
    answer: str
    answer_type: Literal["integer", "float", "symbolic", "string"]
    tolerance: float | None = None
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    tags: list[str] | None = None

    def __post_init__(self) -> None:
        if self.tags is None:
            self.tags = []


def load_benchmark(path: str | Path) -> list[Problem]:
    """Load problems from benchmark JSON file.

    Args:
        path: Path to benchmark JSON file

    Returns:
        List of Problem instances
    """
    path = Path(path)
    with path.open() as f:
        data = json.load(f)

    problems = []
    for prob in data["problems"]:
        problems.append(Problem(**prob))

    return problems
