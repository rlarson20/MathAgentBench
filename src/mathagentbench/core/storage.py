"""Results storage backends."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ResultsStorage(ABC):
    """Abstract base for results storage."""

    @abstractmethod
    def save(self, results: dict[str, Any]) -> None:
        """Save evaluation results."""
        pass

    @abstractmethod
    def load(self, run_id: str) -> dict[str, Any]:
        """Load evaluation results by run ID."""
        pass


class JSONLStorage(ResultsStorage):
    """Store results as JSONL files."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, results: dict[str, Any]) -> None:
        """Save results to JSONL file."""
        # TODO: Implement JSONL writing
        # - One line per problem result
        # - Separate summary JSON file
        raise NotImplementedError

    def load(self, run_id: str) -> dict[str, Any]:
        """Load results from JSONL file."""
        # TODO: Implement JSONL reading
        raise NotImplementedError


class SQLiteStorage(ResultsStorage):
    """Store results in SQLite database."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        # TODO: Initialize schema
        raise NotImplementedError

    def save(self, results: dict[str, Any]) -> None:
        """Save results to SQLite."""
        # TODO: Implement DB writes
        # Tables: runs, problems, traces
        raise NotImplementedError

    def load(self, run_id: str) -> dict[str, Any]:
        """Load results from SQLite."""
        # TODO: Implement DB reads
        raise NotImplementedError
