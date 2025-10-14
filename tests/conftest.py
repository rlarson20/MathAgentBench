"""Pytest configuration and fixtures."""

import pytest
# from pathlib import Path


@pytest.fixture
def sample_problem():
    """Sample problem for testing."""
    from mathagentbench.core.problem import Problem

    return Problem(
        id="test_001",
        question="What is 2 + 2?",
        answer="4",
        answer_type="integer",
        difficulty="easy",
        tags=["arithmetic"],
    )


@pytest.fixture
def sample_benchmark(tmp_path):
    """Create temporary benchmark file."""
    import json

    benchmark = {
        "name": "Test Benchmark",
        "version": "1.0",
        "problems": [
            {
                "id": "test_001",
                "question": "What is 2 + 2?",
                "answer": "4",
                "answer_type": "integer",
                "difficulty": "easy",
                "tags": ["arithmetic"],
            }
        ],
    }
    path = tmp_path / "test_benchmark.json"
    path.write_text(json.dumps(benchmark))
    return path
