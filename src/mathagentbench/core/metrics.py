"""Metrics computation and aggregation."""

from typing import Any


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate metrics from problem results.

    Args:
        results: List of per-problem results

    Returns:
        Dictionary of aggregated metrics
    """
    # TODO: Implement metrics
    # - avg_cost
    # - avg_tool_calls
    # - by_tag breakdown
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    return {
        "pass@1": correct / total
        if total
        else 0.0,  # supposed to be unbiased estimator for EV of correct answer in 1 response
        "total_cost": sum(r["cost"] for r in results),  # TODO: average cost
        "avg_tokens": sum(r["tokens"] for r in results) / total,
        "by_difficulty": {
            d: sum(1 for r in results if r["difficulty"] == d and r["correct"])
            / sum(1 for r in results if r["difficulty"] == d)
            for d in ["easy", "medium", "hard"]
        },
    }


def compare_results(
    results1: dict[str, Any], results2: dict[str, Any]
) -> dict[str, Any]:
    """Compare two evaluation results for drift detection.

    Args:
        results1: First evaluation results
        results2: Second evaluation results

    Returns:
        Comparison report with deltas and warnings
    """
    # TODO: Implement comparison logic
    # - Compute deltas for all metrics
    # - Flag regressions (pass@1 drop >5%)
    # - Flag warnings (cost increase >20%)
    raise NotImplementedError
