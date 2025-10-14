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
    # - pass@1
    # - avg_cost
    # - avg_tokens
    # - avg_tool_calls
    # - by_difficulty breakdown
    # - by_tag breakdown
    raise NotImplementedError


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
