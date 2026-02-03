"""
MLIP (Machine Learning Interatomic Potentials) evaluation module.

Metrics:
- energy_mae: Mean absolute error on energy predictions (meV/atom)
- force_mae: Mean absolute error on force predictions (meV/A)
- force_cosine: Cosine similarity of predicted vs true force vectors
- stress_mae: Mean absolute error on stress predictions (GPa)
"""

import os
import numpy as np
from typing import Dict, List, Optional

# Available metrics for this task
METRICS = ["energy_mae", "force_mae", "force_cosine", "stress_mae"]

# Default test data location (relative to this file)
DEFAULT_TEST_DATA = "data/test.json"


def load_test_data(data_path: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
    """
    Load test data for MLIP evaluation.

    Args:
        data_path: Path to test data file. If None, uses default location.
        limit: Maximum number of test cases to load.

    Returns:
        List of dicts with 'input' (structure data) and ground truth fields.
    """
    if data_path is None:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(this_dir, DEFAULT_TEST_DATA)

    # TODO: Implement data loading based on actual dataset format
    # Expected format: JSON or extxyz with structures, energies, forces, stresses
    raise NotImplementedError(
        f"MLIP test data loading not yet implemented. "
        f"Please add test data to {data_path} and implement this function."
    )


def evaluate(
    predictions: List,
    test_cases: List[Dict],
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Evaluate MLIP predictions.

    Args:
        predictions: List of prediction results from model.run().
        test_cases: List of test cases with ground truth.
        metrics: List of metrics to compute. If None, computes all.

    Returns:
        Dict mapping metric names to scores.
    """
    if metrics is None:
        metrics = METRICS

    for m in metrics:
        if m not in METRICS:
            raise ValueError(f"Unknown metric: {m}. Available: {METRICS}")

    results = {}
    n = len(test_cases)

    # TODO: Implement actual metric computation
    # For now, provide the structure for each metric

    if "energy_mae" in metrics:
        # MAE between predicted and true energies (meV/atom)
        results["energy_mae"] = 0.0

    if "force_mae" in metrics:
        # MAE between predicted and true forces (meV/A)
        results["force_mae"] = 0.0

    if "force_cosine" in metrics:
        # Average cosine similarity between predicted and true force vectors
        results["force_cosine"] = 0.0

    if "stress_mae" in metrics:
        # MAE between predicted and true stresses (GPa)
        results["stress_mae"] = 0.0

    return results
