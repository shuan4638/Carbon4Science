#!/usr/bin/env python3
"""
Unified Benchmark Runner for Carbon4Science

Dynamically loads task-specific evaluation modules from each task directory.
Each task defines its own:
  - METRICS: Available metrics for the task
  - load_test_data(): Load test dataset
  - evaluate(): Compute metrics from predictions

This file is part of the Example/ template. When setting up a new task,
rename "Example" to your task name (e.g. "MolGen") throughout this file.

Usage:
    python run_benchmark.py --model LocalRetro --track_carbon
    python run_benchmark.py --model neuralsym --limit 500
"""

import argparse
import importlib
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add repository root to path (<Task>/benchmarks/ → <Task>/ → repo root)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Task to evaluation module mapping
# When adapting for your task, rename "Example" to your task folder name.
TASKS = {
    "Example": {
        "eval_module": "Example.evaluate",
        "models": {
            "neuralsym": "Example.neuralsym.Inference",
            "LocalRetro": "Example.LocalRetro.Inference",
            "RetroBridge": "Example.RetroBridge.Inference",
            "Chemformer": "Example.Chemformer.Inference",
            "RSGPT": "Example.RSGPT.inference",
            "RSMILES_1x": "Example.RSMILES.Inference",
            "RSMILES_20x": "Example.RSMILES.Inference",
        }
    },
}


def get_task_evaluator(task_name: str):
    """Load the task-specific evaluation module."""
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASKS.keys())}")

    eval_module_name = TASKS[task_name]["eval_module"]
    try:
        eval_module = importlib.import_module(eval_module_name)
        return eval_module
    except ImportError as e:
        raise ImportError(f"Could not load evaluation module for {task_name}: {e}")


def _load_model_config(model_name: str) -> dict:
    """Load model config from models.yaml if available."""
    config_path = Path(__file__).resolve().parent / "configs" / "models.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml
        with open(config_path) as f:
            configs = yaml.safe_load(f)
        return configs.get(model_name, {})
    except ImportError:
        return {}


def get_model_run_func(task_name: str, model_name: str):
    """Load the model's run function, handling model-specific initialization."""
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}")

    models = TASKS[task_name]["models"]
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available for {task_name}: {list(models.keys())}")

    module_name = models[model_name]
    module = importlib.import_module(module_name)

    # If the module has a load_model() function (e.g. Chemformer, RSMILES), call it first
    if hasattr(module, 'load_model') and callable(module.load_model):
        config = _load_model_config(model_name)
        checkpoint = config.get('checkpoint')
        if checkpoint:
            checkpoint = str(ROOT_DIR / checkpoint)
        vocabulary = config.get('vocabulary')
        if vocabulary:
            vocabulary = str(ROOT_DIR / vocabulary)

        kwargs = {}
        if checkpoint:
            kwargs['model_path'] = checkpoint
        if vocabulary:
            kwargs['vocabulary_path'] = vocabulary
        # Pass augmentation_factor if specified in config (e.g. RSMILES_1x vs 20x)
        augmentation_factor = config.get('augmentation_factor')
        if augmentation_factor is not None:
            kwargs['augmentation_factor'] = augmentation_factor
        if kwargs:
            module.load_model(**kwargs)

    return module.run


def count_model_parameters(task_name: str, model_name: str) -> Optional[int]:
    """Count trainable parameters by running a dummy inference to trigger model loading."""
    try:
        import torch
        # After run() is called once, models are typically cached as module globals.
        # Search all loaded modules for PyTorch models.
        module_name = TASKS[task_name]["models"][model_name]
        module = sys.modules.get(module_name)
        if module is None:
            return None

        # Look for nn.Module instances in module globals and common patterns
        for attr_name in dir(module):
            obj = getattr(module, attr_name, None)
            if isinstance(obj, torch.nn.Module):
                return sum(p.numel() for p in obj.parameters())

        # Check common lazy-init patterns: _model, _proposer, etc.
        for var_name in ['_model', '_proposer', '_model_instance']:
            obj = getattr(module, var_name, None)
            if obj is None:
                continue
            # Could be a wrapper object containing a model attribute
            if isinstance(obj, torch.nn.Module):
                return sum(p.numel() for p in obj.parameters())
            # Check if it has a .model attribute (e.g., neuralsym _Proposer.model)
            inner = getattr(obj, 'model', None)
            if isinstance(inner, torch.nn.Module):
                return sum(p.numel() for p in inner.parameters())
    except Exception:
        pass
    return None


def run_benchmark(
    task_name: str,
    model_name: str,
    limit: Optional[int] = None,
    top_k: int = 50,
    metrics: Optional[List[str]] = None,
    track_carbon: bool = False,
    data_path: Optional[str] = None,
    output_path: Optional[str] = None,
    save_predictions: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run benchmark for a model on a task.

    Args:
        task_name: Task name (Retro, MolGen, MatGen)
        model_name: Model name within the task
        limit: Limit number of test samples (optional)
        top_k: Number of predictions per sample
        metrics: Metrics to compute (uses task defaults if None)
        track_carbon: Whether to track carbon emissions
        data_path: Custom test data path (optional)
        output_path: Path to save results JSON (optional)
        verbose: Print progress updates

    Returns:
        Results dictionary with accuracy and carbon metrics
    """
    # Load task-specific evaluator
    evaluator = get_task_evaluator(task_name)
    model_run = get_model_run_func(task_name, model_name)

    # Use task's default metrics if not specified
    if metrics is None:
        metrics = evaluator.METRICS

    # Print config
    if verbose:
        print("=" * 60)
        print(f"Task:    {task_name}")
        print(f"Model:   {model_name}")
        print(f"Metrics: {metrics}")
        print(f"Top-k:   {top_k}")
        print(f"Limit:   {limit or 'Full dataset'}")
        print(f"Carbon:  {'Yes' if track_carbon else 'No'}")
        print("=" * 60)
        print()

    # Load test data
    if verbose:
        print("Loading test data...")
    test_cases = evaluator.load_test_data(data_path=data_path, limit=limit)
    if verbose:
        print(f"Loaded {len(test_cases)} test cases")
        print()

    # Setup carbon tracking
    if track_carbon:
        from carbon_tracker import CarbonTracker
        tracker = CarbonTracker(
            project_name=f"{model_name}_{task_name}_benchmark",
            model_name=model_name,
            task="inference",
            save_results=False  # We'll save manually with full results
        )
    else:
        tracker = None

    # Run inference
    if verbose:
        print(f"Running inference (top_k={top_k})...")
        sys.stdout.flush()

    predictions = []
    start_time = time.time()

    if tracker:
        tracker.start()

    for i, tc in enumerate(test_cases):
        try:
            result = model_run(tc['product'], top_k=top_k)
            predictions.append(result[0] if result else {'input': tc['product'], 'predictions': []})
        except Exception as e:
            predictions.append({'input': tc['product'], 'predictions': []})

        if verbose and (i + 1) % 100 == 0:
            # Compute intermediate accuracy
            intermediate_results = evaluator.evaluate(predictions, test_cases[:i+1], metrics)
            acc_str = ", ".join([f"{m}: {intermediate_results[m]*100:.1f}%" for m in metrics[:2]])
            print(f"  {i+1}/{len(test_cases)} - {acc_str}")
            sys.stdout.flush()

    duration = time.time() - start_time

    if tracker:
        tracker.stop()
        carbon_metrics = tracker.get_metrics()
    else:
        carbon_metrics = {"duration_seconds": duration}

    # Evaluate predictions
    if verbose:
        print()
        print("Evaluating predictions...")
    eval_results = evaluator.evaluate(predictions, test_cases, metrics)

    # Count model parameters (model is loaded after first inference call)
    num_params = count_model_parameters(task_name, model_name)

    # Compile final results
    results = {
        "task": task_name,
        "model": model_name,
        "num_samples": len(test_cases),
        "top_k": top_k,
        "model_params": num_params,
        "metrics": metrics,
        "accuracy": {m: eval_results[m] for m in metrics},
        "correct": eval_results.get("correct", {}),
        "carbon": carbon_metrics,
    }

    # Print results
    if verbose:
        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Samples: {len(test_cases)}")
        if num_params is not None:
            if num_params >= 1_000_000_000:
                print(f"Params:  {num_params:,} ({num_params/1e9:.2f}B)")
            elif num_params >= 1_000_000:
                print(f"Params:  {num_params:,} ({num_params/1e6:.2f}M)")
            else:
                print(f"Params:  {num_params:,} ({num_params/1e3:.1f}K)")
        print()
        print("Accuracy:")
        for m in metrics:
            acc = eval_results[m] * 100
            correct = eval_results.get("correct", {}).get(m, 0)
            print(f"  {m}: {acc:.2f}% ({correct}/{len(test_cases)})")
        print()
        print(f"Duration: {carbon_metrics.get('duration_seconds', duration):.1f}s")
        energy_wh = carbon_metrics.get('energy_wh', 0)
        co2_g = carbon_metrics.get('emissions_g_co2', 0)
        if energy_wh > 0:
            print(f"Energy:   {energy_wh:.4f} Wh")
        if co2_g > 0:
            print(f"CO2:      {co2_g:.4f} g")
        peak_gpu = carbon_metrics.get('peak_gpu_memory_mb', 0)
        peak_cpu = carbon_metrics.get('peak_cpu_memory_mb', 0)
        if peak_gpu > 0:
            print(f"Peak GPU: {peak_gpu:.1f} MB")
        if peak_cpu > 0:
            print(f"Peak CPU: {peak_cpu:.1f} MB")
        print("=" * 60)

    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        if verbose:
            print(f"\nResults saved to: {output_path}")

    # Save raw predictions
    if save_predictions:
        pred_data = []
        for pred, tc in zip(predictions, test_cases):
            pred_data.append({
                "input": tc["product"],
                "ground_truth": tc["ground_truth"],
                "predictions": pred.get("predictions", []) if isinstance(pred, dict) else pred,
            })
        with open(save_predictions, 'w') as f:
            json.dump(pred_data, f, indent=2, default=str)
        if verbose:
            print(f"Predictions saved to: {save_predictions}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Unified Benchmark Runner for Carbon4Science",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Core arguments
    parser.add_argument("--task", type=str, default="Retro",
                        help=f"Task: {', '.join(TASKS.keys())} (default: Retro)")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name within the task")
    parser.add_argument("--metrics", type=str, nargs="+",
                        help="Metrics to compute (uses task defaults if not specified)")

    # Data options
    parser.add_argument("--limit", type=int,
                        help="Limit number of test samples")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Number of predictions per sample (default: 50)")
    parser.add_argument("--data", type=str,
                        help="Custom test data path")

    # Output options
    parser.add_argument("--output", type=str,
                        help="Output JSON file for results")
    parser.add_argument("--track_carbon", action="store_true",
                        help="Track carbon emissions")
    parser.add_argument("--save_predictions", type=str,
                        help="Path to save raw predictions JSON")

    # Info options
    parser.add_argument("--list_tasks", action="store_true",
                        help="List available tasks")
    parser.add_argument("--list_models", action="store_true",
                        help="List available models for a task")
    parser.add_argument("--list_metrics", action="store_true",
                        help="List available metrics for a task")

    args = parser.parse_args()

    # List options
    if args.list_tasks:
        print("Available tasks:")
        for task in TASKS:
            print(f"  {task}")
        return

    if args.list_models:
        if args.task not in TASKS:
            print(f"Unknown task: {args.task}")
            return
        print(f"Available models for {args.task}:")
        for model in TASKS[args.task]["models"]:
            print(f"  {model}")
        return

    if args.list_metrics:
        try:
            evaluator = get_task_evaluator(args.task)
            print(f"Available metrics for {args.task}:")
            for m in evaluator.METRICS:
                print(f"  {m}")
        except ImportError as e:
            print(f"Could not load evaluator for {args.task}: {e}")
        return

    # Run benchmark
    run_benchmark(
        task_name=args.task,
        model_name=args.model,
        limit=args.limit,
        top_k=args.top_k,
        metrics=args.metrics,
        track_carbon=args.track_carbon,
        data_path=args.data,
        output_path=args.output,
        save_predictions=args.save_predictions,
        verbose=True,
    )


if __name__ == "__main__":
    main()
