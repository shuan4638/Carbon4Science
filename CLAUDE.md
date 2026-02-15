# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Rules for Claude Code

### Rule 1: Uniform Inference Interface

When creating or modifying any `Inference.py` file, ALWAYS use this interface:

```python
from typing import List, Dict, Union

def run(input_data, top_k: int = 10) -> List[Dict]:
    """
    Returns:
        [{'input': '...', 'predictions': [{'smiles': '...', 'score': 0.95}, ...]}]
    """
```

**NEVER** deviate from this return format. The benchmark runner depends on it.

### Rule 2: Environment Isolation

Each model has its own conda environment. **NEVER** suggest installing packages globally or mixing environments.

**Retro (Retrosynthesis):**

| Model | Environment Name |
|-------|-----------------|
| neuralsym | `neuralsym` |
| LocalRetro | `rdenv` |
| RetroBridge | `retrobridge` |
| Chemformer | `chemformer` |
| RSGPT | `gpt` |
| RSMILES_1x | `rsmiles` |
| RSMILES_20x | `rsmiles` |

Other tasks (MolGen, MatGen, MLIP) follow the same pattern: one conda env per model.

When running models, use: `./benchmarks/run.sh --model <name>` (handles env switching automatically)

### Rule 3: Benchmark Runner

**ALWAYS** use the wrapper scripts instead of running Python directly:

```bash
# Correct
./benchmarks/run.sh --model neuralsym --limit 1000 --track_carbon

# Wrong - will fail due to environment issues
python benchmarks/run_benchmark.py --model neuralsym
```

### Rule 4: Adding New Models

When adding a new model, you MUST update ALL of these files:

1. `<Task>/<ModelName>/Inference.py` - Implement uniform interface
2. `<Task>/<ModelName>/environment.yml` - Dependencies
3. `<Task>/<ModelName>/CLAUDE.md` - Model-specific guidance
4. `benchmarks/run_benchmark.py` - Add to TASKS dict
5. `benchmarks/run.sh` - Add to MODEL_ENVS mapping
6. `benchmarks/setup_envs.sh` - Add setup function
7. `benchmarks/configs/models.yaml` - Add configuration

### Rule 5: Carbon Tracking

When running benchmarks, ALWAYS include `--track_carbon` flag:

```bash
./benchmarks/run.sh --model LocalRetro --limit 1000 --track_carbon
```

### Rule 6: Device Selection

Always let users specify device. Default to `cuda:0` but support CPU fallback.

### Rule 7: No Hardcoded Paths

Use relative paths from repository root. Never hardcode absolute paths in any script.

### Rule 8: Task-Specific Evaluation

Each task defines its own `evaluate.py` module with:
- `METRICS` - list of available metrics
- `load_test_data(data_path, limit)` - load test dataset
- `evaluate(predictions, test_cases, metrics)` - compute metrics

The benchmark runner dynamically loads the correct evaluator based on the `--task` flag.

### Rule 9: Benchmarking Workflow

When a task leader asks to benchmark their task end-to-end, follow these phases in order:

1. **Setup** — Create task directory structure (`<Task>/`, `<Task>/data/`, model subdirectories)
2. **Evaluate** — Implement `<Task>/evaluate.py` with `METRICS`, `load_test_data()`, and `evaluate()`
3. **Models** — For each model, implement `Inference.py` with the uniform `run()` interface, create `environment.yml`, and write `CLAUDE.md`
4. **Register** — Add all models to `benchmarks/run_benchmark.py` (TASKS dict), `benchmarks/run.sh` (MODEL_ENVS), `benchmarks/setup_envs.sh`, and `benchmarks/configs/models.yaml`
5. **Run** — Execute benchmarks: `./benchmarks/run.sh --model all --limit 1000 --track_carbon`
6. **Plot** — Generate visualization: `python benchmarks/plot_results.py --task <Task> --combined`
7. **Report** — Update `<Task>/README.md` and the root `README.md` with results table and figure

### Rule 10: Plot Generation

Use `benchmarks/plot_results.py` to generate accuracy vs cost plots.

**Commands:**
```bash
# All three plots (carbon, energy, speed) in combined view
python benchmarks/plot_results.py --task <Task> --combined

# Specific sample count
python benchmarks/plot_results.py --task <Task> --combined --samples 500

# Single x-axis metric with custom output
python benchmarks/plot_results.py --task <Task> --combined --xaxis emissions_g_co2 -o my_plot.png

# Per-metric panel view (one subplot per accuracy metric)
python benchmarks/plot_results.py --task <Task>
```

**Adding new models:** When adding a model to a new or existing task, add an entry to `MODEL_STYLES` in `plot_results.py`:
```python
MODEL_STYLES = {
    "MyModel": {"color": "#hex", "marker": "o", "params": "10M", "year": 2024, "venue": "NeurIPS"},
}
```

**Output:** Plots are saved to `benchmarks/figures/<Task>/`.

### Rule 11: Slurm Job Submission

**ALWAYS** submit benchmark runs via Slurm instead of running them as background processes. Use `benchmarks/slurm_benchmark.sh`:

```bash
# Single model
sbatch --job-name=RSGPT benchmarks/slurm_benchmark.sh RSGPT

# Chemformer with proper test set (pickle)
sbatch --job-name=Chemformer benchmarks/slurm_benchmark.sh Chemformer --data Retro/data/uspto_50_chemforner.pickle

# R-SMILES variants
sbatch --job-name=RSMILES_20x benchmarks/slurm_benchmark.sh RSMILES_20x

# Check job status
squeue -u $USER
```

**Cluster details:**
- Partitions: `5000_ada` (GPU), `6000_ada` (GPU), `cpu_only`
- GPU resource: `--gres=gpu:5000ada:1`
- Max walltime: 72 hours (GPU), 48 hours (CPU)
- Override memory with `--mem=32G` for large models (e.g., RSGPT 1B)

Logs are saved to `benchmarks/logs/<jobname>.o<jobid>`.

**NEVER** run long benchmarks as background shell processes. Always use `sbatch`.

### Results JSON Schema

Every benchmark run produces a JSON file following this structure:

```json
{
  "task": "Retro",
  "model": "LocalRetro",
  "num_samples": 1000,
  "top_k": 50,
  "model_params": 8645410,
  "metrics": ["top_1", "top_5", "top_10", "top_50"],
  "accuracy": {
    "top_1": 0.525,
    "top_5": 0.839,
    "top_10": 0.923,
    "top_50": 0.977
  },
  "correct": {
    "top_1": 525,
    "top_5": 839,
    "top_10": 923,
    "top_50": 977
  },
  "carbon": {
    "start_time": "2026-02-03T17:49:37.301875",
    "end_time": "2026-02-03T17:56:18.823578",
    "duration_seconds": 401.52,
    "energy_wh": 41.31,
    "emissions_g_co2": 16.52,
    "gpu_energy_wh": 19.06,
    "cpu_energy_wh": 22.20,
    "ram_energy_wh": 0.05,
    "peak_gpu_memory_mb": 154.2,
    "peak_cpu_memory_mb": 1236.7,
    "project_name": "LocalRetro_Retro_benchmark",
    "model_name": "LocalRetro",
    "task": "inference",
    "hardware": {
      "gpu_model": "NVIDIA RTX 5000 Ada Generation",
      "gpu_count": 8,
      "gpu_memory_gb": 31.99,
      "cpu_model": "INTEL(R) XEON(R) PLATINUM 8558",
      "cpu_cores": 192,
      "ram_gb": 503.04,
      "cuda_version": "Driver 580.82.09",
      "platform": "Linux-5.14.0-..."
    }
  }
}
```

Results are saved to `benchmarks/results/<Task>/<model>_<N>.json`.

---

## Project Overview

**Carbon4Science** benchmarks both predictive performance AND carbon efficiency of generative AI models for scientific discovery.

**Four Tasks:**
1. **Retrosynthesis** (`Retro/`) - Predict reactants from product molecules
2. **Molecule Generation** (`MolGen/`) - Generate novel molecules
3. **Material Generation** (`MatGen/`) - Generate crystal structures
4. **ML Interatomic Potentials** (`MLIP/`) - Predict atomic forces and energies

## Repository Structure

```
Carbon4Science/
├── benchmarks/        # Shared infrastructure (runner, carbon tracker, configs)
├── Retro/             # 7 models: neuralsym, LocalRetro, RetroBridge, Chemformer, RSGPT, RSMILES_1x, RSMILES_20x
├── MolGen/            # Molecule generation models (planned)
├── MatGen/            # Material generation models (planned)
└── MLIP/              # ML interatomic potential models (planned)
```

Each task directory is self-contained with its own evaluation module, test data, and model subdirectories. Each model subdirectory has its own conda environment and CLAUDE.md.

## Quick Inference (All Tasks)

All models provide a `run()` function in `Inference.py`:

```python
# Example: Retrosynthesis
from Retro.LocalRetro.Inference import run
results = run("CCO", top_k=10)

# Example: MolGen (when implemented)
from MolGen.SomeModel.Inference import run
results = run(num_samples=100)
```

## Common Commands

```bash
# Run benchmark for a single model
./benchmarks/run.sh --model <ModelName> --limit 1000 --track_carbon

# Run all models for a task
./benchmarks/run.sh --model all --limit 1000 --track_carbon

# Setup all environments
./benchmarks/setup_envs.sh

# Setup a specific model's environment
./benchmarks/setup_envs.sh <ModelName>
```

## Carbon Measurement

Use the unified `CarbonTracker` wrapper in `benchmarks/carbon_tracker.py`:

```python
from benchmarks.carbon_tracker import CarbonTracker

tracker = CarbonTracker(
    project_name="retrosynthesis_neuralsym",
    model_name="neuralsym",
    task="inference",
)

tracker.start()
# Your inference code here
tracker.stop()
print(tracker.get_metrics())  # energy_wh, emissions_g_co2, duration_seconds
```

## Standardized Protocol

- Report hardware specs (GPU model, CPU, RAM)
- Track both accuracy metrics and carbon footprint
- Use the same test dataset across all models in a task
- Run inference benchmarks and report mean results
