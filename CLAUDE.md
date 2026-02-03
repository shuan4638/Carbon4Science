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

**Retrosynthesis:**

| Model | Environment Name |
|-------|-----------------|
| neuralsym | `neuralsym` |
| LocalRetro | `rdenv` |
| RetroBridge | `retrobridge` |
| Chemformer | `chemformer` |
| RSGPT | `gpt` |

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

---

## Project Overview

**Carbon4Science** benchmarks both predictive performance AND carbon efficiency of generative AI models for scientific discovery.

**Four Tasks:**
1. **Retrosynthesis** (`Retrosynthesis/`) - Predict reactants from product molecules
2. **Molecule Generation** (`MolGen/`) - Generate novel molecules
3. **Material Generation** (`MatGen/`) - Generate crystal structures
4. **ML Interatomic Potentials** (`MLIP/`) - Predict atomic forces and energies

## Repository Structure

```
Carbon4Science/
├── benchmarks/        # Shared infrastructure (runner, carbon tracker, configs)
├── Retrosynthesis/    # 5 models: neuralsym, LocalRetro, RetroBridge, Chemformer, RSGPT
├── MolGen/            # Molecule generation models (planned)
├── MatGen/            # Material generation models (planned)
└── MLIP/              # ML interatomic potential models (planned)
```

Each task directory is self-contained with its own evaluation module, test data, and model subdirectories. Each model subdirectory has its own conda environment and CLAUDE.md.

## Quick Inference (All Tasks)

All models provide a `run()` function in `Inference.py`:

```python
# Example: Retrosynthesis
from Retrosynthesis.LocalRetro.Inference import run
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
