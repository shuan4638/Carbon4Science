# Contributing to Carbon4Science

This guide explains how to add a new model's results to the leaderboard.

---

## How It Works

Each task has its own **orphan branch** (e.g., `Retro`, `Forward`, `MolGen`). You add your model to the task branch. When your benchmark is complete, only two things get merged into `main`:
1. Your result JSON → `results/<Task>/<model>_<N>.json`
2. A new row in the main `README.md` leaderboard table

Your model code, environments, and benchmark scripts stay on the task branch.

---

## Step-by-Step Workflow

### 1. Check out the task branch

```bash
git clone https://github.com/shuan4638/Carbon4Science.git
cd Carbon4Science
git checkout Retro   # or Forward, MolGen, MatGen, MLIP
```

### 2. Add your model

Create your model directory with two required files:

```
<YourModel>/
├── Inference.py       # Uniform run() interface (REQUIRED)
└── environment.yml    # Conda env spec (REQUIRED)
```

See `branch-example/ExampleTask/ExampleModel/` on `main` for a complete template.

### 3. Implement the uniform inference interface

Every `Inference.py` must expose exactly this function:

```python
from typing import List, Dict

def run(input_data, top_k: int = 10) -> List[Dict]:
    """
    Returns:
        [{'input': '...', 'predictions': [{'smiles': '...', 'score': 0.95}, ...]}]
    """
```

### 4. Run the full benchmark with carbon tracking

```bash
# Via Slurm (recommended)
sbatch --job-name=<Model> benchmarks/slurm_benchmark.sh <Model>
squeue -u $USER   # check status

# Or locally
./benchmarks/run.sh --model <Model> --track_carbon --output results/<model>_<N>.json
```

Always run the full test set. Never run long benchmarks as background processes.

### 5. Open a pull request to the task branch

Your PR should contain:

| File | Description |
|------|-------------|
| `<YourModel>/Inference.py` | Uniform `run()` interface |
| `<YourModel>/environment.yml` | Conda env spec |
| `results/<model>_<N>.json` | Result JSON from the full benchmark run |

```bash
git add <YourModel>/ results/<model>_<N>.json
git commit -m "Add <Model> to <Task>"
git push -u origin <Task>
gh pr create --base <Task> --title "Add <Model> to <Task>" --body "Top-1: X%, CO₂/exp: Yg"
```

Shaun reviews and merges the result JSON to `main`.

---

## Result JSON Format

See `branch-example/ExampleTask/results/examplemodel_5007.json` for the full schema.

```json
{
  "task": "YourTask",
  "model": "YourModel",
  "num_samples": 5007,
  "top_k": 50,
  "model_params": 8645410,
  "accuracy": { "top_1": 0.525, "top_50": 0.977 },
  "carbon": {
    "duration_seconds": 2313,
    "energy_wh": 41.31,
    "emissions_g_co2": 62.1,
    "peak_gpu_memory_mb": 154.2,
    "hardware": { "gpu_model": "...", "cpu_model": "..." }
  }
}
```

---

## What NOT to Commit

- Model checkpoints (`*.ckpt`, `*.pt`, `*.bin`, `*.pth`)
- Raw prediction files (`*_predictions.json`)
- Training datasets or large data files

---

## Getting Help

Claude Code reads `CLAUDE.md` automatically and knows the full protocol.

```bash
claude
```

Useful prompts:
- *"I'm [name], adding [Model] to [Task]. Set up my model following the branch-example/ template."*
- *"Write the Inference.py for [Model] following the uniform interface."*
- *"Run the full benchmark for [Model] with carbon tracking and save the result JSON."*
