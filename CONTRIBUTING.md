# Contributing to Carbon4Science

This guide explains how to add a new model's results to the leaderboard.

---

## How It Works

Your branch mirrors the `Example/` folder structure. When your benchmark is complete, only two things get merged into `main`:
1. Your result JSON → `results/<Task>/<model>_<N>.json`
2. A new row in the main `README.md` leaderboard table

Your model code, environments, and benchmark scripts stay on your branch.

---

## Branch Structure

**Never commit directly to `main`.** Create a personal branch:

```
<your-name>/<task>-<model>
```

Examples: `gunwook/molgen-hiervae` · `junkil/matgen-diffcsp` · `junyoung/mlip-mace`

---

## Step-by-Step Workflow

### 1. Set up your branch

```bash
git clone https://github.com/shuan4638/Carbon4Science.git
cd Carbon4Science
git checkout main && git pull
git checkout -b <your-name>/<task>-<model>
```

### 2. Copy the Example/ template

```bash
cp -r Example/ <YourTask>/
```

Adapt it for your task — rename `Example` to your task name throughout `benchmarks/run_benchmark.py`, add your models to the `TASKS` dict, write `evaluate.py` for your metrics, and add your model implementations under `<YourTask>/<Model>/`.

Your branch should look like:

```
<YourTask>/
├── README.md                  # Your results table (see Example/README.md)
├── evaluate.py                # Your task's metrics
├── data/                      # Test dataset
├── benchmarks/
│   ├── run_benchmark.py
│   ├── run.sh
│   ├── slurm_benchmark.sh
│   ├── carbon_tracker.py
│   ├── plot_results.py
│   └── configs/models.yaml
├── results/
│   ├── outputs/               # JSON result files
│   └── figures/               # Plots
└── <ModelA>/
    ├── Inference.py           # Uniform run() interface
    ├── environment.yml
    └── CLAUDE.md
```

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
sbatch --job-name=<Model> <YourTask>/benchmarks/slurm_benchmark.sh <Model>
squeue -u $USER   # check status
```

Always run the full test set. Never run long benchmarks as background processes.

### 5. Open a pull request to main

Your PR should contain only:

| File | Description |
|------|-------------|
| `results/<Task>/<model>_<N>.json` | Result JSON from the full benchmark run |
| `README.md` | New row added to the appropriate leaderboard table |

Your model code, environments, and benchmark scripts stay on your branch — **do not add a task folder to main**.

```bash
git add results/<Task>/<model>_<N>.json README.md
git commit -m "Add <Model> results to <Task> leaderboard"
git push -u origin <your-branch>
gh pr create --title "Add <Model> to <Task>" --body "Top-1: X%, CO₂/exp: Yg"
```

---

## Result JSON Format

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

## What NOT to Commit to main

- Model checkpoints (`*.ckpt`, `*.pt`, `*.bin`, `*.pth`)
- Raw prediction files (`*_predictions.json`)
- Training datasets or large data files
- Your task folder (`<YourTask>/`) — this stays on your branch

---

## Getting Help

Claude Code reads `CLAUDE.md` automatically and knows the full protocol.

```bash
claude
```

Useful prompts:
- *"I'm [name], adding [Model] to [Task]. Set up my branch following the Example/ template."*
- *"Write the Inference.py for [Model] following the uniform interface."*
- *"Run the full benchmark for [Model] with carbon tracking and save the result JSON."*
- *"Add my results to the main README leaderboard table and open a PR."*
