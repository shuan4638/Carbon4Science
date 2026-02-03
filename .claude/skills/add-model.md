# Skill: Add New Model

Guide for adding a new model to the benchmark.

## Usage
```
/add-model [task] [model_name]
```

## Examples
```
/add-model Retrosynthesis MyNewModel
/add-model MolGen VAE
/add-model MatGen CDVAE
/add-model MLIP MACE
```

## Instructions

When the user invokes this skill, guide them through these steps:

### Step 1: Create Model Directory
```
<Task>/<ModelName>/
├── Inference.py        # Required: Uniform interface
├── environment.yml     # Required: Conda environment
├── CLAUDE.md           # Required: Model-specific guidance for Claude Code
├── README.md           # Recommended: Documentation
└── models/             # Model checkpoints (gitignored)
```

### Step 2: Implement Uniform Inference Interface

All tasks use the same return format:

```python
# Inference.py
from typing import List, Dict, Union

def run(input_data, top_k: int = 10) -> List[Dict]:
    """
    Args:
        input_data: Task-specific input (SMILES string, structure, etc.)
        top_k: Number of predictions per input

    Returns:
        [{'input': '...', 'predictions': [{'smiles': '...', 'score': 0.95}, ...]}]
    """
    # Implementation here
    pass
```

For **MolGen** (generative, no input molecule):
```python
def run(num_samples: int = 100, **kwargs) -> List[Dict]:
    """
    Returns:
        [{'input': 'generated', 'predictions': [{'smiles': '...', 'score': 1.0}]}]
    """
    pass
```

### Step 3: Create CLAUDE.md for the Model

Include: project overview, environment setup, commands, architecture, key files.
See `Retrosynthesis/LocalRetro/CLAUDE.md` for a reference.

### Step 4: Create Conda Environment
```yaml
# environment.yml
name: mymodel
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.9
  - pytorch
  - rdkit
  - pip:
    - your-package
```

### Step 5: Register in Benchmark Runner

Update `benchmarks/run_benchmark.py`:
```python
TASKS = {
    "<Task>": {
        "eval_module": "<Task>.evaluate",
        "models": {
            "MyNewModel": "<Task>.MyNewModel.Inference",
        }
    },
}
```

Update `benchmarks/run.sh`:
```bash
declare -A MODEL_ENVS=(
    ["MyNewModel"]="mymodel_env"
)
```

Update `benchmarks/setup_envs.sh`:
```bash
setup_mymodel() {
    echo "Setting up MyNewModel environment..."
    cd ../<Task>/MyNewModel
    conda env create -f environment.yml
    cd -
}
```

### Step 6: Test the Model
```bash
# Quick test with 10 samples
./benchmarks/run.sh --model MyNewModel --limit 10

# Full benchmark with carbon tracking
./benchmarks/run.sh --model MyNewModel --limit 1000 --track_carbon
```

## Checklist
- [ ] `Inference.py` with uniform interface
- [ ] `environment.yml` with dependencies
- [ ] `CLAUDE.md` with model guidance
- [ ] Model registered in `run_benchmark.py`
- [ ] Environment added to `run.sh` and `setup_envs.sh`
- [ ] Config added to `benchmarks/configs/models.yaml`
- [ ] Quick test passes with `--limit 10`
- [ ] Results added to task README.md
