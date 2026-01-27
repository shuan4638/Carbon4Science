# Contributing to Carbon4Science

Thank you for your interest in contributing to The Carbon Cost of Generative AI for Science!

## Table of Contents

- [Reporting Issues](#reporting-issues)
- [Adding a New Model](#adding-a-new-model)
- [Contributing Benchmark Results](#contributing-benchmark-results)
- [Code Style](#code-style)

---

## Reporting Issues

Use [GitHub Issues](https://github.com/shuan4638/Carbon4Science/issues) to report bugs or request features. Please include:

- Hardware configuration (GPU model, RAM)
- Python version and conda environment name
- Steps to reproduce the issue
- Full error traceback

---

## Adding a New Model

### Step 1: Create Model Directory

Create a subdirectory under the appropriate task folder:

```
Retrosynthesis/YourModel/
MolGen/YourModel/
MatGen/YourModel/
```

### Step 2: Required Files

Your model directory must include:

| File | Description |
|------|-------------|
| `Inference.py` | **Required.** Standardized inference interface |
| `requirements.txt` or `environment.yml` | **Required.** All dependencies |
| `README.md` | Model description, paper reference |

### Step 3: Implement the Uniform Interface

**This is critical.** Your `Inference.py` must implement the `run()` function with this exact signature and return format:

```python
# Inference.py
from typing import List, Dict, Union

def run(smiles: Union[str, List[str]], top_k: int = 10) -> List[Dict]:
    """
    Run inference on input SMILES.

    Args:
        smiles: Single SMILES string or list of SMILES strings
        top_k: Number of predictions to return per input

    Returns:
        List of dictionaries with this EXACT format:
        [
            {
                'input': 'CCO',  # The input SMILES
                'predictions': [
                    {'smiles': 'C.O', 'score': 0.95},
                    {'smiles': 'CC.O', 'score': 0.03},
                    ...
                ]
            },
            ...
        ]
    """
    # Handle single SMILES input
    if isinstance(smiles, str):
        smiles = [smiles]

    results = []
    for smi in smiles:
        # Your model inference here
        predictions = your_model.predict(smi, top_k=top_k)

        results.append({
            'input': smi,
            'predictions': [
                {'smiles': pred['smiles'], 'score': pred['confidence']}
                for pred in predictions
            ]
        })

    return results
```

### Step 4: Create Conda Environment

Since models have incompatible dependencies, each needs its own conda environment.

1. **Create `environment.yml`** in your model directory:

```yaml
name: yourmodel
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.8
  - pytorch
  - rdkit
  - pip:
    - your-package
```

2. **Add your model to `benchmarks/setup_envs.sh`**:

```bash
setup_yourmodel() {
    echo "Setting up YourModel environment..."
    cd ../Retrosynthesis/YourModel
    conda env create -f environment.yml
    cd -
    echo "✓ YourModel environment ready"
}

# Add to case statement at bottom:
YourModel|yourmodel) setup_yourmodel ;;
```

### Step 5: Register Model in Benchmark Runner

1. **Edit `benchmarks/run.sh`** - Add your model to the environment mapping:

```bash
declare -A MODEL_ENVS=(
    ["neuralsym"]="neuralsym"
    ["LocalRetro"]="rdenv"
    ["RetroBridge"]="retrobridge"
    ["Chemformer"]="chemformer"
    ["RSGPT"]="gpt"
    ["YourModel"]="yourmodel"  # <-- Add this line
)
```

2. **Edit `benchmarks/run_benchmark.py`** - Add your model to the MODELS dict:

```python
MODELS = {
    # ... existing models ...
    "YourModel": {
        "module": "Retrosynthesis.YourModel.Inference",
        "requires_init": False,  # Set True if you need load_model()
    },
}
```

3. **Edit `benchmarks/configs/models.yaml`**:

```yaml
YourModel:
  env: yourmodel
  checkpoint: Retrosynthesis/YourModel/models/checkpoint.pth
  checkpoint_url: https://example.com/checkpoint.pth  # or null
  data_dir: Retrosynthesis/YourModel/data
  gpu_memory_mb: 4000  # Approximate GPU memory needed
```

### Step 6: Test Your Model

```bash
cd benchmarks

# Setup your environment
./setup_envs.sh YourModel

# Test with a single SMILES
./run.sh --model YourModel --smiles "CCO" --device cuda:0

# Run with carbon tracking
./run.sh --model YourModel --input ../data/test.csv --track_carbon

# Verify output format
./run.sh --model YourModel --smiles "CCO" --output test_output.json
cat results/test_output.json  # Check format matches specification
```

### Step 7: Submit Pull Request

Your PR should include:

- [ ] Model implementation in `Retrosynthesis/YourModel/`
- [ ] `Inference.py` with uniform interface
- [ ] `environment.yml` or `requirements.txt`
- [ ] Updates to `setup_envs.sh`, `run.sh`, `run_benchmark.py`, `models.yaml`
- [ ] Benchmark results in `benchmarks/results/`
- [ ] Model README with paper reference

---

## Contributing Benchmark Results

If you have different hardware and want to contribute benchmark results:

### Step 1: Document Your Hardware

Copy and fill in your hardware configuration:

```bash
cp benchmarks/configs/hardware_template.yaml benchmarks/configs/hardware_yourname.yaml
```

Edit the file:

```yaml
hardware:
  gpu:
    model: "NVIDIA RTX 4090"
    count: 1
    memory_gb: 24
  cpu:
    model: "AMD Ryzen 9 7950X"
    cores: 16
  ram_gb: 64
  cuda_version: "12.1"

contributor:
  name: "Your Name"
  github: "@yourusername"
```

### Step 2: Run Standard Benchmark Protocol

```bash
cd benchmarks

# Run 3 trials for each model
for model in neuralsym LocalRetro RetroBridge Chemformer RSGPT; do
    for run in 1 2 3; do
        ./run.sh --model $model \
            --input ../data/USPTO_50K/test.csv \
            --track_carbon \
            --output results/${model}_run${run}_$(date +%Y%m%d).json
    done
done
```

### Step 3: Submit Results

Submit a PR with:

- Your hardware configuration file
- All benchmark result JSON files
- Summary table of your results

---

## Code Style

- Python: Follow PEP 8
- Use type hints for function signatures
- Document public functions with docstrings

---

## Task Assignments

| Task | Lead | Status |
|------|------|--------|
| Retrosynthesis | @shuan4638 | Active |
| Molecule Generation | TBD | Planned |
| Material Generation | TBD | Planned |

---

## Common Issues

### "ModuleNotFoundError: No module named 'Retrosynthesis'"

Make sure you're running from the repository root:

```bash
cd Carbon4Science  # Not from benchmarks/
python benchmarks/run_benchmark.py --model YourModel --smiles "CCO"
```

### "conda environment not found"

Run the setup script first:

```bash
cd benchmarks
./setup_envs.sh YourModel
```

### GPU out of memory

Use CPU or a smaller batch size:

```bash
./run.sh --model YourModel --device cpu --smiles "CCO"
```

---

## Questions?

Open an issue or contact the maintainers.
