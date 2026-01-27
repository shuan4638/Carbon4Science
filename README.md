# The Carbon Cost of Generative AI for Science

[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A benchmarking framework for evaluating the **carbon efficiency** of generative AI models in scientific discovery.

## Abstract

Artificial intelligence is accelerating scientific discovery, yet current evaluation practices focus almost exclusively on accuracy, neglecting the computational and environmental costs of increasingly complex generative models. This oversight obscures a critical trade-off: **state-of-the-art performance often comes at disproportionate expense**, with order-of-magnitude increases in carbon emissions yielding only marginal improvements.

We present **The Carbon Cost of Generative AI for Science**, a benchmarking framework that systematically evaluates the carbon efficiency of generative models—including diffusion models and large language models—for scientific discovery. Spanning three core tasks (**molecule generation**, **retrosynthesis**, and **material generation**), we assess open-source models using standardized protocols that jointly measure predictive performance and carbon footprint.

**Key Finding**: Simpler, specialized models frequently match or approach state-of-the-art accuracy while consuming **10–100× less compute**.

## Tasks

| Task | Branch | Status |
|------|--------|--------|
| Retrosynthesis | `retrosynthesis` | In Progress |
| Molecule Generation | `molecule_generation` | Planned |
| Material Generation | `material_generation` | Planned |

---

## Getting Started

### Step 1: Clone the Repository

```bash
# Clone main branch (benchmark infrastructure only)
git clone https://github.com/shuan4638/Carbon4Science.git
cd Carbon4Science

# For retrosynthesis models, switch to that branch
git checkout retrosynthesis
```

### Step 2: Setup Environments

**Important**: Each model requires a different conda environment due to incompatible dependencies.

```bash
cd benchmarks

# Setup ALL environments (recommended, takes ~30 min)
chmod +x setup_envs.sh
./setup_envs.sh

# OR setup a specific model's environment
./setup_envs.sh neuralsym
./setup_envs.sh LocalRetro
```

### Step 3: Download Model Checkpoints

```bash
# RetroBridge
mkdir -p Retrosynthesis/RetroBridge/models
wget https://zenodo.org/record/10688201/files/retrobridge.ckpt \
     -O Retrosynthesis/RetroBridge/models/retrobridge.ckpt

# See benchmarks/configs/models.yaml for all checkpoint URLs
```

### Step 4: Run Benchmarks

```bash
cd benchmarks

# Run a single model (auto-switches conda environment)
./run.sh --model neuralsym --smiles "CCO" --device cuda:0

# Run with carbon tracking
./run.sh --model LocalRetro --smiles "CCO" --track_carbon

# Run with evaluation metrics
./run.sh --model neuralsym \
    --input ../data/test.csv \
    --ground_truth ../data/test_reactants.csv \
    --metric top_1 top_10 \
    --track_carbon \
    --output results/neuralsym_results.json

# Run ALL models for comparison
./run.sh --model all --input ../data/test.csv --track_carbon
```

---

## Repository Structure

```
Carbon4Science/
├── README.md                 # This file
├── CONTRIBUTING.md           # How to contribute
├── LICENSE                   # MIT License
│
├── benchmarks/               # Benchmark infrastructure (main branch)
│   ├── run.sh               # Unified runner (handles conda envs)
│   ├── setup_envs.sh        # Environment setup script
│   ├── run_benchmark.py     # Python benchmark script
│   ├── carbon_tracker.py    # Carbon measurement module
│   ├── configs/
│   │   ├── models.yaml      # Model checkpoints & requirements
│   │   └── hardware_template.yaml
│   └── results/             # Benchmark outputs
│
├── Retrosynthesis/          # (retrosynthesis branch only)
│   ├── neuralsym/
│   ├── LocalRetro/
│   ├── RetroBridge/
│   ├── Chemformer/
│   └── RSGPT/
│
├── MolGen/                  # (molecule_generation branch)
└── MatGen/                  # (material_generation branch)
```

---

## For Contributors

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:
- Adding new models
- Running benchmarks
- Submitting results

### Quick Reference: Uniform Model Interface

All models must implement this interface in `Inference.py`:

```python
def run(smiles, top_k=10) -> List[Dict]:
    """
    Args:
        smiles: str or List[str] - Input SMILES
        top_k: int - Number of predictions

    Returns:
        List[Dict] with format:
        [{'input': 'CCO', 'predictions': [{'smiles': '...', 'score': 0.95}, ...]}]
    """
```

---

## Citation

```bibtex
@article{carbon2025,
  title={The Carbon Cost of Generative AI for Science},
  author={...},
  journal={...},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
