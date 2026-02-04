# The Carbon Cost of Generative AI for Science

[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A benchmarking framework for evaluating the **carbon efficiency** of generative AI models in scientific discovery.

## Abstract

Artificial intelligence is accelerating scientific discovery, yet current evaluation practices focus almost exclusively on accuracy, neglecting the computational and environmental costs of increasingly complex generative models. This oversight obscures a critical trade-off: **state-of-the-art performance often comes at disproportionate expense**, with order-of-magnitude increases in carbon emissions yielding only marginal improvements.

We present **The Carbon Cost of Generative AI for Science**, a benchmarking framework that systematically evaluates the carbon efficiency of generative models—including diffusion models and large language models—for scientific discovery. Spanning four core tasks (**retrosynthesis**, **molecule generation**, **material generation**, and **machine learning interatomic potentials**), we assess open-source models using standardized protocols that jointly measure predictive performance and carbon footprint.

**Key Finding**: Simpler, specialized models frequently match or approach state-of-the-art accuracy while consuming **10-100x less compute**.

## Tasks

| Task | Directory | Leader | Status |
|------|-----------|--------|--------|
| Retrosynthesis | `Retro/` | Shuan Chen | Complete |
| Molecule Generation | `MolGen/` | Gunwook Nam | Planned |
| Material Generation | `MatGen/` | Junkil Park | Planned |
| ML Interatomic Potentials | `MLIP/` | Junyoung Choi | Planned |

---

## Benchmarking Methodology

All tasks follow the same standardized protocol to ensure fair, reproducible comparisons:

1. **Same dataset, same metrics, same hardware** — Every model in a task runs on the same test set, is evaluated with the same metrics, and uses the same GPU hardware (reported in results JSON).
2. **Uniform `Inference.py` interface** — Every model exposes a `run()` function with a standardized return format so the benchmark runner can orchestrate any model identically.
3. **Carbon tracking via CodeCarbon** — Energy consumption (Wh), CO2 emissions (g), and wall-clock time are recorded automatically using our `CarbonTracker` wrapper around [CodeCarbon](https://codecarbon.io/).
4. **Environment isolation** — Each model has its own conda environment to prevent dependency conflicts. The runner script (`run.sh`) activates the correct environment automatically.
5. **Normalized comparison** — Cost metrics are normalized to a fixed sample count (e.g., per 500 samples) so models evaluated on different subset sizes can be compared fairly.
6. **Structured JSON results** — Every benchmark run produces a JSON file with accuracy, carbon, hardware, and metadata fields, enabling automated analysis and plotting.
7. **Accuracy vs Cost visualization** — Results are plotted as accuracy (y-axis) vs cost metric (x-axis, log-scale) to reveal the efficiency frontier across models.

---

## Retrosynthesis Results

Five models benchmarked on 1,000 samples from the USPTO-50K test set, evaluated on top-k exact-match accuracy with full carbon tracking.

**Hardware:** NVIDIA RTX 5000 Ada Generation, Intel Xeon Platinum 8558, 503 GB RAM

![Retrosynthesis: Accuracy vs Carbon Cost](benchmarks/figures/Retro/accuracy_vs_carbon_combined_500.png)

| Model | Params | Top-1 | Top-10 | Top-50 | Duration (s) | Energy (Wh) | CO2 (g) | Peak GPU (MB) |
|-------|--------|-------|--------|--------|--------------|-------------|---------|---------------|
| neuralsym | 32.5M | 43.0% | 72.4% | 74.0% | 192 | 21.2 | 8.5 | 504 |
| LocalRetro | 8.6M | 52.5% | 92.3% | 97.7% | 402 | 41.3 | 16.5 | 154 |
| Chemformer | 44.7M | 88.0% | 90.8% | 91.2% | 16,911 | 1,378 | 551 | 207 |
| RetroBridge | 4.6M | 22.1% | 44.5% | 51.7% | 61,974 | 4,566 | 1,966 | 479 |
| RSGPT | ~1.6B | 77.5% | 97.8% | 98.7% | 49,782 | 3,787 | 1,515 | 6,950 |

**Key insight:** LocalRetro achieves 52.5% top-1 accuracy (97.7% top-50) using only 8.5 g CO2 and 154 MB GPU memory — **65x less carbon** than RSGPT and **45x less** than the most expensive model (RetroBridge), while delivering competitive top-k coverage. Chemformer leads on top-1 accuracy (88.0%) but at 65x the carbon cost of LocalRetro. The largest model (RSGPT, ~1.6B params) achieves the best top-50 accuracy (98.7%) but consumes **178x more carbon** than LocalRetro for only a 1 percentage point improvement.

---

## For Task Leaders

Each task leader is responsible for benchmarking models in their domain.
Claude Code is the recommended way to do this — it reads the project's
`CLAUDE.md` files automatically and understands the benchmarking protocol,
directory structure, and conventions.

### Prerequisites

- Linux with NVIDIA GPU(s)
- Conda (Miniconda or Anaconda)
- Git
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code)
  (`npm install -g @anthropic-ai/claude-code`)

### Quick Start

```bash
git clone https://github.com/shuan4638/Carbon4Science.git
cd Carbon4Science
claude
```

Introduce yourself in the first message:

> I'm [name], the task leader for [task]. I need to set up the [task]
> benchmarking pipeline from scratch. Guide me through the process
> step by step.

Claude will walk you through the full workflow, using `Retro/` as the
reference implementation.

### What You'll Build

Claude Code will guide you through each of these steps:

1. **Task directory** — `<Task>/` with README, evaluate.py, data/, and model subdirectories
2. **Evaluation module** — `<Task>/evaluate.py` with your task's metrics and test data loader
3. **Models** — `<Task>/<Model>/Inference.py` with the uniform `run()` interface, conda environment, and CLAUDE.md for each model
4. **Benchmark registration** — Your task and models registered in the benchmark runner (`run_benchmark.py`, `run.sh`, `setup_envs.sh`, `models.yaml`)
5. **Benchmark runs** — All models run with carbon tracking on the same test set and hardware
6. **Results** — Accuracy vs cost plots and a results table in your task README

### Reference

The `Retro/` task is the complete reference implementation. Key files to study:

- `Retro/evaluate.py` — evaluation module structure
- `Retro/LocalRetro/Inference.py` — uniform `run()` interface
- `Retro/LocalRetro/environment.yml` — conda environment spec
- `benchmarks/configs/models.yaml` — model registration format

### Skills

| Skill | Description |
|-------|-------------|
| `/add-model <Task> <ModelName>` | Add a new model to your task |
| `/benchmark <ModelName>` | Run a carbon-tracked benchmark |
| `/evaluate <Task>` | Evaluate model predictions |
| `/plot <Task>` | Generate accuracy vs cost plots |

### Example Prompts

- "Set up my task directory structure following the Retro template"
- "Write the evaluate.py for MolGen with FCD and validity metrics"
- "Create the Inference.py for CDVAE following the uniform interface"
- "Register my models in the benchmark runner"
- "Run benchmarks for all my models with 1000 samples and carbon tracking"
- "Generate plots and update my README with the results table"
- "What does the Retro/LocalRetro/Inference.py look like? I want to follow the same pattern."

---

## Repository Structure

```
Carbon4Science/
├── README.md                 # This file
├── CLAUDE.md                 # Instructions for Claude Code
├── .claude/skills/           # Claude Code skills (add-model, benchmark, evaluate)
│
├── benchmarks/               # Shared benchmark infrastructure
│   ├── run.sh               # Unified runner (handles conda envs)
│   ├── run_benchmark.py     # Python benchmark runner
│   ├── carbon_tracker.py    # Carbon/energy measurement
│   ├── setup_envs.sh        # Environment setup
│   ├── configs/             # Model configs, hardware specs
│   └── results/             # Benchmark outputs (JSON)
│
├── Retro/                   # Retrosynthesis task (Shuan Chen)
│   ├── neuralsym/           # Template-based, Nature 2018
│   ├── LocalRetro/          # MPNN + attention, JACS Au 2021
│   ├── RetroBridge/         # Markov bridges, ICLR 2024
│   ├── Chemformer/          # BART transformer, ML:ST 2022
│   └── RSGPT/               # GPT 1.6B params, Nat. Comm. 2025
│
├── MolGen/                  # Molecule generation (Gunwook Nam)
├── MatGen/                  # Material generation (Junkil Park)
└── MLIP/                    # ML interatomic potentials (Junyoung Choi)
```

---

## Citation

```bibtex
@article{carbon2026,
  title={The Carbon Cost of Generative AI for Science},
  author={...},
  journal={...},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
