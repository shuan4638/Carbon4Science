# The Carbon Cost of Generative AI for Science

[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A benchmarking framework for evaluating the **carbon efficiency** of generative AI models in scientific discovery.

## Abstract

Artificial intelligence is accelerating scientific discovery, yet current evaluation practices focus almost exclusively on accuracy, neglecting the computational and environmental costs of increasingly complex generative models. This oversight obscures a critical trade-off: **state-of-the-art performance often comes at disproportionate expense**, with order-of-magnitude increases in carbon emissions yielding only marginal improvements.

We present **The Carbon Cost of Generative AI for Science**, a benchmarking framework that systematically evaluates the carbon efficiency of generative models—including diffusion models and large language models—for scientific discovery. Spanning three core tasks (**molecule generation**, **retrosynthesis**, and **material generation**), we assess open-source models using standardized protocols that jointly measure predictive performance and carbon footprint.

**Key Finding**: Simpler, specialized models frequently match or approach state-of-the-art accuracy while consuming **10–100× less compute**.

## Tasks

| Task | Status |
|------|--------|
| Retrosynthesis | In Progress |
| Molecule Generation | Planned |
| Material Generation | Planned |

## Quick Start

```bash
git clone https://github.com/shuan4638/Carbon4Science.git
cd Carbon4Science
pip install codecarbon pandas numpy
```

### Carbon Tracking

```python
from benchmarks.carbon_tracker import CarbonTracker

tracker = CarbonTracker(project_name="my_experiment")

with tracker:
    # Your model training or inference
    results = model.predict(test_data)

metrics = tracker.get_metrics()
print(f"Energy: {metrics['energy_kwh']:.4f} kWh")
print(f"CO2: {metrics['emissions_kg_co2']:.4f} kg")
```

## Repository Structure

```
Carbon4Science/
├── benchmarks/          # Carbon measurement infrastructure
├── Retrosynthesis/      # Retrosynthesis models (separate branch)
├── MolGen/              # Molecule generation (planned)
└── MatGen/              # Material generation (planned)
```

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
