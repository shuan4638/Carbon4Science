# Retrosynthesis

**Task Leader:** Shuan Chen

Retrosynthesis prediction: Given a target product molecule, predict the reactants needed to synthesize it.

## Metrics

| Metric | Description |
|--------|-------------|
| `top_1` | Exact match at rank 1 |
| `top_5` | Correct answer in top 5 predictions |
| `top_10` | Correct answer in top 10 predictions |
| `top_50` | Correct answer in top 50 predictions |

## Test Dataset

- **USPTO-50K**: 5,005 test reactions from US Patent data
- Location: `data/test_demapped.csv`

## Models

| Model | Paper | Environment | License |
|-------|-------|-------------|---------|
| neuralsym | [Neural-Symbolic Machine Learning for Retrosynthesis and Reaction Prediction (Nature 2018)](https://www.nature.com/articles/nature25978) | `neuralsym` | MIT |
| LocalRetro | [Deep Retrosynthetic Reaction Prediction using Local Reactivity and Global Attention (JACS Au 2021)](https://pubs.acs.org/doi/10.1021/jacsau.1c00246) | `rdenv` | CC BY-NC-SA 4.0 |
| Chemformer | [Chemformer: A Pre-Trained Transformer for Computational Chemistry (Machine Learning: Science and Technology 2022)](https://iopscience.iop.org/article/10.1088/2632-2153/ac3ffb) | `chemformer` | Apache 2.0 |
| RetroBridge | [RetroBridge: Modeling Retrosynthesis with Markov Bridges (ICLR 2024)](https://openreview.net/forum?id=770DetV8He) | `retrobridge` | CC BY-NC 4.0 |
| RSGPT | [Retrosynthesis prediction with an interpretable deep-learning framework based on molecular assembly tasks (Nature Communications 2025)](https://www.nature.com/articles/s41467-025-62308-6) | `gpt` | MIT |

## Current Results

### Quick Validation (50 samples, beam_size=10)

| Model | Params | Top-1 | Top-5 | Top-10 |
|-------|--------|-------|-------|--------|
| neuralsym | 32.48M | 44.00% | 68.00% | 74.00% |
| LocalRetro | 8.65M | 56.00% | 88.00% | 92.00% |
| RetroBridge | 4.62M | 20.00% | 42.00% | 48.00% |
| Chemformer | ~45M | 88.00% | 94.00% | 94.00% |
| RSGPT | ~1.6B | 78.00% | 94.00% | 98.00% |

### Full Benchmark (1000 samples, top_k=50)

| Model | Params | Top-1 | Top-5 | Top-10 | Top-50 | Duration (s) | Energy (Wh) | CO2 (g) | Peak GPU (MB) |
|-------|--------|-------|-------|--------|--------|--------------|-------------|---------|---------------|
| neuralsym | 32.48M | 43.00% | 67.00% | 72.40% | 74.00% | 192.3 | 21.16 | 8.47 | 504 |
| LocalRetro | 8.65M | 52.50% | 83.90% | 90.40% | 94.90% | 401.5 | 41.31 | 16.52 | 154 |
| RetroBridge | 4.62M | - | - | - | - | - | - | - | - |
| Chemformer | ~45M | - | - | - | - | - | - | - | - |
| RSGPT | ~1.6B | - | - | - | - | - | - | - | - |

*Hardware: NVIDIA RTX 5000 Ada (32GB), Intel Xeon Platinum 8558 (192 cores), 503GB RAM*

## Usage

```python
from Retrosynthesis.evaluate import load_test_data, evaluate, METRICS

# Load test data
test_cases = load_test_data(limit=100)

# Run model inference
from Retrosynthesis.LocalRetro.Inference import run
predictions = [run(tc['product'], top_k=50) for tc in test_cases]

# Evaluate
results = evaluate(predictions, test_cases, metrics=['top_10', 'top_50'])
print(f"Top-10: {results['top_10']*100:.2f}%")
print(f"Top-50: {results['top_50']*100:.2f}%")
```

## Adding a New Model

See `/add-model Retrosynthesis <ModelName>` skill or `../.claude/skills/add-model.md`.
