# Retro (Retrosynthesis)

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

## Results

### Full Benchmark (1,000 samples, top_k=50)

| Model | Params | Top-1 | Top-5 | Top-10 | Top-50 | Duration (s) | Energy (Wh) | CO2 (g) | Peak GPU (MB) |
|-------|--------|-------|-------|--------|--------|--------------|-------------|---------|---------------|
| neuralsym | 32.5M | 43.0% | 67.0% | 72.4% | 74.0% | 192 | 21.2 | 8.5 | 504 |
| LocalRetro | 8.6M | 52.5% | 83.9% | 92.3% | 97.7% | 402 | 41.3 | 16.5 | 154 |
| Chemformer | 44.7M | 88.0% | 90.7% | 90.8% | 91.2% | 16,911 | 1,378 | 551 | 207 |
| RetroBridge | 4.6M | 22.1% | 39.4% | 44.5% | 51.7% | 61,974 | 4,566 | 1,966 | 479 |
| RSGPT | ~1.6B | 77.5% | 96.0% | 97.8% | 98.7% | 49,782 | 3,787 | 1,515 | 6,950 |

*Hardware: NVIDIA RTX 5000 Ada (32GB), Intel Xeon Platinum 8558 (192 cores), 503 GB RAM*

### Validation (500 samples, top_k=50)

| Model | Params | Top-1 | Top-5 | Top-10 | Top-50 | Duration (s) | Energy (Wh) | CO2 (g) | Peak GPU (MB) |
|-------|--------|-------|-------|--------|--------|--------------|-------------|---------|---------------|
| neuralsym | 32.5M | 43.6% | 68.8% | 73.0% | 74.8% | 87 | 6.6 | 2.6 | 504 |
| LocalRetro | 8.6M | 53.2% | 86.0% | 92.3% | 97.7% | 150 | 11.9 | 4.7 | 155 |
| Chemformer | 44.7M | 86.8% | 88.6% | 88.8% | 89.2% | 8,652 | 602 | 241 | 207 |
| RetroBridge | 4.6M | 19.4% | 40.6% | 47.0% | 47.0% | 6,435 | 474 | 204 | 468 |
| RSGPT | ~1.6B | 75.6% | 96.2% | 97.4% | 98.4% | 24,891 | 1,893 | 757 | 6,950 |

## Usage

```python
from Retro.evaluate import load_test_data, evaluate, METRICS

# Load test data
test_cases = load_test_data(limit=100)

# Run model inference
from Retro.LocalRetro.Inference import run
predictions = [run(tc['product'], top_k=50) for tc in test_cases]

# Evaluate
results = evaluate(predictions, test_cases, metrics=['top_10', 'top_50'])
print(f"Top-10: {results['top_10']*100:.2f}%")
print(f"Top-50: {results['top_50']*100:.2f}%")
```

## Adding a New Model

See `/add-model Retro <ModelName>` skill or `../.claude/skills/add-model.md`.
