# MolGen (Molecule Generation)

**Task Leader:** Gunwook Nam

Molecular generation: Generate novel molecules from distribution.

## Metrics

| Metric | Description |
|--------|-------------|
| `V·U·N` | Validity × Uniqueness × Novelty (higher is better) |
| `S.U.N` | SA_norm × Uniqueness × Novelty (higher is better) |
| `FCD` | Fréchet ChemNet Distance between generated and train distributions (lower is better) |

- `validity`: Fraction of valid SMILES strings
- `uniqueness`: Fraction of unique molecules among valid ones
- `novelty`: Fraction of molecules not in training set
- `sa_score_norm`: Normalized SA Score, (10 − SAScore) / 9, averaged over valid molecules (0=hard, 1=easy)
- `sa_score_filter`: Fraction of valid molecules with SA Score < 4 (synthesizable)

## Training Dataset

All models are trained on [ChEMBL 28](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_28/) (1.19M molecules, filtered).

- Location: `data/chembl28/{train,val,test}.txt`
- Split: 80/10/10, seed=42
- Filters: allowed atoms {C, N, O, S, F, Cl, Br}, MW ≤ 500, heavy atoms 3–50

## Models

| Model | Year | Venue | Params | Environment |
|-------|------|-------|--------|-------------|
| REINVENT | 2017 | J. Cheminf. | 4.4M | `carbon-reinvent` |
| JT-VAE | 2018 | ICML | 7.1M | `carbon-jtvae` |
| HierVAE | 2020 | ICML | 8.0M | `carbon-jtvae` |
| MolGPT | 2021 | J. Chem. Inf. Model. | 6.4M | `carbon-reinvent` |
| DiGress | 2023 | ICML | 16.2M | `carbon-digress` |
| REINVENT4 | 2024 | J. Cheminf. | 5.8M | `carbon-reinvent` |
| SmileyLlama | 2024 | arXiv | 8.0B | `carbon-smileyLlama` |
| DeFoG | 2024 | NeurIPS | 16.3M | `carbon-defog` |

## Results

### Benchmark (10,000 molecules)

| Model | Validity | Uniqueness | Novelty | V·U·N (%) | FCD ↓ | CO₂ (g) | Energy (Wh) | ICER (V·U·N) | ICER (FCD) |
|-------|----------|------------|---------|-----------|-------|---------|-------------|:------------:|:----------:|
| REINVENT (ref) | 0.9591 | 0.9996 | 0.9505 | 91.12 | 0.281 | 0.11 | 0.26 | 0 | 0 |
| JT-VAE | 1.0000 | 0.9984 | 0.9964 | 99.48 | 2.644 | 20.45 | 47.49 | +2.228 | +3.240 |
| HierVAE | 0.9853 | 0.9944 | 0.9541 | 93.48 | 3.205 | 14.39 | 33.42 | +2.102 | +3.171 |
| MolGPT | 0.9851 | 0.9992 | 0.9523 | 93.73 | 0.839 | 1.85 | 4.30 | +1.211 | +1.698 |
| DiGress | 0.8019 | 1.0000 | 0.9958 | 79.85 | 1.770 | 391.99 | 910.41 | +3.606 | +4.348 |
| **REINVENT4** | **0.9860** | **0.9997** | **0.9604** | **94.67** | **0.241** | **0.09** | **0.21** | **−0.100** | **−0.149** |
| SmileyLlama | 0.9427 | 0.9998 | 0.9980 | 94.06 | 2.410 | 22.67 | 56.68 | +2.297 | +3.244 |
| DeFoG | 1.0000 | 0.9826 | 0.9920 | 97.47 | 1.235 | 571.96 | 1429.91 | +3.684 | +4.356 |

**ICER** (Incremental Carbon-Effectiveness Ratio) = log<sub>10</sub>((c<sub>i</sub>/c<sub>o</sub>) / (m<sub>i</sub>/m<sub>o</sub>)), where the reference (o) is REINVENT (2017). Lower is more carbon-efficient; negative means better performance with less carbon. For FCD (lower=better), m<sub>i</sub>/m<sub>o</sub> is inverted to FCD<sub>o</sub>/FCD<sub>i</sub>.

### Pre-trained Benchmark (10,000 molecules)

Models evaluated using their original pre-trained checkpoints and training datasets.

| Model | Validity | Uniqueness | Novelty | SA_norm | SA_filter | V·U·N (%) | S.U.N (%) | CO₂ (g) | Energy (Wh) |
|-------|----------|------------|---------|---------|-----------|-----------|-----------|---------|-------------|
| REINVENT | 0.9438 | 0.9997 | 0.9316 | 0.7988 | 0.9193 | 87.90 | 74.40 | 0.18 | 0.41 |
| JT-VAE | 1.0000 | 0.9991 | 0.9147 | 0.8283 | 0.9801 | 91.39 | 75.70 | 10.58 | 24.58 |
| HierVAE | 0.9853 | 0.9944 | 0.9400 | 0.8343 | 0.9670 | 92.10 | 77.98 | 11.97 | 27.81 |
| MolGPT | 0.9937 | 0.9991 | 0.7771 | 0.8372 | 0.9947 | 77.15 | 65.00 | 1.07 | 2.49 |
| DiGress | 0.8759 | 0.9995 | 0.9417 | 0.8362 | 0.9855 | 82.45 | 78.71 | 175.35 | 407.26 |
| **REINVENT4** | **0.9806** | **0.9999** | **0.9603** | 0.7878 | 0.9099 | **94.16** | 75.65 | **0.07** | **0.17** |
| SmileyLlama | 0.9456 | 1.0000 | 0.9968 | 0.7800 | 0.9057 | 94.26 | 77.75 | 21.79 | 54.47 |
| DeFoG | 0.9155 | 0.9996 | 0.8990 | **0.8446** | **0.9938** | 82.27 | 75.90 | 355.24 | 888.09 |

#### Pre-trained Figures

| V·U·N vs Carbon Cost | S.U.N vs Carbon Cost (Pareto) |
|:---------------------:|:-----------------------------:|
| ![VUN vs Carbon](../benchmarks/figures/MolGen/pretrained/vun_vs_carbon.png) | ![SUN Pareto vs Carbon](../benchmarks/figures/MolGen/pretrained/pareto_sun_vs_carbon.png) |

| V·U·N vs Energy Cost | S.U.N vs Energy Cost (Pareto) |
|:--------------------:|:-----------------------------:|
| ![VUN vs Energy](../benchmarks/figures/MolGen/pretrained/vun_vs_energy.png) | ![SUN Pareto vs Energy](../benchmarks/figures/MolGen/pretrained/pareto_sun_vs_energy.png) |

| V·U·N vs Speed | S.U.N vs Speed (Pareto) |
|:--------------:|:-----------------------:|
| ![VUN vs Speed](../benchmarks/figures/MolGen/pretrained/vun_vs_speed.png) | ![SUN Pareto vs Speed](../benchmarks/figures/MolGen/pretrained/pareto_sun_vs_speed.png) |

#### Pre-trained Carbon Efficiency Over Time

| ΔCO₂ vs Year (full scale) | ICER (V·U·N) vs Year |
|:--------------------------:|:--------------------:|
| ![Delta CO2 Full](../benchmarks/figures/MolGen/pretrained/delta_co2_vs_year_full.png) | ![ICER vs Year](../benchmarks/figures/MolGen/pretrained/icer_vs_year.png) |

### Benchmark (10,000 molecules, re-trained on ChEMBL 28)

### Figures

#### Performance vs Carbon Cost

| V·U·N vs Carbon Cost | FCD vs Carbon Cost |
|:---------------------:|:------------------:|
| ![VUN vs Carbon](results/figures/vun_vs_carbon.png) | ![FCD vs Carbon](results/figures/fcd_vs_carbon.png) |

#### Performance vs Energy Cost

| V·U·N vs Energy Cost | FCD vs Energy Cost |
|:--------------------:|:------------------:|
| ![VUN vs Energy](results/figures/vun_vs_energy.png) | ![FCD vs Energy](results/figures/fcd_vs_energy.png) |

#### Performance vs Speed

| V·U·N vs Speed | FCD vs Speed |
|:--------------:|:------------:|
| ![VUN vs Speed](results/figures/vun_vs_speed.png) | ![FCD vs Speed](results/figures/fcd_vs_speed.png) |

#### Carbon Efficiency Over Time

| ΔCO₂ vs Year (full scale) | ΔCO₂ vs Year (broken axis) |
|:--------------------------:|:--------------------------:|
| ![Delta CO2 Full](results/figures/delta_co2_vs_year_full.png) | ![Delta CO2 Wavy](results/figures/delta_co2_vs_year_wavy.png) |

| ICER vs Year |
|:------------:|
| ![ICER vs Year](results/figures/icer_vs_year.png) |
