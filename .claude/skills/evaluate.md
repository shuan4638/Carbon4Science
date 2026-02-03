# Skill: Run Evaluation

Run task-specific evaluation on predictions.

## Usage
```
/evaluate [task]
```

## Examples
```
/evaluate Retrosynthesis
/evaluate MolGen
/evaluate MatGen
/evaluate MLIP
```

## Instructions

When the user invokes this skill:

### For Retrosynthesis

1. **Available metrics:** `top_1`, `top_5`, `top_10`, `top_50`
2. **Test data:** `Retrosynthesis/data/test_demapped.csv`
3. **Run evaluation:**
   ```python
   from Retrosynthesis.evaluate import load_test_data, evaluate, METRICS

   test_cases = load_test_data(limit=100)
   predictions = [model.run(tc['product'], top_k=50) for tc in test_cases]
   results = evaluate(predictions, test_cases)
   ```

### For MolGen

1. **Available metrics:** `validity`, `uniqueness`, `novelty`, `diversity`
2. **Run evaluation:**
   ```python
   from MolGen.evaluate import evaluate, METRICS

   results = evaluate(generated_smiles, reference_smiles=train_smiles)
   ```

### For MatGen

1. **Available metrics:** `validity`, `uniqueness`, `stability`, `coverage`
2. **Run evaluation:**
   ```python
   from MatGen.evaluate import evaluate, METRICS

   results = evaluate(generated_structures, reference_structures=train_structures)
   ```

### For MLIP

1. **Available metrics:** `energy_mae`, `force_mae`, `force_cosine`, `stress_mae`
2. **Run evaluation:**
   ```python
   from MLIP.evaluate import evaluate, METRICS

   results = evaluate(predictions, test_cases)
   ```

## Test Data Locations

- Retrosynthesis: `Retrosynthesis/data/test_demapped.csv`
- MolGen: `MolGen/data/` (to be added)
- MatGen: `MatGen/data/` (to be added)
- MLIP: `MLIP/data/` (to be added)

## Notes
- Each task defines its own metrics in `<Task>/evaluate.py`
- Use `METRICS` constant to see available metrics for a task
- Test data is loaded via `load_test_data(limit=N)`
