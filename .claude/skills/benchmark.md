# Skill: Run Benchmark

Run a carbon-tracked benchmark on a model.

## Usage
```
/benchmark [model] [--limit N] [--track_carbon]
```

## Examples
```
/benchmark LocalRetro --limit 100 --track_carbon
/benchmark neuralsym --limit 500
/benchmark MACE --limit 1000 --track_carbon
```

## Instructions

When the user invokes this skill:

1. **Identify the model and options:**
   - Model name (required)
   - `--limit N`: Number of test samples (default: full dataset)
   - `--track_carbon`: Enable carbon tracking (default: off)
   - `--metrics`: Metrics to compute (default: task defaults)
   - `--task`: Task name (default: auto-detect from model name)

2. **Activate the correct conda environment:**
   ```bash
   source /opt/conda/etc/profile.d/conda.sh
   conda activate <env_name>
   ```

   Known environment mappings (Retro):
   - neuralsym -> `neuralsym`
   - LocalRetro -> `rdenv`
   - RetroBridge -> `retrobridge`
   - Chemformer -> `chemformer`
   - RSGPT -> `gpt`

   For other tasks, check the model's `environment.yml` or `benchmarks/run.sh`.

3. **Run the benchmark:**
   ```bash
   python benchmarks/run_benchmark.py \
       --task <Task> \
       --model <model_name> \
       --limit <N> \
       --track_carbon \
       --output benchmarks/results/<Task>/<model>_<N>.json
   ```

4. **Report results:**
   - Show accuracy metrics (task-specific)
   - Show carbon metrics (duration, energy, CO2, peak GPU memory)
   - Note the output file location
   - Update the task README.md with results

5. **Generate plots:**
   After all models have been benchmarked, generate accuracy vs cost plots:
   ```bash
   python benchmarks/plot_results.py --task <Task> --combined
   ```
   Or use the `/plot` skill. See `.claude/skills/plot.md` for details.

## Notes
- Always run from the repository root directory
- Results are saved to `benchmarks/results/<Task>/<model>_<N>.json`
- Use `--limit` for quick tests to avoid long runtimes
- For parallel GPU execution on different models, use `CUDA_VISIBLE_DEVICES=<N>` prefix to assign each model to a different GPU
