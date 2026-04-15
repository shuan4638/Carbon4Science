#!/bin/bash
#SBATCH -J retro_benchmark
#SBATCH -o benchmarks/logs/%x.o%j
#SBATCH -p 5000_ada
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:5000ada:1
#
# Usage (from task branch root):
#   sbatch --job-name=RSGPT benchmarks/slurm_benchmark.sh RSGPT
#   sbatch --job-name=Chemformer benchmarks/slurm_benchmark.sh Chemformer --data data/uspto_50_chemforner.pickle

MODEL=${1:?Usage: sbatch benchmarks/slurm_benchmark.sh MODEL_NAME [extra args...]}
shift
EXTRA_ARGS="$@"

echo "=============================================="
echo "SUBMIT_DATE           = $(date)"
echo "SLURM_JOBID           = $SLURM_JOBID"
echo "SLURM_JOB_NAME        = $SLURM_JOB_NAME"
echo "MODEL                 = $MODEL"
echo "EXTRA_ARGS            = $EXTRA_ARGS"
echo "=============================================="

cd $SLURM_SUBMIT_DIR

declare -A MODEL_ENVS=(
    ["neuralsym"]="neuralsym"
    ["LocalRetro"]="rdenv"
    ["MEGAN"]="megan2"
    ["Chemformer"]="chemformer"
    ["RSMILES_1x"]="rsmiles"
    ["RSMILES_20x"]="rsmiles"
    ["RetroBridge"]="retrobridge"
    ["RSGPT"]="gpt"
    ["LlaSMol"]="gpt"
)

ENV_NAME=${MODEL_ENVS[$MODEL]}
if [ -z "$ENV_NAME" ]; then
    echo "ERROR: Unknown model '$MODEL'. Known models: ${!MODEL_ENVS[@]}"
    exit 1
fi

# Determine test set size
if echo "$EXTRA_ARGS" | grep -q "chemforner.pickle"; then
    N=5004
else
    N=5007
fi

OUTPUT="results/${MODEL,,}_${N}.json"
PREDICTIONS="results/${MODEL,,}_${N}_predictions.json"

source /home/hakcile/apps/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

if [ "$MODEL" = "Chemformer" ]; then
    export PYTHONPATH="$(pwd)/Chemformer:$PYTHONPATH"
fi

echo "Starting Time is $(date)"
echo "Conda env: $ENV_NAME"
echo "Output: $OUTPUT"

mkdir -p benchmarks/logs results

PYTHONUNBUFFERED=1 python benchmarks/run_benchmark.py \
    --model "$MODEL" \
    --track_carbon \
    --output "$OUTPUT" \
    --save_predictions "$PREDICTIONS" \
    $EXTRA_ARGS

echo "Closing Time is $(date)"
