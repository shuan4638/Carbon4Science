#!/bin/bash
#SBATCH -J benchmark            # job name (overridden by --job-name)
#SBATCH -o Retro/benchmarks/logs/%x.o%j  # output file (%x=job name, %j=jobID)
#SBATCH -p 5000_ada             # partition
#SBATCH -N 1                    # total number of nodes
#SBATCH -n 4                    # CPU cores
#SBATCH --mem=16G               # system RAM
#SBATCH --time=72:00:00         # max walltime
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:5000ada:1    # 1 GPU

# Usage:
#   sbatch --job-name=RSGPT Retro/benchmarks/slurm_benchmark.sh RSGPT
#   sbatch --job-name=RetroBridge Retro/benchmarks/slurm_benchmark.sh RetroBridge
#   sbatch --job-name=Chemformer Retro/benchmarks/slurm_benchmark.sh Chemformer --data Retro/data/uspto_50_chemforner.pickle
#   sbatch --job-name=RSMILES_20x Retro/benchmarks/slurm_benchmark.sh RSMILES_20x

MODEL=${1:?Usage: sbatch Retro/benchmarks/slurm_benchmark.sh MODEL_NAME [extra args...]}
shift
EXTRA_ARGS="$@"

echo "=============================================="
echo "SUBMIT_DATE           = $(date)"
echo "SLURM_JOBID           = $SLURM_JOBID"
echo "SLURM_JOB_NAME        = $SLURM_JOB_NAME"
echo "SLURM_JOB_PARTITION   = $SLURM_JOB_PARTITION"
echo "SLURM_JOB_NODELIST    = $SLURM_JOB_NODELIST"
echo "MODEL                 = $MODEL"
echo "EXTRA_ARGS            = $EXTRA_ARGS"
echo "working directory     = $SLURM_SUBMIT_DIR"
echo "=============================================="

cd $SLURM_SUBMIT_DIR

# Model-to-conda-env mapping
declare -A MODEL_ENVS=(
    ["neuralsym"]="neuralsym"
    ["LocalRetro"]="rdenv"
    ["RetroBridge"]="retrobridge"
    ["Chemformer"]="chemformer"
    ["RSGPT"]="gpt"
    ["RSMILES_1x"]="rsmiles"
    ["RSMILES_20x"]="rsmiles"
)

ENV_NAME=${MODEL_ENVS[$MODEL]}
if [ -z "$ENV_NAME" ]; then
    echo "ERROR: Unknown model '$MODEL'. Known models: ${!MODEL_ENVS[@]}"
    exit 1
fi

# Determine output file names based on test set size
# Default: 5007 samples. Chemformer with pickle: 5004.
if echo "$EXTRA_ARGS" | grep -q "chemforner.pickle"; then
    N=5004
else
    N=5007
fi

OUTPUT="Retro/results/outputs/${MODEL,,}_${N}.json"
PREDICTIONS="Retro/results/outputs/${MODEL,,}_${N}_predictions.json"

# Activate conda
source /home/hakcile/apps/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

# Set PYTHONPATH for Chemformer (needs molbart)
if [ "$MODEL" = "Chemformer" ]; then
    export PYTHONPATH="$(pwd)/Retro/Chemformer:$PYTHONPATH"
fi

echo "Starting Time is $(date)"
echo "Conda env: $ENV_NAME"
echo "Output: $OUTPUT"

PYTHONUNBUFFERED=1 python Retro/benchmarks/run_benchmark.py \
    --model "$MODEL" \
    --track_carbon \
    --output "$OUTPUT" \
    --save_predictions "$PREDICTIONS" \
    $EXTRA_ARGS

echo "Closing Time is $(date)"
