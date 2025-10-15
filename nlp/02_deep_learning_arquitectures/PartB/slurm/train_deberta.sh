#!/bin/bash

#SBATCH --job-name=mdeberta-ft
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=12:00:00

#SBATCH --chdir=/home/est_posgrados_diego.paniagua/deep_learning_arquitectures/PartB
#SBATCH --output=/home/est_posgrados_diego.paniagua/deep_learning_arquitectures/PartB/logs/slurm-%x-%j.out
#SBATCH --error=/home/est_posgrados_diego.paniagua/deep_learning_arquitectures/PartB/errors/slurm-%x-%j.err

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=diego.paniagua@cimat.mx

set -e

MODEL_PATH=${1:-"/home/est_posgrados_diego.paniagua/deep_learning_arquitectures/PartB/models/mdeberta-v3-base"}
EPOCHS=${2:-4}
BATCH_SIZE=${3:-16}
LR=${4:-3e-5}
MAX_LENGTH=${5:-256}
GRAD_ACCUM=${6:-2}
WARMUP=${7:-0.1}

mkdir -p logs models

source /home/est_posgrados_diego.paniagua/miniconda3/etc/profile.d/conda.sh
conda activate nlp-t2

echo "Fine-tuning Transformer con modelo base $MODEL_PATH"

torchrun --nproc_per_node=2 src/train_transformer_classifier.py \
    --data_file data/MeIA_2025_train.csv \
    --model_path "$MODEL_PATH" \
    --save_dir models/mdeberta_cls \
    --log_file logs/train_mdeberta.log \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --max_length "$MAX_LENGTH" \
    --grad_accum "$GRAD_ACCUM" \
    --warmup_ratio "$WARMUP"

echo "Fine-tuning completado."
