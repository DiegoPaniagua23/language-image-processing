#!/bin/bash

#SBATCH --job-name=llama3-ft
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=24:00:00

#SBATCH --chdir=/home/est_posgrados_diego.paniagua/deep_learning_arquitectures/PartA
#SBATCH --output=/home/est_posgrados_diego.paniagua/deep_learning_arquitectures/PartA/logs/slurm-%x-%j.out
#SBATCH --error=/home/est_posgrados_diego.paniagua/deep_learning_arquitectures/PartA/errors/slurm-%x-%j.err

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=diego.paniagua@cimat.mx

set -e

MODEL_PATH=${1:-"/home/est_posgrados_diego.paniagua/deep_learning_arquitectures/PartA/models/llama-3-8b"}
EPOCHS=${2:-10}
BATCH_SIZE=${3:-2}
SEQ_LEN=${4:-256}

mkdir -p logs models

source /home/est_posgrados_diego.paniagua/miniconda3/etc/profile.d/conda.sh
conda activate nlp-t2

echo "Iniciando fine-tuning LoRA de LLaMA-3-8B"

torchrun --nproc_per_node=2 src/train_llama3_lora.py \
    --train_file data/canciones.txt \
    --base_model "$MODEL_PATH" \
    --output_dir models/llama3_lora \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --seq_len "$SEQ_LEN" \
    --grad_accum 8 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.1 \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --bits 4 \
    --sample_len 200

echo "Fine-tuning Llama-3 completado."
