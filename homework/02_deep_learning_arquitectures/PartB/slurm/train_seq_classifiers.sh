#!/bin/bash

#SBATCH --job-name=trainj-cls-seq
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=06:00:00

#SBATCH --chdir=/home/est_posgrados_diego.paniagua/deep_learning_arquitectures/PartB
#SBATCH --output=/home/est_posgrados_diego.paniagua/deep_learning_arquitectures/PartB/logs/slurm-%x-%j.out
#SBATCH --error=/home/est_posgrados_diego.paniagua/deep_learning_arquitectures/PartB/errors/slurm-%x-%j.err

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=diego.paniagua@cimat.mx

set -e

ARCH_LIST=${1:-"cnn,lstm,gru"}
EPOCHS=${2:-10}
BATCH_SIZE=${3:-64}
SEQ_LEN=${4:-200}
EMBED_DIM=${5:-200}
HIDDEN_DIM=${6:-256}
MIN_FREQ=${7:-2}
LR=${8:-0.001}

mkdir -p logs models

source /home/est_posgrados_diego.paniagua/miniconda3/etc/profile.d/conda.sh
conda activate nlp-t2

IFS=',' read -ra ARCHES <<< "$ARCH_LIST"

for arch in "${ARCHES[@]}"; do
    arch_trim=$(echo "$arch" | tr '[:upper:]' '[:lower:]')
    if [[ "$arch_trim" != "cnn" && "$arch_trim" != "rnn" && "$arch_trim" != "lstm" && "$arch_trim" != "gru" ]]; then
        echo "Arquitectura '$arch' no reconocida; se omite." >&2
        continue
    fi

    SAVE_DIR="models/${arch_trim}_cls"
    LOG_FILE="logs/train_${arch_trim}_cls.log"

    echo "Entrenando clasificador $arch_trim con 2 GPUs..."
    python src/train_text_classifier.py \
        --data_file data/MeIA_2025_train.csv \
        --model "$arch_trim" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --seq_len "$SEQ_LEN" \
        --embedding_dim "$EMBED_DIM" \
        --hidden_dim "$HIDDEN_DIM" \
        --min_freq "$MIN_FREQ" \
        --lr "$LR" \
        --save_dir models \
        --log_file "$LOG_FILE" \
        --experiment "${arch_trim}_cls"

    echo "Clasificador $arch_trim completado."

done

echo "Entrenamiento de clasificadores secuenciales finalizado."
