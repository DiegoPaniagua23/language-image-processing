#!/bin/bash

#SBATCH --job-name=train-word-level
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=08:00:00

#SBATCH --chdir=/home/est_posgrados_diego.paniagua/deep_learning_arquitectures/PartA
#SBATCH --output=/home/est_posgrados_diego.paniagua/deep_learning_arquitectures/PartA/logs/slurm-%x-%j.out
#SBATCH --error=/home/est_posgrados_diego.paniagua/deep_learning_arquitectures/PartA/errors/slurm-%x-%j.err

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=diego.paniagua@cimat.mx

set -e

ARCH_LIST=${1:-"gru,lstm,rnn"}
EPOCHS=${2:-20}
BATCH_SIZE=${3:-128}
SEQ_LEN=${4:-128}

mkdir -p logs models

source /home/est_posgrados_diego.paniagua/miniconda3/etc/profile.d/conda.sh
conda activate nlp-t2

IFS=',' read -ra ARCHES <<< "$ARCH_LIST"

for arch in "${ARCHES[@]}"; do
    arch_trim=$(echo "$arch" | tr '[:upper:]' '[:lower:]')
    if [[ "$arch_trim" != "gru" && "$arch_trim" != "lstm" && "$arch_trim" != "rnn" ]]; then
        echo "Arquitectura '$arch' no reconocida; se omite." >&2
        continue
    fi

    SAVE_DIR="models/${arch_trim}_word"
    LOG_CSV="$SAVE_DIR/trainlog.csv"
    SUMMARY_PATH="$SAVE_DIR/summary.json"

    mkdir -p "$SAVE_DIR"

    echo "Iniciando entrenamiento word-level $arch_trim..."
    python src/train_textgen.py \
        --train_file data/canciones.txt \
        --level word \
        --arch "$arch_trim" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --seq_len "$SEQ_LEN" \
        --embedding_dim 256 \
        --hidden_size 512 \
        --num_layers 2 \
        --lr 0.001 \
        --dropout 0.3 \
        --grad_clip 1.0 \
        --min_freq 2 \
        --lowercase \
        --use_multigpu \
        --save_dir "$SAVE_DIR" \
        --log_csv "$LOG_CSV" \
        --summary_path "$SUMMARY_PATH" \
        --sample_every 1 \
        --sample_len 60

    echo "Entrenamiento word-level $arch_trim completado."
done

echo "Todos los entrenamientos word-level finalizaron."
