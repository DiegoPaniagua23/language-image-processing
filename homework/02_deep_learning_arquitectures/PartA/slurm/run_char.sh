#!/bin/bash

#SBATCH --job-name=textgen-char
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=08:00:00

#SBATCH --chdir=/home/est_posgrados_diego.paniagua/deep_learning_arquitectures/PartA
#SBATCH --output=/home/est_posgrados_diego.paniagua/deep_learning_arquitectures/PartA/logs/%x-%j.log
#SBATCH --error=/home/est_posgrados_diego.paniagua/deep_learning_arquitectures/PartA/errors/errors-%x-%j.log

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=diego.paniagua@cimat.mx

set -e

ARCH=${1:-"lstm"}
EPOCHS=${2:-20}
BATCH_SIZE=${3:-128}
SEQ_LEN=${4:-128}

mkdir -p logs models

# Activar el entorno conda
source /home/est_posgrados_diego.paniagua/miniconda3/etc/profile.d/conda.sh
conda activate nlp-t2

echo "Iniciando entrenamiento character-level $ARCH..."
conda run -n nlp-t2 python src/train_textgen.py \
    --train_file data/canciones.txt \
    --level char \
    --arch "$ARCH" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --seq_len "$SEQ_LEN" \
    --embedding_dim 256 \
    --hidden_size 512 \
    --num_layers 2 \
    --lr 0.001 \
    --dropout 0.3 \
    --grad_clip 1.0 \
    --use_multigpu \
    --save_dir models/"${ARCH}_char" \
    --log_csv models/"${ARCH}_char"/trainlog.csv \
    --sample_every 1 \
    --sample_len 400

echo "Entrenamiento completado."
