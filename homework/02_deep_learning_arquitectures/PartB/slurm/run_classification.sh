#!/bin/bash

#SBATCH --job-name=cls-pipeline
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=06:00:00

#SBATCH --chdir=/home/est_posgrados_diego.paniagua/deep_learning_arquitectures/PartB
#SBATCH --output=/home/est_posgrados_diego.paniagua/deep_learning_arquitectures/PartB/logs/slurm-%x-%j.out
#SBATCH --error=/home/est_posgrados_diego.paniagua/deep_learning_arquitectures/PartB/errors/slurm-%x-%j.err

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=diego.paniagua@cimat.mx

set -e

MODE=${1:-"full"}
RUN_ABLATION=${2:-"yes"}
RUN_ANALYSIS=${3:-"yes"}
ABL_BATCHES=${4:-"8,16,32"}
ABL_LRS=${5:-"2e-5,3e-5,5e-5"}
ABL_EPOCHS=${6:-3}
ABL_MAXLEN=${7:-256}
ABL_GRADACC=${8:-2}
ABL_WARMUP=${9:-0.1}
ABL_WEIGHT_DECAY=${10:-0.01}
MODEL_PATH=${11:-"/home/est_posgrados_diego.paniagua/deep_learning_arquitectures/PartB/models/mdeberta-v3-base"}

mkdir -p logs results

source /home/est_posgrados_diego.paniagua/miniconda3/etc/profile.d/conda.sh
conda activate nlp-t2

is_truthy() {
    case "${1,,}" in
        ""|"0"|"no"|"false"|"off") return 1 ;;
        *) return 0 ;;
    esac
}

if is_truthy "$RUN_ABLATION"; then
    echo "Ejecutando ablaciones de Transformer"
    python src/ablation_classify.py \
        --model_path "$MODEL_PATH" \
        --data_file data/MeIA_2025_train.csv \
        --batch_sizes "$ABL_BATCHES" \
        --learning_rates "$ABL_LRS" \
        --epochs "$ABL_EPOCHS" \
        --max_length "$ABL_MAXLEN" \
        --grad_accum "$ABL_GRADACC" \
        --warmup_ratio "$ABL_WARMUP" \
        --weight_decay "$ABL_WEIGHT_DECAY"
else
    echo "[INFO] Ablaciones deshabilitadas."
fi

if is_truthy "$RUN_ANALYSIS"; then
    echo "Generando análisis consolidado"
    python src/analyze_partB.py \
        --models_dir models \
        --results_dir results/analysis \
        --confusion_dir models
else
    echo "[INFO] Análisis deshabilitado."
fi

echo "Pipeline de clasificación completado."
