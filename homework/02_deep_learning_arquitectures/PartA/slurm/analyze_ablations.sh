#!/bin/bash

#SBATCH --job-name=analyze-partA
#SBATCH --partition=C1Mitad1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00

#SBATCH --chdir=/home/est_posgrados_diego.paniagua/deep_learning_arquitectures/PartA
#SBATCH --output=/home/est_posgrados_diego.paniagua/deep_learning_arquitectures/PartA/logs/slurm-%x-%j.out
#SBATCH --error=/home/est_posgrados_diego.paniagua/deep_learning_arquitectures/PartA/errors/slurm-%x-%j.err

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=diego.paniagua@cimat.mx

set -e

ABLATIONS_DIR=${1:-"results/ablations"}
OUTPUT_DIR=${2:-"results/ablations_analysis"}
TOP_N=${3:-10}

mkdir -p logs errors "$OUTPUT_DIR"

source /home/est_posgrados_diego.paniagua/miniconda3/etc/profile.d/conda.sh
conda activate nlp-t2

echo "Análisis de ablaciones - Parte A"
echo "Node: $(hostname)"
echo "Ablations dir: $ABLATIONS_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Top N: $TOP_N"
echo ""

python src/analyze_ablations.py \
    --ablations_dir "$ABLATIONS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --top_n "$TOP_N"

echo ""
echo "Análisis completado. Resultados en: $OUTPUT_DIR"
