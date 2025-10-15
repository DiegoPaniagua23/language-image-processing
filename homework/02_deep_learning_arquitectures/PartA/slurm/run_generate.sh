#!/bin/bash

#SBATCH --job-name=textgen-generate
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=01:30:00

#SBATCH --chdir=/home/est_posgrados_diego.paniagua/deep_learning_arquitectures/PartA
#SBATCH --output=/home/est_posgrados_diego.paniagua/deep_learning_arquitectures/PartA/logs/slurm-%x-%j.out
#SBATCH --error=/home/est_posgrados_diego.paniagua/deep_learning_arquitectures/PartA/errors/slurm-%x-%j.err

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=diego.paniagua@cimat.mx

set -e

# Variables de entorno para optimización
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

MODE=${1:-"all"}

DEFAULT_ARCH_LIST="gru,lstm,rnn"
DEFAULT_LEVEL_LIST="char,word"
DEFAULT_PROMPT=""
DEFAULT_MAX_LENGTH=300
DEFAULT_TEMPERATURE=0.9
DEFAULT_TOP_K=50
DEFAULT_TOP_P=0.95
DEFAULT_NUM_SAMPLES=3
LLAMA_BASE_DEFAULT="/home/est_posgrados_diego.paniagua/deep_learning_arquitectures/PartA/models/llama-3-8b"
DEFAULT_ABLATION_CONTEXTS="128,256,384"
DEFAULT_ABLATION_TEMPERATURES="0.7,0.9,1.1"
DEFAULT_ABLATION_TOP_PS="0.85,0.9,0.95"
DEFAULT_ABLATION_TOP_KS="20,40,60"
RUN_ABLATION_DEFAULT="yes"
RUN_ANALYSIS_DEFAULT="yes"

mkdir -p logs results/samples results/ablations

source /home/est_posgrados_diego.paniagua/miniconda3/etc/profile.d/conda.sh
conda activate nlp-t2

run_generation() {
    local checkpoint="$1"; shift
    local model_type="$1"; shift
    local prompt="$1"; shift
    local max_length="$1"; shift
    local temperature="$1"; shift
    local top_k="$1"; shift
    local top_p="$1"; shift
    local num_samples="$1"; shift
    local output_dir="$1"; shift
    local log_file="$1"; shift
    local metrics_file="$1"; shift
    local extra_args=("$@")

    mkdir -p "$output_dir"

    echo "Generando samples para $checkpoint (tipo: $model_type)"
    python src/generate.py \
        --checkpoint "$checkpoint" \
        --model_type "$model_type" \
        --prompt "$prompt" \
        --max_length "$max_length" \
        --temperature "$temperature" \
        --top_k "$top_k" \
        --top_p "$top_p" \
        --num_samples "$num_samples" \
        --output_dir "$output_dir" \
        --log_file "$log_file" \
        --metrics_file "$metrics_file" \
        "${extra_args[@]}"
}

is_truthy() {
    case "${1,,}" in
        ""|"0"|"no"|"false"|"off") return 1 ;;
        *) return 0 ;;
    esac
}

run_ablation() {
    local checkpoint="$1"
    local model_type="$2"
    local label="$3"
    local base_output="results/ablations/${label}"

    echo "Lanzando ablación para ${label} (${model_type})"
    local cmd=(python src/ablation_generate.py
        --checkpoint "$checkpoint"
        --model_type "$model_type"
        --contexts "$ABLATION_CONTEXTS"
        --temperatures "$ABLATION_TEMPERATURES"
        --top_ps "$ABLATION_TOP_PS"
        --top_ks "$ABLATION_TOP_KS"
        --base_output "$base_output")
    if [[ "$model_type" == "llama" ]]; then
        cmd+=(--use_lora --base_model "$LLAMA_BASE")
    fi
    "${cmd[@]}"
}

if [[ "$MODE" == "all" || "$MODE" == "ALL" ]]; then
    ARCH_LIST=${2:-$DEFAULT_ARCH_LIST}
    LEVEL_LIST=${3:-$DEFAULT_LEVEL_LIST}
    NUM_SAMPLES=${4:-$DEFAULT_NUM_SAMPLES}
    PROMPT=${5:-$DEFAULT_PROMPT}
    MAX_LENGTH=${6:-$DEFAULT_MAX_LENGTH}
    TEMPERATURE=${7:-$DEFAULT_TEMPERATURE}
    TOP_K=${8:-$DEFAULT_TOP_K}
    TOP_P=${9:-$DEFAULT_TOP_P}
    LLAMA_BASE=${10:-$LLAMA_BASE_DEFAULT}
    RUN_ABLATION=${11:-$RUN_ABLATION_DEFAULT}
    RUN_ANALYSIS=${12:-$RUN_ANALYSIS_DEFAULT}
    ABLATION_CONTEXTS=${13:-$DEFAULT_ABLATION_CONTEXTS}
    ABLATION_TEMPERATURES=${14:-$DEFAULT_ABLATION_TEMPERATURES}
    ABLATION_TOP_PS=${15:-$DEFAULT_ABLATION_TOP_PS}
    ABLATION_TOP_KS=${16:-$DEFAULT_ABLATION_TOP_KS}

    IFS=',' read -ra ARCHES <<< "$ARCH_LIST"
    IFS=',' read -ra LEVELS <<< "$LEVEL_LIST"

    declare -a ABLATION_TARGETS=()
    declare -a GENERATION_PIDS=()

    # ===== FASE 1: Generar samples de modelos RNN/LSTM/GRU (en paralelo) =====
    echo "=== FASE 1: Generando samples de modelos RNN/LSTM/GRU ==="

    for level in "${LEVELS[@]}"; do
        level_trim=$(echo "$level" | tr '[:upper:]' '[:lower:]')
        if [[ "$level_trim" != "char" && "$level_trim" != "word" ]]; then
            echo "Nivel '$level' no reconocido; se omite." >&2
            continue
        fi

        for arch in "${ARCHES[@]}"; do
            arch_trim=$(echo "$arch" | tr '[:upper:]' '[:lower:]')
            if [[ "$arch_trim" != "gru" && "$arch_trim" != "lstm" && "$arch_trim" != "rnn" ]]; then
                echo "Arquitectura '$arch' no reconocida; se omite." >&2
                continue
            fi

            CKPT="models/${arch_trim}_${level_trim}/${arch_trim}_${level_trim}_best.pt"
            if [[ ! -f "$CKPT" ]]; then
                echo "[WARN] No se encontró $CKPT; se omite." >&2
                continue
            fi

            TS=$(date +%Y%m%d_%H%M%S)
            OUT_DIR="results/samples/${arch_trim}_${level_trim}"
            LOG_FILE="logs/generate_${arch_trim}_${level_trim}_${TS}.log"
            METRICS_FILE="$OUT_DIR/metrics.jsonl"

            # Ejecutar en background para paralelizar (modelos pequeños)
            echo "  → Lanzando generación: ${arch_trim}_${level_trim}"
            run_generation "$CKPT" "rnn" "$PROMPT" "$MAX_LENGTH" "$TEMPERATURE" "$TOP_K" "$TOP_P" "$NUM_SAMPLES" "$OUT_DIR" "$LOG_FILE" "$METRICS_FILE" &
            GENERATION_PIDS+=($!)
            ABLATION_TARGETS+=("rnn|$CKPT|${arch_trim}_${level_trim}")

            # Limitar concurrencia a 4 jobs (2 GPUs + buffer)
            if [[ ${#GENERATION_PIDS[@]} -ge 4 ]]; then
                echo "  → Esperando a que terminen jobs en progreso..."
                wait ${GENERATION_PIDS[@]}
                GENERATION_PIDS=()
            fi
        done
    done

    # Esperar a que terminen todos los jobs de RNN/LSTM/GRU
    if [[ ${#GENERATION_PIDS[@]} -gt 0 ]]; then
        echo "  → Esperando a que terminen los últimos jobs RNN..."
        wait ${GENERATION_PIDS[@]}
        GENERATION_PIDS=()
    fi
    echo "✓ FASE 1 completada"
    echo ""

    # ===== FASE 2: Generar samples de LLaMA-3 (requiere 2 GPUs) =====
    echo "=== FASE 2: Generando samples de LLaMA-3 LoRA ==="
    LLAMA_CKPT="models/llama3_lora"
    if [[ -d "$LLAMA_CKPT" ]]; then
        TS=$(date +%Y%m%d_%H%M%S)
        OUT_DIR="results/samples/llama3_lora"
        LOG_FILE="logs/generate_llama3_lora_${TS}.log"
        METRICS_FILE="$OUT_DIR/metrics.jsonl"

        echo "  → Generando con LLaMA-3 (device_map='auto' usa ambas GPUs)"
        run_generation "$LLAMA_CKPT" "llama" "$PROMPT" "$MAX_LENGTH" "$TEMPERATURE" "$TOP_K" "$TOP_P" "$NUM_SAMPLES" "$OUT_DIR" "$LOG_FILE" "$METRICS_FILE" --use_lora --base_model "$LLAMA_BASE"
        ABLATION_TARGETS+=("llama|$LLAMA_CKPT|llama3_lora")
        echo "✓ FASE 2 completada"
    else
        echo "[WARN] No se encontró el directorio $LLAMA_CKPT; no se generan muestras de LLaMA 3." >&2
    fi
    echo ""

    # ===== FASE 3: Ejecutar ablaciones (en paralelo cuando sea posible) =====
    if is_truthy "$RUN_ABLATION"; then
        echo "=== FASE 3: Ejecutando ablaciones ==="
        declare -a ABLATION_PIDS=()

        for target in "${ABLATION_TARGETS[@]}"; do
            IFS='|' read -r tgt_type tgt_ckpt tgt_label <<< "$target"

            # RNN/LSTM/GRU: ejecutar en paralelo (modelos pequeños)
            if [[ "$tgt_type" == "rnn" ]]; then
                echo "  → Lanzando ablación: ${tgt_label}"
                run_ablation "$tgt_ckpt" "$tgt_type" "$tgt_label" &
                ABLATION_PIDS+=($!)

                # Limitar concurrencia
                if [[ ${#ABLATION_PIDS[@]} -ge 4 ]]; then
                    echo "  → Esperando a que terminen ablaciones en progreso..."
                    wait ${ABLATION_PIDS[@]}
                    ABLATION_PIDS=()
                fi
            # LLaMA: ejecutar secuencialmente (requiere ambas GPUs)
            else
                # Esperar a que terminen ablaciones RNN pendientes
                if [[ ${#ABLATION_PIDS[@]} -gt 0 ]]; then
                    echo "  → Esperando a que terminen ablaciones RNN..."
                    wait ${ABLATION_PIDS[@]}
                    ABLATION_PIDS=()
                fi
                echo "  → Ejecutando ablación LLaMA-3 (secuencial, requiere 2 GPUs)"
                run_ablation "$tgt_ckpt" "$tgt_type" "$tgt_label"
            fi
        done

        # Esperar ablaciones finales
        if [[ ${#ABLATION_PIDS[@]} -gt 0 ]]; then
            echo "  → Esperando a que terminen las últimas ablaciones..."
            wait ${ABLATION_PIDS[@]}
        fi
        echo "✓ FASE 3 completada"
    else
        echo "[INFO] Ablaciones deshabilitadas por configuración."
    fi
    echo ""

    # ===== FASE 4: Análisis consolidado =====
    if is_truthy "$RUN_ANALYSIS"; then
        echo "=== FASE 4: Consolidando métricas ==="
        python src/analyze_partA.py --models_dir models --results_dir results/analysis --samples_dir results/samples
        echo "✓ FASE 4 completada"
    else
        echo "[INFO] Análisis deshabilitado por configuración."
    fi

    echo ""
    echo "========================================="
    echo " PIPELINE DE GENERACIÓN COMPLETADO
    echo "========================================="
    exit 0
fi

# Modo individual (compatibilidad)
CHECKPOINT=${1:-"models/llama3_lora"}
MODEL_TYPE=${2:-"llama"}
PROMPT=${3:-$DEFAULT_PROMPT}
MAX_LENGTH=${4:-$DEFAULT_MAX_LENGTH}
TEMPERATURE=${5:-$DEFAULT_TEMPERATURE}
TOP_K=${6:-$DEFAULT_TOP_K}
TOP_P=${7:-$DEFAULT_TOP_P}
NUM_SAMPLES=${8:-$DEFAULT_NUM_SAMPLES}
LOG_FILE=${9:-"logs/generate_$(date +%Y%m%d_%H%M%S).log"}
OUTPUT_DIR=${10:-"results/samples/custom"}
METRICS_FILE=${11:-"${OUTPUT_DIR}/metrics.jsonl"}
USE_LORA_FLAG=${12:-"--use_lora"}
BASE_MODEL=${13:-$LLAMA_BASE_DEFAULT}

EXTRA_ARGS=()
if [[ "$USE_LORA_FLAG" == "--use_lora" ]]; then
    EXTRA_ARGS=(--use_lora --base_model "$BASE_MODEL")
fi

run_generation "$CHECKPOINT" "$MODEL_TYPE" "$PROMPT" "$MAX_LENGTH" "$TEMPERATURE" "$TOP_K" "$TOP_P" "$NUM_SAMPLES" "$OUTPUT_DIR" "$LOG_FILE" "$METRICS_FILE" "${EXTRA_ARGS[@]}"

echo "Generación individual completada. Samples en $OUTPUT_DIR"

if is_truthy "$RUN_ANALYSIS_DEFAULT"; then
    echo "Recuerda ejecutar analyze_partA.py manualmente si requieres consolidar métricas."
fi
