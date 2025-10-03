#!/usr/bin/env bash
set -euo pipefail

# ======================================================
# Configuración rápida (ajústala si lo necesitas)
# ======================================================
PY=python
TRAIN_PY="PartA/src/train_textgen.py"
CORPUS="PartA/data/canciones.txt"

# --- Modo rápido (smoke test) ---
EPOCHS=1
BATCH=64
SEQ_CHAR=64              # secuencia para nivel carácter
SEQ_WORD=20              # secuencia para nivel palabra
MIN_FREQ=3               # recortar vocab en word-level
LOWERCASE="--lowercase"  # quita esto si NO quieres minúsculas
MAX_TRAIN_BATCHES=12     # limita batches para terminar rápido
MAX_VAL_BATCHES=3
PROGRESS_EVERY=4
SAMPLE_EVERY=1
SAMPLE_LEN_CHAR=120
SAMPLE_LEN_WORD=60
SAMPLE_TEMP=0.9

# Si quieres corridas más largas, cambia a:
# EPOCHS=10; MAX_TRAIN_BATCHES=; MAX_VAL_BATCHES=; PROGRESS_EVERY=50; SEQ_CHAR=128; SEQ_WORD=32; MIN_FREQ=1

# ------------------------------------------------------
# Comprobaciones básicas
# ------------------------------------------------------
if [[ ! -f "$TRAIN_PY" ]]; then
  echo "[ERROR] No se encuentra $TRAIN_PY"; exit 1
fi
if [[ ! -f "$CORPUS" ]]; then
  echo "[ERROR] No se encuentra el corpus $CORPUS"; exit 1
fi

mkdir -p PartA/models PartA/logs

run_char () {
  local ARCH="$1"
  local SAVE_DIR="PartA/models/${ARCH}_char"
  local LOG_CSV="${SAVE_DIR}/trainlog.csv"
  mkdir -p "$SAVE_DIR"
  echo "=== [CHAR] ${ARCH} ==="
  $PY "$TRAIN_PY" \
    --level char --arch "$ARCH" \
    --train_file "$CORPUS" \
    --epochs "$EPOCHS" --batch_size "$BATCH" \
    --seq_len "$SEQ_CHAR" \
    --save_dir "$SAVE_DIR" \
    --log_csv  "$LOG_CSV" \
    --max_train_batches "${MAX_TRAIN_BATCHES:-}" \
    --max_val_batches   "${MAX_VAL_BATCHES:-}" \
    --progress_every    "${PROGRESS_EVERY:-0}" \
    --sample_every "$SAMPLE_EVERY" \
    --sample_len  "$SAMPLE_LEN_CHAR" \
    --sample_temp "$SAMPLE_TEMP"
  echo
}

run_word () {
  local ARCH="$1"
  local SAVE_DIR="PartA/models/${ARCH}_word"
  local LOG_CSV="${SAVE_DIR}/trainlog.csv"
  mkdir -p "$SAVE_DIR"
  echo "=== [WORD] ${ARCH} ==="
  $PY "$TRAIN_PY" \
    --level word --arch "$ARCH" \
    --train_file "$CORPUS" \
    --epochs "$EPOCHS" --batch_size "$BATCH" \
    --seq_len "$SEQ_WORD" \
    --min_freq "$MIN_FREQ" $LOWERCASE \
    --save_dir "$SAVE_DIR" \
    --log_csv  "$LOG_CSV" \
    --max_train_batches "${MAX_TRAIN_BATCHES:-}" \
    --max_val_batches   "${MAX_VAL_BATCHES:-}" \
    --progress_every    "${PROGRESS_EVERY:-0}" \
    --sample_every "$SAMPLE_EVERY" \
    --sample_len  "$SAMPLE_LEN_WORD" \
    --sample_temp "$SAMPLE_TEMP"
  echo
}

# ======================================================
# Lanzamientos
# ======================================================
echo ">>> Iniciando corridas smoke-test en CPU (char + word, RNN/GRU/LSTM)"
echo "Corpus: $CORPUS"
echo

# CHAR-LEVEL
run_char rnn
run_char gru
run_char lstm

# WORD-LEVEL
run_word rnn
run_word gru
run_word lstm

echo ">>> Listo. Revisa PartA/models/*/{trainlog.csv, *_best.pt, sample_epoch01.txt}"
