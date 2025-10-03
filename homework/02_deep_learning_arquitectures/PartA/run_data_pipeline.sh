#!/usr/bin/env bash
set -euo pipefail

# === Configuración rápida ===
ARTISTS_FILE="PartA/data/artists_en.txt"
RAW_JSONL="PartA/data/rap_raw_en.jsonl"
RAW_META="PartA/data/rap_raw_en_meta.csv"
CLEAN_TXT="PartA/data/canciones.txt"
CLEAN_META="PartA/data/canciones_meta.csv"

MAX_PER_ARTIST=10
LANG="en"           # en | es | any
MIN_WORDS=80

# === Comprobaciones previas ===
if [[ -z "${GENIUS_ACCESS_TOKEN:-}" ]]; then
  echo "[ERROR] La variable de entorno GENIUS_ACCESS_TOKEN no está definida."
  echo "        Exporta tu token antes de correr:  export GENIUS_ACCESS_TOKEN='xxxxxxxx'"
  exit 1
fi

if [[ ! -f "$ARTISTS_FILE" ]]; then
  echo "[ERROR] No existe el archivo de artistas: $ARTISTS_FILE"
  exit 1
fi

# === Crear carpetas si no existen ===
mkdir -p PartA/data PartA/logs PartA/models PartA/results PartA/src

# === Paso 1: Descarga RAW desde Genius ===
echo "[1/3] Descargando canciones RAW con lyricsgenius..."
python PartA/src/genius_fetch.py \
  --artists_file "$ARTISTS_FILE" \
  --max_per_artist "$MAX_PER_ARTIST" \
  --lang "$LANG" \
  --out_jsonl "$RAW_JSONL" \
  --out_meta  "$RAW_META"

echo "[INFO] Canciones RAW: $(wc -l < "$RAW_JSONL") (esperado ~$(wc -l < "$ARTISTS_FILE") * $MAX_PER_ARTIST)"

# === Paso 2: Limpieza + unificación con delimitadores ===
echo "[2/3] Limpiando y unificando corpus..."
python PartA/src/clean_lyrics.py \
  --in_jsonl "$RAW_JSONL" \
  --out_txt  "$CLEAN_TXT" \
  --out_csv  "$CLEAN_META" \
  --lang "$LANG" \
  --min_words "$MIN_WORDS"

# === Paso 3: Validaciones rápidas ===
echo "[3/3] Validaciones:"
COUNT_SONGS=$(grep -c "<|startsong|>" "$CLEAN_TXT" || true)
echo "   - Canciones delimitadas: $COUNT_SONGS"
echo "   - Primeras líneas del corpus:"
head -n 20 "$CLEAN_TXT" | sed 's/^/     /'
echo "   - Meta (top 5):"
python - << 'PY'
import pandas as pd
m = pd.read_csv("PartA/data/canciones_meta.csv")
print(m.groupby('artist').size().sort_values(ascending=False).head(10))
print("\nPreview:")
print(m.head(5))
PY

echo "[DONE] Corpus listo en: $CLEAN_TXT"
