#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# run_data_pipeline.sh
# Pipeline de descarga y limpieza de letras de canciones desde Genius API
#
# Requisitos:
#   - GENIUS_ACCESS_TOKEN debe estar definido como variable de entorno
#   - Python 3.x con paquetes: lyricsgenius, langdetect, pandas
#
# Uso:
#   export GENIUS_ACCESS_TOKEN="tu_token_aqui"
#   bash PartA/run_data_pipeline.sh
# =============================================================================

# Obtener directorio del script para rutas relativas
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Pipeline: Descarga y Limpieza de Letras"
echo "============================================================"
echo "Directorio de trabajo: $SCRIPT_DIR"
echo ""

# === Configuración ===
ARTISTS_FILE="data/artists_en.txt"
RAW_JSONL="data/rap_raw_en.jsonl"
RAW_META="data/rap_raw_en_meta.csv"
CLEAN_TXT="data/canciones.txt"
CLEAN_META="data/canciones_meta.csv"

MAX_PER_ARTIST=10
LANG="en"           # en | es | any
MIN_WORDS=80

echo "[CONFIG] Parámetros:"
echo "  - Artistas:       $ARTISTS_FILE"
echo "  - Max/artista:    $MAX_PER_ARTIST"
echo "  - Idioma:         $LANG"
echo "  - Min palabras:   $MIN_WORDS"
echo ""

# === Validaciones previas ===
echo "[CHECK] Validando requisitos..."

# Verificar token de Genius
if [[ -z "${GENIUS_ACCESS_TOKEN:-}" ]]; then
  echo ""
  echo "❌ [ERROR] La variable de entorno GENIUS_ACCESS_TOKEN no está definida."
  echo ""
  echo "Solución:"
  echo "  1. Obtén tu token en: https://genius.com/api-clients"
  echo "  2. Exporta la variable:"
  echo "     export GENIUS_ACCESS_TOKEN='tu_token_aqui'"
  echo ""
  exit 1
fi
echo "  ✓ Token de Genius encontrado"

# Verificar archivo de artistas
if [[ ! -f "$ARTISTS_FILE" ]]; then
  echo ""
  echo "❌ [ERROR] No existe el archivo de artistas: $ARTISTS_FILE"
  echo ""
  exit 1
fi
NUM_ARTISTS=$(grep -v '^#' "$ARTISTS_FILE" | grep -v '^[[:space:]]*$' | wc -l)
echo "  ✓ Archivo de artistas: $NUM_ARTISTS artista(s)"

# Verificar que existen los scripts de Python
if [[ ! -f "src/genius_fetch.py" ]]; then
  echo "❌ [ERROR] No se encuentra src/genius_fetch.py"
  exit 1
fi
if [[ ! -f "src/clean_lyrics.py" ]]; then
  echo "❌ [ERROR] No se encuentra src/clean_lyrics.py"
  exit 1
fi
echo "  ✓ Scripts de Python encontrados"

# Verificar dependencias de Python
echo "  Verificando dependencias de Python..."
python -c "import lyricsgenius, langdetect, pandas" 2>/dev/null || {
  echo ""
  echo "❌ [ERROR] Faltan dependencias de Python"
  echo "Instala con: pip install lyricsgenius langdetect pandas"
  echo ""
  exit 1
}
echo "  ✓ Dependencias de Python instaladas"
echo ""

# === Crear carpetas si no existen ===
mkdir -p data logs models results

# === Inicio del pipeline ===
START_TIME=$(date +%s)

# === Paso 1: Descarga RAW desde Genius ===
echo "============================================================"
echo "[PASO 1/3] Descargando canciones desde Genius API"
echo "============================================================"
echo "Estimado: ~$((NUM_ARTISTS * MAX_PER_ARTIST)) canciones máximo"
echo "Esto puede tardar varios minutos..."
echo ""

STEP1_START=$(date +%s)

python src/genius_fetch.py \
  --artists_file "$ARTISTS_FILE" \
  --max_per_artist "$MAX_PER_ARTIST" \
  --lang "$LANG" \
  --out_jsonl "$RAW_JSONL" \
  --out_meta  "$RAW_META"

STEP1_END=$(date +%s)
STEP1_TIME=$((STEP1_END - STEP1_START))

# Verificar que se generaron los archivos
if [[ ! -f "$RAW_JSONL" ]]; then
  echo "❌ [ERROR] No se generó el archivo $RAW_JSONL"
  exit 1
fi

RAW_COUNT=$(wc -l < "$RAW_JSONL")
echo ""
echo "✓ Paso 1 completado en ${STEP1_TIME}s"
echo "✓ Canciones descargadas: $RAW_COUNT"
echo ""

# === Paso 2: Limpieza + unificación con delimitadores ===
echo "============================================================"
echo "[PASO 2/3] Limpiando y procesando letras"
echo "============================================================"
echo ""

STEP2_START=$(date +%s)

python src/clean_lyrics.py \
  --in_jsonl "$RAW_JSONL" \
  --out_txt  "$CLEAN_TXT" \
  --out_csv  "$CLEAN_META" \
  --lang "$LANG" \
  --min_words "$MIN_WORDS"

STEP2_END=$(date +%s)
STEP2_TIME=$((STEP2_END - STEP2_START))

# Verificar que se generaron los archivos
if [[ ! -f "$CLEAN_TXT" ]]; then
  echo "❌ [ERROR] No se generó el archivo $CLEAN_TXT"
  exit 1
fi

echo ""
echo "✓ Paso 2 completado en ${STEP2_TIME}s"
echo ""

# === Paso 3: Validaciones y estadísticas ===
echo "============================================================"
echo "[PASO 3/3] Validaciones y estadísticas finales"
echo "============================================================"
echo ""

COUNT_SONGS=$(grep -c "<|startsong|>" "$CLEAN_TXT" || true)
FILE_SIZE=$(du -h "$CLEAN_TXT" | cut -f1)
TOTAL_LINES=$(wc -l < "$CLEAN_TXT")

echo "[INFO] Estadísticas del corpus:"
echo "  - Canciones:      $COUNT_SONGS"
echo "  - Tamaño:         $FILE_SIZE"
echo "  - Líneas totales: $TOTAL_LINES"
echo ""

echo "[INFO] Preview del corpus (primeras 15 líneas):"
head -n 15 "$CLEAN_TXT" | sed 's/^/     /'
echo "     ..."
echo ""

echo "[INFO] Estadísticas por artista (top 10):"
python - << 'PY'
import pandas as pd
import sys

try:
    df = pd.read_csv("data/canciones_meta.csv")
    print("\n  Top 10 artistas por número de canciones:")
    top = df.groupby('artist').size().sort_values(ascending=False).head(10)
    for artist, count in top.items():
        print(f"    {count:3d} - {artist}")

    print(f"\n  Total de canciones: {len(df)}")
    print(f"  Artistas únicos: {df['artist'].nunique()}")
    print(f"  Palabras promedio: {df['n_words'].mean():.1f}")
    print(f"  Caracteres promedio: {df['n_chars'].mean():.1f}")
    print(f"  Líneas promedio: {df['n_lines'].mean():.1f}")
except Exception as e:
    print(f"  Error al procesar metadatos: {e}")
    sys.exit(1)
PY

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "✓ PIPELINE COMPLETADO EXITOSAMENTE"
echo "============================================================"
echo "Tiempo total: ${TOTAL_TIME}s ($((TOTAL_TIME / 60))m $((TOTAL_TIME % 60))s)"
echo ""
echo "Archivos generados:"
echo "  1. $RAW_JSONL   (canciones crudas)"
echo "  2. $RAW_META    (metadatos crudos)"
echo "  3. $CLEAN_TXT   (corpus limpio) ← PRINCIPAL"
echo "  4. $CLEAN_META  (metadatos limpios)"
echo ""
echo "============================================================"
