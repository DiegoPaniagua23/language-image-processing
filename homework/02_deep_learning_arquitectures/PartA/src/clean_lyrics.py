#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
clean_lyrics.py
Lee JSONL (lyrics_raw) y:
 - Limpia encabezados [Chorus], [Verse], "You might also like", "Embed", etc.
 - Normaliza saltos de línea y espacios.
 - Filtra por idioma si se desea.
 - Escribe corpus unificado con <|startsong|>/<|endsong|> (una línea cada uno).
 - Crea CSV meta con longitudes y conteos.
"""

import re
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional

def clean_text(t: Optional[str]) -> str:
    """
    Limpia el texto de la letra de una canción eliminando elementos no deseados.

    Operaciones de limpieza:
    - Normaliza saltos de línea (\\r\\n y \\r → \\n)
    - Elimina encabezados de sección: [Chorus], [Verse 2: Artist], etc.
    - Elimina texto de Genius: "You might also like", "Embed"
    - Reduce múltiples saltos de línea a máximo 2
    - Elimina espacios/tabs al final de cada línea

    Args:
        t: Texto crudo de la letra o None

    Returns:
        str: Texto limpio y normalizado (vacío si la entrada es None)
    """
    if t is None:
        return ""

    # Normalizar saltos de línea a formato Unix
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    # Eliminar encabezados de sección tipo [Chorus], [Verse 1: Kendrick Lamar]
    t = re.sub(r"^\s*\[[^\]]+\]\s*$", "", t, flags=re.MULTILINE)

    # Eliminar texto insertado por Genius
    t = re.sub(r"^\s*You might also like\s*$", "", t, flags=re.MULTILINE|re.IGNORECASE)
    t = re.sub(r"\bEmbed\s*$", "", t, flags=re.IGNORECASE|re.MULTILINE)

    # Reducir múltiples saltos de línea consecutivos (3 o más → 2)
    t = re.sub(r"\n{3,}", "\n\n", t)

    # Eliminar espacios y tabs al final de cada línea
    t = re.sub(r"[ \t]+$", "", t, flags=re.MULTILINE)

    return t.strip()

def main():
    """
    Función principal que procesa y limpia letras de canciones desde JSONL.

    Lee un archivo JSONL con letras crudas, las limpia, filtra por idioma
    y número mínimo de palabras, y genera:
    1. Un archivo de texto con todas las letras delimitadas por marcadores
    2. Un CSV con metadatos y estadísticas de cada canción
    """
    # Configurar argumentos de línea de comandos
    ap = argparse.ArgumentParser(
        description="Limpia y procesa letras de canciones desde archivo JSONL"
    )
    ap.add_argument("--in_jsonl", required=True,
                    help="Archivo JSONL de entrada con letras crudas")
    ap.add_argument("--out_txt", required=True,
                    help="Archivo de texto de salida con letras limpias")
    ap.add_argument("--out_csv", required=True,
                    help="Archivo CSV de salida con metadatos")
    ap.add_argument("--lang", type=str, default="any", choices=["en","es","any"],
                    help="Filtrar por idioma: en, es o any (default: any)")
    ap.add_argument("--min_words", type=int, default=80,
                    help="Mínimo de palabras para incluir canción (default: 80)")
    args = ap.parse_args()

    # Validar que el archivo de entrada existe
    in_file = Path(args.in_jsonl)
    if not in_file.exists():
        raise FileNotFoundError(f"ERROR: No se encuentra {in_file}")

    print(f"[INFO] Procesando: {in_file}")
    print(f"[INFO] Filtros: lang={args.lang}, min_words={args.min_words}")

    # Crear carpetas de salida si no existen
    Path(args.out_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    # Inicializar contadores y estructuras de datos
    kept = []              # Lista de registros para CSV
    total_in = 0           # Total de canciones leídas
    parse_errors = 0       # Canciones con error de JSON
    filtered_lang = 0      # Canciones filtradas por idioma
    filtered_words = 0     # Canciones filtradas por palabras mínimas

    # Procesar archivo JSONL línea por línea
    with open(args.out_txt, "w", encoding="utf-8") as fout, \
         open(args.in_jsonl, "r", encoding="utf-8") as fin:

        for line in fin:
            total_in += 1

            # Parsear JSON de la línea
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue

            # Filtrar por idioma si se especificó en o es
            lang = rec.get("lang", "unk")
            if args.lang in ("en", "es") and lang != args.lang:
                filtered_lang += 1
                continue

            # Limpiar letra cruda
            raw = rec.get("lyrics_raw", "")
            txt = clean_text(raw)

            # Contar palabras (cualquier secuencia de caracteres no-espacio)
            n_words = len(re.findall(r"\S+", txt))

            # Filtrar canciones muy cortas
            if n_words < args.min_words:
                filtered_words += 1
                continue

            # Escribir letra en formato delimitado
            # Formato: cada canción entre marcadores especiales (útil para LLMs)
            fout.write("<|startsong|>\n")
            fout.write(txt.strip() + "\n")
            fout.write("<|endsong|>\n")

            # Calcular estadísticas para el CSV
            n_chars = len(txt)
            n_lines = len([line for line in txt.split("\n") if line.strip()])

            # Guardar metadatos
            kept.append({
                "artist": rec.get("artist", ""),
                "title": rec.get("title", ""),
                "lang": lang,
                "n_words": n_words,
                "n_chars": n_chars,
                "n_lines": n_lines
            })

    # Guardar metadatos en CSV
    pd.DataFrame(kept).to_csv(args.out_csv, index=False)

    # Mostrar resumen detallado del procesamiento
    print(f"\n{'='*60}")
    print(f"[RESUMEN] Procesamiento completado")
    print(f"{'='*60}")
    print(f"  Total leídas:         {total_in}")
    print(f"  Errores de parsing:   {parse_errors}")
    print(f"  Filtradas por idioma: {filtered_lang}")
    print(f"  Filtradas por palabras: {filtered_words}")
    print(f"  ✓ Guardadas:          {len(kept)}")
    print(f"{'='*60}")
    print(f"[DONE] Corpus limpio → {args.out_txt}")
    print(f"[DONE] Metadatos CSV → {args.out_csv}")

    # Advertir si no se guardó ninguna canción
    if len(kept) == 0:
        print("\n[WARN] No se guardó ninguna canción. Revisa los filtros aplicados.")

if __name__ == "__main__":
    main()
