#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
genius_fetch.py
Descarga letras desde Genius por artista usando lyricsgenius.
Guarda: JSONL (raw por canción) + CSV meta.
Opciones: --artists "A,B,..." o --artists_file; --max_per_artist; --lang en|es|any
Requiere: export GENIUS_ACCESS_TOKEN=...
"""

import os
import re
import json
import argparse
import pandas as pd
import lyricsgenius as lg
from pathlib import Path

# Libs externas
try:
    from langdetect import detect, LangDetectException
except ImportError:
    detect = None
    LangDetectException = Exception

def guess_lang(text):
    """
    Intenta detectar el idioma del texto usando langdetect.

    Args:
        text: Texto a analizar (str) o None
    Returns:
        str: "en", "es" o "unk" (desconocido)
    """
    if text is None or len(text.strip()) < 20:
        return "unk"
    if detect is None:
        return "unk"
    try:
        # quitar encabezados tipo [Chorus] para detectar mejor
        t = re.sub(r"\[[^\]]+\]", " ", text)
        t = re.sub(r"\s+", " ", t).strip()
        return detect(t)
    except (LangDetectException, Exception):
        return "unk"

def load_artists(args):
    """
    Carga la lista de artistas desde un archivo.

    Args:
        args: Argumentos de línea de comandos (Namespace)
    Returns:
        list: Lista de artistas (sin duplicados, en orden)
    """
    arts = []
    if args.artists:
        arts += [a.strip() for a in args.artists.split(",") if a.strip()]

    if args.artists_file:
        artists_file = Path(args.artists_file)

        if not artists_file.exists():
            raise FileNotFoundError(f"ERROR: No se encuentra {artists_file}")

        with open(artists_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    arts.append(s)

    return list(dict.fromkeys(arts))

def main():
    """
    Función principal que descarga letras de canciones desde Genius API.

    El token de acceso debe estar definido como variable de entorno:
        export GENIUS_ACCESS_TOKEN="token_aqui"

    Obtén token en: https://genius.com/api-clients
    """
    # Configurar argumentos de línea de comandos
    ap = argparse.ArgumentParser(
        description="Descarga letras de canciones desde Genius API"
    )
    ap.add_argument("--artists", type=str, default="",
                    help="Lista separada por comas: 'Artista1,Artista2'")
    ap.add_argument("--artists_file", type=str,
                    help="Archivo con 1 artista por línea")
    ap.add_argument("--max_per_artist", type=int, default=10,
                    help="Máximo número de canciones por artista")
    ap.add_argument("--lang", type=str, default="any", choices=["en","es","any"],
                    help="Filtrar por idioma: en, es o any")
    ap.add_argument("--out_jsonl", type=str, required=True,
                    help="Ruta de salida para archivo JSONL con letras")
    ap.add_argument("--out_meta",  type=str, required=True,
                    help="Ruta de salida para archivo CSV con metadatos")
    args = ap.parse_args()

    # Obtener token de acceso desde variable de entorno, debe estar definido en la terminal
    token = os.environ.get("GENIUS_ACCESS_TOKEN")
    if not token:
        raise SystemExit(
            "ERROR: Define GENIUS_ACCESS_TOKEN en el entorno.\n"
            "Ejecuta: export GENIUS_ACCESS_TOKEN='token_aqui'\n"
            "Obtén token en: https://genius.com/api-clients"
        )

    # Cargar artistas desde argumentos o archivo
    artists = load_artists(args)
    if not artists:
        raise SystemExit(
            "ERROR: Proporciona al menos un artista.\n"
            "Usa --artists 'Artista1,Artista2' o --artists_file ruta/archivo.txt"
        )

    print(f"[INFO] Se buscarán letras para {len(artists)} artista(s)")

    # Crear carpetas de salida si no existen
    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_meta).parent.mkdir(parents=True, exist_ok=True)

    # Configurar cliente Genius API
    genius = lg.Genius(
        token,
        timeout=15,              # Tiempo máximo de espera por request (segundos)
        retries=3,               # Número de reintentos si falla
        sleep_time=0.7,          # Pausa entre requests para evitar rate limiting
        skip_non_songs=True,     # Omitir videos, podcasts, etc.
        excluded_terms=["(Remix)","(Live)","(Snippet)","(Demo)","Skit"],  # Términos a excluir
        remove_section_headers=False,  # Mantener headers [Chorus], [Verse], etc.
        verbose=False,           # No mostrar logs detallados
    )

    # Inicializar estructuras de datos
    seen_titles = set()      # Para evitar duplicados (mismo artista + título)
    records = []             # Lista de registros para CSV
    total_kept = 0           # Contador total de canciones guardadas

    # Abrir archivo JSONL para escritura
    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        # Iterar sobre cada artista
        for a in artists:
            print(f"[INFO] Buscando: {a}")

            # Buscar artista en Genius API
            try:
                art = genius.search_artist(
                    a,
                    max_songs=args.max_per_artist,
                    sort="popularity",       # Ordenar por popularidad
                    include_features=False   # No incluir canciones donde aparece como feature
                )
            except Exception as e:
                print(f"[WARN] Falló search_artist({a}): {e}")
                continue

            # Validar que se encontró el artista y tiene canciones
            if not art or not art.songs:
                print(f"[WARN] Sin canciones para {a}")
                continue

            # Procesar canciones del artista
            kept_a = 0  # Contador de canciones guardadas para este artista
            for s in art.songs:
                # Extraer información básica de la canción
                title = s.title.strip() if getattr(s, "title", None) else "Unknown"

                # Crear clave única para detectar duplicados
                key = f"{a}|{title}".lower()
                if key in seen_titles:
                    continue  # Saltar si ya fue procesada

                lyrics = getattr(s, "lyrics", None)
                url = getattr(s, "url", None)

                # Validar que la letra tenga contenido mínimo
                if not lyrics or len(lyrics.strip()) < 50:
                    continue

                # Detectar idioma de la letra
                lang = guess_lang(lyrics)

                # Filtrar por idioma si se especificó en o es
                if args.lang in ("en","es"):
                    if lang != args.lang:
                        continue  # Saltar si no coincide con el idioma deseado

                # Crear registro con toda la información
                rec = {
                    "artist": a,
                    "title": title,
                    "url": url,
                    "lang": lang,
                    "lyrics_raw": lyrics
                }

                # Guardar en JSONL (una línea por canción)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

                # Marcar como vista y actualizar contadores
                seen_titles.add(key)
                kept_a += 1
                total_kept += 1

                # Calcular estadísticas para el CSV de metadatos
                n_chars = len(lyrics)
                n_words = len(re.findall(r"\S+", lyrics))
                records.append({
                    "artist": a,
                    "title": title,
                    "lang": lang,
                    "url": url,
                    "n_chars": n_chars,
                    "n_words": n_words
                })

                # Limitar a max_per_artist canciones por artista
                if kept_a >= args.max_per_artist:
                    break

            print(f"[OK] {a}: {kept_a} canciones guardadas.")

    # Guardar metadatos en CSV
    pd.DataFrame(records).to_csv(args.out_meta, index=False)

    # Mostrar resumen final
    print(f"[DONE] Guardadas {total_kept} canciones en {args.out_jsonl}")
    print(f"[DONE] Meta CSV: {args.out_meta}")

if __name__ == "__main__":
    main()
