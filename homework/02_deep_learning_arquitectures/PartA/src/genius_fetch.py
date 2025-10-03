"""
genius_fetch.py
Descarga letras desde Genius por artista usando lyricsgenius.
Guarda: JSONL (raw por canción) + CSV meta rápido.
Opciones: --artists "A,B,..." o --artists_file; --max_per_artist; --lang en|es|any
Requiere: export GENIUS_ACCESS_TOKEN=...
"""
import os, json, argparse, time, re
from pathlib import Path
import pandas as pd

# libs externas
try:
    from langdetect import detect
except Exception:
    detect = None
from unidecode import unidecode
import lyricsgenius as lg

def guess_lang(text):
    if text is None or len(text.strip()) < 20:
        return "unk"
    if detect is None:
        return "unk"
    try:
        # quitar encabezados tipo [Chorus] para detectar mejor
        t = re.sub(r"\[[^\]]+\]", " ", text)
        t = re.sub(r"\s+", " ", t).strip()
        return detect(t)
    except Exception:
        return "unk"

def load_artists(args):
    arts = []
    if args.artists:
        arts += [a.strip() for a in args.artists.split(",") if a.strip()]
    if args.artists_file:
        with open(args.artists_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    arts.append(s)
    return list(dict.fromkeys(arts))  # dedupe, keep order

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artists", type=str, default="", help="Lista separada por comas")
    ap.add_argument("--artists_file", type=str, help="Archivo con 1 artista por línea")
    ap.add_argument("--max_per_artist", type=int, default=10)
    ap.add_argument("--lang", type=str, default="any", choices=["en","es","any"])
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--out_meta",  type=str, required=True)
    args = ap.parse_args()

    token = os.environ.get("GENIUS_ACCESS_TOKEN")
    if not token:
        raise SystemExit("ERROR: Define GENIUS_ACCESS_TOKEN en el entorno.")

    artists = load_artists(args)
    if not artists:
        raise SystemExit("ERROR: Proporciona --artists o --artists_file")

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_meta).parent.mkdir(parents=True, exist_ok=True)

    genius = lg.Genius(
        token,
        timeout=15,
        retries=3,
        sleep_time=0.7,
        skip_non_songs=True,
        excluded_terms=["(Remix)","(Live)","(Snippet)","(Demo)","Skit"],
        remove_section_headers=False,  # headers se limpian después
        verbose=False,
    )

    seen_titles = set()
    records = []
    total_kept = 0

    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for a in artists:
            print(f"[INFO] Buscando: {a}")
            try:
                art = genius.search_artist(a, max_songs=args.max_per_artist,
                                           sort="popularity", include_features=False)
            except Exception as e:
                print(f"[WARN] Falló search_artist({a}): {e}")
                continue

            if not art or not art.songs:
                print(f"[WARN] Sin canciones para {a}")
                continue

            kept_a = 0
            for s in art.songs:
                title = s.title.strip() if getattr(s, "title", None) else "Unknown"
                key = f"{a}|{title}".lower()
                if key in seen_titles:
                    continue

                lyrics = getattr(s, "lyrics", None)
                url = getattr(s, "url", None)

                if not lyrics or len(lyrics.strip()) < 50:
                    continue

                lang = guess_lang(lyrics)
                if args.lang in ("en","es"):
                    if lang != args.lang:
                        continue

                rec = {
                    "artist": a,
                    "title": title,
                    "url": url,
                    "lang": lang,
                    "lyrics_raw": lyrics
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                seen_titles.add(key)
                kept_a += 1
                total_kept += 1

                # meta para CSV
                n_chars = len(lyrics)
                n_words = len(re.findall(r"\S+", lyrics))
                records.append({
                    "artist": a, "title": title, "lang": lang, "url": url,
                    "n_chars": n_chars, "n_words": n_words
                })

                if kept_a >= args.max_per_artist:
                    break

            print(f"[OK] {a}: {kept_a} canciones guardadas.")

    pd.DataFrame(records).to_csv(args.out_meta, index=False)
    print(f"[DONE] Guardadas {total_kept} canciones en {args.out_jsonl}")
    print(f"[DONE] Meta CSV: {args.out_meta}")

if __name__ == "__main__":
    main()
