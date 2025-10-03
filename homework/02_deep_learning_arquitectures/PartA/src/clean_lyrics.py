
"""
clean_lyrics.py
Lee JSONL (lyrics_raw) y:
 - Limpia encabezados [Chorus], [Verse], "You might also like", "Embed", etc.
 - Normaliza saltos de línea y espacios.
 - Filtra por idioma si se desea.
 - Escribe corpus unificado con <|startsong|>/<|endsong|> (una línea cada uno).
 - Crea CSV meta con longitudes y conteos.
"""
import argparse, json, re, csv
from pathlib import Path
import pandas as pd

def clean_text(t: str) -> str:
    if t is None:
        return ""
    # quitar basura común de Genius
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # líneas SOLO con headers tipo [Chorus], [Verse 2: ...]
    t = re.sub(r"^\s*\[[^\]]+\]\s*$", "", t, flags=re.MULTILINE)
    # quitar "You might also like" y derivados
    t = re.sub(r"^\s*You might also like\s*$", "", t, flags=re.MULTILINE|re.IGNORECASE)
    # quitar "Embed" suelto al final
    t = re.sub(r"\bEmbed\s*$", "", t, flags=re.IGNORECASE|re.MULTILINE)
    # colapsar repeticiones de líneas vacías
    t = re.sub(r"\n{3,}", "\n\n", t)
    # eliminar espacios al final de línea
    t = re.sub(r"[ \t]+$", "", t, flags=re.MULTILINE)
    return t.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_txt",  required=True)
    ap.add_argument("--out_csv",  required=True)
    ap.add_argument("--lang", type=str, default="any", choices=["en","es","any"])
    ap.add_argument("--min_words", type=int, default=80)
    args = ap.parse_args()

    Path(args.out_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    kept = []
    total_in = 0

    with open(args.out_txt, "w", encoding="utf-8") as fout, \
         open(args.in_jsonl, "r", encoding="utf-8") as fin:
        for line in fin:
            total_in += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue
            lang = rec.get("lang", "unk")
            if args.lang in ("en","es") and lang != args.lang:
                continue

            raw = rec.get("lyrics_raw", "")
            txt = clean_text(raw)
            n_words = len(re.findall(r"\S+", txt))
            if n_words < args.min_words:
                continue

            # escribir delimitado
            fout.write("<|startsong|>\n")
            fout.write(txt.strip() + "\n")
            fout.write("<|endsong|>\n")

            kept.append({
                "artist": rec.get("artist",""),
                "title":  rec.get("title",""),
                "lang":   lang,
                "n_words": n_words,
                "n_chars": len(txt),
                "n_lines": txt.count("\n")+1
            })

    pd.DataFrame(kept).to_csv(args.out_csv, index=False)
    print(f"[DONE] {len(kept)} / {total_in} canciones limpias → {args.out_txt}")
    print(f"[DONE] Meta CSV → {args.out_csv}")

if __name__ == "__main__":
    main()
