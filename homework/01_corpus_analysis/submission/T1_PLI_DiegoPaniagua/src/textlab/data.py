
import os
import re
import ftfy
import pandas as pd
from collections import Counter
from typing import List, Tuple, Dict

# NLTK stopwords (lazy import / download)
def _ensure_stopwords():
    import nltk
    try:
        from nltk.corpus import stopwords  # noqa
        _ = stopwords.words("spanish")
    except LookupError:
        try:
            nltk.download("stopwords")
        except Exception:
            pass

_ensure_stopwords()
from nltk.corpus import stopwords  # type: ignore

# -----------------------------
# Tokenization & Cleaning
# -----------------------------
WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+(?:[-'][A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+)?")

def tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return WORD_RE.findall(text.lower())

def remove_scrape_artifacts(text: str) -> str:
    if not isinstance(text, str):
        return text
    # Quita "Más..." o variantes al final
    return re.sub(r"Más[\\s\\W]*$", "", text)

def load_corpus(csv_path:str, text_col:str="Review") -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    df = df.dropna(subset=[text_col])
    df = df[df[text_col].astype(str).str.strip()!=""].copy()
    df.reset_index(drop=True, inplace=True)

    # Mojibake fix + scrape artifacts
    df["Review_clean"] = df[text_col].apply(ftfy.fix_text).apply(remove_scrape_artifacts)

    # Tokenize
    df["tokens"] = df["Review_clean"].apply(tokenize)

    # Stopwords
    stop_es = set(stopwords.words("spanish"))
    df["tokens_clean"] = df["tokens"].apply(lambda toks: [t for t in toks if t not in stop_es])
    df["n_tokens"] = df["tokens"].apply(len)
    df["n_tokens_clean"] = df["tokens_clean"].apply(len)
    return df

def corpus_token_stats(tokens_list:List[List[str]]) -> Dict[str,int]:
    counter = Counter()
    for toks in tokens_list:
        counter.update(toks)
    total_tokens = sum(counter.values())
    vocab_size   = len(counter)
    hapax_count  = sum(1 for _,c in counter.items() if c==1)
    return dict(total_tokens=total_tokens, vocab_size=vocab_size, hapax_count=hapax_count)
