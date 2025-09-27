
import os
import pandas as pd
from collections import Counter
from typing import List, Tuple

def _load_spacy():
    import spacy
    try:
        return spacy.load("es_core_news_sm")
    except Exception as e:
        raise RuntimeError("spaCy model 'es_core_news_sm' no encontrado. Instala con: "
                           "`python -m spacy download es_core_news_sm`") from e

def get_pos_tags(text:str) -> List[str]:
    nlp = _load_spacy()
    doc = nlp(text if isinstance(text,str) else "")
    return [t.pos_ for t in doc]

def add_pos_columns(df: pd.DataFrame) -> pd.DataFrame:
    nlp = _load_spacy()
    df = df.copy()
    df["pos_seq"] = df["Review_clean"].apply(lambda t: [tok.pos_ for tok in nlp(t if isinstance(t,str) else "")])
    df["pos_4grams"] = df["pos_seq"].apply(lambda seq: [tuple(seq[i:i+4]) for i in range(len(seq)-3)])
    return df

def top_pos_4grams_per_class(df:pd.DataFrame, class_col:str, k:int=20) -> pd.DataFrame:
    frames = []
    for val, g in df.groupby(class_col, dropna=False):
        c = Counter()
        for ngrams in g["pos_4grams"]:
            c.update(ngrams)
        topk = c.most_common(k)
        tmp = pd.DataFrame(topk, columns=["pos_4gram","freq"])
        tmp.insert(0, "class_col", class_col)
        tmp.insert(1, "class_value", val)
        tmp["rank"] = range(1, len(tmp)+1)
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True)
