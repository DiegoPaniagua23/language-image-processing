
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple
from .data import corpus_token_stats

def basic_corpus_summary(df: pd.DataFrame) -> Dict[str, float]:
    stats = corpus_token_stats(df["tokens"].tolist())
    hapax_prop_over_vocab = stats["hapax_count"]/max(stats["vocab_size"],1)
    hapax_prop_over_tokens = stats["hapax_count"]/max(stats["total_tokens"],1)
    # stopwords already removed in tokens_clean; recover totals
    total_stop = (df["n_tokens"] - df["n_tokens_clean"]).sum()
    stopword_pct = 100.0 * total_stop / max(stats["total_tokens"],1)
    return {
        **stats,
        "hapax_prop_over_vocab": hapax_prop_over_vocab,
        "hapax_prop_over_tokens": hapax_prop_over_tokens,
        "total_stop": int(total_stop),
        "stopword_pct": float(stopword_pct),
    }

def per_group_stats(df: pd.DataFrame, class_col:str) -> pd.DataFrame:
    def _agg(g):
        c = Counter()
        for toks in g["tokens"]:
            c.update(toks)
        return pd.Series({
            "n_docs": len(g),
            "n_tokens": int(sum(c.values())),
            "vocab_size": int(len(c))
        })
    out = (df.groupby(class_col, dropna=False)
             .apply(_agg, include_groups=False)
             .reset_index()
             .rename(columns={class_col:"class_value"}))
    out.insert(0, "class_col", class_col)
    return out[["class_col","class_value","n_docs","n_tokens","vocab_size"]]

def build_zipf_frame(df: pd.DataFrame) -> pd.DataFrame:
    counter = Counter()
    for toks in df["tokens"]:
        counter.update(toks)
    freq_df = (
        pd.DataFrame(counter.items(), columns=["token","freq"])
        .sort_values(["freq","token"], ascending=[False, True])
        .reset_index(drop=True)
    )
    freq_df["rank"] = np.arange(1, len(freq_df)+1)
    freq_df["log_rank"] = np.log(freq_df["rank"])
    freq_df["log_freq"] = np.log(freq_df["freq"])
    return freq_df

def zipf_linfit(freq_df: pd.DataFrame) -> Dict[str,float]:
    x = freq_df["log_rank"].values
    y = freq_df["log_freq"].values
    a, b = np.polyfit(x, y, deg=1)  # y = a x + b
    y_hat = a*x + b
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res/ss_tot
    s = -a
    C = float(np.exp(b))
    f1 = int(freq_df.loc[0, "freq"])
    err_abs = abs(C - f1)
    err_rel = 100.0 * err_abs / max(f1,1)
    return dict(a=float(a), b=float(b), s=float(s), C=float(C), r2=float(r2),
                f1=float(f1), err_abs=float(err_abs), err_rel=float(err_rel))
