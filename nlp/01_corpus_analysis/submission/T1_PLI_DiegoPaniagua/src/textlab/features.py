
import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder

def docs_from_tokens(df:pd.DataFrame, col:str="tokens_clean"):
    return df[col].apply(lambda toks: " ".join(toks)).tolist()

def bow_matrices(df:pd.DataFrame) -> Dict[str,object]:
    docs = docs_from_tokens(df)
    vec_tf    = CountVectorizer()
    vec_tfidf = TfidfVectorizer()
    X_tf      = vec_tf.fit_transform(docs)
    X_tfidf   = vec_tfidf.fit_transform(docs)
    return dict(
        X_tf=X_tf, X_tfidf=X_tfidf,
        vec_tf=vec_tf, vec_tfidf=vec_tfidf
    )

def chi2_ranking(X, vocab, y, classes, topk:int=20):
    # Global
    scores, pvals = chi2(X, y)
    idx = np.argsort(scores)[::-1][:topk]
    global_rank = pd.DataFrame({"token": vocab[idx], "chi2": scores[idx], "p_value": pvals[idx]})
    # One-vs-rest
    frames = []
    for i, cname in enumerate(classes):
        y_bin = (y==i).astype(int)
        s, p = chi2(X, y_bin)
        j = np.argsort(s)[::-1][:topk]
        frames.append(pd.DataFrame({"class": cname, "token": vocab[j], "chi2": s[j], "p_value": p[j]}))
    ovr_rank = pd.concat(frames, ignore_index=True)
    return global_rank, ovr_rank

def bigram_bow(df:pd.DataFrame, min_df:int=3):
    docs = docs_from_tokens(df)
    vec_tf    = CountVectorizer(ngram_range=(2,2), min_df=min_df)
    vec_tfidf = TfidfVectorizer(ngram_range=(2,2), min_df=min_df)
    X_tf      = vec_tf.fit_transform(docs)
    X_tfidf   = vec_tfidf.fit_transform(docs)
    return dict(
        X_bi_tf=X_tf, X_bi_tfidf=X_tfidf,
        vec_bi_tf=vec_tf, vec_bi_tfidf=vec_tfidf
    )

def encode_labels(y_raw):
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str).values)
    return y, le.classes_
