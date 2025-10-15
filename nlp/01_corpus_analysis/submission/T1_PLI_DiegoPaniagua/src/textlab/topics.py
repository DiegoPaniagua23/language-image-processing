
import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif

def lsa_topics(df, n_topics:int=50, min_df:int=5):
    docs = df["tokens_clean"].apply(lambda t: " ".join(t)).tolist()
    vec = TfidfVectorizer(min_df=min_df)
    X = vec.fit_transform(docs)
    svd = TruncatedSVD(n_components=n_topics, random_state=42)
    Z = svd.fit_transform(X)
    terms = np.array(vec.get_feature_names_out())
    return dict(Z=Z, svd=svd, terms=terms, tfidf_vec=vec, X=X)

def topic_top_terms(svd, terms, top_terms:int=10) -> pd.DataFrame:
    rows = []
    for i, comp in enumerate(svd.components_):
        idx = np.argsort(comp)[::-1][:top_terms]
        rows.append({"topic": i, "terms": ", ".join(terms[idx])})
    return pd.DataFrame(rows)

def topic_informativeness_mi(Z, y, top:int=10):
    mi = mutual_info_classif(Z, y, discrete_features=False, random_state=42)
    dfmi = pd.DataFrame({"topic": np.arange(Z.shape[1]), "MI": mi}).sort_values("MI", ascending=False).reset_index(drop=True)
    return dfmi.head(top), dfmi

def encode_labels(df, class_col:str):
    le = LabelEncoder()
    y = le.fit_transform(df[class_col].astype(str).values)
    return y, le.classes_
