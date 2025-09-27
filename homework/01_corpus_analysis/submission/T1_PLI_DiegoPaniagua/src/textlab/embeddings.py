
import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from gensim.models import Word2Vec
from sklearn.cluster import KMeans

def train_word2vec(sentences:List[List[str]], vector_size:int=100, window:int=5,
                   min_count:int=3, sg:int=1, epochs:int=10, seed:int=42) -> Word2Vec:
    model = Word2Vec(
        sentences=sentences, vector_size=vector_size, window=window,
        min_count=min_count, sg=sg, workers=4, epochs=epochs, seed=seed
    )
    return model

def doc_embeddings(tokens_list:List[List[str]], wv, dim:int) -> np.ndarray:
    def _emb(tokens):
        vecs = [wv[t] for t in tokens if t in wv]
        return np.mean(vecs, axis=0) if len(vecs)>0 else np.zeros(dim)
    return np.vstack([_emb(toks) for toks in tokens_list])

def kmeans_clusters(emb:np.ndarray, k:int=5, seed:int=42):
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(emb)
    return km, labels

def closest_docs_to_centroid(emb:np.ndarray, km, cluster_id:int, topn:int=5):
    centroid = km.cluster_centers_[cluster_id]
    dists = np.linalg.norm(emb - centroid, axis=1)
    idx = np.argsort(dists)[:topn]
    return idx, dists[idx]

def try_analogy(wv, positives:List[str], negatives:List[str], topn:int=10):
    V = set(wv.index_to_key)
    missing = [t for t in positives+negatives if t not in V]
    if missing:
        return None, missing
    res = wv.most_similar(positive=positives, negative=negatives, topn=topn)
    return pd.DataFrame(res, columns=["word","cosine"]), []
