
#!/usr/bin/env python3
import os, argparse, json
import pandas as pd

from textlab.data import load_corpus
from textlab.analysis import basic_corpus_summary, per_group_stats, build_zipf_frame, zipf_linfit
from textlab.viz import plot_zipf, plot_confusion
from textlab.features import bow_matrices, chi2_ranking, bigram_bow, encode_labels
from textlab.syntax import add_pos_columns, top_pos_4grams_per_class
from textlab.embeddings import train_word2vec, doc_embeddings, kmeans_clusters, closest_docs_to_centroid
from textlab.models import experiments_70_30
from textlab.topics import lsa_topics, topic_top_terms, topic_informativeness_mi, encode_labels as enc_y_topics

def main():
    ap = argparse.ArgumentParser(description="Simple NLP pipeline (modular, minimal).")
    ap.add_argument("--csv", required=True, help="Ruta al CSV")
    ap.add_argument("--text-col", default="Review", help="Columna de texto original")
    ap.add_argument("--class-col", default="Polarity", help="Columna de clase para análisis supervisado")
    ap.add_argument("--model", default="logreg", choices=["logreg","svm"], help="Clasificador para 70/30")
    ap.add_argument("--out-dir", default="../", help="Directorio base para reports/ y data/processed/")
    args = ap.parse_args()

    base = os.path.abspath(args.out_dir)
    figs = os.path.join(base, "reports/figures")
    tabs = os.path.join(base, "reports/tables")
    proc = os.path.join(base, "data/processed")
    for d in (figs, tabs, proc): os.makedirs(d, exist_ok=True)

    # 1. Carga y limpieza
    df = load_corpus(args.csv, text_col=args.text_col)
    df.to_csv(os.path.join(proc, "df_clean_tokens.csv"), index=False)

    # 1) Descriptivos
    summary = basic_corpus_summary(df)
    pd.DataFrame([summary]).to_csv(os.path.join(tabs,"corpus_summary.csv"), index=False)

    # Por clase
    present_cols = [c for c in ["Polarity","Town","Region","Type"] if c in df.columns]
    stats_all = []
    for col in present_cols:
        stats_all.append(per_group_stats(df, col))
    if stats_all:
        stats_per_class = pd.concat(stats_all, ignore_index=True)
        stats_per_class.to_csv(os.path.join(proc,"stats_per_class.csv"), index=False)

    # 2) Zipf
    freq_df = build_zipf_frame(df)
    freq_df.to_csv(os.path.join(proc,"zipf_freqs.csv"), index=False)
    zstats = zipf_linfit(freq_df)
    pd.DataFrame([zstats]).to_csv(os.path.join(proc,"zipf_fit_stats.csv"), index=False)
    plot_zipf(freq_df, os.path.join(figs, "zipf_loglog_scatter.png"))

    # 5) BoW + chi2
    bow = bow_matrices(df)
    y_raw = df[args.class_col].astype(str).values
    y, classes = encode_labels(df[args.class_col])
    vocab_tf    = bow["vec_tf"].get_feature_names_out()
    vocab_tfidf = bow["vec_tfidf"].get_feature_names_out()

    glob_tf, ovr_tf = chi2_ranking(bow["X_tf"], vocab_tf, y, classes, topk=20)
    glob_tf.to_csv(os.path.join(tabs,"top20_chi2_global_tf.csv"), index=False)
    ovr_tf.to_csv(os.path.join(tabs,"top20_chi2_ovr_tf.csv"), index=False)

    glob_tfidf, ovr_tfidf = chi2_ranking(bow["X_tfidf"], vocab_tfidf, y, classes, topk=20)
    glob_tfidf.to_csv(os.path.join(tabs,"top20_chi2_global_tfidf.csv"), index=False)
    ovr_tfidf.to_csv(os.path.join(tabs,"top20_chi2_ovr_tfidf.csv"), index=False)

    # 6) Bigramas
    big = bigram_bow(df, min_df=3)
    vocab_bi_tf    = big["vec_bi_tf"].get_feature_names_out()
    vocab_bi_tfidf = big["vec_bi_tfidf"].get_feature_names_out()

    glob_bi_tf, ovr_bi_tf = chi2_ranking(big["X_bi_tf"], vocab_bi_tf, y, classes, topk=20)
    glob_bi_tf.to_csv(os.path.join(tabs,"top20_chi2_global_bigram_tf.csv"), index=False)
    ovr_bi_tf.to_csv(os.path.join(tabs,"top20_chi2_ovr_bigram_tf.csv"), index=False)

    glob_bi_tfidf, ovr_bi_tfidf = chi2_ranking(big["X_bi_tfidf"], vocab_bi_tfidf, y, classes, topk=20)
    glob_bi_tfidf.to_csv(os.path.join(tabs,"top20_chi2_global_bigram_tfidf.csv"), index=False)
    ovr_bi_tfidf.to_csv(os.path.join(tabs,"top20_chi2_ovr_bigram_tfidf.csv"), index=False)

    # 4) POS 4-gramas
    try:
        df_pos = add_pos_columns(df)
        for col in present_cols:
            pos_tab = top_pos_4grams_per_class(df_pos, col, k=20)
            pos_tab.to_csv(os.path.join(tabs, f"top_pos4grams_{col.lower()}.csv"), index=False)
    except RuntimeError as e:
        # spaCy no instalado; continúa sin POS
        with open(os.path.join(proc,"pos_warning.txt"), "w") as f:
            f.write(str(e))

    # 7-8) Word2Vec + Embeddings + KMeans
    try:
        from gensim.models import Word2Vec  # quick check to avoid failing environments
        w2v = train_word2vec(df["tokens_clean"].tolist(), vector_size=100, window=5, min_count=3, sg=1, epochs=10)
        emb = doc_embeddings(df["tokens_clean"].tolist(), wv=w2v.wv, dim=w2v.wv.vector_size)
        km, labels = kmeans_clusters(emb, k=5, seed=42)
        df["cluster"] = labels
        df[["Review_clean","cluster"]].to_csv(os.path.join(proc,"clusters_assignments.csv"), index=False)

        rows = []
        for c in range(5):
            idxs, dists = closest_docs_to_centroid(emb, km, c, topn=5)
            for rank, (i, d) in enumerate(zip(idxs, dists), start=1):
                rows.append({"cluster":c, "rank":rank, "doc_id":int(i), "distance":float(d), "text": df["Review_clean"].iloc[i]})
        pd.DataFrame(rows).to_csv(os.path.join(tabs,"cluster_top_docs.csv"), index=False)
    except Exception as e:
        with open(os.path.join(proc,"w2v_warning.txt"), "w") as f:
            f.write(str(e))

    # 9) Clasificación 70/30
    results, summary = experiments_70_30(df, class_col=args.class_col, text_col="Review_clean",
                                         model_type=args.model, out_dir=tabs)
    summary.to_csv(os.path.join(tabs,"classification_summary.csv"), index=False)
    # figuras de matrices de confusión
    for r in results:
        out_png = os.path.join(figs, f"cm_{r['name'].replace(' ','_').replace('(','').replace(')','')}.png")
        plot_confusion(r["confusion"], f"Matriz de Confusión — {r['name']}", out_png)

    # 10) LSA 50 topics
    lsa = lsa_topics(df, n_topics=50, min_df=5)
    top_terms = topic_top_terms(lsa["svd"], lsa["terms"], top_terms=10)
    top_terms.to_csv(os.path.join(tabs,"lsa_topics_terms.csv"), index=False)
    y_topics, _ = enc_y_topics(df, args.class_col)
    top_mi, df_mi = topic_informativeness_mi(lsa["Z"], y_topics, top=10)
    top_mi.to_csv(os.path.join(tabs,"lsa_topics_top_mi.csv"), index=False)
    df_mi.to_csv(os.path.join(tabs,"lsa_topics_mi_all.csv"), index=False)

    print("✅ Listo. Archivos en:", base)

if __name__ == "__main__":
    main()
