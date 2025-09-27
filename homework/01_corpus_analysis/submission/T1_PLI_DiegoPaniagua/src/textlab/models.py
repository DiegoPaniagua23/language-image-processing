
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Any
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
import re

WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{2,}(?:[-'][A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+)?")
stemmer = SnowballStemmer("spanish")

def tokenize_basic(text: str, lower: bool = False):
    if not isinstance(text, str):
        return []
    if lower:
        text = text.lower()
    return WORD_RE.findall(text)

def tokenizer_stem(text: str):
    toks = tokenize_basic(text, lower=True)
    return [stemmer.stem(t) for t in toks]

def pick_classifier(model_type:str):
    if model_type == "svm":
        return LinearSVC(random_state=42)
    elif model_type == "logreg":
        return LogisticRegression(max_iter=2000, multi_class="auto", random_state=42)
    else:
        raise ValueError(f"model_type desconocido: {model_type}. Usa 'svm' o 'logreg'.")

def run_experiment(name:str, X_train, X_test, y_train, y_test, vectorizer, classifier) -> Dict[str,Any]:
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)
    clf = classifier.fit(Xtr, y_train)
    y_pred = clf.predict(Xte)

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    f1w = f1_score(y_test, y_pred, average="weighted")

    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    cm_df = pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_)

    return dict(name=name, accuracy=acc, f1_macro=f1m, f1_weighted=f1w,
                report=report, confusion=cm_df, classes=clf.classes_, model=clf)

def experiments_70_30(df, class_col:str="Polarity", text_col:str="Review_clean", model_type:str="logreg",
                      out_dir:str="../reports/tables"):
    X = df[text_col].astype(str).values
    y = df[class_col].astype(str).values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    clf = pick_classifier(model_type)

    results = []

    # (a) Sin preproc
    vec_a = TfidfVectorizer(lowercase=False, token_pattern=r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{2,}")
    res_a = run_experiment("(a) Sin preproc", X_tr, X_te, y_tr, y_te, vec_a, clf)
    results.append(res_a)

    # (b) Minúsculas
    vec_b = TfidfVectorizer(lowercase=True, token_pattern=r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{2,}")
    res_b = run_experiment("(b) Minúsculas", X_tr, X_te, y_tr, y_te, vec_b, clf)
    results.append(res_b)

    # (c) Minúsculas + stemming
    vec_c = TfidfVectorizer(lowercase=False, tokenizer=tokenizer_stem, preprocessor=None, token_pattern=None)
    res_c = run_experiment("(c) Minúsculas + Stem", X_tr, X_te, y_tr, y_te, vec_c, clf)
    results.append(res_c)

    # (d) + min_df=10
    vec_d = TfidfVectorizer(lowercase=False, tokenizer=tokenizer_stem, preprocessor=None, token_pattern=None, min_df=10)
    res_d = run_experiment("(d) Minúsculas + Stem + min_df=10", X_tr, X_te, y_tr, y_te, vec_d, clf)
    results.append(res_d)

    # Guardar matrices de confusión y resumen
    os.makedirs(out_dir, exist_ok=True)
    for r in results:
        cm_path = os.path.join(out_dir, f"confusion_matrix_{r['name'].replace(' ','_').replace('(','').replace(')','')}.csv")
        r["confusion"].to_csv(cm_path, index=True)

    summary = pd.DataFrame([{k:v for k,v in r.items() if k in ("name","accuracy","f1_macro","f1_weighted")} for r in results])
    return results, summary
