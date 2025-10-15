
# nlp — Modularization pipeline

## Estructura
```
nlp_/
  ├── src/textlab/
  │   ├── __init__.py
  │   ├── data.py          # carga/limpieza/tokenización/stopwords
  │   ├── analysis.py      # descriptivos + Ley de Zipf
  │   ├── features.py      # BoW (TF/TF-IDF), chi2, bigramas
  │   ├── syntax.py        # POS + POS 4-gramas (spaCy)
  │   ├── embeddings.py    # Word2Vec + doc-emb + KMeans
  │   ├── models.py        # 4 experimentos 70/30 (SVM o LogReg)
  │   └── topics.py        # LSA (50 tópicos) + MI
  ├── scripts/run_all.py   # orquestador minimal
  ├── requirements.txt
  ├── reports/{figures,tables}/   # salidas
  └── data/processed/             # intermedios
```

## Uso
1) Crea venv e instala deps:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download es_core_news_sm   # para la parte de POS
```

2) Ejecuta todo con un solo comando (modifica rutas/columnas si quieres):
```bash
python -m scripts.run_all --csv "/ruta/MeIA_2025_train.csv" \
  --text-col Review --class-col Polarity --model logreg \
  --out-dir .
```

- Cambia `--model` a `svm` si prefieres SVM.
- Las salidas aparecen en `reports/` y `data/processed/`.

