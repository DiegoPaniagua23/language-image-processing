# Homework 01: Corpus Analysis (NLP)

This assignment analyzes the Spanish hospitality review dataset `MeIA_2025_train.csv` to build a reproducible NLP pipeline that spans corpus exploration, feature engineering, and baseline modeling.

## Assignment Overview

### 1. Data Acquisition & Cleaning
- Load raw opinions from `data/raw/MeIA_2025_train.csv`
- Normalize text, remove stopwords, and generate token statistics stored in `data/interim/`

### 2. Corpus Analytics
- Compute vocabulary size, token counts, and hapax proportions
- Summarize distributions by rating (`Polarity`) and metadata fields (`Town`, `Region`, `Type`)
- Build Zipf frequency tables and fit log–log regressions

### 3. Feature Engineering
- Create bag-of-words and TF-IDF representations with optional stemming
- Evaluate frequency thresholds (`min_df`) and lowercase pipelines
- Prepare syntactic features (POS n-grams) and embeddings for downstream tasks

### 4. Modeling & Evaluation
- Train logistic regression and linear SVM baselines on TF-IDF vectors
- Compare preprocessing variants across accuracy and (macro/weighted) F1 scores
- Export confusion matrices and classification reports for analysis

### 5. Reporting & Deliverables
- Store figures and tables under `reports/`
- Provide narrative answers, methodology, and conclusions in `submission/`

## Dataset

The raw file `MeIA_2025_train.csv` contains the following fields:

| Column | Description |
|--------|-------------|
| `Review` | Spanish review text |
| `Polarity` | Rating from 1.0 to 5.0 |
| `Town` | City where the opinion was collected |
| `Region` | Geographic region |
| `Type` | Establishment category (Restaurant / Hotel / Attractive) |

`data/interim/` and `data/processed/` host intermediate artifacts generated during cleaning and feature extraction.

## Project Structure

```
01_corpus_analysis/
├── data/
│   ├── raw/                     # Original dataset (MeIA_2025_train.csv, sample)
│   ├── interim/                 # Intermediate artifacts
│   └── processed/               # Cleaned corpora ready for modeling
├── notebooks/                   # Exploratory notebooks and drafts
├── reports/
│   ├── figures/                 # Generated plots
│   └── tables/                  # Tabular outputs
├── requirements.txt             # Core dependencies for the assignment
├── submission/
│   ├── ADVANCED_SETUP.md        # Extended instructions and environment notes
│   ├── T1_PLI_DiegoPaniagua/    # Modularized implementation (scripts & package)
│   ├── T1_PTI_DiegoPaniagua.pdf # Written report
│   └── Tarea 1.pdf              # Original assignment statement
└── README.md
```

## Reproducibility

1. **Set up the environment**

```bash
cd homework/01_corpus_analysis/submission/T1_PLI_DiegoPaniagua
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

2. **Run the end-to-end pipeline**

```bash
python -m scripts.run_all --csv ../../data/raw/MeIA_2025_train.csv \
  --text-col Review --class-col Polarity --model logreg \
  --out-dir .
```

- Set `--model svm` to train the linear SVM baseline.
- Outputs are written to `reports/` and intermediate artifacts to `data/processed/` within the submission package.

3. **Review results**
- Metrics and summaries: `reports/tables/`
- Confusion matrices: CSV files in the chosen `--out-dir`
- Narrative answers: `submission/T1_PTI_DiegoPaniagua.pdf`

## Related Materials

- Slide decks: `../../diapositivas/`
- Lecture notebooks: `../../2024/notebooks/`
- Course outline: `../../temario/temario_procesamiento_texto_imagenes.pdf`
