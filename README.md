# Language & Image Processing

![License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![LaTeX](https://img.shields.io/badge/Docs-LaTeX-%23008080.svg)

This repository contains assignments and projects for the Language & Image Processing course in the Master's program at CIMAT (Center for Research in Mathematics). The coursework emphasizes practical applications in Natural Language Processing (NLP) and Computer Vision (CV) using Python.


📚 Course Content
- Natural Language Processing: corpus analysis, tokenization, normalization, feature engineering, embeddings (word2vec/fastText), classic ML for text, sequence models, Transformers.
- Computer Vision: image preprocessing, feature extraction, CNNs, transfer learning, augmentation, and evaluation.


🗂️ Repository Structure
The repository follows the next structure:

```
language-image-processing/
├── nlp/
│   └── 01_Corpus_Analysis/            # Homework 01: Corpus analysis (NLP)
│       ├── data/
│       │   └── raw/
│       │       └── MeIA_2025_train.csv
│       ├── scripts/
│       │   └── 01_description.py      # Entrypoint (to be implemented)
│       ├── src/
│       │   └── corpus_analysis/
│       │       ├── __init__.py
│       │       ├── constants.py
│       │       ├── descriptives.py
│       │       ├── io.py
│       │       └── nlp.py
│       └── requirements.txt
├── LICENSE                            # MIT license
└── README.md                          # This file
```


📝 Assignments
The current course progress:

| Assignment | Topic | Key Methods | Link |
| --- | --- | --- | --- |
| 01 | Corpus Analysis (NLP) | pandas, numpy, spaCy, matplotlib | [📁 View](nlp/01_Corpus_Analysis/) |
| 02 | … | … | … |
| 03 | … | … | … |
| 04 | … | … | … |


🛠️ Technical Stack
Programming & Analysis
- Python (≥3.10): data processing, modeling, and analysis
- Key Python Packages: pandas, numpy, scikit-learn, scipy, matplotlib, spaCy, gensim, tqdm, joblib

Documentation & Reporting
- LaTeX for mathematical typesetting (reports)
- Markdown for repository documentation
- Git for version control

Development Tools
- VS Code / Jupyter for interactive development
- Virtual environments (venv) for reproducible setups


🚀 Getting Started
Tarea 1 (NLP — Corpus Analysis)
1) Create a virtual environment and install dependencies:
   - `cd nlp/01_Corpus_Analysis`
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
   - (optional) Install Spanish spaCy model: `python -m spacy download es_core_news_sm`

2) Expected usage (once the script is implemented):
   - `python scripts/01_description.py --input data/raw/MeIA_2025_train.csv --outdir outputs/description`

Data Notes
- `MeIA_2025_train.csv` columns: `Review` (Spanish text), `Polarity` (1.0–5.0), `Town`, `Region`, `Type` (Restaurant/Hotel/Attractive).

Assignment PDF
- The PDF describing Homework 1 is not yet in this repository. Once available (e.g., `docs/hw1/consigna.pdf`), it will be linked here.


📄 License
This project is released under the MIT License — see `LICENSE`.
