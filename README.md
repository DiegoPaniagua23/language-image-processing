# Language & Image Processing

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![LaTeX](https://img.shields.io/badge/Docs-LaTeX-008080.svg)](https://www.latex-project.org/)

This repository collects assignments, lecture material, and supporting resources for the **Language & Image Processing** course in the Master's program at [CIMAT](https://www.cimat.mx/). The coursework blends Natural Language Processing (NLP) and Computer Vision (CV) with a focus on reproducible pipelines in Python.

## 📚 Course Content

1. **Natural Language Processing Foundations**
   - Corpus preparation, tokenization, normalization, exploratory analysis
2. **Representation Learning & Transformers**
   - Word embeddings, sequence models, attention mechanisms, Transformer architectures
3. **Computer Vision Fundamentals**
   - Image preprocessing, feature extraction, convolutional networks, transfer learning, augmentation
4. **Modeling & Evaluation**
   - Classical ML baselines, deep learning pipelines, cross-validation, error analysis
5. **Applied Projects & Reproducibility**
   - End-to-end pipelines for Spanish reviews, experiment tracking, documentation-first workflows

## 📁 Repository Structure

```
language-image-processing/
├── 2024/                         # Classroom notebooks and interactive demos
│   └── notebooks/                # Session Jupyter notebooks
├── diapositivas/                 # Slide decks for NLP & CV lectures
├── homework/                     # Course assignments and submissions
│   ├── 01_corpus_analysis/       # Homework 01: Corpus analysis pipeline
│   └── 02_deep_learning_arquitecturas/  # Homework 02: Deep learning reading & report
├── temario/                      # Course syllabus and outline
├── LICENSE
└── README.md
```

## 📊 Assignments

| Assignment | Module | Key Methods | Link |
|------------|--------|-------------|------|
| **01** | **Corpus Analysis (NLP)** | Token statistics, Zipf law, TF-IDF, Logistic/SVM baselines | [📂 View](./homework/01_corpus_analysis/) |
| **02** | **Deep Learning Architectures** | CNNs, sequence models, Transformers (reading + report) | [📂 View](./homework/02_deep_learning_arquitecturas/) |
| 03 | ... | ... | ... |
| 04 | ... | ... | ... |

## 🛠 Technical Stack

**Programming & Analysis:**
- Python (≥3.10) for preprocessing, modeling, and experimentation
- Core libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `spaCy`, `gensim`, `tqdm`

**Documentation & Reporting:**
- LaTeX for formal reports
- Markdown for repository documentation
- Git for version control and collaboration

**Development Tools:**
- JupyterLab / VS Code for interactive exploration
- Virtual environments (`venv`) for isolated dependencies
- spaCy language models (`es_core_news_sm`) for Spanish NLP tasks

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- `pip` and `virtualenv` (or `python -m venv`)
- Optional: GPU-enabled PyTorch/TensorFlow for advanced experiments
- LaTeX distribution for compiling reports

### Quickstart (Homework 01)

```bash
cd homework/01_corpus_analysis/submission/T1_PLI_DiegoPaniagua
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download es_core_news_sm
python -m scripts.run_all --csv ../../data/raw/MeIA_2025_train.csv \
  --text-col Review --class-col Polarity --model logreg \
  --out-dir .
```

Outputs are stored in `reports/` and intermediate artifacts in `data/processed/`.

### Additional Resources
- Notebooks: `2024/notebooks`
- Slide decks: `diapositivas/`
- Syllabus: `temario/temario_procesamiento_texto_imagenes.pdf`

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
