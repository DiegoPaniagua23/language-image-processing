# Deep Learning Architectures for NLP: Text Generation and Classification

**Author:** Diego Paniagua  
**Institution:** CIMAT (Centro de Investigación en Matemáticas)  
**Course:** Language and Image Processing  
**Assignment:** Task 2 - Deep Learning Architectures for NLP

---

## 1. ASSIGNMENT OVERVIEW

This project implements and compares different deep learning architectures for two fundamental NLP tasks:

### PART A: TEXT GENERATION (Song Lyrics)

**Objective:** Train and evaluate generative models to create song lyrics using:
- Recurrent architectures: RNN, LSTM, and GRU (character-level and word-level)
- Transformer architecture: LLaMA-3 8B with LoRA fine-tuning

**Key deliverables:**
- Perplexity (PPL) metrics on validation set
- Generated samples (minimum 3 songs per model)
- Ablation studies on context length and decoding parameters (temperature, top-p)
- Training time, GPU memory usage, and parameter count analysis

### PART B: TEXT CLASSIFICATION

**Objective:** Train and evaluate classification models for sentiment analysis using:
- Sequential architectures: CNN, RNN, LSTM, and GRU
- Transformer architecture: mDeBERTa-v3-base fine-tuning

**Key deliverables:**
- Accuracy and F1-Score metrics on test set
- Confusion matrices and error analysis
- Ablation studies on batch size and learning rate
- Comparative analysis of RNN vs Transformer performance

### COMPARATIVE ANALYSIS

The project includes comprehensive comparison between recurrent and transformer architectures across both tasks, analyzing:
- Performance metrics (PPL, Accuracy, F1-Score)
- Computational cost (training time, inference time, memory usage)
- Model complexity (parameter count)
- Practical considerations for deployment

---

## 2. DATASETS

### PART A: Song Lyrics Corpus

**Source:** Custom compiled corpus  
**Location:** `PartA/data/canciones.txt`  
**Size:** 100+ songs from selected music genre  
**Format:** UTF-8 encoded text with song delimiters (`</startsong|>` ... `<startsong|>`)

**Preprocessing:**
- Normalized to UTF-8 encoding
- Removed metadata, headers, and advertisements
- Cleaned noise and formatting inconsistencies
- Documented in `PartA/src/clean_lyrics.py`

### PART B: Sentiment Classification Dataset

**Source:** MeIA 2025 training dataset  
**Location:** `PartB/data/MeIA_2025_train.csv`  
**Task:** Multi-class sentiment classification  
**Format:** CSV with text and label columns  
**Splits:** Automatically divided into train/validation/test

---

## 3. PROJECT STRUCTURE

```
deep_learning_arquitectures/
│
├── README.md                   # This file
├── requirements.txt            # Python dependencies (pip)
├── environment.yml             # Conda environment specification
│
├── docs/                       # Technical documentation
│   └── SESSION_SUMMARY.md     # Complete session summary (troubleshooting, fixes)
│
├── PartA/                      # TEXT GENERATION
│   ├── data/
│   │   ├── canciones.txt      # Song lyrics corpus
│   │   └── *.csv              # Metadata files
│   ├── models/
│   │   ├── rnn_char/          # RNN character-level model
│   │   ├── rnn_word/          # RNN word-level model
│   │   ├── lstm_char/         # LSTM character-level model
│   │   ├── lstm_word/         # LSTM word-level model
│   │   ├── gru_char/          # GRU character-level model
│   │   ├── gru_word/          # GRU word-level model
│   │   └── llama3_lora/       # LLaMA-3 LoRA adapters
│   ├── results/
│   │   ├── samples/           # Generated text samples
│   │   ├── ablations/         # Ablation study results
│   │   └── analysis/          # Consolidated metrics and plots
│   ├── logs/                  # Training logs (stdout)
│   ├── errors/                # Error logs (stderr)
│   ├── src/
│   │   ├── train_textgen.py          # Train RNN/LSTM/GRU models
│   │   ├── train_llama3_lora.py      # Fine-tune LLaMA-3 with LoRA
│   │   ├── generate.py               # Generate text samples
│   │   ├── ablation_generate.py      # Ablation experiments
│   │   ├── analyze_partA.py          # Consolidate metrics and plots
│   │   └── *.py                      # Utility scripts
│   ├── train_char_level.sh    # SLURM: Train char-level models
│   ├── train_word_level.sh    # SLURM: Train word-level models
│   ├── train_llama3.sh        # SLURM: Fine-tune LLaMA-3
│   └── run_generate.sh        # SLURM: Generate samples and analysis
│
├── PartB/                      # TEXT CLASSIFICATION
│   ├── data/
│   │   └── MeIA_2025_train.csv       # Classification dataset
│   ├── models/
│   │   ├── cnn_cls/           # CNN classifier
│   │   ├── lstm_cls/          # LSTM classifier
│   │   ├── gru_cls/           # GRU classifier
│   │   └── mdeberta_cls/      # mDeBERTa-v3 fine-tuned
│   ├── results/
│   │   └── analysis/          # Consolidated metrics and plots
│   ├── logs/                  # Training logs (stdout)
│   ├── errors/                # Error logs (stderr)
│   ├── src/
│   │   ├── train_text_classifier.py        # Train CNN/RNN/LSTM/GRU
│   │   ├── train_transformer_classifier.py # Fine-tune mDeBERTa
│   │   ├── ablation_classify.py            # Ablation experiments
│   │   ├── analyze_partB.py                # Consolidate metrics and plots
│   │   └── *.py                            # Utility scripts
│   ├── train_seq_classifiers.sh   # SLURM: Train sequential models
│   ├── train_deberta.sh           # SLURM: Fine-tune transformer
│   └── run_classification.sh      # SLURM: Full pipeline with analysis
│
└──
```

### MODEL ARTIFACTS (per trained model)

Each trained model directory contains:
- `model_best.pt` / `*.pt` - Best checkpoint (lowest validation loss)
- `trainlog.csv` - Per-epoch metrics (loss, PPL/accuracy, time)
- `summary.json` - Model config, training stats, best metrics
- `vocab.json` - Vocabulary mapping (RNN models only)
- `metrics_test.json` - Test set evaluation (classification only)
- `confusion_matrix.csv` - Confusion matrix (classification only)
- `sample_epoch*.txt` - Generated samples per epoch (generation only)

---

## 4. REPRODUCIBILITY

### ENVIRONMENT SETUP

1. **Create conda environment:**
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate environment:**
   ```bash
   conda activate nlp-t2
   ```

3. **Verify installation:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

**Key dependencies:**
- Python 3.11
- PyTorch with CUDA 12.1
- Transformers, Datasets, Accelerate (Hugging Face)
- PEFT (Parameter-Efficient Fine-Tuning)
- bitsandbytes >= 0.43 (for 4-bit quantization)
- scikit-learn, pandas, matplotlib, seaborn

### CLUSTER EXECUTION (Lab-SB CIMAT)

All scripts are configured for SLURM execution on the Lab-SB cluster at CIMAT.

**IMPORTANT: Multi-GPU Configuration**
- The cluster does NOT support `--gres=gpu:2` SLURM directive
- GPU partition automatically assigns 2 GPUs per node
- For training (DDP): Use `torchrun --nproc_per_node=2` (implemented in scripts)
- For inference: Use `device_map="auto"` (automatic GPU distribution)

**SLURM configuration:**
- Partition: GPU (for models requiring GPU) or C1Mitad1 (for CPU-only analysis)
- Nodes: 1
- GPUs: 2 per node (automatic assignment)
- Notifications: email alerts on BEGIN/END/FAIL

### PART A: TEXT GENERATION

**Step 1: Train RNN/LSTM/GRU models**

```bash
cd deep_learning_arquitectures/PartA

# Character-level (all architectures)
sbatch train_char_level.sh

# Word-level (all architectures)
sbatch train_word_level.sh

# Custom parameters: architecture list, epochs, batch size, sequence length
sbatch train_char_level.sh gru,lstm 30 128 200
```

**Step 2: Fine-tune LLaMA-3 with LoRA**

```bash
sbatch train_llama3.sh

# Custom: model path, epochs, batch size, context length
sbatch train_llama3.sh models/llama-3-8b 5 2 256
```

> **Note:** This script uses torchrun for multi-GPU training (DDP + LoRA).  
> Recent fixes applied:
> - Gradient checkpointing disabled to avoid DDP conflicts
> - Proper LOCAL_RANK device mapping for distributed training
> - Optimized for 2x Titan RTX GPUs (~20-22GB per GPU)

**Step 3: Generate samples and run analysis**

```bash
sbatch run_generate.sh
```

This script (optimized for 2 GPUs):
- Generates samples from all trained models (parallel execution for RNN models)
- Runs ablation experiments (context length, temperature, top-p, top-k)
- Consolidates metrics and creates plots
- Uses `device_map='auto'` for LLaMA-3 inference (automatic GPU distribution)
- Outputs: `results/samples/`, `results/ablations/`, `results/analysis/`

### PART B: TEXT CLASSIFICATION

**Step 1: Train CNN/RNN/LSTM/GRU classifiers**

```bash
cd deep_learning_arquitectures/PartB

sbatch train_seq_classifiers.sh

# Custom: architectures, epochs, batch size, seq_len, emb_dim, hidden_dim, min_freq, lr
sbatch train_seq_classifiers.sh cnn,lstm,gru 12 64 200 200 256 2 0.001
```

**Step 2: Fine-tune mDeBERTa transformer**

```bash
sbatch train_deberta.sh

# Custom: model path, epochs, batch size, learning rate, max_len, grad_accum, warmup
sbatch train_deberta.sh models/mdeberta-v3-base 4 16 3e-5 256 2 0.1
```

**Step 3: Run full classification pipeline with analysis**

```bash
sbatch run_classification.sh
```

This script:
- Runs ablation studies (batch size, learning rate sweeps)
- Consolidates metrics and confusion matrices
- Creates learning curves and comparative plots
- Outputs: `results/analysis/`

### MONITORING JOBS

**View active jobs:**
```bash
squeue -u $USER
```

**View real-time logs:**
```bash
tail -f PartA/logs/slurm-*.out
tail -f PartB/logs/slurm-*.out
```

**View errors:**
```bash
tail -f PartA/errors/slurm-*.err
tail -f PartB/errors/slurm-*.err
```

**Job statistics:**
```bash
sacct -j <job_id> --format=JobID,JobName,Elapsed,MaxRSS,MaxVMSize
```

### SEED AND REPRODUCIBILITY

All training scripts use fixed random seeds for reproducibility:
- Python random seed: 42
- NumPy seed: 42
- PyTorch seed: 42
- CUDA deterministic operations: enabled where possible

---

## CONTACT AND SUPPORT

For questions or issues related to this project:
- **Author:** Diego Paniagua
- **Email:** diego.paniagua@cimat.mx
- **Institution:** CIMAT
