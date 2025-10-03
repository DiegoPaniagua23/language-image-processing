#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_textgen.py
Entrena modelos RNN/GRU/LSTM para generación de texto (letras de rap).

CARACTERÍSTICAS:
- Soporta character-level y word-level modeling
- Arquitecturas: RNN, GRU, LSTM (configurable con capas y dimensiones)
- Multi-GPU con DataParallel
- Logging exclusivo a archivos
- Cálculo de Perplexity (PPL) en validación
- Guardado automático del mejor modelo
- Generación de samples durante entrenamiento

REQUISITOS:
- Lee corpus desde PartA/data/canciones.txt con delimitadores <|startsong|>/<|endsong|>
- level=char: vocabulario de caracteres
- level=word: vocabulario de palabras (con filtro min_freq, opcional lowercase)
- Logs automáticos en ../logs/ con timestamp
- Modelos guardados en ../models/

"""

import os, sys, math, random, argparse, csv, time, re, logging
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------- Repro ----------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------- Logging Setup ----------
def setup_logger(log_file: str):
    """Configura logging a archivo (NO stdout para Lab-SB)"""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_textgen")
    logger.setLevel(logging.INFO)
    # Eliminar handlers previos si existen
    logger.handlers.clear()
    # Solo file handler, NO console
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

# ---------- IO ----------
def load_corpus_txt(path: str, remove_delims=True):
    """
    Carga el corpus de texto desde archivo.

    Args:
        path: Ruta al archivo de texto
        remove_delims: Si True, elimina delimitadores <|startsong|>/<|endsong|>

    Returns:
        Texto completo como string
    """
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    if remove_delims:
        txt = txt.replace("<|startsong|>", "").replace("<|endsong|>", "")
    return txt.strip()

# ---------- Tokenización (word-level) ----------
def word_tokenize(text: str, lowercase: bool):
    """
    Tokeniza texto en palabras (word-level).

    Args:
        text: Texto a tokenizar
        lowercase: Si True, convierte todo a minúsculas

    Returns:
        Lista de tokens (palabras)
    """
    if lowercase:
        text = text.lower()
    # colapsar espacios
    text = re.sub(r"[ \t]+", " ", text.replace("\r", "").replace("\t", " "))
    # mantener saltos como separadores normales
    tokens = re.findall(r"\S+", text)
    return tokens

# ---------- Vocabs ----------
def build_char_vocab(text: str):
    """
    Construye vocabulario de caracteres.

    Args:
        text: Texto completo del corpus

    Returns:
        stoi: Dict str->int (char to index)
        itos: Dict int->str (index to char)
    """
    chars = sorted(list(set(text)))
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for ch,i in stoi.items()}
    return stoi, itos

def build_word_vocab(tokens, min_freq=1):
    """
    Construye vocabulario de palabras con frecuencia mínima.

    Args:
        tokens: Lista de tokens (palabras)
        min_freq: Frecuencia mínima para incluir palabra en vocab

    Returns:
        stoi: Dict str->int (word to index)
        itos: Dict int->str (index to word)
        PAD: Token de padding
        UNK: Token para palabras desconocidas
    """
    from collections import Counter
    PAD = "<pad>"; UNK = "<unk>"
    cnt = Counter(tokens)
    vocab = [PAD, UNK] + [w for w,c in sorted(cnt.items(), key=lambda x:(-x[1], x[0])) if c >= min_freq]
    stoi = {w:i for i,w in enumerate(vocab)}
    itos = {i:w for w,i in stoi.items()}
    return stoi, itos, PAD, UNK

# ---------- Datasets ----------
class SeqDataset(Dataset):
    """
    Dataset para modelado de lenguaje (char o word level).
    Genera ventanas deslizantes de secuencias (x, y) donde y = x desplazado 1 posición.

    Args:
        indices: Array 1D de índices (vocabulario numérico)
        seq_len: Longitud de cada secuencia
    """
    def __init__(self, indices, seq_len: int):
        super().__init__()
        self.data = indices
        self.seq_len = seq_len
        self.N = len(self.data) - self.seq_len

    def __len__(self):
        return max(0, self.N)

    def __getitem__(self, idx):
        # x: secuencia de entrada (t:t+seq_len)
        x = torch.tensor(self.data[idx:idx+self.seq_len], dtype=torch.long)
        # y: secuencia objetivo (t+1:t+seq_len+1) - predecir siguiente token
        y = torch.tensor(self.data[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return x, y

# ---------- Modelo ----------
class CharWordLM(nn.Module):
    """
    Modelo de lenguaje basado en RNN/GRU/LSTM.
    Soporta tanto character-level como word-level.

    Args:
        vocab_size: Tamaño del vocabulario
        embed_dim: Dimensión de embeddings
        hidden_dim: Dimensión de hidden state del RNN
        num_layers: Número de capas recurrentes
        arch: Arquitectura ("rnn", "gru", o "lstm")
        dropout: Probabilidad de dropout
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, arch="lstm", dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[arch]
        self.arch = arch
        self.rnn = rnn_cls(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, h=None):
        """
        Forward pass.

        Args:
            x: Tensor de entrada (batch_size, seq_len)
            h: Hidden state opcional (para generación autoregresiva)

        Returns:
            logits: (batch_size, seq_len, vocab_size)
            h: Hidden state actualizado
        """
        emb = self.embed(x)
        out, h = self.rnn(emb, h)
        out = self.drop(out)
        logits = self.fc(out)
        return logits, h

# ---------- Muestreo ----------
@torch.no_grad()
def sample_char(model, device, itos, stoi, start_text="", max_new_tokens=400, temperature=1.0):
    """
    Genera texto character-level usando el modelo entrenado.

    Args:
        model: Modelo CharWordLM entrenado
        device: torch.device (cpu/cuda)
        itos: Dict index->char
        stoi: Dict char->index
        start_text: Texto inicial (prompt)
        max_new_tokens: Número de caracteres a generar
        temperature: Control de aleatoriedad (0.7=conservador, 1.2=creativo)

    Returns:
        Texto generado (string)
    """
    model.eval()
    if not start_text:
        start_text = " "
    input_ids = torch.tensor([[stoi.get(ch, 0) for ch in start_text]], dtype=torch.long, device=device)
    h = None
    out_text = start_text
    for _ in range(max_new_tokens):
        logits, h = model(input_ids, h)
        last = logits[:, -1, :] / max(1e-6, temperature)
        probs = torch.softmax(last, dim=-1)
        idx_next = torch.multinomial(probs, 1)
        out_text += itos[idx_next.item()]
        input_ids = idx_next
    return out_text

@torch.no_grad()
def sample_word(model, device, itos, stoi, start_tokens=None, max_new_tokens=60, temperature=1.0, unk_id=None):
    """
    Genera texto word-level usando el modelo entrenado.

    Args:
        model: Modelo CharWordLM entrenado
        device: torch.device (cpu/cuda)
        itos: Dict index->word
        stoi: Dict word->index
        start_tokens: Lista de tokens iniciales (prompt)
        max_new_tokens: Número de palabras a generar
        temperature: Control de aleatoriedad
        unk_id: ID del token <unk>

    Returns:
        Texto generado (string con palabras separadas por espacios)
    """
    model.eval()
    if not start_tokens:
        # semilla vacía → token más frecuente que no sea <pad>/<unk>
        start_tokens = []
    if len(start_tokens) == 0:
        # elige un token “popular” (id 2 suele ser el más frecuente distinto de specials)
        start_ids = [2]
    else:
        start_ids = [stoi.get(t, unk_id if unk_id is not None else 1) for t in start_tokens]
    input_ids = torch.tensor([start_ids], dtype=torch.long, device=device)
    h = None
    out_ids = list(start_ids)
    for _ in range(max_new_tokens):
        logits, h = model(input_ids, h)
        last = logits[:, -1, :] / max(1e-6, temperature)
        probs = torch.softmax(last, dim=-1)
        idx_next = torch.multinomial(probs, 1)
        out_ids.append(idx_next.item())
        input_ids = idx_next
    # reconstruir
    toks = [itos[i] for i in out_ids]
    # quita <pad>/<unk> sueltos del inicio si aparecen
    toks = [t for t in toks if t not in ("<pad>",)]
    return " ".join(toks)

# ---------- Train / Eval (con límites de batches para smoke) ----------
def run_epoch(model, loader, criterion, optimizer, device, logger, grad_clip=None, max_batches=None, progress_every=0):
    """
    Ejecuta una época completa de entrenamiento.

    Args:
        model: Modelo a entrenar
        loader: DataLoader de entrenamiento
        criterion: Función de pérdida
        optimizer: Optimizador
        device: torch.device
        logger: Logger para escribir a archivo
        grad_clip: Valor para gradient clipping (None = sin clipping)
        max_batches: Límite de batches para smoke test (None = todos)
        progress_every: Log cada N batches (0 = sin logs intermedios)

    Returns:
        Pérdida promedio de la época
    """
    model.train()
    total_loss = 0.0; seen = 0
    for i, (x, y) in enumerate(loader, 1):
        x = x.to(device); y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        seen += x.size(0)
        if progress_every and (i % progress_every == 0):
            logger.info(f"  [train] batch {i}/{len(loader)} loss={loss.item():.4f}")
        if max_batches and i >= max_batches:
            break
    return total_loss / max(seen, 1)

@torch.no_grad()
def evaluate(model, loader, criterion, device, logger, max_batches=None, progress_every=0):
    """
    Evalúa el modelo en el conjunto de validación.

    Args:
        model: Modelo a evaluar
        loader: DataLoader de validación
        criterion: Función de pérdida
        device: torch.device
        logger: Logger para escribir a archivo
        max_batches: Límite de batches (None = todos)
        progress_every: Log cada N batches (0 = sin logs)

    Returns:
        avg_loss: Pérdida promedio
        ppl: Perplexity (exp(avg_loss))
    """
    model.eval()
    total_loss = 0.0; seen = 0
    for i, (x, y) in enumerate(loader, 1):
        x = x.to(device); y = y.to(device)
        logits, _ = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item() * x.size(0)
        seen += x.size(0)
        if progress_every and (i % progress_every == 0):
            logger.info(f"  [valid] batch {i}/{len(loader)} loss={loss.item():.4f}")
        if max_batches and i >= max_batches:
            break
    avg = total_loss / max(seen, 1)
    ppl = math.exp(min(20, avg))
    return avg, ppl

def save_csv_row(csv_path, row_dict, header_order):
    """
    Guarda una fila en archivo CSV (append mode).
    Crea el archivo y escribe headers si no existe.

    Args:
        csv_path: Ruta al archivo CSV
        row_dict: Diccionario con datos de la fila
        header_order: Lista con orden de columnas
    """
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header_order)
        if not exists:
            w.writeheader()
        w.writerow(row_dict)

# ---------- Main ----------
def main():
    """
    Función principal para entrenar modelos RNN/GRU/LSTM de generación de texto.

    Soporta:
    - Character-level y word-level modeling
    - Multi-GPU con DataParallel (optimizado para Lab-SB)
    - Logging a archivos (sin stdout, requisito del servidor)
    - Smoke testing con límite de batches
    - Guardado de checkpoints y samples
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", type=str, default="../data/canciones.txt")
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--level", type=str, default="char", choices=["char","word"])
    ap.add_argument("--arch", type=str, default="lstm", choices=["rnn","gru","lstm"])
    ap.add_argument("--embedding_dim", type=int, default=256)
    ap.add_argument("--hidden_size", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", type=str, default="../models/text_lm")
    ap.add_argument("--log_csv", type=str, default="../models/text_lm/trainlog.csv")
    ap.add_argument("--sample_every", type=int, default=1)
    ap.add_argument("--sample_temp", type=float, default=0.9)
    ap.add_argument("--sample_len", type=int, default=400)
    # word options
    ap.add_argument("--min_freq", type=int, default=1)
    ap.add_argument("--lowercase", action="store_true")
    # smoke/progreso
    ap.add_argument("--max_train_batches", type=int, default=None)
    ap.add_argument("--max_val_batches", type=int, default=None)
    ap.add_argument("--progress_every", type=int, default=0)
    # logging y multi-GPU
    ap.add_argument("--log_file", type=str, default=None, help="Path al archivo de log (default: auto-genera en ../logs/)")
    ap.add_argument("--use_multigpu", action="store_true", help="Usar DataParallel para multi-GPU")
    args = ap.parse_args()

    set_seed(args.seed)

    # Setup logging
    if args.log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = f"../logs/train_{args.arch}_{args.level}_{timestamp}.log"
    logger = setup_logger(args.log_file)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    logger.info("="*60)
    logger.info(f"INICIO ENTRENAMIENTO: {args.arch.upper()} {args.level.upper()}")
    logger.info("="*60)
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(n_gpus):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    logger.info(f"Args: {vars(args)}")

    # Cargar corpus
    logger.info(f"Cargando corpus desde: {args.train_file}")
    if not os.path.exists(args.train_file):
        logger.error(f"ERROR: No se encuentra el archivo {args.train_file}")
        logger.error("Ejecuta primero: bash run_data_pipeline.sh")
        sys.exit(1)
    text = load_corpus_txt(args.train_file)
    file_size_mb = os.path.getsize(args.train_file) / (1024*1024)
    logger.info(f"Corpus cargado: {len(text):,} caracteres ({file_size_mb:.2f} MB)")

    # Construir vocabulario y convertir a índices
    if args.level == "char":
        logger.info(f"Corpus chars: {len(text):,}")
        stoi, itos = build_char_vocab(text)
        vocab_size = len(stoi)
        data = np.array([stoi[ch] for ch in text], dtype=np.int64)
        logger.info(f"Vocab size: {vocab_size}")

    else:  # word-level
        tokens = word_tokenize(text, lowercase=args.lowercase)
        logger.info(f"Corpus tokens: {len(tokens):,}")
        stoi, itos, PAD, UNK = build_word_vocab(tokens, min_freq=args.min_freq)
        vocab_size = len(stoi)
        unk_id = stoi[UNK]
        data = np.array([stoi.get(t, unk_id) for t in tokens], dtype=np.int64)
        logger.info(f"Vocab size: {vocab_size} (min_freq={args.min_freq}, lowercase={args.lowercase})")

    # Train/val split
    n = len(data)
    n_val = int(n * args.val_split)
    train_ids = data[:-n_val] if n_val > 0 else data
    val_ids   = data[-n_val:] if n_val > 0 else data[-1:]

    # Crear datasets y dataloaders
    train_ds = SeqDataset(train_ids, seq_len=args.seq_len)
    val_ds   = SeqDataset(val_ids,   seq_len=args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=True)

    # Crear modelo
    # Crear modelo
    model = CharWordLM(vocab_size, args.embedding_dim, args.hidden_size, args.num_layers, args.arch, args.dropout).to(device)

    # Multi-GPU support (DataParallel para 2x Titan RTX en Lab-SB)
    if args.use_multigpu and n_gpus > 1:
        logger.info(f"Usando DataParallel con {n_gpus} GPUs")
        model = nn.DataParallel(model)

    # Configurar optimizador y pérdida
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Preparar directorios para guardar resultados
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    log_csv = Path(args.log_csv); log_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["epoch","train_loss","val_loss","val_ppl","best"]
    best_val = float("inf")
    best_path = Path(args.save_dir) / f"{args.arch}_{args.level}_best.pt"

    # Training loop
    # Training loop
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        # Entrenar una época
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, logger,
                               grad_clip=args.grad_clip,
                               max_batches=args.max_train_batches,
                               progress_every=args.progress_every)
        # Evaluar en validación
        val_loss, val_ppl = evaluate(model, val_loader, criterion, device, logger,
                                     max_batches=args.max_val_batches,
                                     progress_every=args.progress_every)
        dt = time.time() - t0
        is_best = ""

        # Guardar mejor modelo (basado en val_loss)
        if val_loss < best_val:
            best_val = val_loss
            # Guardar state_dict correcto (sin DataParallel wrapper)
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({
                "model_state": model_to_save.state_dict(),
                "stoi": stoi, "itos": itos,
                "args": vars(args)
            }, best_path)
            is_best = "YES"

        # Log resultados de la época
        logger.info(f"[Epoch {epoch:02d}] Train {train_loss:.4f} | Val {val_loss:.4f} | PPL {val_ppl:.2f} | {dt:.1f}s {('**BEST**' if is_best else '')}")
        save_csv_row(str(log_csv), {
            "epoch": epoch, "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_loss:.6f}", "val_ppl": f"{val_ppl:.6f}",
            "best": is_best
        }, header)

        # Generar samples periódicamente
        if args.sample_every > 0 and (epoch % args.sample_every == 0):
            # Muestreo: usar modelo sin wrapper si es DataParallel
            model_for_sample = model.module if isinstance(model, nn.DataParallel) else model
            out_path = Path(args.save_dir) / f"sample_epoch{epoch:02d}.txt"

            if args.level == "char":
                sample = sample_char(model_for_sample, device, itos, stoi,
                                    start_text="", max_new_tokens=args.sample_len,
                                    temperature=args.sample_temp)
            else:  # word-level
                sample = sample_word(model_for_sample, device, itos, stoi,
                                    start_tokens=None, max_new_tokens=args.sample_len,
                                    temperature=args.sample_temp, unk_id=unk_id)

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(sample)
            logger.info(f"Sample guardado en {out_path}")

    # Resumen final
    logger.info("="*60)
    logger.info("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    logger.info("="*60)
    logger.info(f"Mejor modelo: {best_path}")
    if os.path.exists(best_path):
        size_mb = os.path.getsize(best_path) / (1024*1024)
        logger.info(f"Tamaño del modelo: {size_mb:.1f} MB")
    logger.info(f"Mejor val_loss: {best_val:.4f} | Mejor PPL: {math.exp(min(20, best_val)):.2f}")
    logger.info(f"Métricas CSV: {log_csv}")
    logger.info(f"Samples: {args.save_dir}/sample_epoch*.txt")
    logger.info(f"Log completo: {args.log_file}")
    logger.info("="*60)

if __name__ == "__main__":
    main()
