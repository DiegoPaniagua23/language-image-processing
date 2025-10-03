#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_textgen.py (char & word level)
RNN/GRU/LSTM LM para letras.
- Lee PartA/data/canciones.txt con <|startsong|>/<|endsong|>
- level=char: vocab de caracteres
- level=word: vocab de palabras (freq>=min_freq, opcional lowercase)
- Entrena/valida (PPL), guarda mejor checkpoint y samples
"""

import os, math, random, argparse, csv, time, re
from pathlib import Path
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

# ---------- IO ----------
def load_corpus_txt(path: str, remove_delims=True):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    if remove_delims:
        txt = txt.replace("<|startsong|>", "").replace("<|endsong|>", "")
    return txt.strip()

# ---------- Tokenización (word-level) ----------
def word_tokenize(text: str, lowercase: bool):
    if lowercase:
        text = text.lower()
    # colapsar espacios
    text = re.sub(r"[ \t]+", " ", text.replace("\r", "").replace("\t", " "))
    # mantener saltos como separadores normales
    tokens = re.findall(r"\S+", text)
    return tokens

# ---------- Vocabs ----------
def build_char_vocab(text: str):
    chars = sorted(list(set(text)))
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for ch,i in stoi.items()}
    return stoi, itos

def build_word_vocab(tokens, min_freq=1):
    from collections import Counter
    PAD = "<pad>"; UNK = "<unk>"
    cnt = Counter(tokens)
    vocab = [PAD, UNK] + [w for w,c in sorted(cnt.items(), key=lambda x:(-x[1], x[0])) if c >= min_freq]
    stoi = {w:i for i,w in enumerate(vocab)}
    itos = {i:w for w,i in stoi.items()}
    return stoi, itos, PAD, UNK

# ---------- Datasets ----------
class SeqDataset(Dataset):
    """Funciona para char o word: recibe ya índices (1D array)"""
    def __init__(self, indices, seq_len: int):
        super().__init__()
        self.data = indices
        self.seq_len = seq_len
        self.N = len(self.data) - self.seq_len
    def __len__(self):
        return max(0, self.N)
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return x, y

# ---------- Modelo ----------
class CharWordLM(nn.Module):
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
        emb = self.embed(x)
        out, h = self.rnn(emb, h)
        out = self.drop(out)
        logits = self.fc(out)
        return logits, h

# ---------- Muestreo ----------
@torch.no_grad()
def sample_char(model, device, itos, stoi, start_text="", max_new_tokens=400, temperature=1.0):
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
def run_epoch(model, loader, criterion, optimizer, device, grad_clip=None, max_batches=None, progress_every=0):
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
            print(f"  [train] batch {i}/{len(loader)} loss={loss.item():.4f}")
        if max_batches and i >= max_batches:
            break
    return total_loss / max(seen, 1)

@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches=None, progress_every=0):
    model.eval()
    total_loss = 0.0; seen = 0
    for i, (x, y) in enumerate(loader, 1):
        x = x.to(device); y = y.to(device)
        logits, _ = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item() * x.size(0)
        seen += x.size(0)
        if progress_every and (i % progress_every == 0):
            print(f"  [valid] batch {i}/{len(loader)} loss={loss.item():.4f}")
        if max_batches and i >= max_batches:
            break
    avg = total_loss / max(seen, 1)
    ppl = math.exp(min(20, avg))
    return avg, ppl

def save_csv_row(csv_path, row_dict, header_order):
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header_order)
        if not exists:
            w.writeheader()
        w.writerow(row_dict)

# ---------- Main ----------
def main():
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
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    text = load_corpus_txt(args.train_file)

    if args.level == "char":
        print(f"[INFO] corpus chars: {len(text):,}")
        stoi, itos = build_char_vocab(text)
        vocab_size = len(stoi)
        data = np.array([stoi[ch] for ch in text], dtype=np.int64)

    else:  # word-level
        tokens = word_tokenize(text, lowercase=args.lowercase)
        print(f"[INFO] corpus tokens: {len(tokens):,}")
        stoi, itos, PAD, UNK = build_word_vocab(tokens, min_freq=args.min_freq)
        vocab_size = len(stoi)
        unk_id = stoi[UNK]
        data = np.array([stoi.get(t, unk_id) for t in tokens], dtype=np.int64)
        print(f"[INFO] vocab_size: {vocab_size} (min_freq={args.min_freq}, lowercase={args.lowercase})")

    if args.level == "char":
        print(f"[INFO] vocab_size: {vocab_size}")

    # split
    n = len(data)
    n_val = int(n * args.val_split)
    train_ids = data[:-n_val] if n_val > 0 else data
    val_ids   = data[-n_val:] if n_val > 0 else data[-1:]

    train_ds = SeqDataset(train_ids, seq_len=args.seq_len)
    val_ds   = SeqDataset(val_ids,   seq_len=args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=True)

    model = CharWordLM(vocab_size, args.embedding_dim, args.hidden_size, args.num_layers, args.arch, args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    log_csv = Path(args.log_csv); log_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["epoch","train_loss","val_loss","val_ppl","best"]
    best_val = float("inf")
    best_path = Path(args.save_dir) / f"{args.arch}_{args.level}_best.pt"

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device,
                               grad_clip=args.grad_clip,
                               max_batches=args.max_train_batches,
                               progress_every=args.progress_every)
        val_loss, val_ppl = evaluate(model, val_loader, criterion, device,
                                     max_batches=args.max_val_batches,
                                     progress_every=args.progress_every)
        dt = time.time() - t0
        is_best = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "stoi": stoi, "itos": itos,
                "args": vars(args)
            }, best_path)
            is_best = "YES"

        print(f"[Epoch {epoch:02d}] Train {train_loss:.4f} | Val {val_loss:.4f} | PPL {val_ppl:.2f} | {dt:.1f}s {('**BEST**' if is_best else '')}")
        save_csv_row(str(log_csv), {
            "epoch": epoch, "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_loss:.6f}", "val_ppl": f"{val_ppl:.6f}",
            "best": is_best
        }, header)

        if args.sample_every > 0 and (epoch % args.sample_every == 0):
            if args.level == "char":
                sample = sample_char(model, device, itos, stoi, start_text="", max_new_tokens=args.sample_len, temperature=args.sample_temp)
            else:
                sample = sample_word(model, device, itos, stoi, start_tokens=None, max_new_tokens=max(20, args.sample_len//4), temperature=args.sample_temp, unk_id=stoi.get("<unk>",1))
            out_path = Path(args.save_dir) / f"sample_epoch{epoch:02d}.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(sample)
            print(f"[SAMPLE] escrito en {out_path}")

    print(f"[INFO] Mejor modelo guardado en {best_path}")
    print(f"[INFO] Log CSV en {log_csv}")

if __name__ == "__main__":
    main()
