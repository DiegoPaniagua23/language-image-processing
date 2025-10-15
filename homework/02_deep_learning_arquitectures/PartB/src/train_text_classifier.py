#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entrena clasificadores basados en embeddings (CNN/RNN/LSTM/GRU)."""

import argparse
import json
import logging
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

DEFAULT_PAD = "<pad>"
DEFAULT_UNK = "<unk>"


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_text_classifier")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def basic_tokenizer(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, seq_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.seq_len = seq_len
        self.pad_id = vocab[DEFAULT_PAD]
        self.unk_id = vocab[DEFAULT_UNK]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = basic_tokenizer(self.texts[idx])
        ids = [self.vocab.get(tok, self.unk_id) for tok in tokens][: self.seq_len]
        length = len(ids)
        if length < self.seq_len:
            ids += [self.pad_id] * (self.seq_len - length)
        return torch.tensor(ids, dtype=torch.long), length, self.labels[idx]


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, model_type="cnn", hidden_dim=128, num_layers=1, dropout=0.3):
        super().__init__()
        self.model_type = model_type
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        if model_type == "cnn":
            channels = hidden_dim
            kernel_sizes = [3, 4, 5]
            self.convs = nn.ModuleList(
                [nn.Conv1d(embed_dim, channels, k, padding=k // 2) for k in kernel_sizes]
            )
            self.fc = nn.Linear(channels * len(kernel_sizes), num_classes)
        else:
            rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[model_type]
            self.rnn = rnn_cls(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, lengths):
        emb = self.embedding(input_ids)
        if self.model_type == "cnn":
            x = emb.transpose(1, 2)
            feats = [torch.relu(conv(x)) for conv in self.convs]
            pooled = [torch.max(feat, dim=2).values for feat in feats]
            concat = torch.cat(pooled, dim=1)
            logits = self.fc(self.dropout(concat))
            return logits

        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, hidden = self.rnn(packed)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        last = hidden[-1]
        logits = self.fc(self.dropout(last))
        return logits


def collate_batch(batch):
    inputs, lengths, labels = zip(*batch)
    inputs = torch.stack(inputs)
    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    return inputs, lengths, labels


def build_vocab(texts, min_freq):
    counter = Counter()
    for txt in texts:
        counter.update(basic_tokenizer(txt))
    vocab = {DEFAULT_PAD: 0, DEFAULT_UNK: 1}
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    preds, trues = [], []
    with torch.no_grad():
        for inputs, lengths, labels in loader:
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            logits = model(inputs, lengths)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            trues.extend(labels.cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average="macro")
    report = classification_report(trues, preds, output_dict=True, zero_division=0)
    return avg_loss, acc, f1, preds, trues, report


def main():
    parser = argparse.ArgumentParser(description="Entrena modelos de clasificación secuencial")
    parser.add_argument("--data_file", default="../data/MeIA_2025_train.csv")
    parser.add_argument("--model", choices=["cnn", "rnn", "lstm", "gru"], default="cnn")
    parser.add_argument("--embedding_dim", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seq_len", type=int, default=200)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--log_file", type=str, default="../logs/partb_train_text.log")
    parser.add_argument("--save_dir", type=str, default="../models")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--log_csv", type=str, default=None)
    parser.add_argument("--summary_path", type=str, default=None)
    parser.add_argument("--metrics_path", type=str, default=None)
    parser.add_argument("--confusion_path", type=str, default=None)
    parser.add_argument("--report_path", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    df = pd.read_csv(args.data_file)
    df = df.dropna(subset=["Review", "Polarity"])

    label_values = sorted(df["Polarity"].unique())
    label2id = {lbl: idx for idx, lbl in enumerate(label_values)}
    id2label = {idx: lbl for lbl, idx in label2id.items()}
    df["label_id"] = df["Polarity"].map(label2id)

    texts = df["Review"].tolist()
    labels = df["label_id"].astype(int).tolist()

    vocab = build_vocab(texts, args.min_freq)

    strat = df["label_id"].tolist()
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=args.val_size + args.test_size, stratify=strat, random_state=args.seed
    )
    temp_strat = temp_labels
    val_ratio = args.val_size / (args.val_size + args.test_size)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=1 - val_ratio,
        stratify=temp_strat,
        random_state=args.seed,
    )

    train_ds = TextDataset(train_texts, train_labels, vocab, args.seq_len)
    val_ds = TextDataset(val_texts, val_labels, vocab, args.seq_len)
    test_ds = TextDataset(test_texts, test_labels, vocab, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    model_name = args.experiment or f"{args.model}_cls"
    save_dir = Path(args.save_dir) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    log_file = Path(args.log_file)
    if log_file.is_dir():
        log_file = log_file / f"train_text_{model_name}.log"
    logger = setup_logger(log_file)

    if args.log_csv is None:
        args.log_csv = str(save_dir / "trainlog.csv")
    if args.summary_path is None:
        args.summary_path = str(save_dir / "summary.json")
    if args.metrics_path is None:
        args.metrics_path = str(save_dir / "metrics_test.json")
    if args.confusion_path is None:
        args.confusion_path = str(save_dir / "confusion_matrix.csv")
    if args.report_path is None:
        args.report_path = str(save_dir / "classification_report.json")

    logger.info("Iniciando entrenamiento %s", model_name)
    logger.info("Vocab size: %d", len(vocab))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TextClassifier(
        vocab_size=len(vocab),
        embed_dim=args.embedding_dim,
        num_classes=len(label2id),
        model_type=args.model,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    header = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1"]
    with open(args.log_csv, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")

    best_f1 = -math.inf
    best_state = None
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        preds_train, trues_train = [], []
        for inputs, lengths, labels_batch in train_loader:
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            logits = model(inputs, lengths)
            loss = loss_fn(logits, labels_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * labels_batch.size(0)
            preds_train.extend(torch.argmax(logits, dim=1).cpu().tolist())
            trues_train.extend(labels_batch.cpu().tolist())

        train_loss = epoch_loss / len(train_ds)
        train_acc = accuracy_score(trues_train, preds_train)
        val_loss, val_acc, val_f1, _, _, _ = evaluate(model, val_loader, device)

        logger.info(
            "Epoch %d/%d - train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f val_f1=%.4f",
            epoch,
            args.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_f1,
        )
        with open(args.log_csv, "a", encoding="utf-8") as f:
            row = [str(epoch), f"{train_loss:.6f}", f"{train_acc:.6f}", f"{val_loss:.6f}", f"{val_acc:.6f}", f"{val_f1:.6f}"]
            f.write(",".join(row) + "\n")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {
                "model_state": model.state_dict(),
                "config": {
                    "model": args.model,
                    "embedding_dim": args.embedding_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "seq_len": args.seq_len,
                    "vocab_size": len(vocab),
                    "label2id": label2id,
                    "id2label": id2label,
                },
            }
            torch.save(best_state, save_dir / "model_best.pt")

    train_time = time.time() - train_start

    if best_state is not None:
        model.load_state_dict(best_state["model_state"])

    test_loss, test_acc, test_f1, test_preds, test_trues, report = evaluate(model, test_loader, device)
    cm = confusion_matrix(test_trues, test_preds)

    np.savetxt(args.confusion_path, cm, fmt="%d", delimiter=",")
    with open(args.metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "test_f1_macro": test_f1,
                "best_val_f1": best_f1,
                "num_samples_test": len(test_trues),
            },
            f,
            indent=2,
        )
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    time_per_epoch = train_time / max(1, args.epochs)
    summary = {
        "model_name": model_name,
        "architecture": args.model,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "train_time_sec": train_time,
        "time_per_epoch_sec": time_per_epoch,
        "device": str(device),
        "vocab_size": len(vocab),
        "label2id": label2id,
        "id2label": id2label,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "best_val_f1": best_f1,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "log_file": str(log_file),
        "log_csv": args.log_csv,
        "metrics_path": args.metrics_path,
        "confusion_path": args.confusion_path,
    }
    with open(args.summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(save_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f)

    logger.info(
        "Entrenamiento finalizado. Test acc=%.4f test f1=%.4f. Artefactos guardados en %s",
        test_acc,
        test_f1,
        save_dir,
    )


if __name__ == "__main__":
    main()
