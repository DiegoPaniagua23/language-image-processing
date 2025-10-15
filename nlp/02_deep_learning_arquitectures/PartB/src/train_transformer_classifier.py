#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fine-tuning de modelos Transformer para clasificación de reseñas."""

import argparse
import inspect
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer,
                          TrainingArguments)


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_transformer_classifier")
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


def prepare_datasets(df: pd.DataFrame, seed: int, test_size: float, val_size: float):
    labels = df["label_id"].tolist()
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=labels, random_state=seed
    )
    train_labels = train_df["label_id"].tolist()
    val_ratio = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_ratio,
        stratify=train_labels,
        random_state=seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int):
    def _tokenize(batch):
        return tokenizer(
            batch["Review"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(_tokenize, batched=True)


def build_trainlog_csv(log_history, csv_path: Path) -> None:
    rows = []
    for entry in log_history:
        if "epoch" not in entry:
            continue
        epoch = int(entry["epoch"])
        row = {
            "epoch": epoch,
            "train_loss": entry.get("loss"),
            "eval_loss": entry.get("eval_loss"),
            "eval_accuracy": entry.get("eval_accuracy"),
            "eval_f1": entry.get("eval_f1"),
        }
        rows.append(row)
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("epoch").drop_duplicates(subset="epoch", keep="last")
    df.to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning Transformer para clasificación")
    parser.add_argument("--data_file", default="../data/MeIA_2025_train.csv")
    parser.add_argument("--model_path", default="../models/mdeberta-v3-base")
    parser.add_argument("--save_dir", default="../models/mdeberta_cls")
    parser.add_argument("--log_file", default="../logs/partb_train_transformer.log")
    parser.add_argument("--log_csv", default=None)
    parser.add_argument("--summary_path", default=None)
    parser.add_argument("--metrics_path", default=None)
    parser.add_argument("--confusion_path", default=None)
    parser.add_argument("--report_path", default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    log_file = Path(args.log_file)
    if log_file.is_dir():
        log_file = log_file / "train_transformer.log"
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

    df = pd.read_csv(args.data_file).dropna(subset=["Review", "Polarity"])
    label_values = sorted(df["Polarity"].unique())
    label2id = {lbl: idx for idx, lbl in enumerate(label_values)}
    id2label = {idx: lbl for lbl, idx in label2id.items()}
    df["label_id"] = df["Polarity"].map(label2id)

    train_df, val_df, test_df = prepare_datasets(df, args.seed, args.test_size, args.val_size)

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    test_ds = Dataset.from_pandas(test_df)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token

    train_ds_tok = tokenize_dataset(train_ds, tokenizer, args.max_length)
    val_ds_tok = tokenize_dataset(val_ds, tokenizer, args.max_length)
    test_ds_tok = tokenize_dataset(test_ds, tokenizer, args.max_length)

    train_ds_tok = train_ds_tok.rename_column("label_id", "labels")
    val_ds_tok = val_ds_tok.rename_column("label_id", "labels")
    test_ds_tok = test_ds_tok.rename_column("label_id", "labels")

    train_ds_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    hf_id2label = {idx: str(lbl) for idx, lbl in id2label.items()}
    hf_label2id = {str(lbl): idx for lbl, idx in label2id.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=len(label2id),
        id2label=hf_id2label,
        label2id=hf_label2id,
        ignore_mismatched_sizes=True,
    )

    # Configuración de argumentos del Trainer con compatibilidad retroactiva.
    training_kwargs = {
        "output_dir": str(save_dir / "hf_run"),
        "logging_dir": str(save_dir / "tb"),
        "overwrite_output_dir": True,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "save_strategy": "epoch",
        "evaluation_strategy": "epoch",
        "logging_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_f1",
        "greater_is_better": True,
        "fp16": args.fp16,
        "seed": args.seed,
        "report_to": [],
    }

    training_signature = inspect.signature(TrainingArguments.__init__)
    accepted_params = set(training_signature.parameters.keys())
    steps_per_epoch = max(1, math.ceil(len(train_ds_tok) / (args.batch_size * args.grad_accum)))

    evaluation_supported = "evaluation_strategy" in accepted_params

    if not evaluation_supported:
        training_kwargs.pop("evaluation_strategy", None)
        if "evaluate_during_training" in accepted_params:
            training_kwargs["evaluate_during_training"] = True
        logger.warning("evaluation_strategy no disponible; se activa evaluate_during_training si procede")

    if "save_strategy" not in accepted_params:
        training_kwargs.pop("save_strategy", None)
        if "save_steps" in accepted_params:
            training_kwargs["save_steps"] = steps_per_epoch
        logger.warning("save_strategy no disponible; se usa save_steps=%d", training_kwargs.get("save_steps"))

    if "logging_strategy" not in accepted_params:
        training_kwargs.pop("logging_strategy", None)
        if "logging_steps" in accepted_params:
            training_kwargs["logging_steps"] = steps_per_epoch
        logger.warning("logging_strategy no disponible; se usa logging_steps=%d", training_kwargs.get("logging_steps"))

    if "warmup_ratio" not in accepted_params and "warmup_steps" in accepted_params:
        training_kwargs.pop("warmup_ratio", None)
        total_training_steps = steps_per_epoch * args.epochs
        warmup_steps = int(total_training_steps * args.warmup_ratio)
        training_kwargs["warmup_steps"] = max(0, warmup_steps)
        logger.warning("warmup_ratio no disponible; se usa warmup_steps=%d", training_kwargs["warmup_steps"])

    if not evaluation_supported:
        for key in ["load_best_model_at_end", "metric_for_best_model", "greater_is_better"]:
            if key in training_kwargs:
                logger.warning("%s requiere evaluation_strategy; se elimina del TrainingArguments", key)
                training_kwargs.pop(key)

    for legacy_key in ["metric_for_best_model", "greater_is_better", "load_best_model_at_end", "fp16", "report_to"]:
        if legacy_key in training_kwargs and legacy_key not in accepted_params:
            logger.warning("%s no disponible; se elimina del TrainingArguments", legacy_key)
            training_kwargs.pop(legacy_key)

    training_args = TrainingArguments(**training_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "Fine-tuning en %s | epochs=%d batch=%d grad_accum=%d lr=%.2e",
        device,
        args.epochs,
        args.batch_size,
        args.grad_accum,
        args.lr,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")
        return {"accuracy": acc, "f1": f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds_tok,
        eval_dataset=val_ds_tok,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    train_start = time.time()
    train_result = trainer.train()
    train_time = time.time() - train_start
    logger.info("Entrenamiento completado en %.2f minutos", train_time / 60)

    build_trainlog_csv(trainer.state.log_history, Path(args.log_csv))

    metrics_val = trainer.evaluate()
    logger.info(
        "Mejor modelo en validation -> loss=%.4f acc=%.4f f1=%.4f",
        metrics_val.get("eval_loss", math.nan),
        metrics_val.get("eval_accuracy", math.nan),
        metrics_val.get("eval_f1", math.nan),
    )

    test_predictions = trainer.predict(test_ds_tok)
    test_logits = test_predictions.predictions
    test_labels = test_predictions.label_ids
    test_preds = np.argmax(test_logits, axis=1)

    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average="macro")
    test_loss = math.nan
    if "test_loss" in test_predictions.metrics:
        test_loss = test_predictions.metrics["test_loss"]

    cm = confusion_matrix(test_labels, test_preds)
    np.savetxt(args.confusion_path, cm.astype(int), fmt="%d", delimiter=",")

    report = classification_report(test_labels, test_preds, output_dict=True, zero_division=0)
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(args.metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_accuracy": test_acc,
                "test_f1_macro": test_f1,
                "test_loss": test_loss,
                "eval_accuracy": metrics_val.get("eval_accuracy"),
                "eval_f1": metrics_val.get("eval_f1"),
                "eval_loss": metrics_val.get("eval_loss"),
            },
            f,
            indent=2,
        )

    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    summary = {
        "model_name": Path(args.model_path).name,
        "save_dir": str(save_dir),
        "num_labels": len(label2id),
        "label2id": label2id,
        "id2label": id2label,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_length": args.max_length,
        "train_samples": len(train_ds_tok),
        "val_samples": len(val_ds_tok),
        "test_samples": len(test_ds_tok),
        "train_time_sec": train_time,
        "train_runtime": train_result.metrics.get("train_runtime"),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
        "best_eval_f1": metrics_val.get("eval_f1"),
        "best_eval_accuracy": metrics_val.get("eval_accuracy"),
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "log_file": str(log_file),
        "log_csv": args.log_csv,
        "metrics_path": args.metrics_path,
        "confusion_path": args.confusion_path,
    }
    with open(args.summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Evaluación test -> acc=%.4f f1=%.4f", test_acc, test_f1)


if __name__ == "__main__":
    main()
