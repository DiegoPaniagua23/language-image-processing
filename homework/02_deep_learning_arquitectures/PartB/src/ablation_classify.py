#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lanza ablaciones variando batch size y learning rate para el Transformer."""

import argparse
import itertools
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_list(arg: str):
    values = []
    for item in arg.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(item)
    return values


def main():
    parser = argparse.ArgumentParser(description="Ablación clasificadores Transformer")
    parser.add_argument("--model_path", default="models/mdeberta-v3-base")
    parser.add_argument("--data_file", default="data/MeIA_2025_train.csv")
    parser.add_argument("--output_root", default="results/ablations_classif")
    parser.add_argument("--batch_sizes", default="8,16,32")
    parser.add_argument("--learning_rates", default="2e-5,3e-5,5e-5")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", help="Solo mostrar comandos")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    batches = [int(x) for x in parse_list(args.batch_sizes)]
    lrs = [float(x) for x in parse_list(args.learning_rates)]

    manifest_rows = []
    for batch_size, lr in itertools.product(batches, lrs):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_root / f"bs{batch_size}_lr{lr:.0e}"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_file = run_dir / f"train_transformer_{stamp}.log"
        log_csv = run_dir / "trainlog.csv"
        summary_path = run_dir / "summary.json"
        metrics_path = run_dir / "metrics_test.json"
        confusion_path = run_dir / "confusion_matrix.csv"
        report_path = run_dir / "classification_report.json"

        cmd = [
            sys.executable,
            "src/train_transformer_classifier.py",
            "--data_file",
            args.data_file,
            "--model_path",
            args.model_path,
            "--save_dir",
            str(run_dir / "model"),
            "--log_file",
            str(log_file),
            "--log_csv",
            str(log_csv),
            "--summary_path",
            str(summary_path),
            "--metrics_path",
            str(metrics_path),
            "--confusion_path",
            str(confusion_path),
            "--report_path",
            str(report_path),
            "--batch_size",
            str(batch_size),
            "--lr",
            str(lr),
            "--epochs",
            str(args.epochs),
            "--max_length",
            str(args.max_length),
            "--grad_accum",
            str(args.grad_accum),
            "--warmup_ratio",
            str(args.warmup_ratio),
            "--weight_decay",
            str(args.weight_decay),
            "--seed",
            str(args.seed),
        ]

        manifest_rows.append(
            {
                "timestamp": stamp,
                "batch_size": batch_size,
                "learning_rate": lr,
                "output_dir": str(run_dir),
                "log_file": str(log_file),
                "command": " ".join(cmd),
            }
        )

        if args.dry_run:
            print("[DRY-RUN]", " ".join(cmd))
        else:
            env = os.environ.copy()
            print("[RUN]", " ".join(cmd))
            result = subprocess.run(cmd, cwd=Path(__file__).parent.parent, env=env)
            if result.returncode != 0:
                print(f"[WARN] Comando falló con código {result.returncode}")

    if manifest_rows:
        pd.DataFrame(manifest_rows).to_csv(output_root / "ablation_manifest.csv", index=False)


if __name__ == "__main__":
    main()
