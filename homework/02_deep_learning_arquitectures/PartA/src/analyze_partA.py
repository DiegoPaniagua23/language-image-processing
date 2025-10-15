#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility para consolidar métricas, curvas y resúmenes de la Parte A."""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    print(f"[WARN] Matplotlib no disponible: {exc}", file=sys.stderr)
    plt = None


def _float_or_none(val):
    if val in (None, "", "None"):
        return None
    try:
        return float(val)
    except Exception:
        return None


def plot_trainlog(csv_path: Path, out_dir: Path):
    if plt is None:
        return None
    try:
        # Leer CSV con manejo robusto de líneas malformadas
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        # Si está vacío después de skip, intentar con método alternativo
        if df.empty:
            # Leer como texto y procesar manualmente
            with open(csv_path, 'r') as f:
                lines = f.readlines()
            # Tomar header y arreglar líneas
            header = lines[0].strip()
            fixed_lines = [header]
            num_cols = len(header.split(','))
            for line in lines[1:]:
                parts = line.strip().rstrip(',').split(',')
                # Rellenar con valores vacíos si faltan columnas
                while len(parts) < num_cols:
                    parts.append('')
                fixed_lines.append(','.join(parts[:num_cols]))
            # Crear DataFrame desde las líneas corregidas
            from io import StringIO
            df = pd.read_csv(StringIO('\n'.join(fixed_lines)))
    except Exception as e:
        print(f"[WARN] No se pudo leer {csv_path}: {e}", file=sys.stderr)
        return None
    if "epoch" not in df.columns:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df["epoch"], df.get("train_loss", 0), label="train")
    axes[0].plot(df["epoch"], df.get("val_loss", 0), label="val")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss por época")
    axes[0].legend()

    if "val_ppl" in df.columns:
        ppl = df["val_ppl"].apply(_float_or_none)
        axes[1].plot(df["epoch"], ppl)
    else:
        axes[1].plot(df["epoch"], [math.nan] * len(df))
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("PPL")
    axes[1].set_title("Perplejidad por época")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    out_path = out_dir / f"{csv_path.parent.name}_curvas.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def collect_summaries(path: Path):
    rows = []
    for summary_file in path.rglob("summary.json"):
        try:
            with open(summary_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        data["model_dir"] = str(summary_file.parent)
        rows.append(data)
    return rows


def collect_metrics(metrics_root: Path):
    records = []
    for metrics_file in metrics_root.rglob("metrics.jsonl"):
        try:
            with open(metrics_file, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rec["metrics_file"] = str(metrics_file)
                    records.append(rec)
        except FileNotFoundError:
            continue
    return records


def main():
    ap = argparse.ArgumentParser(description="Genera plots y tablas de la Parte A")
    ap.add_argument("--models_dir", type=str, default="../models",
                    help="Directorio raíz que contiene checkpoints entrenados")
    ap.add_argument("--results_dir", type=str, default="../results/analysis",
                    help="Directorio donde se escribirá la salida")
    ap.add_argument("--samples_dir", type=str, default="../results",
                    help="Directorio raíz con samples y metrics.jsonl")
    args = ap.parse_args()

    models_dir = Path(args.models_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    samples_dir = Path(args.samples_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Curvas para cada trainlog.csv
    plot_manifest = []
    for csv_path in models_dir.rglob("trainlog.csv"):
        out_path = plot_trainlog(csv_path, plots_dir)
        if out_path:
            plot_manifest.append({
                "trainlog": str(csv_path),
                "plot": str(out_path)
            })

    if plot_manifest:
        pd.DataFrame(plot_manifest).to_csv(results_dir / "plots_manifest.csv", index=False)

    # Consolidar summary.json
    summaries = collect_summaries(models_dir)
    if summaries:
        pd.DataFrame(summaries).to_csv(results_dir / "summaries.csv", index=False)

    # Consolidar metrics.jsonl de generación
    metrics = collect_metrics(samples_dir)
    if metrics:
        pd.DataFrame(metrics).to_csv(results_dir / "generation_metrics.csv", index=False)

    print("[DONE] Análisis generado en", results_dir)


if __name__ == "__main__":
    main()
