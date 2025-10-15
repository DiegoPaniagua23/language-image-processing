#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Consolida métricas, reportes y gráficas para la Parte B."""

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def collect_jsons(root: Path, filename: str):
    rows = []
    for path in root.rglob(filename):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                data["source"] = str(path)
                rows.append(data)
        except Exception:
            continue
    return rows


def plot_confusion_matrix(csv_path: Path, out_path: Path, labels):
    if plt is None:
        return None
    df = pd.read_csv(csv_path, header=None)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(df.values, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(csv_path.parent.name)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, df.iat[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_learning_curves(csv_path: Path, out_path: Path):
    if plt is None:
        return None
    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns:
        return None
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(df["epoch"], df.get("train_loss", 0), label="train_loss", marker="o")
    if "eval_loss" in df.columns:
        ax1.plot(df["epoch"], df["eval_loss"], label="val_loss", marker="o")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    if "eval_accuracy" in df.columns or "val_acc" in df.columns:
        ax2 = ax1.twinx()
        metric_key = "eval_accuracy" if "eval_accuracy" in df.columns else "val_acc"
        ax2.plot(df["epoch"], df[metric_key], color="tab:green", label="val_accuracy", marker="s")
        ax2.set_ylabel("Accuracy", color="tab:green")
        ax2.tick_params(axis="y", labelcolor="tab:green")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="lower right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Análisis Parte B")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--results_dir", default="results/analysis")
    parser.add_argument("--confusion_dir", default="models")
    args = parser.parse_args()

    models_dir = Path(args.models_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    confusion_dir = Path(args.confusion_dir).resolve()

    results_dir.mkdir(parents=True, exist_ok=True)

    summaries = collect_jsons(models_dir, "summary.json")
    if summaries:
        pd.DataFrame(summaries).to_csv(results_dir / "summaries.csv", index=False)

    metrics = collect_jsons(models_dir, "metrics_test.json")
    if metrics:
        pd.DataFrame(metrics).to_csv(results_dir / "metrics_test.csv", index=False)

    reports = collect_jsons(models_dir, "classification_report.json")
    if reports:
        pd.DataFrame(reports).to_json(results_dir / "classification_reports.json", orient="records", indent=2)

    plots_dir = results_dir / "plots"
    cm_dir = results_dir / "confusion_plots"
    manifest_rows = []

    for csv_path in models_dir.rglob("trainlog.csv"):
        out_path = plots_dir / f"{csv_path.parent.name}_curves.png"
        res = plot_learning_curves(csv_path, out_path)
        if res:
            manifest_rows.append({"trainlog": str(csv_path), "plot": str(res)})

    confusion_rows = []
    for cm_path in confusion_dir.rglob("confusion_matrix.csv"):
        labels = []
        summary_path = cm_path.parent / "summary.json"
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as fh:
                summary = json.load(fh)
                label_map = summary.get("id2label") or {}
                labels = [str(label_map[str(i)] if isinstance(label_map, dict) else label_map[i])
                          if isinstance(label_map, dict) and str(i) in label_map
                          else str(label_map.get(i, i))
                          for i in range(len(summary.get("label2id", {})))]
        if not labels:
            labels = [str(i) for i in range(pd.read_csv(cm_path, header=None).shape[0])]
        out_path = cm_dir / f"{cm_path.parent.name}_cm.png"
        res = plot_confusion_matrix(cm_path, out_path, labels)
        if res:
            confusion_rows.append({"confusion_csv": str(cm_path), "plot": str(res)})

    if manifest_rows:
        pd.DataFrame(manifest_rows).to_csv(results_dir / "plots_manifest.csv", index=False)
    if confusion_rows:
        pd.DataFrame(confusion_rows).to_csv(results_dir / "confusion_manifest.csv", index=False)

    print("[DONE] Resultados consolidados en", results_dir)


if __name__ == "__main__":
    main()
