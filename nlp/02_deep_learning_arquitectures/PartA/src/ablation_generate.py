#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lanza barridos de generación para análisis de ablación."""

import argparse
import itertools
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_list(arg, cast=float):
    vals = []
    for item in arg.split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(cast(item))
    return vals


def main():
    ap = argparse.ArgumentParser(description="Ablaciones de generación")
    ap.add_argument("--checkpoint", required=True, help="Ruta al checkpoint")
    ap.add_argument("--model_type", choices=["rnn", "gpt"], required=True)
    ap.add_argument("--contexts", type=str, default="256",
                    help="Lista separada por comas de longitudes de contexto (-> max_length)")
    ap.add_argument("--temperatures", type=str, default="0.7,1.0,1.2")
    ap.add_argument("--top_ps", type=str, default="0.9,0.95")
    ap.add_argument("--top_ks", type=str, default="40,60")
    ap.add_argument("--num_samples", type=int, default=3)
    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base_output", type=str, default="../results/ablations",
                    help="Directorio raíz para guardar resultados")
    ap.add_argument("--use_lora", action="store_true", help="Activa modo LoRA en generate.py")
    ap.add_argument("--base_model", type=str, default="",
                    help="Ruta al modelo base cuando se usa LoRA")
    args = ap.parse_args()

    contexts = parse_list(args.contexts, int)
    temps = parse_list(args.temperatures, float)
    top_ps = parse_list(args.top_ps, float)
    top_ks = parse_list(args.top_ks, int)

    base_output = Path(args.base_output).resolve()
    base_output.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    combos = itertools.product(contexts, temps, top_ps, top_ks)
    for ctx, temp, top_p, top_k in combos:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base_output / f"ctx{ctx}_temp{temp:.2f}_topp{top_p:.2f}_topk{top_k}"
        run_dir.mkdir(parents=True, exist_ok=True)

        log_file = run_dir / f"generate_{stamp}.log"
        metrics_file = run_dir / "metrics.jsonl"

        cmd = [
            sys.executable, "src/generate.py",
            "--checkpoint", args.checkpoint,
            "--model_type", args.model_type,
            "--max_length", str(ctx),
            "--temperature", f"{temp}",
            "--top_p", f"{top_p}",
            "--top_k", str(top_k),
            "--num_samples", str(args.num_samples),
            "--seed", str(args.seed),
            "--output_dir", str(run_dir),
            "--log_file", str(log_file),
            "--metrics_file", str(metrics_file)
        ]
        if args.prompt:
            cmd += ["--prompt", args.prompt]
        if args.use_lora:
            cmd.append("--use_lora")
            if args.base_model:
                cmd += ["--base_model", args.base_model]

        env = os.environ.copy()
        print("[RUN]", " ".join(cmd))
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent, env=env)
        status = "ok" if result.returncode == 0 else f"fail({result.returncode})"

        manifest_rows.append({
            "timestamp": stamp,
            "context": ctx,
            "temperature": temp,
            "top_p": top_p,
            "top_k": top_k,
            "status": status,
            "output_dir": str(run_dir),
            "log_file": str(log_file),
            "metrics_file": str(metrics_file)
        })

    if manifest_rows:
        pd.DataFrame(manifest_rows).to_csv(base_output / "ablation_manifest.csv", index=False)


if __name__ == "__main__":
    main()
