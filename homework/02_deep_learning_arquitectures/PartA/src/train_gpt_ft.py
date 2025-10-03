#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_gpt_ft.py
Fine-tuning de modelos Causal LM (GPT-2 / TinyLlama) sobre un corpus unificado
con <|startsong|>/<|endsong|>. Soporta:
 - FT completo (por defecto) o LoRA/QLoRA (--use_lora, --bits {16,8,4})
 - Control de pasos (--max_steps) para smoke tests
 - Logging CSV y muestras de generación

Requiere: transformers, datasets, accelerate, peft
Opcional p/QLoRA: bitsandbytes (en el servidor)
"""

import os, math, argparse, csv, time, random
from pathlib import Path
import numpy as np
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)

# PEFT / LoRA
from peft import LoraConfig, get_peft_model, TaskType

# --------- Utils ----------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def save_csv_row(csv_path, row_dict, header_order):
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header_order)
        if not exists:
            w.writeheader()
        w.writerow(row_dict)

def load_corpus_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    # quitamos delimitadores del entrenamiento
    txt = txt.replace("<|startsong|>", "").replace("<|endsong|>", "")
    return txt.strip()

# chunking a bloques de longitud fija
def chunk_tokens(token_ids, block_size):
    # crea ventanas consecutivas sin solape
    for i in range(0, len(token_ids) - block_size, block_size):
        yield token_ids[i:i+block_size]

@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt="", max_new_tokens=120, temperature=0.9):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

# --------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", type=str, default="../data/canciones.txt")
    ap.add_argument("--val_split", type=float, default=0.1)

    ap.add_argument("--model_name", type=str, default="gpt2",
                    help="p.ej. gpt2, gpt2-medium, TinyLlama/TinyLlama-1.1B-Chat-v1.0, etc.")
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_steps", type=int, default=-1, help=">0 para limitar pasos (smoke test)")

    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--bits", type=int, default=16, choices=[16,8,4], help="16=floating; 8/4 requieren bitsandbytes")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--gradient_checkpointing", action="store_true")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", type=str, default="../models/gpt_ft")
    ap.add_argument("--log_csv", type=str, default="../models/gpt_ft/trainlog.csv")
    ap.add_argument("--sample_every", type=int, default=1)
    ap.add_argument("--sample_temp", type=float, default=0.9)
    ap.add_argument("--sample_len", type=int, default=200)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    # --------- Tokenizer / Model ----------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {}
    if args.bits in (4,8):
        try:
            import bitsandbytes as bnb  # noqa
        except Exception as e:
            raise SystemExit(f"[ERROR] Pediste bits={args.bits} pero no está instalado bitsandbytes: {e}")
        load_kwargs.update(dict(
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_8bit=(args.bits==8),
            load_in_4bit=(args.bits==4),
        ))
    else:
        load_kwargs.update(dict(torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32))

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.use_lora:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, bias="none", target_modules=["c_attn","attn","q_proj","v_proj","k_proj","o_proj"]
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # --------- Datos ----------
    text = load_corpus_txt(args.train_file)
    # tokenizar todo y partir en bloques
    token_ids = tokenizer(text, return_tensors=None, add_special_tokens=False)["input_ids"]
    blocks = list(chunk_tokens(token_ids, args.seq_len))
    if len(blocks) < 10:
        print("[WARN] Muy pocos bloques; considera bajar seq_len o aumentar corpus.")
    n = len(blocks)
    n_val = max(1, int(n * args.val_split))
    train_blocks = blocks[:-n_val] if n_val < n else blocks
    val_blocks   = blocks[-n_val:] if n_val < n else blocks[-1:]

    train_ds = Dataset.from_dict({"input_ids": train_blocks})
    val_ds   = Dataset.from_dict({"input_ids": val_blocks})

    # data collator para LM
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --------- Trainer ----------
    out_dir = Path(args.save_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_csv = Path(args.log_csv); log_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["epoch_or_step","train_loss","eval_loss","eval_ppl","is_best","steps","lr"]

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs if args.max_steps <= 0 else 1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        evaluation_strategy="steps" if args.max_steps>0 else "epoch",
        save_strategy="steps" if args.max_steps>0 else "epoch",
        save_total_limit=2,
        logging_steps=50,
        logging_dir=str(out_dir / "tb"),
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        report_to=[],
        max_steps=args.max_steps if args.max_steps>0 else -1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    # hook para escribir CSV cuando haya evaluación
    best_eval = float("inf")
    def compute_and_log_metrics(eval_loss, step_or_epoch, steps_done, lr):
        nonlocal best_eval
        eval_ppl = math.exp(min(20, eval_loss)) if eval_loss is not None else None
        is_best = ""
        if (eval_loss is not None) and (eval_loss < best_eval):
            best_eval = eval_loss
            is_best = "YES"
        save_csv_row(str(log_csv), {
            "epoch_or_step": step_or_epoch,
            "train_loss": "",
            "eval_loss": f"{eval_loss:.6f}" if eval_loss is not None else "",
            "eval_ppl": f"{eval_ppl:.6f}" if eval_ppl is not None else "",
            "is_best": is_best,
            "steps": steps_done,
            "lr": f"{lr:.8f}" if lr is not None else ""
        }, header)

    # Entrenamiento
    t0 = time.time()
    trainer.train()
    dt = time.time() - t0

    # Eval final y logging
    metrics = trainer.evaluate()
    eval_loss = metrics.get("eval_loss", None)
    compute_and_log_metrics(eval_loss, "final", trainer.state.global_step, trainer.state.get_last_lr()[0] if trainer.state.get_last_lr() else None)
    print(f"[DONE] eval_loss={eval_loss:.4f} | ppl={math.exp(eval_loss):.2f} | {dt/60:.1f} min")

    # Muestra generada
    try:
        prompt = ""
        sample = generate_sample(model, tokenizer, device, prompt=prompt, max_new_tokens=args.sample_len, temperature=args.sample_temp)
        with open(out_dir / "sample.txt", "w", encoding="utf-8") as f:
            f.write(sample)
        print(f"[SAMPLE] guardado en {out_dir/'sample.txt'}")
    except Exception as e:
        print(f"[WARN] generación falló: {e}")

    # Guardar (adapters si LoRA; de lo contrario el modelo completo)
    if args.use_lora:
        model.save_pretrained(str(out_dir / "lora_adapters"))
        tokenizer.save_pretrained(str(out_dir))
        print("[SAVE] Adapters LoRA en lora_adapters/")
    else:
        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        print("[SAVE] Modelo completo guardado")

if __name__ == "__main__":
    main()
