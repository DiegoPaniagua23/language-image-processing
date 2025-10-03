#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_gpt_ft.py
===============

Fine-tuning de modelos Causal LM (GPT-2 / TinyLlama) sobre corpus de letras.

CARACTERÍSTICAS:
- Fine-tuning completo o LoRA/QLoRA (eficiente en memoria)
- Soporte para GPT-2 (124M), GPT-2-medium (355M), TinyLlama (1.1B)
- Cuantización 4-bit/8-bit con bitsandbytes (QLoRA)
- Multi-GPU automático con Trainer API
- Logging exclusivo a archivos (requisito Lab-SB)
- Gradient checkpointing para ahorrar memoria

USO BÁSICO:
    # Fine-tuning completo GPT-2:
    python train_gpt_ft.py --model_name gpt2 --epochs 3 --batch_size 8

    # LoRA (eficiente):
    python train_gpt_ft.py --model_name gpt2 --use_lora --epochs 5

    # QLoRA 4-bit (máxima eficiencia):
    python train_gpt_ft.py --model_name gpt2-medium --use_lora --bits 4

SERVIDOR LAB-SB:
    # Con 2x Titan RTX (multi-GPU automático):
    python train_gpt_ft.py --model_name gpt2 --use_lora \\
        --batch_size 16 --grad_accum 2 --epochs 5 \\
        --gradient_checkpointing

OUTPUTS:
- Modelo: {save_dir}/ (completo) o {save_dir}/lora_adapters/ (LoRA)
- Log CSV: {log_csv}
- Sample: {save_dir}/sample.txt
- Logs: {log_file}

MODIFICADO PARA LAB-SB:
- Sin print() - solo logging a archivos
- Multi-GPU automático con Trainer
- Soporte LoRA/QLoRA para memoria limitada

Requiere: transformers, datasets, accelerate, peft
Opcional (servidor): bitsandbytes
"""

import os, math, argparse, csv, time, random, inspect, logging
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)

# PEFT / LoRA
from peft import LoraConfig, get_peft_model, TaskType

# --------- Logging Setup ----------
def setup_logger(log_file: str):
    """
    Configura logging a archivo (NO stdout para Lab-SB).

    Args:
        log_file: Ruta al archivo de log

    Returns:
        Logger configurado
    """
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_gpt_ft")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    # Solo file handler, NO console
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

# --------- Utils ----------
def set_seed(seed=42):
    """Configura semilla para reproducibilidad."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def save_csv_row(csv_path, row_dict, header_order):
    """
    Guarda una fila en archivo CSV (append mode).

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

def load_corpus_txt(path):
    """
    Carga el corpus de texto eliminando delimitadores.

    Args:
        path: Ruta al archivo de corpus

    Returns:
        Texto completo sin delimitadores
    """
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    # quitamos delimitadores del entrenamiento
    txt = txt.replace("<|startsong|>", "").replace("<|endsong|>", "")
    return txt.strip()

# chunking a bloques de longitud fija
def chunk_tokens(token_ids, block_size):
    """
    Divide lista de tokens en bloques de longitud fija sin solapamiento.

    Args:
        token_ids: Lista de IDs de tokens
        block_size: Longitud de cada bloque

    Yields:
        Bloques de longitud block_size
    """
    # crea ventanas consecutivas sin solape
    for i in range(0, len(token_ids) - block_size, block_size):
        yield token_ids[i:i+block_size]

@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt="", max_new_tokens=120, temperature=0.9):
    """
    Genera texto usando el modelo entrenado.

    Args:
        model: Modelo de lenguaje
        tokenizer: Tokenizer
        device: torch.device
        prompt: Texto inicial (vacío = generación desde cero)
        max_new_tokens: Número de tokens a generar
        temperature: Control de aleatoriedad

    Returns:
        Texto generado
    """
    model.eval()
    # Si el prompt vacío da longitud 0, arrancamos con EOS o un espacio
    if not prompt:
        if tokenizer.eos_token_id is not None:
            input_ids = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids, device=device)
        else:
            enc = tokenizer(" ", return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
    else:
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        if enc["input_ids"].shape[-1] == 0:
            enc = tokenizer(" ", return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def _get_current_lr(trainer):
    """
    Obtiene el learning rate actual del trainer.

    Args:
        trainer: Transformers Trainer

    Returns:
        Learning rate actual o None si no disponible
    """
    # intenta por el scheduler
    try:
        if getattr(trainer, "lr_scheduler", None) is not None:
            lrs = trainer.lr_scheduler.get_last_lr()
            if isinstance(lrs, (list, tuple)) and len(lrs)>0:
                return float(lrs[0])
    except Exception:
        pass
    # intenta por el optimizer
    try:
        return float(trainer.optimizer.param_groups[0]["lr"])
    except Exception:
        return None

# --------- Main ----------
def main():
    """
    Función principal para fine-tuning de GPT-2/TinyLlama.

    Workflow:
    1. Carga modelo y tokenizer (con cuantización opcional)
    2. Aplica LoRA si se especifica
    3. Tokeniza corpus y crea bloques
    4. Entrena con Trainer API (multi-GPU automático)
    5. Evalúa y genera samples
    6. Guarda modelo o adapters
    """
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
    ap.add_argument("--log_file", type=str, default=None, help="Path al archivo de log (default: auto-genera)")
    ap.add_argument("--sample_every", type=int, default=1)
    ap.add_argument("--sample_temp", type=float, default=0.9)
    ap.add_argument("--sample_len", type=int, default=200)
    args = ap.parse_args()

    set_seed(args.seed)

    # Setup logging
    if args.log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = args.model_name.replace("/", "_")
        args.log_file = f"../logs/train_gpt_ft_{model_short}_{timestamp}.log"
    logger = setup_logger(args.log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    logger.info(f"Device: {device}, GPUs disponibles: {n_gpus}")
    logger.info(f"Args: {vars(args)}")

    logger.info(f"Args: {vars(args)}")

    # --------- Tokenizer / Model ----------
    logger.info(f"Cargando tokenizer y modelo: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Evitar warnings por longitudes grandes al tokenizar el corpus completo:
    tokenizer.model_max_length = int(1e9)

    # Configuración de carga (cuantización 4-bit/8-bit)
    load_kwargs = {}
    if args.bits in (4,8):
        try:
            import bitsandbytes as bnb  # noqa
            logger.info(f"Usando cuantización {args.bits}-bit con bitsandbytes")
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
        logger.info("Gradient checkpointing habilitado")

    # Aplicar LoRA si se especifica
    if args.use_lora:
        logger.info(f"Aplicando LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, bias="none", target_modules=["c_attn","attn","q_proj","v_proj","k_proj","o_proj"]
        )
        model = get_peft_model(model, lora_cfg)
        # Log trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Parámetros entrenables: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # --------- Datos ----------
    logger.info(f"Cargando corpus desde {args.train_file}")
    text = load_corpus_txt(args.train_file)

    # tokenizar todo y partir en bloques
    logger.info(f"Tokenizando corpus (seq_len={args.seq_len})...")
    token_ids = tokenizer(text, return_tensors=None, add_special_tokens=False, truncation=False)["input_ids"]
    blocks = list(chunk_tokens(token_ids, args.seq_len))

    if len(blocks) < 10:
        logger.warning(f"Solo {len(blocks)} bloques disponibles. Considera bajar seq_len o aumentar corpus.")

    logger.info(f"Total de bloques: {len(blocks)}")

    # Train/val split
    n = len(blocks)
    n_val = max(1, int(n * args.val_split))
    train_blocks = blocks[:-n_val] if n_val < n else blocks
    val_blocks   = blocks[-n_val:] if n_val < n else blocks[-1:]

    logger.info(f"Train blocks: {len(train_blocks)}, Val blocks: {len(val_blocks)}")

    train_ds = Dataset.from_dict({"input_ids": train_blocks})
    val_ds   = Dataset.from_dict({"input_ids": val_blocks})

    # data collator para LM
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --------- Trainer ----------
    logger.info("Configurando Trainer (multi-GPU automático si hay varias GPUs)")
    out_dir = Path(args.save_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_csv = Path(args.log_csv); log_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["epoch_or_step","train_loss","eval_loss","eval_ppl","is_best","steps","lr"]

    base_kwargs = dict(
        output_dir=str(out_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs if args.max_steps <= 0 else 1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        save_total_limit=2,
        logging_steps=50,
        logging_dir=str(out_dir / "tb"),
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        report_to=[],  # NO reportar a tensorboard/wandb (escribimos a archivo)
        max_steps=args.max_steps if args.max_steps>0 else -1,
    )


    # Detecta si la firma usa eval_strategy (4.56.x) o evaluation_strategy (otras)
    ta_sig = inspect.signature(TrainingArguments.__init__).parameters
    extra = {}
    if "eval_strategy" in ta_sig:
        extra["eval_strategy"] = ("steps" if args.max_steps>0 else "epoch")
    else:
        extra["evaluation_strategy"] = ("steps" if args.max_steps>0 else "epoch")

    # save_strategy sí existe en 4.56.x
    if "save_strategy" in ta_sig:
        extra["save_strategy"] = ("steps" if args.max_steps>0 else "epoch")

    training_args = TrainingArguments(**base_kwargs, **extra)

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
        """Callback interno para logging de métricas."""
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
    logger.info("Iniciando entrenamiento...")
    t0 = time.time()
    trainer.train()
    dt = time.time() - t0
    logger.info(f"Entrenamiento completado en {dt/60:.1f} minutos")

    logger.info(f"Entrenamiento completado en {dt/60:.1f} minutos")

    # Eval final y logging
    logger.info("Evaluación final...")
    metrics = trainer.evaluate()
    eval_loss = metrics.get("eval_loss", None)
    compute_and_log_metrics(
        eval_loss,
        "final",
        trainer.state.global_step,
        _get_current_lr(trainer)
    )
    eval_ppl = math.exp(eval_loss) if eval_loss else float('inf')
    logger.info(f"Eval loss: {eval_loss:.4f}, PPL: {eval_ppl:.2f}")

    # Muestra generada
    logger.info("Generando sample de texto...")
    try:
        prompt = ""
        sample = generate_sample(model, tokenizer, device, prompt=prompt, max_new_tokens=args.sample_len, temperature=args.sample_temp)
        with open(out_dir / "sample.txt", "w", encoding="utf-8") as f:
            f.write(sample)
        logger.info(f"Sample guardado en {out_dir/'sample.txt'}")
    except Exception as e:
        logger.warning(f"Generación falló: {e}")

    # Guardar (adapters si LoRA; de lo contrario el modelo completo)
    if args.use_lora:
        model.save_pretrained(str(out_dir / "lora_adapters"))
        tokenizer.save_pretrained(str(out_dir))
        logger.info("Adapters LoRA guardados en lora_adapters/")
    else:
        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        logger.info("Modelo completo guardado")

    logger.info(f"Log CSV: {log_csv}")
    logger.info("Fine-tuning completado exitosamente")

if __name__ == "__main__":
    main()
