#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fine-tuning LoRA de LLaMA 3 8B sin dependencias de GPT-2."""

import argparse
import inspect
import json
import logging
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


DEFAULT_BASE_MODEL = Path("../models/llama-3-8b").resolve()
DEFAULT_OUTPUT_DIR = Path("../models/llama3_lora").resolve()


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_llama3_lora")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_corpus(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    return text.replace("<|startsong|>", "").replace("<|endsong|>", "").strip()


def chunk_tokens(token_ids, block_size, stride=None):
    """
    Divide tokens en bloques con overlap para maximizar datos de entrenamiento.

    Args:
        token_ids: Lista de IDs de tokens
        block_size: Longitud de cada bloque
        stride: Paso entre bloques (default: block_size // 4 para más overlap)

    Yields:
        Bloques de tokens de longitud block_size
    """
    if stride is None:
        # Reducir stride para generar MÁS bloques con overlap
        # Esto ayuda a que el modelo vea más veces el contenido de canciones
        stride = max(1, block_size // 4)  # Cambiado de // 2 a // 4
    for i in range(0, len(token_ids) - block_size + 1, stride):
        yield token_ids[i:i + block_size]


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning LoRA de LLaMA 3 8B sobre canciones")
    parser.add_argument("--train_file", default="../data/canciones.txt")
    parser.add_argument("--base_model", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8, 16])
    parser.add_argument("--sample_len", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_file", default="logs/train_llama3_lora.log")
    parser.add_argument("--summary_path", default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log_file).resolve()
    logger = setup_logger(log_file)

    logger.info("Inicializando fine-tuning LoRA para %s", args.base_model)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {}
    if args.bits in (4, 8):
        try:
            import bitsandbytes as bnb  # noqa: F401
            from transformers import BitsAndBytesConfig
        except Exception as exc:  # pragma: no cover
            raise SystemExit(f"[ERROR] bitsandbytes requerido para bits={args.bits}: {exc}")

        # Obtener el device local correcto para entrenamiento distribuido
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Configuración correcta para cuantización en multi-GPU
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=(args.bits == 4),
            load_in_8bit=(args.bits == 8),
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        load_kwargs = dict(
            quantization_config=bnb_config,
            device_map={"": local_rank},
            torch_dtype=torch.float16,
        )
    else:
        load_kwargs = dict(torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    if args.bits in (4, 8):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    # Deshabilitar gradient checkpointing para evitar conflictos con DDP
    model.config.use_cache = False
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Parámetros entrenables: %d (%.2f%% del total %d)", trainable, 100 * trainable / total_params, total_params)

    text = load_corpus(Path(args.train_file).resolve())
    token_ids = tokenizer(text, return_tensors=None, add_special_tokens=False)["input_ids"]
    blocks = list(chunk_tokens(token_ids, args.seq_len))
    if len(blocks) < 2:
        raise SystemExit("[ERROR] Corpus insuficiente para crear bloques de entrenamiento")
    logger.info("Bloques disponibles: %d", len(blocks))

    n_val = max(1, int(0.1 * len(blocks)))
    train_blocks = blocks[:-n_val]
    val_blocks = blocks[-n_val:]

    train_ds = Dataset.from_dict({"input_ids": train_blocks})
    val_ds = Dataset.from_dict({"input_ids": val_blocks})
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_kwargs = {
        "output_dir": str(output_dir / "hf_run"),
        "overwrite_output_dir": True,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_dir": str(output_dir / "tb"),
        "logging_strategy": "epoch",
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "report_to": [],
        "fp16": torch.cuda.is_available(),
        "seed": args.seed,
        "gradient_checkpointing": False,  # Deshabilitar para evitar conflictos con DDP+LoRA
        "ddp_find_unused_parameters": False,  # Optimización para DDP
    }

    training_signature = inspect.signature(TrainingArguments.__init__)
    accepted_params = set(training_signature.parameters.keys())
    steps_per_epoch = max(1, math.ceil(len(train_ds) / (args.batch_size * args.grad_accum)))

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
        training_kwargs["warmup_steps"] = max(0, int(total_training_steps * args.warmup_ratio))
        logger.warning("warmup_ratio no disponible; se usa warmup_steps=%d", training_kwargs["warmup_steps"])

    if not evaluation_supported:
        for key in ["load_best_model_at_end", "metric_for_best_model", "greater_is_better"]:
            if key in training_kwargs:
                logger.warning("%s requiere evaluation_strategy; se elimina del TrainingArguments", key)
                training_kwargs.pop(key)

    for legacy_key in ["report_to", "fp16", "save_total_limit"]:
        if legacy_key in training_kwargs and legacy_key not in accepted_params:
            logger.warning("%s no disponible; se elimina del TrainingArguments", legacy_key)
            training_kwargs.pop(legacy_key)

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    logger.info("Comienza entrenamiento...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    logger.info("Entrenamiento completado en %.2f min", train_time / 60)

    metrics = trainer.evaluate()
    eval_loss = metrics.get("eval_loss")
    eval_ppl = math.exp(eval_loss) if eval_loss is not None else None
    logger.info("Eval loss=%.4f, ppl=%s", eval_loss if eval_loss is not None else float("nan"),
                f"{eval_ppl:.3f}" if eval_ppl else "na")

    adapters_dir = output_dir / "lora_adapters"
    model.save_pretrained(str(adapters_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Adapters guardados en %s", adapters_dir)

    if args.summary_path:
        summary_path = Path(args.summary_path).resolve()
    else:
        summary_path = output_dir / "summary.json"

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_mem = 0.0

    summary = {
        "base_model": args.base_model,
        "output_dir": str(output_dir),
        "epochs": args.epochs,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "bits": args.bits,
        "train_blocks": len(train_blocks),
        "val_blocks": len(val_blocks),
        "train_time_sec": train_time,
        "eval_loss": eval_loss,
        "eval_ppl": eval_ppl,
        "peak_gpu_mem_mb": peak_mem,
        "log_file": str(log_file),
        "trainable_params": trainable,
        "total_params": total_params,
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Resumen guardado en %s", summary_path)


if __name__ == "__main__":
    main()
