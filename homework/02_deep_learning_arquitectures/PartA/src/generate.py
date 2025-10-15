#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate.py
===========

Script para generar texto con modelos entrenados (RNN/LSTM/GRU o LLaMA-3 LoRA).

CARACTERÍSTICAS:
- Soporta modelos RNN/LSTM/GRU (train_textgen.py)
- Soporta LLaMA-3 8B con LoRA adapters (train_llama3_lora.py)
- Generación con control de temperatura, top-k, top-p
- Múltiples samples con diferentes seeds
- Logging a archivos (requisito Lab-SB)
- Prompt automático para modelos LoRA fine-tuned en canciones

USO BÁSICO:
    # Generar con RNN/LSTM/GRU:
    python generate.py --checkpoint ../models/lstm_char/lstm_char_best.pt \\
        --model_type rnn --num_samples 3

    # Generar con LLaMA-3 LoRA adapters:
    python generate.py --checkpoint ../models/llama3_lora \\
        --model_type llama --use_lora \\
        --base_model ../models/llama-3-8b \\
        --num_samples 5

PARÁMETROS DE GENERACIÓN:
    --temperature FLOAT    # 0.7=conservador, 1.0=balanceado, 1.2=creativo
    --top_k INT            # Limita a top-k tokens más probables (solo LLaMA)
    --top_p FLOAT          # Nucleus sampling 0.9-0.95 recomendado (solo LLaMA)
    --max_length INT       # Longitud máxima del texto generado
    --prompt STR           # Prompt inicial (vacío = auto-genera para LLaMA)

OUTPUTS:
- Samples: {output_dir}/sample_{i}_seed{seed}.txt
- Metrics: {output_dir}/metrics.jsonl
- Log: {log_file}

"""

import os, argparse, logging, random, json, time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn

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
    logger = logging.getLogger("generate")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def set_seed(seed=42):
    """Configura semilla para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --------- Modelo RNN/LSTM/GRU (train_textgen.py) ----------
class CharWordLM(nn.Module):
    """
    Modelo de lenguaje RNN/GRU/LSTM (copia de train_textgen.py).
    Necesario para cargar checkpoints entrenados con train_textgen.py
    """
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

# --------- Funciones de Generación ----------
@torch.no_grad()
def generate_char_rnn(model, device, itos, stoi, start_text="", max_length=500, temperature=0.9, logger=None):
    """
    Genera texto character-level con modelo RNN/LSTM/GRU.

    Args:
        model: Modelo CharWordLM entrenado
        device: torch.device
        itos: Dict index->char
        stoi: Dict char->index
        start_text: Prompt inicial
        max_length: Longitud máxima del texto
        temperature: Control de aleatoriedad
        logger: Logger opcional

    Returns:
        Texto generado
    """
    model.eval()
    if not start_text:
        start_text = " "

    if logger:
        logger.info(f"Generando char-level (start='{start_text[:20]}...', max_len={max_length}, temp={temperature})")

    input_ids = torch.tensor([[stoi.get(ch, 0) for ch in start_text]], dtype=torch.long, device=device)
    h = None
    out_text = start_text

    for i in range(max_length):
        logits, h = model(input_ids, h)
        last = logits[:, -1, :] / max(1e-6, temperature)
        probs = torch.softmax(last, dim=-1)
        idx_next = torch.multinomial(probs, 1)
        out_text += itos[idx_next.item()]
        input_ids = idx_next

    return out_text

@torch.no_grad()
def generate_word_rnn(model, device, itos, stoi, start_tokens=None, max_length=100, temperature=0.9, unk_id=1, logger=None):
    """
    Genera texto word-level con modelo RNN/LSTM/GRU.

    Args:
        model: Modelo CharWordLM entrenado
        device: torch.device
        itos: Dict index->word
        stoi: Dict word->index
        start_tokens: Lista de tokens iniciales
        max_length: Número de palabras a generar
        temperature: Control de aleatoriedad
        unk_id: ID del token <unk>
        logger: Logger opcional

    Returns:
        Texto generado
    """
    model.eval()
    if not start_tokens:
        start_tokens = []

    if logger:
        logger.info(f"Generando word-level (start_tokens={start_tokens}, max_len={max_length}, temp={temperature})")

    if len(start_tokens) == 0:
        # Token frecuente (id 2 suele ser común, no <pad>/<unk>)
        start_ids = [2]
    else:
        start_ids = [stoi.get(t, unk_id) for t in start_tokens]

    input_ids = torch.tensor([start_ids], dtype=torch.long, device=device)
    h = None
    out_ids = list(start_ids)

    for i in range(max_length):
        logits, h = model(input_ids, h)
        last = logits[:, -1, :] / max(1e-6, temperature)
        probs = torch.softmax(last, dim=-1)
        idx_next = torch.multinomial(probs, 1)
        out_ids.append(idx_next.item())
        input_ids = idx_next

    # Reconstruir texto
    toks = [itos[i] for i in out_ids if itos[i] not in ("<pad>",)]
    return " ".join(toks)

@torch.no_grad()
def generate_llama(model, tokenizer, device, prompt="", max_length=300, temperature=0.9, top_k=50, top_p=0.95, logger=None):
    """
    Genera texto con modelo LLaMA-3 (con o sin LoRA).

    Args:
        model: Modelo AutoModelForCausalLM o PeftModel
        tokenizer: Tokenizer de LLaMA
        device: torch.device
        prompt: Texto inicial (si está vacío, se usa prompt por defecto)
        max_length: Número de tokens a generar
        temperature: Control de aleatoriedad (0.7-1.2)
        top_k: Top-k sampling (20-60 típico)
        top_p: Nucleus sampling (0.85-0.95 típico)
        logger: Logger opcional

    Returns:
        Texto generado
    """
    model.eval()

    if logger:
        logger.info(f"Generando LLaMA (prompt='{prompt[:40]}...', max_len={max_length}, temp={temperature}, top_k={top_k}, top_p={top_p})")

    # Preparar input
    if not prompt:
        # Usar espacio como fallback mínimo
        enc = tokenizer(" ", return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
    else:
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        if enc["input_ids"].shape[-1] == 0:
            enc = tokenizer(" ", return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

    # Generar
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_length,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(out[0], skip_special_tokens=True)

# --------- Main ----------
def main():
    """
    Función principal para generación de texto.

    Workflow:
    1. Carga checkpoint (RNN/LSTM/GRU o LLaMA-3 LoRA)
    2. Configura parámetros de generación
    3. Genera múltiples samples con diferentes seeds
    4. Guarda samples y métricas en archivos
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path al checkpoint (.pt para RNN, directorio para LLaMA)")
    ap.add_argument("--model_type", type=str, required=True, choices=["rnn", "llama"],
                    help="Tipo de modelo: 'rnn' (RNN/LSTM/GRU) o 'llama' (LLaMA-3 LoRA)")

    # Parámetros de generación
    ap.add_argument("--prompt", type=str, default="",
                    help="Texto inicial (vacío = auto-genera prompt de canciones para LLaMA)")
    ap.add_argument("--max_length", type=int, default=300,
                    help="Longitud máxima del texto generado (tokens para LLaMA, chars/words para RNN)")
    ap.add_argument("--temperature", type=float, default=0.9,
                    help="Temperatura (0.7=conservador, 1.0=balanceado, 1.2=creativo)")
    ap.add_argument("--top_k", type=int, default=50,
                    help="Top-k sampling (solo LLaMA, valores típicos: 20-60)")
    ap.add_argument("--top_p", type=float, default=0.95,
                    help="Nucleus sampling (solo LLaMA, valores típicos: 0.85-0.95)")
    ap.add_argument("--num_samples", type=int, default=1,
                    help="Número de samples a generar con diferentes seeds")

    # Para modelos LoRA (LLaMA-3)
    ap.add_argument("--use_lora", action="store_true",
                    help="Si el checkpoint contiene LoRA adapters (solo para LLaMA)")
    ap.add_argument("--base_model", type=str, default="../models/llama-3-8b",
                    help="Path al modelo base LLaMA-3 8B (requerido para LoRA)")

    # Output
    ap.add_argument("--output_dir", type=str, default="../results/samples",
                    help="Directorio para guardar samples")
    ap.add_argument("--log_file", type=str, default=None,
                    help="Archivo de log (default: auto-genera)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Semilla inicial (se incrementa por sample)")
    ap.add_argument("--metrics_file", type=str, default=None,
                    help="Archivo JSONL para registrar métricas de inferencia")

    args = ap.parse_args()

    # Setup logging
    if args.log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = f"../logs/generate_{timestamp}.log"
    logger = setup_logger(args.log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Args: {vars(args)}")

    # Crear directorio de output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.metrics_file is None:
        args.metrics_file = str(output_dir / "metrics.jsonl")
    metrics_path = Path(args.metrics_file)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_rows = []

    # --------- Cargar Modelo ----------
    if args.model_type == "rnn":
        logger.info(f"Cargando modelo RNN desde {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)

        # Extraer info del checkpoint
        model_args = checkpoint["args"]
        stoi = checkpoint["stoi"]
        itos = checkpoint["itos"]
        vocab_size = len(stoi)

        # Reconstruir modelo
        model = CharWordLM(
            vocab_size=vocab_size,
            embed_dim=model_args["embedding_dim"],
            hidden_dim=model_args["hidden_size"],
            num_layers=model_args["num_layers"],
            arch=model_args["arch"],
            dropout=model_args.get("dropout", 0.3)
        ).to(device)

        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        level = model_args.get("level", "char")
        logger.info(f"Modelo cargado: {model_args['arch']} ({level}-level, vocab={vocab_size})")

    elif args.model_type == "llama":
        logger.info(f"Cargando modelo LLaMA-3 desde {args.checkpoint}")

        from transformers import AutoTokenizer, AutoModelForCausalLM

        if args.use_lora:
            # Cargar modelo base + LoRA adapters
            logger.info(f"Cargando base model: {args.base_model}")
            from peft import PeftModel

            base_model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )

            lora_path = Path(args.checkpoint) / "lora_adapters"
            if not lora_path.exists():
                lora_path = Path(args.checkpoint)

            logger.info(f"Cargando LoRA adapters desde {lora_path}")
            model = PeftModel.from_pretrained(base_model, str(lora_path))
            model.eval()

            tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        else:
            # Cargar modelo LLaMA completo (full fine-tuning)
            logger.warning("Cargando LLaMA sin LoRA. Esto requiere mucha memoria.")
            tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
            model = AutoModelForCausalLM.from_pretrained(
                args.checkpoint,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Modelo LLaMA-3 cargado exitosamente")

    # --------- Generar Samples ----------
    logger.info(f"Generando {args.num_samples} samples...")

    for i in range(args.num_samples):
        sample_seed = args.seed + i
        set_seed(sample_seed)

        logger.info(f"\n=== Sample {i+1}/{args.num_samples} (seed={sample_seed}) ===")

        try:
            # Inicializar current_prompt para ambos tipos de modelos
            current_prompt = args.prompt

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t_start = time.time()
            if args.model_type == "rnn":
                if level == "char":
                    text = generate_char_rnn(
                        model, device, itos, stoi,
                        start_text=args.prompt,
                        max_length=args.max_length,
                        temperature=args.temperature,
                        logger=logger
                    )
                else:  # word-level
                    start_tokens = args.prompt.split() if args.prompt else None
                    unk_id = stoi.get("<unk>", 1)
                    text = generate_word_rnn(
                        model, device, itos, stoi,
                        start_tokens=start_tokens,
                        max_length=args.max_length,
                        temperature=args.temperature,
                        unk_id=unk_id,
                        logger=logger
                    )

            elif args.model_type == "llama":
                # Si no hay prompt, usar uno que guíe hacia letras de canciones
                current_prompt = args.prompt
                if not current_prompt and args.use_lora:
                    # Prompt por defecto para modelos fine-tuned en canciones
                    current_prompt = "<|startsong|> I'm down for whatever, I gotta blow son, is now or never\n"
                    logger.info("Usando prompt por defecto para canciones")

                text = generate_llama(
                    model, tokenizer, device,
                    prompt=current_prompt,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    logger=logger
                )
            infer_time = time.time() - t_start
            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
            else:
                peak_mem = 0.0

            # Guardar sample
            output_file = output_dir / f"sample_{i+1}_seed{sample_seed}.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"# Sample {i+1}\n")
                f.write(f"# Seed: {sample_seed}\n")
                f.write(f"# Temperature: {args.temperature}\n")
                # Guardar el prompt real usado (puede ser diferente del argumento)
                actual_prompt = current_prompt if args.model_type == "llama" else args.prompt
                f.write(f"# Prompt: '{actual_prompt}'\n")
                f.write(f"# {'='*60}\n\n")
                f.write(text)

            logger.info(f"Sample guardado en {output_file}")
            logger.info(f"Longitud: {len(text)} caracteres")
            logger.info(f"Preview: {text[:100]}...")

            metrics_rows.append({
                "timestamp": datetime.now().isoformat(),
                "sample_id": i + 1,
                "seed": sample_seed,
                "model_type": args.model_type,
                "checkpoint": args.checkpoint,
                "prompt": args.prompt,
                "max_length": args.max_length,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "inference_time_sec": infer_time,
                "peak_gpu_mem_mb": peak_mem,
                "output_file": str(output_file)
            })

        except Exception as e:
            logger.error(f"Error generando sample {i+1}: {e}")
            continue

    logger.info(f"\n{'='*60}")
    logger.info(f"Generación completada: {args.num_samples} samples en {output_dir}")
    logger.info(f"Log completo: {args.log_file}")

    if metrics_rows:
        with open(metrics_path, "a", encoding="utf-8") as f:
            for row in metrics_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(f"Métricas registradas en {metrics_path}")

if __name__ == "__main__":
    main()
