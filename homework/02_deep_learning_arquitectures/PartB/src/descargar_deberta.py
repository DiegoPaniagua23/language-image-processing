#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Descarga el modelo de lenguaje mdeberta-v3-base."""

from transformers import AutoModelForSequenceClassification, DebertaV2Tokenizer
import os

MODEL_NAME = "microsoft/mdeberta-v3-base"
SAVE_DIRECTORY = "/home/diego23/Downloads/mdeberta-v3-base"


if not os.path.exists(SAVE_DIRECTORY):
    os.makedirs(SAVE_DIRECTORY)
    print(f"Directorio '{SAVE_DIRECTORY}' creado.")

print("-" * 50)

try:
    print(f"Descargando tokenizador de '{MODEL_NAME}'...")
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(SAVE_DIRECTORY)
    print(f"Tokenizador guardado exitosamente en '{SAVE_DIRECTORY}'")
    print("-" * 50)

    print(f"Descargando modelo '{MODEL_NAME}'...")
    print("Esto puede tardar unos minutos dependiendo de tu conexión...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.save_pretrained(SAVE_DIRECTORY)
    print(f"Modelo guardado exitosamente en '{SAVE_DIRECTORY}'")
    print("-" * 50)

    print("\n Proceso completado")

except Exception as e:
    print(f"\n Ocurrió un error durante la descarga: {e}")
    print("Verifica tu conexión a internet o el identificador del modelo.")
