#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Descarga el modelo de lenguaje Meta-Llama-3-8B."""

import os
from huggingface_hub import snapshot_download


LOCAL_MODEL_PATH = "/home/diego23/Downloads/llama-3-8b"
MODEL_ID = "meta-llama/Meta-Llama-3-8B"

print(f"Iniciando la descarga de {MODEL_ID}...")

try:
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_MODEL_PATH,
        local_dir_use_symlinks=False,
        # token=os.environ.get("HF_TOKEN"), # Se usa si iniciaste sesión con `huggingface-cli login`
    )

    print(f"\n Descarga completa. El modelo se encuentra en el directorio: {os.path.abspath(LOCAL_MODEL_PATH)}")
    print("El directorio contiene todos los archivos necesarios para cargar el modelo 'offline'.")

except Exception as e:
    print(f"\n Error durante la descarga. Asegúrate de haber aceptado los términos de uso de Llama 3 en Hugging Face y haber iniciado sesión con `huggingface-cli login`.")
    print(f"Detalles del error: {e}")
