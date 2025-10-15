
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_zipf(freq_df:pd.DataFrame, out_png:str):
    plt.figure(figsize=(7.5,6))
    plt.scatter(freq_df["log_rank"], freq_df["log_freq"], s=2, alpha=0.5, rasterized=True)
    plt.xlabel("log(rango)")
    plt.ylabel("log(frecuencia)")
    plt.title("Ley de Zipf — Dispersión log-log")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

def plot_confusion(cm_df:pd.DataFrame, title:str, out_png:str):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title(title); plt.ylabel("Etiqueta real"); plt.xlabel("Predicción")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()
