#!/usr/bin/env python3
"""
Análisis Consolidado de Ablaciones - Parte A
==============================================

Script para analizar resultados de ablaciones de generación de texto.
Genera visualizaciones comparativas y tablas de ranking para los diferentes
modelos (RNN, LSTM, GRU char/word, LLaMA-3) con distintas configuraciones de
hiperparámetros de muestreo (temperature, top_p, top_k, context_length).
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo para gráficas profesionales
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16


def load_ablation_results(ablations_dir: Path, model_name: str) -> pd.DataFrame:
    """
    Carga los resultados de ablaciones desde archivos metrics.jsonl.

    Args:
        ablations_dir: Directorio raíz de ablaciones (results/ablations/)
        model_name: Nombre del modelo (e.g., 'gru_char', 'llama3_lora')

    Returns:
        DataFrame con las métricas agregadas por configuración
    """
    model_dir = ablations_dir / model_name

    if not model_dir.exists():
        print(f"Advertencia: No se encontró {model_dir}")
        return pd.DataFrame()

    print(f"Cargando ablaciones de: {model_name}")

    results = []
    config_dirs = [d for d in model_dir.iterdir() if d.is_dir()]

    for config_dir in sorted(config_dirs):
        metrics_file = config_dir / "metrics.jsonl"

        if not metrics_file.exists():
            continue

        # Parsear nombre del directorio para extraer hiperparámetros
        # Formato: ctx{context}_temp{temp}_topp{topp}_topk{topk}
        config_name = config_dir.name
        try:
            parts = config_name.split('_')
            context = int(parts[0].replace('ctx', ''))
            temperature = float(parts[1].replace('temp', ''))
            top_p = float(parts[2].replace('topp', ''))
            top_k = int(parts[3].replace('topk', ''))
        except (ValueError, IndexError):
            print(f"No se pudo parsear: {config_name}")
            continue

        # Leer métricas de todas las muestras
        samples_metrics = []
        with open(metrics_file, 'r') as f:
            for line in f:
                if line.strip():
                    samples_metrics.append(json.loads(line))

        if not samples_metrics:
            continue

        # Agregar métricas (promedio de las 3 muestras)
        inference_times = [m['inference_time_sec'] for m in samples_metrics]
        gpu_mems = [m['peak_gpu_mem_mb'] for m in samples_metrics]

        results.append({
            'model': model_name,
            'context_length': context,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'avg_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'avg_gpu_memory': np.mean(gpu_mems),
            'std_gpu_memory': np.std(gpu_mems),
            'num_samples': len(samples_metrics),
            'config_dir': str(config_dir)
        })

    df = pd.DataFrame(results)
    print(f"   ✓ {len(df)} configuraciones cargadas")

    return df


def load_all_models(ablations_dir: Path) -> pd.DataFrame:
    """
    Carga resultados de ablaciones de todos los modelos disponibles.

    Args:
        ablations_dir: Directorio raíz de ablaciones

    Returns:
        DataFrame consolidado con todos los modelos
    """
    all_models = [
        'rnn_char', 'rnn_word',
        'lstm_char', 'lstm_word',
        'gru_char', 'gru_word',
        'llama3_lora'
    ]

    dfs = []
    for model in all_models:
        df = load_ablation_results(ablations_dir, model)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        raise ValueError(" No se encontraron ablaciones")

    return pd.concat(dfs, ignore_index=True)


def plot_temperature_vs_inference_time(df: pd.DataFrame, output_dir: Path):
    """
    Gráfica: Temperature vs Tiempo de Inferencia (líneas por context_length).
    """
    print("\n[2/8] Generando gráfica: Temperature vs Inference Time...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Efecto de Temperature en Tiempo de Inferencia por Tipo de Modelo',
                 fontsize=16, fontweight='bold')

    model_groups = [
        (['rnn_char', 'lstm_char', 'gru_char'], 'Character-Level Models'),
        (['rnn_word', 'lstm_word', 'gru_word'], 'Word-Level Models'),
    ]

    for idx, (models, title) in enumerate(model_groups):
        ax = axes[idx, 0]
        subset = df[df['model'].isin(models)]

        if subset.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(title)
            continue

        for ctx in sorted(subset['context_length'].unique()):
            ctx_data = subset[subset['context_length'] == ctx]
            for model in models:
                model_data = ctx_data[ctx_data['model'] == model].sort_values('temperature')
                if not model_data.empty:
                    ax.plot(model_data['temperature'],
                           model_data['avg_inference_time'],
                           marker='o', label=f'{model} (ctx={ctx})', linewidth=2)

        ax.set_xlabel('Temperature', fontsize=12)
        ax.set_ylabel('Avg Inference Time (sec)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    # LLaMA-3 en subplot separado (si existe)
    ax = axes[0, 1]
    llama_data = df[df['model'] == 'llama3_lora']
    if not llama_data.empty:
        for ctx in sorted(llama_data['context_length'].unique()):
            ctx_data = llama_data[llama_data['context_length'] == ctx].sort_values('temperature')
            ax.plot(ctx_data['temperature'],
                   ctx_data['avg_inference_time'],
                   marker='s', label=f'ctx={ctx}', linewidth=2, markersize=8)

        ax.set_xlabel('Temperature', fontsize=12)
        ax.set_ylabel('Avg Inference Time (sec)', fontsize=12)
        ax.set_title('LLaMA-3 with LoRA', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'LLaMA-3 data not available', ha='center', va='center')
        ax.set_title('LLaMA-3 with LoRA', fontsize=14)

    # Comparación entre arquitecturas
    ax = axes[1, 1]
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        # Promediar sobre todas las configuraciones
        grouped = model_data.groupby('temperature')['avg_inference_time'].mean()
        ax.plot(grouped.index, grouped.values, marker='o',
               label=model, linewidth=2, markersize=8)

    ax.set_xlabel('Temperature', fontsize=12)
    ax.set_ylabel('Avg Inference Time (sec)', fontsize=12)
    ax.set_title('Comparación entre Arquitecturas', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "temperature_vs_inference_time.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Gráfica guardada: {output_file.name}")


def plot_context_vs_memory(df: pd.DataFrame, output_dir: Path):
    """
    Gráfica: Context Length vs GPU Memory Usage (barras agrupadas por modelo).
    """
    print("\n[3/8] Generando gráfica: Context Length vs GPU Memory...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Uso de Memoria GPU por Longitud de Contexto',
                 fontsize=16, fontweight='bold')

    # Subplot 1: RNN/LSTM/GRU
    ax = axes[0]
    rnn_models = ['rnn_char', 'rnn_word', 'lstm_char', 'lstm_word', 'gru_char', 'gru_word']
    rnn_data = df[df['model'].isin(rnn_models)]

    if not rnn_data.empty:
        pivot = rnn_data.groupby(['model', 'context_length'])['avg_gpu_memory'].mean().unstack()
        pivot.plot(kind='bar', ax=ax, width=0.8, colormap='tab10')
        ax.set_xlabel('Modelo', fontsize=12)
        ax.set_ylabel('GPU Memory (MB)', fontsize=12)
        ax.set_title('RNN/LSTM/GRU Models', fontsize=14, fontweight='bold')
        ax.legend(title='Context Length', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')

    # Subplot 2: LLaMA-3
    ax = axes[1]
    llama_data = df[df['model'] == 'llama3_lora']

    if not llama_data.empty:
        pivot = llama_data.groupby('context_length')['avg_gpu_memory'].mean()
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(pivot)))
        pivot.plot(kind='bar', ax=ax, width=0.6, color=colors)
        ax.set_xlabel('Context Length', fontsize=12)
        ax.set_ylabel('GPU Memory (MB)', fontsize=12)
        ax.set_title('LLaMA-3 with LoRA', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

        # Añadir valores sobre las barras
        for i, v in enumerate(pivot.values):
            ax.text(i, v + pivot.max() * 0.02, f'{v:.1f} MB',
                   ha='center', va='bottom', fontsize=10)
    else:
        ax.text(0.5, 0.5, 'LLaMA-3 data not available', ha='center', va='center')

    plt.tight_layout()
    output_file = output_dir / "context_vs_memory.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Gráfica guardada: {output_file.name}")


def plot_sampling_heatmaps(df: pd.DataFrame, output_dir: Path):
    """
    Heatmaps: top_p vs top_k coloreado por inference time (uno por temperatura).
    """
    print("\n[4/8] Generando heatmaps: Sampling Parameters...")

    temperatures = sorted(df['temperature'].unique())
    n_temps = len(temperatures)

    fig, axes = plt.subplots(1, n_temps, figsize=(6 * n_temps, 5))
    if n_temps == 1:
        axes = [axes]

    fig.suptitle('Tiempo de Inferencia: Top-P vs Top-K por Temperature',
                 fontsize=16, fontweight='bold')

    # Promediar sobre todos los modelos y contextos
    for idx, temp in enumerate(temperatures):
        ax = axes[idx]
        temp_data = df[df['temperature'] == temp]

        # Crear pivot table
        pivot = temp_data.pivot_table(
            values='avg_inference_time',
            index='top_p',
            columns='top_k',
            aggfunc='mean'
        )

        if pivot.empty:
            ax.text(0.5, 0.5, f'No data for temp={temp}', ha='center', va='center')
            ax.set_title(f'Temperature = {temp}')
            continue

        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd',
                   cbar_kws={'label': 'Inference Time (sec)'}, ax=ax,
                   linewidths=0.5, linecolor='gray')

        ax.set_title(f'Temperature = {temp}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Top-K', fontsize=12)
        ax.set_ylabel('Top-P', fontsize=12)

    plt.tight_layout()
    output_file = output_dir / "sampling_heatmaps.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Heatmaps guardados: {output_file.name}")


def plot_model_comparison_bars(df: pd.DataFrame, output_dir: Path):
    """
    Gráfica de barras: Comparación de modelos (tiempo promedio de inferencia y memoria).
    """
    print("\n[5/8] Generando gráfica: Model Comparison...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Comparación entre Modelos: Eficiencia Computacional',
                 fontsize=16, fontweight='bold')

    # Agrupar por modelo (promedio global)
    model_stats = df.groupby('model').agg({
        'avg_inference_time': ['mean', 'std'],
        'avg_gpu_memory': ['mean', 'std']
    }).reset_index()

    model_stats.columns = ['model', 'mean_time', 'std_time', 'mean_mem', 'std_mem']
    model_stats = model_stats.sort_values('mean_time')

    # Subplot 1: Tiempo de inferencia
    ax = axes[0]
    colors = plt.cm.tab10(np.arange(len(model_stats)))
    bars = ax.bar(range(len(model_stats)), model_stats['mean_time'],
                  yerr=model_stats['std_time'], capsize=5, color=colors, alpha=0.8)

    ax.set_xticks(range(len(model_stats)))
    ax.set_xticklabels(model_stats['model'], rotation=45, ha='right')
    ax.set_ylabel('Avg Inference Time (sec)', fontsize=12)
    ax.set_title('Tiempo de Inferencia Promedio', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Añadir valores
    for i, (bar, val) in enumerate(zip(bars, model_stats['mean_time'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.3f}s', ha='center', va='bottom', fontsize=9)

    # Subplot 2: Memoria GPU
    ax = axes[1]
    bars = ax.bar(range(len(model_stats)), model_stats['mean_mem'],
                  yerr=model_stats['std_mem'], capsize=5, color=colors, alpha=0.8)

    ax.set_xticks(range(len(model_stats)))
    ax.set_xticklabels(model_stats['model'], rotation=45, ha='right')
    ax.set_ylabel('GPU Memory (MB)', fontsize=12)
    ax.set_title('Uso de Memoria GPU Promedio', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Añadir valores
    for i, (bar, val) in enumerate(zip(bars, model_stats['mean_mem'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f} MB', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_file = output_dir / "model_comparison_bars.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Gráfica guardada: {output_file.name}")


def create_best_configs_table(df: pd.DataFrame, output_dir: Path, top_n: int = 10):
    """
    Crea tabla CSV con las mejores configuraciones (menor tiempo de inferencia).
    """
    print(f"\n[6/8] Creando tabla de Top {top_n} configuraciones...")

    # Ordenar por tiempo de inferencia
    df_sorted = df.sort_values('avg_inference_time').head(top_n).copy()

    # Crear columna de ranking
    df_sorted.insert(0, 'rank', range(1, len(df_sorted) + 1))

    # Seleccionar y formatear columnas
    columns_to_save = [
        'rank', 'model', 'context_length', 'temperature', 'top_p', 'top_k',
        'avg_inference_time', 'std_inference_time', 'avg_gpu_memory', 'std_gpu_memory'
    ]

    df_export = df_sorted[columns_to_save].copy()

    # Formatear números
    df_export['avg_inference_time'] = df_export['avg_inference_time'].map('{:.4f}'.format)
    df_export['std_inference_time'] = df_export['std_inference_time'].map('{:.4f}'.format)
    df_export['avg_gpu_memory'] = df_export['avg_gpu_memory'].map('{:.2f}'.format)
    df_export['std_gpu_memory'] = df_export['std_gpu_memory'].map('{:.2f}'.format)

    # Guardar CSV
    output_file = output_dir / "best_configurations_ranking.csv"
    df_export.to_csv(output_file, index=False)

    print(f"   ✓ Tabla guardada: {output_file.name}")

    return df_sorted


def create_consolidated_data(df: pd.DataFrame, output_dir: Path):
    """
    Guarda DataFrame consolidado con todas las ablaciones.
    """
    print("\n[7/8] Guardando DataFrame consolidado...")

    output_file = output_dir / "ablations_consolidated.csv"
    df.to_csv(output_file, index=False)

    print(f"   ✓ DataFrame guardado: {output_file.name}")


def create_summary_report(df: pd.DataFrame, output_dir: Path):
    """
    Crea reporte de texto con estadísticas descriptivas.
    """
    print("\n[8/8] Generando reporte estadístico...")

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("REPORTE DE ANÁLISIS DE ABLACIONES - PARTE A")
    report_lines.append("Generación de Texto con Diferentes Arquitecturas y Hiperparámetros")
    report_lines.append("=" * 80)
    report_lines.append(f"\nFecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"\nTotal de configuraciones analizadas: {len(df)}")
    report_lines.append(f"Total de modelos: {df['model'].nunique()}")
    report_lines.append(f"Modelos analizados: {', '.join(sorted(df['model'].unique()))}")

    # Estadísticas globales
    report_lines.append("\n" + "=" * 80)
    report_lines.append("ESTADÍSTICAS GLOBALES")
    report_lines.append("=" * 80)

    report_lines.append(f"\nTiempo de Inferencia:")
    report_lines.append(f"  - Media: {df['avg_inference_time'].mean():.4f} sec")
    report_lines.append(f"  - Mediana: {df['avg_inference_time'].median():.4f} sec")
    report_lines.append(f"  - Std Dev: {df['avg_inference_time'].std():.4f} sec")
    report_lines.append(f"  - Mínimo: {df['avg_inference_time'].min():.4f} sec")
    report_lines.append(f"  - Máximo: {df['avg_inference_time'].max():.4f} sec")

    report_lines.append(f"\nMemoria GPU:")
    report_lines.append(f"  - Media: {df['avg_gpu_memory'].mean():.2f} MB")
    report_lines.append(f"  - Mediana: {df['avg_gpu_memory'].median():.2f} MB")
    report_lines.append(f"  - Std Dev: {df['avg_gpu_memory'].std():.2f} MB")
    report_lines.append(f"  - Mínimo: {df['avg_gpu_memory'].min():.2f} MB")
    report_lines.append(f"  - Máximo: {df['avg_gpu_memory'].max():.2f} MB")

    # Mejor configuración global
    report_lines.append("\n" + "=" * 80)
    report_lines.append("MEJOR CONFIGURACIÓN (menor tiempo de inferencia)")
    report_lines.append("=" * 80)

    best_config = df.loc[df['avg_inference_time'].idxmin()]
    report_lines.append(f"\nModelo: {best_config['model']}")
    report_lines.append(f"Context Length: {best_config['context_length']}")
    report_lines.append(f"Temperature: {best_config['temperature']}")
    report_lines.append(f"Top-P: {best_config['top_p']}")
    report_lines.append(f"Top-K: {best_config['top_k']}")
    report_lines.append(f"Tiempo de Inferencia: {best_config['avg_inference_time']:.4f} ± {best_config['std_inference_time']:.4f} sec")
    report_lines.append(f"Memoria GPU: {best_config['avg_gpu_memory']:.2f} ± {best_config['std_gpu_memory']:.2f} MB")

    # Análisis por modelo
    report_lines.append("\n" + "=" * 80)
    report_lines.append("ANÁLISIS POR MODELO")
    report_lines.append("=" * 80)

    for model in sorted(df['model'].unique()):
        model_data = df[df['model'] == model]
        report_lines.append(f"\n{model.upper()}:")
        report_lines.append(f"  - Configuraciones: {len(model_data)}")
        report_lines.append(f"  - Tiempo medio: {model_data['avg_inference_time'].mean():.4f} sec")
        report_lines.append(f"  - Memoria media: {model_data['avg_gpu_memory'].mean():.2f} MB")

        best_model_config = model_data.loc[model_data['avg_inference_time'].idxmin()]
        report_lines.append(f"  - Mejor config: ctx={best_model_config['context_length']}, "
                          f"temp={best_model_config['temperature']}, "
                          f"top_p={best_model_config['top_p']}, "
                          f"top_k={best_model_config['top_k']}")
        report_lines.append(f"    → Tiempo: {best_model_config['avg_inference_time']:.4f} sec")

    # Análisis de efecto de hiperparámetros
    report_lines.append("\n" + "=" * 80)
    report_lines.append("EFECTO DE HIPERPARÁMETROS")
    report_lines.append("=" * 80)

    report_lines.append("\nContext Length:")
    for ctx in sorted(df['context_length'].unique()):
        ctx_data = df[df['context_length'] == ctx]
        report_lines.append(f"  - {ctx}: {ctx_data['avg_inference_time'].mean():.4f} sec "
                          f"(memoria: {ctx_data['avg_gpu_memory'].mean():.2f} MB)")

    report_lines.append("\nTemperature:")
    for temp in sorted(df['temperature'].unique()):
        temp_data = df[df['temperature'] == temp]
        report_lines.append(f"  - {temp}: {temp_data['avg_inference_time'].mean():.4f} sec")

    report_lines.append("\nTop-P:")
    for topp in sorted(df['top_p'].unique()):
        topp_data = df[df['top_p'] == topp]
        report_lines.append(f"  - {topp}: {topp_data['avg_inference_time'].mean():.4f} sec")

    report_lines.append("\nTop-K:")
    for topk in sorted(df['top_k'].unique()):
        topk_data = df[df['top_k'] == topk]
        report_lines.append(f"  - {topk}: {topk_data['avg_inference_time'].mean():.4f} sec")

    # Insights
    report_lines.append("\n" + "=" * 80)
    report_lines.append("INSIGHTS Y RECOMENDACIONES")
    report_lines.append("=" * 80)

    # Comparar char vs word
    char_models = df[df['model'].str.contains('char')]
    word_models = df[df['model'].str.contains('word')]

    if not char_models.empty and not word_models.empty:
        report_lines.append(f"\nChar-level vs Word-level:")
        report_lines.append(f"  - Char-level tiempo promedio: {char_models['avg_inference_time'].mean():.4f} sec")
        report_lines.append(f"  - Word-level tiempo promedio: {word_models['avg_inference_time'].mean():.4f} sec")
        faster = "Character-level" if char_models['avg_inference_time'].mean() < word_models['avg_inference_time'].mean() else "Word-level"
        report_lines.append(f"  → {faster} es más rápido en promedio")

    # Comparar RNN vs LSTM vs GRU
    rnn_models = df[df['model'].str.contains('rnn_')]
    lstm_models = df[df['model'].str.contains('lstm')]
    gru_models = df[df['model'].str.contains('gru')]

    if not rnn_models.empty and not lstm_models.empty and not gru_models.empty:
        report_lines.append(f"\nRNN vs LSTM vs GRU:")
        report_lines.append(f"  - RNN tiempo promedio: {rnn_models['avg_inference_time'].mean():.4f} sec")
        report_lines.append(f"  - LSTM tiempo promedio: {lstm_models['avg_inference_time'].mean():.4f} sec")
        report_lines.append(f"  - GRU tiempo promedio: {gru_models['avg_inference_time'].mean():.4f} sec")

        times = [
            (rnn_models['avg_inference_time'].mean(), 'RNN'),
            (lstm_models['avg_inference_time'].mean(), 'LSTM'),
            (gru_models['avg_inference_time'].mean(), 'GRU')
        ]
        fastest = min(times)[1]
        report_lines.append(f"  → {fastest} es la arquitectura más rápida")

    # Impacto del context length
    ctx_correlation = df[['context_length', 'avg_inference_time']].corr().iloc[0, 1]
    report_lines.append(f"\nImpacto del Context Length:")
    report_lines.append(f"  - Correlación con tiempo de inferencia: {ctx_correlation:.3f}")
    if abs(ctx_correlation) > 0.5:
        direction = "aumenta" if ctx_correlation > 0 else "disminuye"
        report_lines.append(f"  → El tiempo de inferencia {direction} significativamente con context length")

    report_lines.append("\n" + "=" * 80)
    report_lines.append("FIN DEL REPORTE")
    report_lines.append("=" * 80)

    # Guardar reporte
    output_file = output_dir / "ablations_summary_report.txt"
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"   ✓ Reporte guardado: {output_file.name}")


def print_summary_table(df_best: pd.DataFrame):
    """
    Imprime tabla resumen en consola.
    """
    print("\n" + "=" * 80)
    print("TOP 5 MEJORES CONFIGURACIONES (por tiempo de inferencia)")
    print("=" * 80)

    display_cols = ['rank', 'model', 'context_length', 'temperature',
                   'top_p', 'top_k', 'avg_inference_time', 'avg_gpu_memory']

    df_display = df_best.head(5)[display_cols].copy()
    df_display.columns = ['Rank', 'Model', 'Ctx', 'Temp', 'Top-P', 'Top-K',
                         'Inf Time (s)', 'GPU Mem (MB)']

    print(df_display.to_string(index=False))
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Análisis consolidado de ablaciones - Parte A (Generación de Texto)'
    )
    parser.add_argument(
        '--ablations_dir',
        type=str,
        default='results/ablations',
        help='Directorio raíz de ablaciones (default: results/ablations)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/ablations_analysis',
        help='Directorio de salida para análisis (default: results/ablations_analysis)'
    )
    parser.add_argument(
        '--top_n',
        type=int,
        default=10,
        help='Número de mejores configuraciones en ranking (default: 10)'
    )

    args = parser.parse_args()

    # Convertir a Path
    ablations_dir = Path(args.ablations_dir)
    output_dir = Path(args.output_dir)

    # Verificar directorio de ablaciones
    if not ablations_dir.exists():
        print(f"Error: No se encontró el directorio {ablations_dir}")
        sys.exit(1)

    # Crear directorio de salida
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" ANÁLISIS DE ABLACIONES - PARTE A")
    print("=" * 80)
    print(f"\n Directorio de ablaciones: {ablations_dir}")
    print(f" Directorio de salida: {output_dir}")

    # Cargar datos
    print("\n[1/8] Cargando resultados de ablaciones...")
    df = load_all_models(ablations_dir)
    print(f"\n✓ Total de configuraciones cargadas: {len(df)}")
    print(f"✓ Total de modelos: {df['model'].nunique()}")

    # Generar análisis
    plot_temperature_vs_inference_time(df, output_dir)
    plot_context_vs_memory(df, output_dir)
    plot_sampling_heatmaps(df, output_dir)
    plot_model_comparison_bars(df, output_dir)
    df_best = create_best_configs_table(df, output_dir, args.top_n)
    create_consolidated_data(df, output_dir)
    create_summary_report(df, output_dir)

    # Resumen final
    print_summary_table(df_best)

    print("\n ANÁLISIS COMPLETADO")
    print(f"\n Archivos generados en: {output_dir}")
    print("   - temperature_vs_inference_time.png")
    print("   - context_vs_memory.png")
    print("   - sampling_heatmaps.png")
    print("   - model_comparison_bars.png")
    print("   - best_configurations_ranking.csv")
    print("   - ablations_consolidated.csv")
    print("   - ablations_summary_report.txt")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
