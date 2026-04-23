# -*- coding: utf-8 -*-
"""
Gera três gráficos de linha (uma imagem para cada métrica):
- Acurácia vs Número de Clientes
- Precisão vs Número de Clientes
- Recall vs Número de Clientes

Cada ponto agora representa o RESULTADO FINAL (último round) do cenário
(2, 4, 6 clientes), lido de `evolution_metrics.csv` em cada pasta.
Deixamos de usar médias e não há linha constante do Centralizado aqui.

O estilo visual replica o padrão do projeto PhishingDetection001.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_final_metric(csv_path: str, metric_name: str):
    """
    Lê `evolution_metrics.csv` e retorna o valor FINAL (último round)
    para a métrica solicitada.

    Espera colunas: 'Round','Acurácia (%)','Precisão (%)','Recall (%)','F1-Score (%)'
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")

    df = pd.read_csv(csv_path, encoding='utf-8')
    if 'Round' not in df.columns:
        raise ValueError(f"Coluna 'Round' não encontrada em {csv_path}")

    # Pega a última ocorrência do maior Round
    max_round = df['Round'].max()
    final_rows = df[df['Round'] == max_round]
    if final_rows.empty:
        raise ValueError(f"Não foi possível localizar o último round em: {csv_path}")

    col_map = {
        'Acurácia': ['Acurácia (%)', 'Accuracy (%)'],
        'Precisão': ['Precisão (%)', 'Precision (%)'],
        'Recall':   ['Recall (%)', 'Sensibilidade (%)'],
    }
    target_cols = col_map.get(metric_name)
    if not target_cols:
        raise ValueError(f"Métrica não suportada: {metric_name}")

    for col in target_cols:
        if col in final_rows.columns:
            val = float(final_rows.iloc[-1][col])
            return val

    raise ValueError(f"Coluna correspondente a '{metric_name}' não encontrada em {csv_path}")


def plot_metric_vs_clients(metric_name: str, x_clients, y_final_vals, output_dir: str,
                           color_line: str, title: str, ylabel: str, filename_png: str, filename_csv: str):
    """Plota e salva um gráfico de linha para uma métrica específica (valor final por cenário)."""
    # Estilo consistente
    plt.style.use('seaborn-v0_8')
    seaborn_palette_set = False
    try:
        sns.set_palette('husl')
        seaborn_palette_set = True
    except Exception:
        seaborn_palette_set = False

    plt.figure(figsize=(12, 8))

    # Linha com resultados finais por cenário (último round)
    plt.plot(
        x_clients,
        y_final_vals,
        marker='o',
        linewidth=3,
        markersize=8,
        color=color_line,
        markerfacecolor='white',
        markeredgecolor=color_line,
        markeredgewidth=2,
        alpha=0.85,
        label='Federado (resultado final por cenário)'
    )

    # Anotações nos pontos federados
    for xc, yt in zip(x_clients, y_final_vals):
        plt.annotate(f"{yt:.2f}%", (xc, yt), textcoords="offset points", xytext=(0, 10),
                     ha='center', fontweight='bold', fontsize=10)

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Número de Clientes', fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(x_clients)
    # Iniciar eixo Y em 0, limitar ao máximo + margem, sem ultrapassar 100
    y_max = max(y_final_vals)
    margin = max(0.02 * y_max, 1.0)
    plt.ylim(0, min(100, y_max + margin))
    plt.legend(loc='best', frameon=True)
    plt.tight_layout()

    # Saída
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, filename_png)
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()

    # CSV com os dados usados
    df_out = pd.DataFrame({
        'Numero de Clientes': x_clients,
        'Valor Final Federado (%)': [round(v, 4) for v in y_final_vals],
    })
    csv_path = os.path.join(output_dir, filename_csv)
    df_out.to_csv(csv_path, index=False, encoding='utf-8')

    print(f"Gráfico salvo: {png_path}")
    print(f"Dados CSV salvos: {csv_path}")


def generate_all_charts(output_dir: str):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    datasets = {
        2: os.path.join(base_dir, '2clientes', 'evolution_metrics.csv'),
        4: os.path.join(base_dir, '4clientes', 'evolution_metrics.csv'),
        6: os.path.join(base_dir, '6clientes', 'evolution_metrics.csv'),
    }

    x_clients = []

    # Métricas
    metrics = [
        ('Acurácia', '#2E86AB', 'Acurácia vs Número de Clientes', 'Acurácia (%)', 'accuracy_vs_clientes.png', 'accuracy_vs_clientes.csv'),
        ('Precisão', '#A23B72', 'Precisão vs Número de Clientes', 'Precisão (%)', 'precision_vs_clientes.png', 'precision_vs_clientes.csv'),
        ('Recall', '#F18F01', 'Recall vs Número de Clientes', 'Recall (%)', 'recall_vs_clientes.png', 'recall_vs_clientes.csv'),
    ]

    # Acumular valores por métrica
    final_vals_by_metric = {m[0]: [] for m in metrics}

    for n_clients, path in datasets.items():
        x_clients.append(n_clients)
        for metric_name, *_ in metrics:
            final_val = load_final_metric(path, metric_name)
            final_vals_by_metric[metric_name].append(final_val)

    # Para cada métrica, plotar série final por cenário
    for metric_name, color_line, title, ylabel, filename_png, filename_csv in metrics:
        plot_metric_vs_clients(
            metric_name,
            x_clients,
            final_vals_by_metric[metric_name],
            output_dir,
            color_line,
            title,
            ylabel,
            filename_png,
            filename_csv,
        )


if __name__ == '__main__':
    out_dir = os.path.join(os.path.dirname(__file__), 'graphs')
    try:
        generate_all_charts(out_dir)
    except Exception as e:
        print(f"Erro ao gerar gráficos: {e}")
        sys.exit(1)
    sys.exit(0)