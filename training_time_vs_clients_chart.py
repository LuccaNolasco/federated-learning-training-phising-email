# -*- coding: utf-8 -*-
"""
Gera um gráfico de linha onde o eixo X é o número de clientes (2, 4, 6)
e o eixo Y é o tempo médio de treino dos clientes (em segundos).

Também plota uma linha constante representando o tempo médio de treino
do modelo centralizado, calculado como a média do tempo da entidade
"Centralizado" nos arquivos resource_usage_comparison.csv dos diretórios
../2clientes, ../4clientes e ../6clientes.

O gráfico segue o padrão estético usado no projeto PhishingDetection001.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_times_from_csv(csv_path: str):
    """Lê um CSV de uso de recursos e retorna:
    - média dos tempos dos clientes
    - lista de tempos dos clientes
    - tempo da entidade Centralizado (se existir, senão None)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8")

    # Normalizar coluna de entidade para evitar problemas de espaço/capitalização
    df['Entidade'] = df['Entidade'].astype(str).str.strip()

    # Filtrar linhas de clientes
    client_mask = df['Entidade'].str.startswith('Cliente')
    client_times = df.loc[client_mask, 'Tempo (segundos)'].astype(float).tolist()
    avg_client_time = sum(client_times) / len(client_times) if client_times else None

    # Tempo do centralizado
    central_rows = df.loc[df['Entidade'] == 'Centralizado']
    central_time = None
    if not central_rows.empty:
        # Caso haja múltiplas linhas, usar a média
        central_time = central_rows['Tempo (segundos)'].astype(float).mean()

    return avg_client_time, client_times, central_time


def generate_chart(output_dir: str):
    """Gera o gráfico e salva PNG e CSV em output_dir."""
    # Estilo consistente com o projeto
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    datasets = {
        2: os.path.join(base_dir, '2clientes', 'resource_usage_comparison.csv'),
        4: os.path.join(base_dir, '4clientes', 'resource_usage_comparison.csv'),
        6: os.path.join(base_dir, '6clientes', 'resource_usage_comparison.csv'),
    }

    x_clients = []
    y_avg_times = []
    central_times = []

    for n_clients, path in datasets.items():
        avg_client_time, _, central_time = load_times_from_csv(path)
        if avg_client_time is None:
            raise ValueError(f"Não foi possível calcular tempo médio dos clientes em: {path}")
        x_clients.append(n_clients)
        y_avg_times.append(avg_client_time)
        if central_time is not None:
            central_times.append(central_time)

    # Média do tempo centralizado ao longo dos três cenários
    if not central_times:
        raise ValueError("Nenhum tempo de 'Centralizado' encontrado nos CSVs informados.")
    central_avg = sum(central_times) / len(central_times)

    # Plot
    plt.figure(figsize=(12, 8))

    # Linha federada (média dos clientes por cenário)
    plt.plot(
        x_clients,
        y_avg_times,
        marker='o',
        linewidth=3,
        markersize=8,
        color='#2E86AB',
        markerfacecolor='white',
        markeredgecolor='#2E86AB',
        markeredgewidth=2,
        alpha=0.85,
        label='Federado (média dos clientes)'
    )

    # Linha constante do centralizado
    plt.plot(
        x_clients,
        [central_avg] * len(x_clients),
        linestyle='--',
        linewidth=2.5,
        color='#FF6B35',
        alpha=0.9,
        label='Centralizado (média)'
    )

    # Anotações nos pontos da linha federada
    for xc, yt in zip(x_clients, y_avg_times):
        plt.annotate(f"{yt:.2f}s", (xc, yt), textcoords="offset points", xytext=(0, 10),
                     ha='center', fontweight='bold', fontsize=10)

    # Anotação única para a linha centralizada (usar posição média no eixo X)
    mid_x = x_clients[len(x_clients)//2]
    plt.annotate(f"{central_avg:.2f}s", (mid_x, central_avg), textcoords="offset points", xytext=(0, -18),
                 ha='center', fontweight='bold', fontsize=10, color='#FF6B35')

    plt.title('Tempo Médio de Treino vs Número de Clientes', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Número de Clientes', fontsize=12, fontweight='bold')
    plt.ylabel('Tempo médio de treino (segundos)', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(x_clients)
    # Iniciar eixo Y em 0 para melhor fidelidade visual
    y_max = max(y_avg_times + [central_avg])
    margin = max(0.05 * y_max, 1.0)
    plt.ylim(0, y_max + margin)
    plt.legend(loc='best', frameon=True)
    plt.tight_layout()

    # Saída
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, 'training_time_vs_clientes.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()

    # CSV com os dados usados
    df_out = pd.DataFrame({
        'Numero de Clientes': x_clients,
        'Tempo Federado (s)': [round(v, 4) for v in y_avg_times],
        'Tempo Centralizado Medio (s)': [round(central_avg, 4)] * len(x_clients),
    })
    csv_path = os.path.join(output_dir, 'training_time_vs_clientes.csv')
    df_out.to_csv(csv_path, index=False, encoding='utf-8')

    print(f"Gráfico salvo: {png_path}")
    print(f"Dados CSV salvos: {csv_path}")


if __name__ == '__main__':
    # Diretório padrão de saída seguindo o projeto
    out_dir = os.path.join(os.path.dirname(__file__), 'graphs')
    try:
        generate_chart(out_dir)
    except Exception as e:
        print(f"Erro ao gerar gráfico: {e}")
        sys.exit(1)
    sys.exit(0)