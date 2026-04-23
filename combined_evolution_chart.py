#!/usr/bin/env python3
"""
Script para gerar gráfico de evolução combinada das métricas
com diferentes números de clientes a partir do arquivo evolution_metrics.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Configurar estilo visual
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_combined_evolution_chart(csv_file_path, output_dir="graphs"):
    """
    Cria gráfico de evolução combinada das métricas com diferentes números de clientes.
    
    Args:
        csv_file_path (str): Caminho para o arquivo CSV com os dados
        output_dir (str): Diretório de saída para os gráficos
    """
    
    # Verificar se o arquivo existe
    if not os.path.exists(csv_file_path):
        print(f"Erro: Arquivo {csv_file_path} não encontrado!")
        return None
    
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Carregar dados do CSV
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8')
        print(f"Dados carregados com sucesso: {len(df)} registros")
    except Exception as e:
        print(f"Erro ao carregar CSV: {e}")
        return None
    
    # Verificar colunas necessárias
    required_columns = ['Round', 'Acurácia (%)', 'Precisão (%)', 'Recall (%)', 'F1-Score (%)', 'n_clientes']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Erro: Colunas faltando no CSV: {missing_columns}")
        return None
    
    # Obter números únicos de clientes
    n_clientes_unique = sorted(df['n_clientes'].unique())
    print(f"Números de clientes encontrados: {n_clientes_unique}")
    
    # Configurar cores para cada número de clientes
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#8B5A3C']
    color_map = {n: colors[i % len(colors)] for i, n in enumerate(n_clientes_unique)}
    
    # Criar figura com subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Métricas para plotar
    metrics = ['Acurácia (%)', 'Precisão (%)', 'Recall (%)', 'F1-Score (%)']
    titles = ['Evolução da Acurácia', 'Evolução da Precisão', 'Evolução do Recall', 'Evolução do F1-Score']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        
        # Plotar linha para cada número de clientes
        for n_clientes in n_clientes_unique:
            # Filtrar dados para este número de clientes
            data_subset = df[df['n_clientes'] == n_clientes].sort_values('Round')
            
            if len(data_subset) > 0:
                rounds = data_subset['Round'].values
                values = data_subset[metric].values
                
                # Plotar linha
                ax.plot(rounds, values, 
                       marker='o', 
                       linewidth=3, 
                       markersize=8,
                       color=color_map[n_clientes],
                       markerfacecolor='white',
                       markeredgecolor=color_map[n_clientes],
                       markeredgewidth=2,
                       alpha=0.8,
                       label=f'{n_clientes} clientes')
                
                # Adicionar valores nos pontos (apenas no último ponto para não poluir)
                if len(rounds) > 0:
                    last_round = rounds[-1]
                    last_value = values[-1]
                    ax.annotate(f'{last_value:.1f}%', 
                               (last_round, last_value),
                               textcoords="offset points", 
                               xytext=(10, 5),
                               ha='left', 
                               va='bottom',
                               fontweight='bold', 
                               fontsize=9,
                               color=color_map[n_clientes])
        
        # Configurar eixos e título
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Rounds de Agregação', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=10)
        
        # Definir ticks do eixo X baseado nos rounds únicos
        all_rounds = sorted(df['Round'].unique())
        ax.set_xticks(all_rounds)
        
        # Ajustar limites do eixo Y
        all_values = df[metric].values
        min_val = min(all_values) - 1
        max_val = max(all_values) + 1
        ax.set_ylim(max(0, min_val), min(100, max_val))
    
    # Título geral
    plt.suptitle('Evolução das Métricas por Número de Clientes', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Ajustar layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Salvar gráfico
    output_file = os.path.join(output_dir, "combined_evolution_metrics.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico de evolução combinada salvo: {output_file}")
    
    # Criar também um CSV organizado para referência
    output_csv = os.path.join(output_dir, "combined_evolution_metrics.csv")
    df_sorted = df.sort_values(['n_clientes', 'Round'])
    df_sorted.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Dados organizados salvos: {output_csv}")
    
    return output_file

def main():
    """Função principal para executar o script."""
    
    # Caminho para o arquivo CSV
    csv_file = "evolucao_combinada/evolution_metrics.csv"
    
    # Verificar se estamos no diretório correto
    if not os.path.exists(csv_file):
        print(f"Arquivo {csv_file} não encontrado no diretório atual.")
        print("Certifique-se de executar o script no diretório raiz do projeto.")
        return
    
    # Gerar gráfico
    result = create_combined_evolution_chart(csv_file)
    
    if result:
        print("\n✅ Gráfico gerado com sucesso!")
        print(f"📊 Arquivo salvo: {result}")
    else:
        print("\n❌ Erro ao gerar gráfico.")

if __name__ == "__main__":
    main()