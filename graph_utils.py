# -*- coding: utf-8 -*-
"""
Utilitários para geração de gráficos das métricas de desempenho
do sistema de aprendizado federado de detecção de phishing.
"""

import os
import time
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
import pandas as pd

# Importar bibliotecas para monitoramento real de GPU
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("Warning: pynvml not available. GPU monitoring will use CPU estimation.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MetricsCollector:
    """Coletor de métricas de desempenho para o sistema federado."""
    
    def __init__(self):
        self.server_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'energy_consumption': [],
            'processing_time': []
        }
        self.client_metrics = {}
        self.round_data = []
        
    def add_server_metrics(self, round_num: int, accuracy: float, precision: float, 
                          recall: float, f1: float, energy: float = 0.0, 
                          processing_time: float = 0.0):
        """Adiciona métricas do servidor centralizado."""
        self.server_metrics['accuracy'].append(accuracy)
        self.server_metrics['precision'].append(precision)
        self.server_metrics['recall'].append(recall)
        self.server_metrics['f1'].append(f1)
        self.server_metrics['energy_consumption'].append(energy)
        self.server_metrics['processing_time'].append(processing_time)
        
    def add_client_metrics(self, client_id: int, round_num: int, accuracy: float, 
                          precision: float, recall: float, f1: float, 
                          energy: float = 0.0, processing_time: float = 0.0):
        """Adiciona métricas de um cliente específico."""
        if client_id not in self.client_metrics:
            self.client_metrics[client_id] = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1': [],
                'energy_consumption': [],
                'processing_time': []
            }
        
        self.client_metrics[client_id]['accuracy'].append(accuracy)
        self.client_metrics[client_id]['precision'].append(precision)
        self.client_metrics[client_id]['recall'].append(recall)
        self.client_metrics[client_id]['f1'].append(f1)
        self.client_metrics[client_id]['energy_consumption'].append(energy)
        self.client_metrics[client_id]['processing_time'].append(processing_time)

class GraphGenerator:
    """Gerador de gráficos para visualização das métricas."""
    
    def __init__(self, output_dir: str = "graphs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def create_performance_comparison_chart(self, server_metrics: Dict, 
                                           client_metrics: Dict, 
                                           metric_name: str, 
                                           title: str,
                                           ylabel: str):
        """Cria gráfico de barras comparando servidor e clientes."""
        plt.figure(figsize=(12, 8))
        
        # Preparar dados
        entities = ['Servidor']
        values = [np.mean(server_metrics[metric_name]) if server_metrics[metric_name] else 0]
        colors = ['#2E86AB']  # Azul para servidor
        
        # Adicionar clientes
        client_colors = ['#A23B72', '#F18F01', '#C73E1D', '#8B5A3C', '#6A994E']
        for i, (client_id, metrics) in enumerate(client_metrics.items()):
            entities.append(f'Cliente {client_id}')
            values.append(np.mean(metrics[metric_name]) if metrics[metric_name] else 0)
            colors.append(client_colors[i % len(client_colors)])
        
        # Criar gráfico de barras
        bars = plt.bar(entities, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Entidades', fontsize=12, fontweight='bold')
        plt.ylabel(ylabel, fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Salvar gráfico
        filename = f"{metric_name}_comparison.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gráfico salvo: {filepath}")
        return filepath
        
    def create_ml_metrics_chart(self, server_metrics: Dict, client_metrics: Dict):
        """Cria gráfico comparativo das métricas de ML (acurácia, precisão, recall)."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        metrics = ['accuracy', 'precision', 'recall']
        titles = ['Acurácia', 'Precisão', 'Recall']
        ylabels = ['Acurácia (%)', 'Precisão (%)', 'Recall (%)']
        
        for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
            ax = axes[i]
            
            # Preparar dados
            entities = ['Servidor']
            values = [np.mean(server_metrics[metric]) * 100 if server_metrics[metric] else 0]
            colors = ['#2E86AB']
            
            # Adicionar clientes
            client_colors = ['#A23B72', '#F18F01', '#C73E1D', '#8B5A3C', '#6A994E']
            for j, (client_id, client_data) in enumerate(client_metrics.items()):
                entities.append(f'Cliente {client_id}')
                values.append(np.mean(client_data[metric]) * 100 if client_data[metric] else 0)
                colors.append(client_colors[j % len(client_colors)])
            
            # Criar barras
            bars = ax.bar(entities, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
            
            # Adicionar valores nas barras
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Entidades', fontsize=10, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 105)
        
        plt.suptitle('Métricas de Desempenho de Machine Learning', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Salvar gráfico
        filepath = os.path.join(self.output_dir, "ml_metrics_comparison.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gráfico de métricas ML salvo: {filepath}")
        return filepath
        
    def create_resource_usage_chart(self, server_metrics: Dict, client_metrics: Dict):
        """Cria gráfico comparativo de consumo de recursos (energia e tempo)."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfico de consumo de energia
        ax1 = axes[0]
        entities = ['Servidor']
        energy_values = [np.mean(server_metrics['energy_consumption']) if server_metrics['energy_consumption'] else 0]
        colors = ['#2E86AB']
        
        client_colors = ['#A23B72', '#F18F01', '#C73E1D', '#8B5A3C', '#6A994E']
        for i, (client_id, client_data) in enumerate(client_metrics.items()):
            entities.append(f'Cliente {client_id}')
            energy_values.append(np.mean(client_data['energy_consumption']) if client_data['energy_consumption'] else 0)
            colors.append(client_colors[i % len(client_colors)])
        
        bars1 = ax1.bar(entities, energy_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        for bar, value in zip(bars1, energy_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.2f}W', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_title('Consumo de Energia', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Entidades', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Energia (Watts)', fontsize=10, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Gráfico de tempo de processamento
        ax2 = axes[1]
        time_values = [np.mean(server_metrics['processing_time']) if server_metrics['processing_time'] else 0]
        
        for i, (client_id, client_data) in enumerate(client_metrics.items()):
            time_values.append(np.mean(client_data['processing_time']) if client_data['processing_time'] else 0)
        
        bars2 = ax2.bar(entities, time_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        for bar, value in zip(bars2, time_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_title('Tempo de Processamento', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Entidades', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Tempo (segundos)', fontsize=10, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Consumo de Recursos', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Salvar gráfico
        filepath = os.path.join(self.output_dir, "resource_usage_comparison.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gráfico de recursos salvo: {filepath}")
        return filepath

class PerformanceMonitor:
    """Monitor de desempenho para medir consumo de energia e tempo de processamento.
    
    Medições:
    - Tempo: Duração completa da operação (treinamento do cliente ou agregação do servidor)
    - Energia: Consumo real da GPU usando NVML ou estimativa baseada em CPU
    
    Para clientes: Mede o tempo de treinamento do modelo
    Para servidor: Mede o tempo de agregação dos parâmetros dos clientes
    """
    
    def __init__(self):
        self.start_time = None
        self.start_energy = None
        self.process = psutil.Process()
        self.gpu_handle = None
        self.use_gpu_monitoring = False
        
        # Inicializar monitoramento de GPU se disponível
        if NVML_AVAILABLE and TORCH_AVAILABLE:
            try:
                pynvml.nvmlInit()
                if torch.cuda.is_available():
                    device_count = pynvml.nvmlDeviceGetCount()
                    if device_count > 0:
                        # Usar a primeira GPU disponível
                        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        self.use_gpu_monitoring = True
                        print("GPU monitoring initialized successfully")
                    else:
                        print("No GPU devices found")
                else:
                    print("CUDA not available")
            except Exception as e:
                print(f"Failed to initialize GPU monitoring: {e}")
                self.use_gpu_monitoring = False
        
    def start_monitoring(self):
        """Inicia o monitoramento de desempenho."""
        self.start_time = time.time()
        
        if self.use_gpu_monitoring and self.gpu_handle:
            try:
                # Obter consumo de energia inicial da GPU
                self.start_energy = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0  # Converter para Watts
            except Exception as e:
                print(f"Error getting initial GPU power: {e}")
                # Fallback para monitoramento de CPU
                self.start_energy = self.process.cpu_percent()
                self.use_gpu_monitoring = False
        else:
            # Fallback para monitoramento de CPU
            self.start_energy = self.process.cpu_percent()
        
    def stop_monitoring(self) -> tuple:
        """Para o monitoramento e retorna (energia_consumida, tempo_processamento)."""
        if self.start_time is None:
            return 0.0, 0.0
            
        end_time = time.time()
        processing_time = end_time - self.start_time
        
        if self.use_gpu_monitoring and self.gpu_handle:
            try:
                # Obter consumo de energia final da GPU
                end_energy = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0  # Converter para Watts
                # Calcular energia média consumida durante o período
                avg_power = (self.start_energy + end_energy) / 2
                energy_consumption = avg_power * processing_time  # Watts * segundos = Joules
                
                print(f"GPU Power - Start: {self.start_energy:.2f}W, End: {end_energy:.2f}W, Avg: {avg_power:.2f}W")
                print(f"Energy consumed: {energy_consumption:.2f}J over {processing_time:.2f}s")
                
                return energy_consumption, processing_time
                
            except Exception as e:
                print(f"Error getting final GPU power: {e}")
                # Fallback para estimativa de CPU
                end_cpu = self.process.cpu_percent()
                avg_cpu = (self.start_energy + end_cpu) / 2
                energy_consumption = (avg_cpu / 100) * 2.0 * processing_time
                return energy_consumption, processing_time
        else:
            # Estimativa baseada no uso de CPU
            end_cpu = self.process.cpu_percent()
            avg_cpu = (self.start_energy + end_cpu) / 2
            # Estimativa: ~2W por 1% de CPU (valor aproximado)
            energy_consumption = (avg_cpu / 100) * 2.0 * processing_time
            return energy_consumption, processing_time
    
    def __del__(self):
        """Cleanup NVML quando o objeto é destruído."""
        if self.use_gpu_monitoring:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

def generate_sample_data():
    """Gera dados de exemplo para demonstração dos gráficos."""
    collector = MetricsCollector()
    
    # Dados do servidor (simulados)
    collector.add_server_metrics(1, 0.92, 0.89, 0.94, 0.91, 15.5, 45.2)
    
    # Dados dos clientes (simulados)
    collector.add_client_metrics(1, 1, 0.88, 0.85, 0.90, 0.87, 8.2, 32.1)
    collector.add_client_metrics(2, 1, 0.90, 0.87, 0.92, 0.89, 9.1, 35.4)
    collector.add_client_metrics(3, 1, 0.86, 0.83, 0.88, 0.85, 7.8, 29.8)
    
    return collector

def main():
    """Função principal para teste dos gráficos."""
    print("Gerando gráficos de exemplo...")
    
    # Gerar dados de exemplo
    collector = generate_sample_data()
    
    # Criar gerador de gráficos
    graph_gen = GraphGenerator()
    
    # Gerar gráficos
    graph_gen.create_ml_metrics_chart(collector.server_metrics, collector.client_metrics)
    graph_gen.create_resource_usage_chart(collector.server_metrics, collector.client_metrics)
    
    print("Gráficos gerados com sucesso no diretório 'graphs'!")

if __name__ == "__main__":
    main()