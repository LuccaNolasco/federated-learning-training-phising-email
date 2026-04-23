# -*- coding: utf-8 -*-
"""
Servidor Flower para aprendizado federado de detecção de phishing
com TinyBert adaptado.
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import pickle
from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

import flwr as fl
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import (
    EvaluateIns, 
    EvaluateRes, 
    FitIns, 
    FitRes, 
    Parameters, 
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from data_utils import load_and_preprocess_dataset, split_dataset, tokenize_dataset, load_tokenizer, test_phrase
from config import (
    MODEL_NAME, NUM_LABELS, FLOWER_CONFIG, 
    SERVER_CONFIG, TRAINING_CONFIG
)
from graph_utils import MetricsCollector, GraphGenerator, PerformanceMonitor
from centralized_model import CentralizedModel

class PhishingFedAvg(FedAvg):
    """Estratégia FedAvg personalizada para detecção de phishing."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.round_metrics = []
        self.metrics_collector = MetricsCollector()
        self.performance_monitor = PerformanceMonitor()
        self.centralized_model = None
        self.centralized_metrics = None
        
        # Novos atributos para tracking de loss e matrizes de confusão
        self.loss_per_round = []
        self.train_loss_history = []
        self.avg_train_loss_per_round = []
        self.test_dataset = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Preparar dataset de teste para avaliação do modelo agregado
        self._prepare_test_data()
    
    def _prepare_test_data(self):
        """Prepara o dataset de teste para avaliação do modelo agregado."""
        print("Preparando dataset de teste para avaliação do modelo agregado...")
        
        # Carregar e preprocessar dataset
        dataset = load_and_preprocess_dataset()
        train_ds, test_ds = split_dataset(dataset)
        
        # Carregar tokenizer
        self.tokenizer = load_tokenizer()
        
        # Tokenizar dataset de teste
        self.test_dataset = tokenize_dataset(test_ds, self.tokenizer)
        
        print(f"Dataset de teste preparado com {len(self.test_dataset)} amostras")
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Agrega os resultados de treinamento dos clientes."""
        
        # Iniciar monitoramento de performance do servidor
        self.performance_monitor.start_monitoring()
        
        print(f"\n=== Rodada {server_round} - Agregação de Treinamento ===")
        print(f"Clientes que completaram o treinamento: {len(results)}")
        print(f"Clientes que falharam: {len(failures)}")
        
        if failures:
            print("Falhas detectadas:")
            for i, failure in enumerate(failures):
                print(f"  Falha {i+1}: {failure}")
        
        # Chamar agregação padrão do FedAvg
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_parameters is not None:
            print(f"Agregação de parâmetros concluída com sucesso")
            
            # Coletar métricas de treinamento
            train_losses = []
            for client_proxy, fit_res in results:
                train_loss = fit_res.metrics.get("train_loss")
                if train_loss is None:
                    continue

                train_loss = float(train_loss)
                train_losses.append(train_loss)
                self.train_loss_history.append(
                    {"round": server_round, "client_id": client_proxy.cid, "train_loss": train_loss}
                )
            
            if train_losses:
                avg_train_loss = np.mean(train_losses)
                aggregated_metrics["avg_train_loss"] = avg_train_loss
                print(f"Loss médio de treinamento: {avg_train_loss:.4f}")
                self.avg_train_loss_per_round.append(
                    {"round": server_round, "avg_train_loss": float(avg_train_loss)}
                )
            
            # NOVA FUNCIONALIDADE: Calcular loss do modelo agregado no conjunto de teste
            print(f"[DEBUG] Tentando calcular loss do modelo agregado para round {server_round}")
            aggregated_loss = self._evaluate_aggregated_model_loss(aggregated_parameters, server_round)
            print(f"[DEBUG] Resultado da avaliação de loss: {aggregated_loss}")
            if aggregated_loss is not None:
                self.loss_per_round.append({
                    'round': server_round,
                    'loss': aggregated_loss
                })
                print(f"Loss do modelo agregado no conjunto de teste: {aggregated_loss:.4f}")
                
                # Salvar loss em CSV após cada round
                self._save_loss_to_csv()
            else:
                print(f"[DEBUG] Loss retornado foi None - não foi possível calcular")
        
        # Coletar apenas métricas de recursos dos clientes (energia e tempo)
        for i, (_, fit_res) in enumerate(results):
            client_id = i  # ID baseado na posição (0, 1) para corresponder aos clientes reais
            # Verificar se já existe entrada para este cliente neste round
            if client_id not in self.metrics_collector.client_metrics:
                self.metrics_collector.client_metrics[client_id] = {
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'f1': [],
                    'energy_consumption': [],
                    'processing_time': []
                }
            
            # Adicionar apenas métricas de recursos
            self.metrics_collector.client_metrics[client_id]['energy_consumption'].append(
                fit_res.metrics.get("energy_consumption", 0.0)
            )
            self.metrics_collector.client_metrics[client_id]['processing_time'].append(
                fit_res.metrics.get("processing_time", 0.0)
            )
        
        # Parar monitoramento e coletar métricas do servidor
        self.temp_server_energy, self.temp_server_time = self.performance_monitor.stop_monitoring()
        
        return aggregated_parameters, aggregated_metrics
    


    def set_final_parameters(self, parameters: Parameters):
        """Define os parâmetros finais para geração da matriz de confusão."""
        self._final_parameters = parameters

    def _evaluate_aggregated_model_loss(self, parameters: Parameters, server_round: int) -> Optional[float]:
        """Avalia o loss do modelo agregado no conjunto de teste."""
        try:
            print(f"[DEBUG] Iniciando avaliação de loss para round {server_round}")
            
            # Carregar modelo
            print(f"[DEBUG] Carregando modelo {MODEL_NAME}")
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, 
                num_labels=NUM_LABELS
            )
            model.to(self.device)
            print(f"[DEBUG] Modelo carregado e movido para {self.device}")
            
            # Aplicar parâmetros agregados
            print(f"[DEBUG] Aplicando parâmetros agregados")
            params_arrays = parameters_to_ndarrays(parameters)
            params_dict = zip(model.parameters(), params_arrays)
            for param, new_param in params_dict:
                param.data = torch.tensor(new_param, dtype=param.dtype, device=param.device)
            print(f"[DEBUG] Parâmetros aplicados com sucesso")
            
            model.eval()
            
            # Preparar DataLoader
            print(f"[DEBUG] Preparando DataLoader com {len(self.test_dataset)} amostras")
            
            # Configurar formato do dataset para PyTorch
            self.test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
            
            test_dataloader = DataLoader(
                self.test_dataset, 
                batch_size=16, 
                shuffle=False
            )
            print(f"[DEBUG] DataLoader criado com {len(test_dataloader)} batches")
            
            total_loss = 0.0
            total_samples = 0
            
            print(f"[DEBUG] Iniciando loop de avaliação")
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_dataloader):
                    print(f"[DEBUG] Processando batch {batch_idx + 1}/{len(test_dataloader)}")
                    
                    # Extrair dados do batch
                    if isinstance(batch, dict):
                        input_ids = batch.get('input_ids')
                        attention_mask = batch.get('attention_mask')
                        labels = batch.get('labels') or batch.get('label')  # Tentar ambas as chaves
                    else:
                        # Se batch for uma tupla/lista
                        input_ids = batch[0] if len(batch) > 0 else None
                        attention_mask = batch[1] if len(batch) > 1 else None
                        labels = batch[2] if len(batch) > 2 else batch[-1]  # labels geralmente são o último elemento
                    
                    print(f"[DEBUG] Batch shapes - input_ids: {input_ids.shape if hasattr(input_ids, 'shape') else type(input_ids)}, labels: {labels.shape if hasattr(labels, 'shape') else type(labels)}")
                    
                    # Verificar se temos os dados necessários
                    if input_ids is None or labels is None:
                        print(f"[DEBUG] Dados insuficientes no batch - pulando")
                        continue
                    
                    # Converter para tensores se necessário
                    if not isinstance(input_ids, torch.Tensor):
                        if isinstance(input_ids, list):
                            # Se for lista de listas, converter para tensor
                            input_ids = torch.tensor(input_ids, dtype=torch.long)
                        else:
                            print(f"[DEBUG] Tipo inesperado para input_ids: {type(input_ids)}")
                            continue
                    
                    if not isinstance(labels, torch.Tensor):
                        if isinstance(labels, list):
                            labels = torch.tensor(labels, dtype=torch.long)
                        else:
                            print(f"[DEBUG] Tipo inesperado para labels: {type(labels)}")
                            continue
                    
                    # Criar attention_mask se não existir
                    if attention_mask is None:
                        attention_mask = torch.ones_like(input_ids)
                    elif not isinstance(attention_mask, torch.Tensor):
                        if isinstance(attention_mask, list):
                            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
                    
                    print(f"[DEBUG] Tensores convertidos e movidos para {self.device}")
                    print(f"[DEBUG] Executando forward pass")
                    
                    # Mover tensores para o dispositivo correto
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    
                    print(f"[DEBUG] Loss calculado: {loss.item()}")
                    
                    total_loss += loss.item() * labels.size(0)
                    total_samples += labels.size(0)
            
            avg_loss = total_loss / total_samples if total_samples > 0 else None
            print(f"[DEBUG] Loss médio calculado: {avg_loss}")
            return avg_loss
            
        except Exception as e:
            print(f"[DEBUG] Erro detalhado ao avaliar loss do modelo agregado:")
            print(f"[DEBUG] Tipo do erro: {type(e).__name__}")
            print(f"[DEBUG] Mensagem: {str(e)}")
            import traceback
            print(f"[DEBUG] Traceback completo:")
            traceback.print_exc()
            return None
    
    def _save_loss_to_csv(self):
        """Salva os dados de loss por round em um arquivo CSV."""
        if not self.loss_per_round:
            return
            
        df = pd.DataFrame(self.loss_per_round)
        
        # Criar diretório se não existir
        os.makedirs("loss_tracking", exist_ok=True)
        
        # Salvar CSV
        csv_path = "loss_tracking/aggregated_loss_per_round.csv"
        df.to_csv(csv_path, index=False)
        print(f"Loss por round salvo em: {csv_path}")
    
    def _generate_confusion_matrices(self):
        """Gera matrizes de confusão para cada cliente e o modelo agregado final."""
        print("\n=== Gerando Matrizes de Confusão ===")
        
        # Usar o GraphGenerator para gerar as matrizes no diretório Graphs
        try:
            graph_generator = GraphGenerator()
            confusion_matrices_path = graph_generator.create_confusion_matrices()
            print(f"Matrizes de confusão geradas com sucesso em: {confusion_matrices_path}")
        except Exception as e:
            print(f"Erro ao gerar matrizes de confusão com GraphGenerator: {e}")
            # Fallback para o método antigo
            self._generate_confusion_matrices_fallback()
    
    def _generate_confusion_matrices_fallback(self):
        """Método de fallback para gerar matrizes de confusão."""
        # Criar diretório para matrizes de confusão
        os.makedirs("Graphs/confusion_matrices", exist_ok=True)
        
        # Gerar matriz de confusão para o modelo agregado final
        self._generate_aggregated_model_confusion_matrix()
        
        # Gerar matrizes de confusão para cada cliente
        self._generate_client_confusion_matrices()
    
    def _generate_aggregated_model_confusion_matrix(self):
        """Gera matriz de confusão para o modelo agregado final."""
        try:
            # Usar os parâmetros mais recentes (último round)
            if not hasattr(self, '_final_parameters') or self._final_parameters is None:
                print("Parâmetros finais não disponíveis para matriz de confusão do modelo agregado")
                return
            
            # Carregar modelo
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, 
                num_labels=NUM_LABELS
            )
            model.to(self.device)
            
            # Aplicar parâmetros finais
            params_arrays = parameters_to_ndarrays(self._final_parameters)
            params_dict = zip(model.parameters(), params_arrays)
            for param, new_param in params_dict:
                param.data = torch.tensor(new_param, dtype=param.dtype, device=param.device)
            
            model.eval()
            
            # Fazer predições
            y_true, y_pred = self._get_predictions(model)
            
            # Gerar matriz de confusão
            cm = confusion_matrix(y_true, y_pred)
            
            # Plotar matriz de confusão
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Legítimo', 'Phishing'], 
                       yticklabels=['Legítimo', 'Phishing'])
            plt.title('Matriz de Confusão - Modelo Agregado Final')
            plt.ylabel('Verdadeiro')
            plt.xlabel('Predito')
            plt.tight_layout()
            plt.savefig('Graphs/confusion_matrices/aggregated_model_confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Calcular e salvar métricas
            report = classification_report(y_true, y_pred, target_names=['Legítimo', 'Phishing'], output_dict=True)
            
            # Salvar relatório em CSV
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv('Graphs/confusion_matrices/aggregated_model_metrics.csv')
            
            print(f"Matriz de confusão do modelo agregado salva em: Graphs/confusion_matrices/aggregated_model_confusion_matrix.png")
            print(f"Métricas do modelo agregado salvas em: Graphs/confusion_matrices/aggregated_model_metrics.csv")
            
        except Exception as e:
            print(f"Erro ao gerar matriz de confusão do modelo agregado: {e}")
    
    def _generate_client_confusion_matrices(self):
        """Gera matrizes de confusão para cada cliente individual."""
        print("Gerando matrizes de confusão para clientes individuais...")
        
        # Criar diretório se não existir
        os.makedirs("Graphs/confusion_matrices", exist_ok=True)
        
        # Procurar arquivos de predições dos clientes
        predictions_dir = "client_predictions"
        if not os.path.exists(predictions_dir):
            print("Diretório de predições dos clientes não encontrado")
            return
        
        client_files = [f for f in os.listdir(predictions_dir) if f.endswith('.pkl')]
        
        if not client_files:
            print("Nenhum arquivo de predições de cliente encontrado")
            return
        
        # Gerar matriz para cada cliente
        for filename in client_files:
            try:
                filepath = os.path.join(predictions_dir, filename)
                
                # Carregar predições
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                
                client_id = data['client_id']
                y_true = data['y_true']
                y_pred = data['y_pred']
                
                # Gerar matriz de confusão
                cm = confusion_matrix(y_true, y_pred)
                
                # Criar visualização
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Legítimo', 'Phishing'],
                           yticklabels=['Legítimo', 'Phishing'])
                plt.title(f'Matriz de Confusão - Cliente {client_id}', fontsize=14, fontweight='bold')
                plt.xlabel('Predição', fontsize=12)
                plt.ylabel('Valor Real', fontsize=12)
                plt.tight_layout()
                
                # Salvar matriz
                matrix_path = f"Graphs/confusion_matrices/client_{client_id}_confusion_matrix.png"
                plt.savefig(matrix_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Matriz de confusão do Cliente {client_id} salva em: {matrix_path}")
                
                # Gerar relatório de classificação
                report = classification_report(y_true, y_pred, 
                                             target_names=['Legítimo', 'Phishing'],
                                             output_dict=True)
                
                # Salvar relatório em CSV
                report_df = pd.DataFrame(report).transpose()
                report_path = f"Graphs/confusion_matrices/client_{client_id}_classification_report.csv"
                report_df.to_csv(report_path)
                
                print(f"Relatório de classificação do Cliente {client_id} salvo em: {report_path}")
                
            except Exception as e:
                print(f"Erro ao processar predições do cliente {filename}: {e}")
        
        print(f"Matrizes de confusão geradas para {len(client_files)} clientes")
    
    def _get_predictions(self, model):
        """Obtém predições do modelo no conjunto de teste."""
        test_dataloader = DataLoader(
            self.test_dataset, 
            batch_size=16, 
            shuffle=False
        )
        
        y_true = []
        y_pred = []
        
        model.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = torch.tensor(batch['input_ids']).to(self.device)
                attention_mask = torch.tensor(batch['attention_mask']).to(self.device)
                labels = torch.tensor(batch['label']).to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())
        
        return y_true, y_pred
    
    def _generate_loss_chart(self):
        """Gera gráfico de linha com a evolução do loss por round."""
        if not self.loss_per_round:
            print("Nenhum dado de loss disponível para gerar gráfico")
            return
        
        # Criar diretório se não existir
        os.makedirs("loss_tracking", exist_ok=True)
        
        # Extrair dados
        rounds = [item['round'] for item in self.loss_per_round]
        losses = [item['loss'] for item in self.loss_per_round]
        
        # Criar gráfico
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, losses, marker='o', linewidth=2, markersize=6)
        plt.title('Evolução do Loss do Modelo Agregado por Round', fontsize=14, fontweight='bold')
        plt.xlabel('Round de Agregação', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Salvar gráfico
        chart_path = "loss_tracking/loss_evolution_chart.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gráfico de evolução do loss salvo em: {chart_path}")
        
        # Também salvar dados em CSV (caso não tenha sido salvo ainda)
        self._save_loss_to_csv()
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Agrega resultados de avaliação dos clientes."""
        print(f"\n=== Agregação de Avaliação - Rodada {server_round} ===")
        print(f"Clientes que completaram avaliação: {len(results)}")
        print(f"Clientes que falharam na avaliação: {len(failures)}")
        
        if not results:
            print("Nenhum resultado de avaliação disponível")
            return 0.0, {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "total_samples": 0.0,
                "num_clients": 0.0
            }
        
        # Coletar métricas de todos os clientes
        losses = []
        accuracies = []
        f1_scores = []
        precisions = []
        recalls = []
        total_samples = 0
        
        for _, evaluate_res in results:
            losses.append(evaluate_res.loss)
            total_samples += evaluate_res.num_examples
            
            if "accuracy" in evaluate_res.metrics:
                accuracies.append(evaluate_res.metrics["accuracy"])
            if "f1" in evaluate_res.metrics:
                f1_scores.append(evaluate_res.metrics["f1"])
            if "precision" in evaluate_res.metrics:
                precisions.append(evaluate_res.metrics["precision"])
            if "recall" in evaluate_res.metrics:
                recalls.append(evaluate_res.metrics["recall"])
        
        # Calcular médias
        avg_loss = np.mean(losses)
        aggregated_metrics = {
            "total_samples": total_samples,
            "num_clients": len(results)
        }
        
        if accuracies:
            avg_accuracy = np.mean(accuracies)
            aggregated_metrics["accuracy"] = avg_accuracy
            print(f"Accuracy média: {avg_accuracy:.4f}")
        
        if f1_scores:
            avg_f1 = np.mean(f1_scores)
            aggregated_metrics["f1"] = avg_f1
            print(f"F1-Score médio: {avg_f1:.4f}")
        
        if precisions:
            avg_precision = np.mean(precisions)
            aggregated_metrics["precision"] = avg_precision
            print(f"Precision média: {avg_precision:.4f}")
        
        if recalls:
            avg_recall = np.mean(recalls)
            aggregated_metrics["recall"] = avg_recall
            print(f"Recall médio: {avg_recall:.4f}")
        
        print(f"Loss médio: {avg_loss:.4f}")
        print(f"Total de amostras avaliadas: {total_samples}")
        
        # Armazenar métricas da rodada
        round_summary = {
            "round": server_round,
            "loss": avg_loss,
            "accuracy": aggregated_metrics.get("accuracy", 0.0),
            "f1": aggregated_metrics.get("f1", 0.0),
            "precision": aggregated_metrics.get("precision", 0.0),
            "recall": aggregated_metrics.get("recall", 0.0),
            "samples": total_samples,
            "clients": len(results)
        }
        self.round_metrics.append(round_summary)
        
        # Adicionar métricas completas do servidor (incluindo desempenho do modelo unificado)
        server_energy = getattr(self, 'temp_server_energy', 0.0)
        server_time = getattr(self, 'temp_server_time', 0.0)
        
        self.metrics_collector.add_server_metrics(
            server_round,
            aggregated_metrics.get("accuracy", 0.0),
            aggregated_metrics.get("precision", 0.0),
            aggregated_metrics.get("recall", 0.0),
            aggregated_metrics.get("f1", 0.0),
            server_energy,
            server_time
        )
        
        # Adicionar métricas de evolução para gráficos
        self.metrics_collector.add_evolution_metrics(
            server_round,
            aggregated_metrics.get("accuracy", 0.0),
            aggregated_metrics.get("precision", 0.0),
            aggregated_metrics.get("recall", 0.0),
            aggregated_metrics.get("f1", 0.0)
        )
        
        # Coletar apenas métricas de qualidade dos clientes (accuracy, precision, recall, f1)
        for i, (_, evaluate_res) in enumerate(results):
            client_id = i  # ID baseado na posição (0, 1) para corresponder aos clientes reais
            # Verificar se já existe entrada para este cliente
            if client_id not in self.metrics_collector.client_metrics:
                self.metrics_collector.client_metrics[client_id] = {
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'f1': [],
                    'energy_consumption': [],
                    'processing_time': []
                }
            
            # Adicionar apenas métricas de qualidade
            self.metrics_collector.client_metrics[client_id]['accuracy'].append(
                evaluate_res.metrics.get("accuracy", 0.0)
            )
            self.metrics_collector.client_metrics[client_id]['precision'].append(
                evaluate_res.metrics.get("precision", 0.0)
            )
            self.metrics_collector.client_metrics[client_id]['recall'].append(
                evaluate_res.metrics.get("recall", 0.0)
            )
            self.metrics_collector.client_metrics[client_id]['f1'].append(
                evaluate_res.metrics.get("f1", 0.0)
            )
        
        return avg_loss, aggregated_metrics

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Avaliação centralizada no servidor (opcional)."""
        print(f"\n=== Avaliação Centralizada - Rodada {server_round} ===")
        
        # Por enquanto, não implementamos avaliação centralizada
        # Pode ser adicionada posteriormente se necessário
        print("Avaliação centralizada não implementada - usando apenas avaliação federada")
        
        return None

def get_initial_parameters():
    """Obtém os parâmetros iniciais do modelo."""
    print("Carregando modelo inicial para obter parâmetros...")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=NUM_LABELS
    )
    
    # Extrair parâmetros
    parameters = []
    for param in model.parameters():
        parameters.append(param.detach().numpy())
    
    print(f"Parâmetros iniciais obtidos: {len(parameters)} tensores")
    
    return ndarrays_to_parameters(parameters)

def test_final_model(parameters: Parameters):
    """Testa o modelo final com algumas frases de exemplo."""
    print("\n=== Testando Modelo Final ===")
    
    # Carregar modelo e tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=NUM_LABELS
    )
    tokenizer = load_tokenizer()
    
    # Definir parâmetros finais
    params_arrays = parameters_to_ndarrays(parameters)
    params_dict = zip(model.parameters(), params_arrays)
    for param, new_param in params_dict:
        param.data = torch.tensor(new_param, dtype=param.dtype, device=param.device)
    
    # Frases de teste
    test_phrases = [
        "Click here to reset your account password now.",
        "Meeting confirmed for 10 AM tomorrow in the office.",
        "This is your confirmation code. Do not share it with anyone. Type in your credit card information",
        "Payment confirmed. Your screening for the movie 'Superman' will be at 4 p.m. in room 5.",
        "Hello? Do you remember me? We met at Game Stop last weekend and you gave me your email."
    ]
    
    print("Resultados dos testes:")
    for i, phrase in enumerate(test_phrases, 1):
        label, confidence = test_phrase(model, tokenizer, phrase)
        print(f"{i}. '{phrase[:50]}...'")
        print(f"   → {label} (confiança: {confidence:.3f})")

def server_fn(context):
    """Função para configurar o servidor."""
    # Obter parâmetros iniciais
    initial_parameters = get_initial_parameters()
    
    # Configurar estratégia
    strategy = PhishingFedAvg(
        fraction_fit=FLOWER_CONFIG["fraction_fit"],
        fraction_evaluate=FLOWER_CONFIG["fraction_evaluate"],
        min_fit_clients=FLOWER_CONFIG["min_fit_clients"],
        min_evaluate_clients=FLOWER_CONFIG["min_evaluate_clients"],
        min_available_clients=FLOWER_CONFIG["min_available_clients"],
        initial_parameters=initial_parameters,
    )
    
    # Configurar servidor
    config = ServerConfig(num_rounds=FLOWER_CONFIG["num_rounds"])
    
    return config, strategy

def print_final_summary(strategy: PhishingFedAvg):
    """Imprime um resumo final do treinamento."""
    print("\n" + "="*60)
    print("RESUMO FINAL DO TREINAMENTO FEDERADO")
    print("="*60)
    
    if not strategy.round_metrics:
        print("Nenhuma métrica disponível.")
        return
    
    print(f"Total de rodadas: {len(strategy.round_metrics)}")
    print("\nProgresso por rodada:")
    print(f"{'Rodada':<8} {'Loss':<8} {'Accuracy':<10} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Clientes':<9}")
    print("-" * 70)
    
    for metrics in strategy.round_metrics:
        print(f"{metrics['round']:<8} {metrics['loss']:<8.4f} {metrics['accuracy']:<10.4f} "
              f"{metrics['f1']:<8.4f} {metrics['precision']:<10.4f} {metrics['recall']:<8.4f} "
              f"{metrics['clients']:<9}")
    
    # Métricas finais
    final_metrics = strategy.round_metrics[-1]
    print("\nMétricas Finais do Modelo Federado:")
    print(f"  Loss: {final_metrics['loss']:.4f}")
    print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  F1-Score: {final_metrics['f1']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f}")
    print(f"  Recall: {final_metrics['recall']:.4f}")
    print(f"  Amostras totais: {final_metrics['samples']}")
    print(f"  Clientes participantes: {final_metrics['clients']}")
    
    # Treinar modelo centralizado
    print("\n" + "="*60)
    print("TREINAMENTO DO MODELO CENTRALIZADO")
    print("="*60)
    
    try:
        strategy.centralized_model = CentralizedModel()
        strategy.centralized_metrics = strategy.centralized_model.train()
        
        print("\nMétricas Finais do Modelo Centralizado:")
        print(f"  Accuracy: {strategy.centralized_metrics['accuracy']:.4f}")
        print(f"  F1-Score: {strategy.centralized_metrics['f1']:.4f}")
        print(f"  Precision: {strategy.centralized_metrics['precision']:.4f}")
        print(f"  Recall: {strategy.centralized_metrics['recall']:.4f}")
        print(f"  Tempo de processamento: {strategy.centralized_metrics['processing_time']:.2f}s")
        print(f"  Consumo de energia: {strategy.centralized_metrics['energy_consumption']:.2f}J")
        
        # Adicionar métricas do modelo centralizado ao coletor
        strategy.metrics_collector.centralized_metrics = strategy.centralized_metrics
        
    except Exception as e:
        print(f"\nErro ao treinar modelo centralizado: {e}")
        print("Continuando sem modelo centralizado...")
    
    # Gerar gráficos
    print("\n" + "="*60)
    print("GERANDO GRÁFICOS DE DESEMPENHO")
    print("="*60)
    
    try:
        graph_generator = GraphGenerator()
        
        # Gerar gráficos de métricas de ML
        ml_chart_path = graph_generator.create_ml_metrics_chart(
            strategy.metrics_collector.server_metrics,
            strategy.metrics_collector.client_metrics,
            strategy.centralized_metrics if hasattr(strategy, 'centralized_metrics') else None
        )
        
        # Gerar gráficos de recursos
        resource_chart_path = graph_generator.create_resource_usage_chart(
            strategy.metrics_collector.server_metrics,
            strategy.metrics_collector.client_metrics,
            strategy.centralized_metrics if hasattr(strategy, 'centralized_metrics') else None
        )
        
        # Gerar gráfico de evolução das métricas
        evolution_chart_path = graph_generator.create_evolution_metrics_chart(
            strategy.metrics_collector.evolution_metrics
        )
        
        # Gerar gráfico de evolução do loss
        loss_chart_path = None
        if strategy.loss_per_round:
            loss_data = {
                'rounds': [item['round'] for item in strategy.loss_per_round],
                'losses': [item['loss'] for item in strategy.loss_per_round]
            }
            loss_chart_path = graph_generator.create_loss_evolution_chart(loss_data)

        avg_train_loss_chart_path = None
        if strategy.avg_train_loss_per_round:
            avg_train_loss_data = {
                'rounds': [item['round'] for item in strategy.avg_train_loss_per_round],
                'losses': [item['avg_train_loss'] for item in strategy.avg_train_loss_per_round],
            }
            avg_train_loss_chart_path = graph_generator.create_avg_train_loss_evolution_chart(
                avg_train_loss_data
            )

        client_train_loss_csv_path = None
        if strategy.train_loss_history:
            client_train_loss_csv_path = graph_generator.save_client_train_loss_history(
                strategy.train_loss_history
            )
        
        print(f"\nGráficos salvos com sucesso:")
        print(f"  - Métricas ML: {ml_chart_path}")
        print(f"  - Uso de recursos: {resource_chart_path}")
        if evolution_chart_path:
            print(f"  - Evolução das métricas: {evolution_chart_path}")
        if loss_chart_path:
            print(f"  - Evolução do loss: {loss_chart_path}")
        if avg_train_loss_chart_path:
            print(f"  - Evolução do avg_train_loss: {avg_train_loss_chart_path}")
        if client_train_loss_csv_path:
            print(f"  - Train loss por cliente: {client_train_loss_csv_path}")
        
    except Exception as e:
        print(f"\nErro ao gerar gráficos: {e}")
        print("Continuando sem gráficos...")
    
    # NOVA FUNCIONALIDADE: Gerar gráfico de loss por round
    print("\n" + "="*60)
    print("GERANDO GRÁFICO DE LOSS POR ROUND")
    print("="*60)
    
    try:
        strategy._generate_loss_chart()
    except Exception as e:
        print(f"Erro ao gerar gráfico de loss: {e}")
    
    # NOVA FUNCIONALIDADE: Gerar matrizes de confusão
    print("\n" + "="*60)
    print("GERANDO MATRIZES DE CONFUSÃO")
    print("="*60)
    
    try:
        strategy._generate_confusion_matrices()
    except Exception as e:
        print(f"Erro ao gerar matrizes de confusão: {e}")

def main():
    """Função principal para executar o servidor."""
    parser = argparse.ArgumentParser(description="Servidor Flower para detecção de phishing")
    parser.add_argument(
        "--host", 
        type=str, 
        default=SERVER_CONFIG["host"], 
        help="Endereço do servidor"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=SERVER_CONFIG["port"], 
        help="Porta do servidor"
    )
    parser.add_argument(
        "--rounds", 
        type=int, 
        default=FLOWER_CONFIG["num_rounds"], 
        help="Número de rodadas de treinamento"
    )
    parser.add_argument(
        "--min-clients", 
        type=int, 
        default=FLOWER_CONFIG["min_available_clients"], 
        help="Número mínimo de clientes"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print("SERVIDOR FLOWER - DETECÇÃO DE PHISHING")
    print("="*50)
    print(f"Endereço: {args.host}:{args.port}")
    print(f"Rodadas: {args.rounds}")
    print(f"Clientes mínimos: {args.min_clients}")
    print(f"Modelo: {MODEL_NAME}")
    
    # Atualizar configurações com argumentos
    FLOWER_CONFIG["num_rounds"] = args.rounds
    FLOWER_CONFIG["min_available_clients"] = args.min_clients
    FLOWER_CONFIG["min_fit_clients"] = args.min_clients
    FLOWER_CONFIG["min_evaluate_clients"] = args.min_clients
    
    # Obter parâmetros iniciais
    initial_parameters = get_initial_parameters()
    
    # Configurar estratégia
    strategy = PhishingFedAvg(
        fraction_fit=FLOWER_CONFIG["fraction_fit"],
        fraction_evaluate=FLOWER_CONFIG["fraction_evaluate"],
        min_fit_clients=FLOWER_CONFIG["min_fit_clients"],
        min_evaluate_clients=FLOWER_CONFIG["min_evaluate_clients"],
        min_available_clients=FLOWER_CONFIG["min_available_clients"],
        initial_parameters=initial_parameters,
    )
    
    print("\nIniciando servidor Flower...")
    
    try:
        # Iniciar servidor
        fl.server.start_server(
            server_address=f"{args.host}:{args.port}",
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
        )
        
    except KeyboardInterrupt:
        print("\nServidor interrompido pelo usuário.")
    
    except Exception as e:
        print(f"\nErro no servidor: {e}")
    
    finally:
        # Definir parâmetros finais para geração de matrizes de confusão
        if hasattr(strategy, '_last_parameters') and strategy._last_parameters:
            strategy.set_final_parameters(strategy._last_parameters)
        
        # Imprimir resumo final
        print_final_summary(strategy)
        
        # Testar modelo final se disponível
        if hasattr(strategy, '_last_parameters') and strategy._last_parameters:
            test_final_model(strategy._last_parameters)
        
        print("\nServidor finalizado.")

if __name__ == "__main__":
    main()
