# -*- coding: utf-8 -*-
"""
Servidor Flower para aprendizado federado de detecção de phishing
com TinyBert adaptado.
"""

import argparse
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict

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

from transformers import AutoModelForSequenceClassification
from data_utils import load_and_preprocess_dataset, split_dataset, tokenize_dataset, load_tokenizer, test_phrase
from config import (
    MODEL_NAME, NUM_LABELS, FLOWER_CONFIG, 
    SERVER_CONFIG, TRAINING_CONFIG
)

class PhishingFedAvg(FedAvg):
    """Estratégia FedAvg personalizada para detecção de phishing."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.round_metrics = []
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Agrega os resultados de treinamento dos clientes."""
        
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
            for _, fit_res in results:
                if "train_loss" in fit_res.metrics:
                    train_losses.append(fit_res.metrics["train_loss"])
            
            if train_losses:
                avg_train_loss = np.mean(train_losses)
                aggregated_metrics["avg_train_loss"] = avg_train_loss
                print(f"Loss médio de treinamento: {avg_train_loss:.4f}")
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Agrega os resultados de avaliação dos clientes."""
        
        print(f"\n=== Rodada {server_round} - Agregação de Avaliação ===")
        print(f"Clientes que completaram a avaliação: {len(results)}")
        print(f"Clientes que falharam: {len(failures)}")
        
        if not results:
            print("Nenhum resultado de avaliação disponível")
            return None, {}
        
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
    print("\nMétricas Finais:")
    print(f"  Loss: {final_metrics['loss']:.4f}")
    print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  F1-Score: {final_metrics['f1']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f}")
    print(f"  Recall: {final_metrics['recall']:.4f}")
    print(f"  Amostras totais: {final_metrics['samples']}")
    print(f"  Clientes participantes: {final_metrics['clients']}")

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
        # Imprimir resumo final
        print_final_summary(strategy)
        
        # Testar modelo final se disponível
        if hasattr(strategy, '_last_parameters') and strategy._last_parameters:
            test_final_model(strategy._last_parameters)
        
        print("\nServidor finalizado.")

if __name__ == "__main__":
    main()