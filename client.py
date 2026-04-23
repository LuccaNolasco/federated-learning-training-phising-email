# -*- coding: utf-8 -*-
"""
Cliente Flower para aprendizado federado de detecção de phishing
com TinyBert adaptado.
"""

import argparse
import torch
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple
import os
import pickle

import flwr as fl
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from data_utils import prepare_client_data
from config import (
    MODEL_NAME, NUM_LABELS, get_training_config, 
    FLOWER_CONFIG, SERVER_CONFIG
)
from config_performance import print_current_mode
from graph_utils import PerformanceMonitor

class PhishingFlowerClient(NumPyClient):
    """Cliente Flower para detecção de phishing com TinyBert."""
    
    def __init__(self, client_id: int, num_clients: int):
        self.client_id = client_id
        self.num_clients = num_clients
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.test_dataset = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.performance_monitor = PerformanceMonitor()
        
        print(f"[Cliente {client_id}] Inicializando cliente...")
        print(f"[Cliente {client_id}] Dispositivo: {self.device}")
        
        # Mostrar modo de performance atual
        print_current_mode()
        
        # Preparar dados e modelo
        self._setup_data_and_model()
    
    def _setup_data_and_model(self):
        """Configura dados e modelo para o cliente."""
        # Preparar dados do cliente
        self.train_dataset, self.test_dataset, self.tokenizer = prepare_client_data(
            self.client_id, self.num_clients
        )
        
        # Carregar modelo
        print(f"[Cliente {self.client_id}] Carregando modelo: {MODEL_NAME}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, 
            num_labels=NUM_LABELS
        )
        self.model.to(self.device)
        
        print(f"[Cliente {self.client_id}] Configuração concluída!")
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Retorna os parâmetros atuais do modelo."""
        print(f"[Cliente {self.client_id}] Obtendo parâmetros do modelo")
        
        parameters = []
        for param in self.model.parameters():
            parameters.append(param.cpu().detach().numpy())
        
        return parameters
    
    def set_parameters(self, parameters: List[np.ndarray]):
        """Define os parâmetros do modelo."""
        print(f"[Cliente {self.client_id}] Definindo parâmetros do modelo")
        
        params_dict = zip(self.model.parameters(), parameters)
        for param, new_param in params_dict:
            param.data = torch.tensor(new_param, dtype=param.dtype, device=param.device)
    
    def compute_metrics(self, pred):
        """Calcula métricas de avaliação."""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Treina o modelo com os parâmetros recebidos."""
        print(f"[Cliente {self.client_id}] Iniciando treinamento - Rodada {config.get('server_round', 'N/A')}")
        
        # Inicializar monitor de desempenho preciso
        from training_time_monitor import PrecisePerformanceMonitor
        performance_monitor = PrecisePerformanceMonitor()
        
        # Iniciar monitoramento de energia
        performance_monitor.start_energy_monitoring()
        
        # Definir parâmetros recebidos do servidor
        self.set_parameters(parameters)
        
        # Obter configurações dinâmicas
        training_config = get_training_config()
        
        # Configurar argumentos de treinamento
        training_args = TrainingArguments(
            output_dir=f"./results_client_{self.client_id}",
            eval_strategy=training_config["eval_strategy"],
            save_strategy=training_config["save_strategy"],
            learning_rate=training_config["learning_rate"],
            per_device_train_batch_size=training_config["per_device_train_batch_size"],
            per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
            num_train_epochs=training_config["num_train_epochs"],
            weight_decay=training_config["weight_decay"],
            logging_dir=f"./logs_client_{self.client_id}",
            logging_steps=training_config["logging_steps"],
            load_best_model_at_end=training_config["load_best_model_at_end"],
            metric_for_best_model=training_config["metric_for_best_model"],
            dataloader_pin_memory=training_config["dataloader_pin_memory"],
            dataloader_num_workers=training_config["dataloader_num_workers"],
            prediction_loss_only=training_config["prediction_loss_only"],
            disable_tqdm=training_config["disable_tqdm"],
            fp16=training_config["fp16"],
            gradient_checkpointing=training_config["gradient_checkpointing"],
            report_to=None,  # Desabilitar relatórios
            save_total_limit=1,  # Manter apenas o último checkpoint
            log_level="error",  # Reduzir verbosidade dos logs
        )
        
        # Configurar trainer com callback de tempo preciso
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[performance_monitor.get_training_callback()],
        )
        
        # Treinar modelo
        print(f"\n{'🚀'*20}")
        print(f"🚀 INICIANDO TREINAMENTO - Cliente {self.client_id}")
        print(f"{'🚀'*20}")
        train_result = trainer.train()
        
        # Obter métricas de treinamento
        train_loss = train_result.training_loss
        train_samples = len(self.train_dataset)
        
        # Parar monitoramento e obter métricas (tempo preciso, excluindo avaliações)
        energy_consumption, processing_time = performance_monitor.stop_monitoring()
        
        # Obter métricas detalhadas para debug
        detailed_metrics = performance_monitor.get_detailed_metrics()
        
        # Exibir resumo final organizado
        print(f"\n{'='*50}")
        print(f"📊 RESUMO DO CLIENTE {self.client_id}")
        print(f"{'='*50}")
        print(f"🎯 Loss de Treinamento: {train_loss:.4f}")
        print(f"⚡ Energia Consumida: {energy_consumption:.2f}W")
        print(f"⏱️  Tempo de Processamento: {processing_time:.2f}s")
        print(f"📈 Amostras Treinadas: {train_samples}")
        print(f"{'='*50}\n")
        
        # Retornar parâmetros atualizados
        updated_parameters = self.get_parameters({})
        
        return updated_parameters, train_samples, {
            "train_loss": train_loss,
            "energy_consumption": energy_consumption,
            "processing_time": processing_time
        }
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Avalia o modelo com os parâmetros recebidos."""
        print(f"\n{'─'*40}")
        print(f"🔍 AVALIAÇÃO - Cliente {self.client_id}")
        print(f"{'─'*40}")
        
        # Definir parâmetros recebidos do servidor
        self.set_parameters(parameters)
        
        # Obter configurações dinâmicas
        training_config = get_training_config()
        
        # Configurar argumentos para avaliação
        eval_args = TrainingArguments(
            output_dir=f"./eval_client_{self.client_id}",
            per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
            dataloader_pin_memory=training_config["dataloader_pin_memory"],
            dataloader_num_workers=training_config["dataloader_num_workers"],
            prediction_loss_only=training_config["prediction_loss_only"],
            disable_tqdm=training_config["disable_tqdm"],
            report_to=None,
        )
        
        # Criar trainer para avaliação
        trainer = Trainer(
            model=self.model,
            args=eval_args,
            eval_dataset=self.test_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )
        
        # Avaliar modelo
        eval_result = trainer.evaluate()
        
        # NOVA FUNCIONALIDADE: Coletar predições para matriz de confusão
        predictions = trainer.predict(self.test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Salvar predições para geração posterior de matriz de confusão
        self._save_predictions(y_true, y_pred)
        
        eval_loss = eval_result["eval_loss"]
        eval_samples = len(self.test_dataset)
        
        # Extrair métricas adicionais
        metrics = {
            "accuracy": eval_result.get("eval_accuracy", 0.0),
            "f1": eval_result.get("eval_f1", 0.0),
            "precision": eval_result.get("eval_precision", 0.0),
            "recall": eval_result.get("eval_recall", 0.0)
        }
        
        # Exibir resultados da avaliação de forma organizada
        print(f"\n📋 RESULTADOS DA AVALIAÇÃO:")
        print(f"  🎯 Loss: {eval_loss:.4f}")
        print(f"  ✅ Accuracy: {metrics['accuracy']:.4f}")
        print(f"  🎪 F1-Score: {metrics['f1']:.4f}")
        print(f"  🎯 Precision: {metrics['precision']:.4f}")
        print(f"  📊 Recall: {metrics['recall']:.4f}")
        print(f"  📈 Amostras Avaliadas: {eval_samples}")
        print(f"{'─'*40}\n")
        
        return eval_loss, eval_samples, metrics
    
    def _save_predictions(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Salva as predições do cliente para geração posterior de matriz de confusão."""
        # Criar diretório se não existir
        os.makedirs("client_predictions", exist_ok=True)
        
        # Salvar predições
        predictions_data = {
            'client_id': self.client_id,
            'y_true': y_true,
            'y_pred': y_pred
        }
        
        filename = f"client_predictions/client_{self.client_id}_predictions.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(predictions_data, f)
        
        print(f"[Cliente {self.client_id}] Predições salvas em: {filename}")

def client_fn(context: Context) -> fl.client.Client:
    """Função para criar instância do cliente Flower."""
    # Obter ID do cliente do contexto
    client_id = context.node_config["partition-id"]
    num_clients = context.node_config.get("num-partitions", FLOWER_CONFIG["min_available_clients"])
    
    # Criar e retornar cliente
    client = PhishingFlowerClient(client_id, num_clients)
    return client.to_client()

def main():
    """Função principal para executar o cliente."""
    parser = argparse.ArgumentParser(description="Cliente Flower para detecção de phishing")
    parser.add_argument(
        "--client-id", 
        type=int, 
        required=True, 
        help="ID único do cliente (0, 1, 2, ...)"
    )
    parser.add_argument(
        "--num-clients", 
        type=int, 
        default=FLOWER_CONFIG["min_available_clients"], 
        help="Número total de clientes"
    )
    parser.add_argument(
        "--server-address", 
        type=str, 
        default=f"{SERVER_CONFIG['host']}:{SERVER_CONFIG['port']}", 
        help="Endereço do servidor (host:porta)"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'🌟'*25}")
    print(f"🌟 INICIANDO CLIENTE FLOWER {args.client_id} 🌟")
    print(f"{'🌟'*25}")
    print(f"🌐 Servidor: {args.server_address}")
    print(f"👥 Total de clientes: {args.num_clients}")
    print(f"{'🌟'*25}\n")
    
    # Criar cliente
    client = PhishingFlowerClient(args.client_id, args.num_clients)
    
    # Conectar ao servidor
    print(f"🔗 Conectando ao servidor...")
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client
    )
    
    print(f"\n{'👋'*20}")
    print(f"👋 Cliente {args.client_id} desconectado do servidor")
    print(f"{'👋'*20}\n")

if __name__ == "__main__":
    main()