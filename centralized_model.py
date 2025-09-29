# -*- coding: utf-8 -*-
"""
Modelo centralizado para comparação com o aprendizado federado.
Treina com todos os dados disponíveis sem federação.
"""

import torch
import numpy as np
from typing import Dict, Tuple
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from data_utils import load_and_preprocess_dataset, tokenize_dataset, load_tokenizer
from config import MODEL_NAME, NUM_LABELS, get_training_config
from graph_utils import PerformanceMonitor

class CentralizedModel:
    """Modelo centralizado para detecção de phishing."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.test_dataset = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.performance_monitor = PerformanceMonitor()
        self.metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'energy_consumption': 0.0,
            'processing_time': 0.0
        }
        
        print(f"[Modelo Centralizado] Inicializando...")
        print(f"[Modelo Centralizado] Dispositivo: {self.device}")
    
    def setup_data_and_model(self):
        """Configura dados e modelo para treinamento centralizado."""
        print("[Modelo Centralizado] Carregando dados e modelo...")
        
        # Carregar tokenizer
        self.tokenizer = load_tokenizer()
        
        # Carregar e preprocessar dataset completo
        dataset = load_and_preprocess_dataset()
        
        # Tokenizar dataset
        tokenized_dataset = tokenize_dataset(dataset, self.tokenizer)
        
        # Dividir em treino e teste (usando todos os dados)
        train_test_split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
        self.train_dataset = train_test_split['train']
        self.test_dataset = train_test_split['test']
        
        print(f"[Modelo Centralizado] Dados de treino: {len(self.train_dataset)} amostras")
        print(f"[Modelo Centralizado] Dados de teste: {len(self.test_dataset)} amostras")
        
        # Inicializar modelo
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS
        )
        self.model.to(self.device)
        
        print("[Modelo Centralizado] Modelo e dados configurados com sucesso!")
    
    def compute_metrics(self, pred):
        """Calcula métricas de avaliação."""
        predictions, labels = pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self) -> Dict:
        """Treina o modelo centralizado com todos os dados."""
        print("\n[Modelo Centralizado] Iniciando treinamento...")
        
        if self.model is None or self.train_dataset is None:
            self.setup_data_and_model()
        
        # Iniciar monitoramento de performance
        self.performance_monitor.start_monitoring()
        
        # Configurar argumentos de treinamento
        training_config = get_training_config()
        
        training_args = TrainingArguments(
            output_dir="./centralized_results",
            num_train_epochs=training_config.get("num_train_epochs", 3),
            per_device_train_batch_size=training_config.get("per_device_train_batch_size", 8),
            per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 8),
            warmup_steps=training_config.get("warmup_steps", 500),
            weight_decay=training_config.get("weight_decay", 0.01),
            logging_dir="./centralized_logs",
            logging_steps=100000,
            eval_strategy="epoch",  # Corrigido: evaluation_strategy -> eval_strategy
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model=training_config.get("metric_for_best_model", "f1"),
            learning_rate=training_config.get("learning_rate", 2e-5),
            report_to=None,  # Desabilitar wandb
        )
        
        # Criar trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        # Treinar modelo
        print("[Modelo Centralizado] Executando treinamento...")
        train_result = trainer.train()
        
        # Parar monitoramento de performance
        energy_consumption, processing_time = self.performance_monitor.stop_monitoring()
        
        print(f"[Modelo Centralizado] Treinamento concluído!")
        print(f"[Modelo Centralizado] Tempo de processamento: {processing_time:.2f}s")
        print(f"[Modelo Centralizado] Consumo de energia: {energy_consumption:.2f}J")
        
        # Avaliar modelo final
        eval_result = trainer.evaluate()
        
        # Armazenar métricas
        self.metrics = {
            'accuracy': eval_result.get('eval_accuracy', 0.0),
            'precision': eval_result.get('eval_precision', 0.0),
            'recall': eval_result.get('eval_recall', 0.0),
            'f1': eval_result.get('eval_f1', 0.0),
            'energy_consumption': energy_consumption,
            'processing_time': processing_time
        }
        
        print(f"[Modelo Centralizado] Métricas finais:")
        print(f"  - Acurácia: {self.metrics['accuracy']:.4f}")
        print(f"  - Precisão: {self.metrics['precision']:.4f}")
        print(f"  - Recall: {self.metrics['recall']:.4f}")
        print(f"  - F1-Score: {self.metrics['f1']:.4f}")
        
        return self.metrics
    
    def get_metrics(self) -> Dict:
        """Retorna as métricas do modelo centralizado."""
        return self.metrics.copy()
    
    def evaluate(self) -> Dict:
        """Avalia o modelo treinado."""
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda!")
        
        print("[Modelo Centralizado] Avaliando modelo...")
        
        # Criar trainer apenas para avaliação
        training_args = TrainingArguments(
            output_dir="./centralized_results",
            per_device_eval_batch_size=8,
            report_to=None,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            eval_dataset=self.test_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        eval_result = trainer.evaluate()
        
        return {
            'accuracy': eval_result.get('eval_accuracy', 0.0),
            'precision': eval_result.get('eval_precision', 0.0),
            'recall': eval_result.get('eval_recall', 0.0),
            'f1': eval_result.get('eval_f1', 0.0)
        }