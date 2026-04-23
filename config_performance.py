# -*- coding: utf-8 -*-
"""
Configuração de Performance com detecção automática de hardware.
Usa configurações originais do Colab para GPU e otimizadas para CPU.
"""

import torch

# Detecta automaticamente se há GPU disponível
HAS_GPU = torch.cuda.is_available()
DEVICE_TYPE = "gpu" if HAS_GPU else "cpu"

def get_performance_config():
    """Retorna configurações otimizadas baseadas no hardware disponível."""
    
    if HAS_GPU:
        # Configurações originais do Colab (otimizadas para GPU)
        return {
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "num_train_epochs": 1,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "logging_steps": 100000,
            "load_best_model_at_end": True,
            "dataloader_pin_memory": True,  # GPU pode usar pin_memory
            "dataloader_num_workers": 2,    # GPU pode usar workers
            "prediction_loss_only": False,
            "disable_tqdm": False,
            "fp16": True,  # Acelera treinamento em GPU
            "gradient_checkpointing": False,
        }
    else:
        # Configurações otimizadas para CPU (mais rápidas)
        return {
            "eval_strategy": "no",           # Sem avaliação intermediária
            "save_strategy": "no",           # Sem salvamento intermediário
            "num_train_epochs": 1,           # Menos épocas
            "per_device_train_batch_size": 32,  # Batch maior para CPU
            "per_device_eval_batch_size": 64,   # Batch maior para avaliação
            "logging_steps": 100000,            # Menos logs
            "load_best_model_at_end": False, # Sem carregamento do melhor modelo
            "dataloader_pin_memory": False,  # CPU não precisa de pin_memory
            "dataloader_num_workers": 0,     # CPU sem workers paralelos
            "prediction_loss_only": False,    # Todas as métricas
            "disable_tqdm": True,            # Sem barra de progresso
            "fp16": False,                   # CPU não suporta fp16
            "gradient_checkpointing": False,
        }

def print_current_mode():
    """Imprime informações sobre o hardware e configurações."""
    config = get_performance_config()
    
    print(f"🖥️  Hardware Detectado: {DEVICE_TYPE.upper()}")
    if HAS_GPU:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"   - GPU: {gpu_name}")
        print(f"   - Configurações: Originais do Colab (otimizadas para GPU)")
    else:
        print(f"   - CPU: Configurações otimizadas para velocidade")
    
    print(f"\n⚙️  Configurações de Treinamento:")
    print(f"   - Épocas: {config['num_train_epochs']}")
    print(f"   - Batch Size (treino): {config['per_device_train_batch_size']}")
    print(f"   - Batch Size (eval): {config['per_device_eval_batch_size']}")
    print(f"   - Avaliação: {'Ativa' if config['eval_strategy'] != 'no' else 'Desabilitada'}")
    print(f"   - Pin Memory: {config['dataloader_pin_memory']}")
    print(f"   - Workers: {config['dataloader_num_workers']}")
    if HAS_GPU:
        print(f"   - FP16: {config['fp16']}")
    print()