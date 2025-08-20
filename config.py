# -*- coding: utf-8 -*-
"""
Configuração para o sistema de aprendizado federado de detecção de phishing
com TinyBert adaptado usando Flower Framework.
"""

import os
from config_performance import get_performance_config, print_current_mode

# Configurações do modelo
MODEL_NAME = "prajjwal1/bert-tiny"
NUM_LABELS = 2
MAX_LENGTH = 512

# Configurações do dataset
DATASET_NAME = "zefang-liu/phishing-email-dataset"
TEST_SIZE = 0.2
RANDOM_SEED = 42

# Configurações base de treinamento (serão mescladas com configurações de performance)
BASE_TRAINING_CONFIG = {
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "metric_for_best_model": "f1",
}

def get_training_config():
    """Retorna configurações de treinamento mescladas com configurações de performance."""
    performance_config = get_performance_config()
    
    # Mescla configurações base com configurações de performance
    training_config = {**BASE_TRAINING_CONFIG, **performance_config}
    
    return training_config

# Configuração de treinamento (será atualizada dinamicamente)
TRAINING_CONFIG = get_training_config()

# Configurações do Flower
FLOWER_CONFIG = {
    "num_rounds": 5,
    "min_fit_clients": 2,
    "min_evaluate_clients": 2,
    "min_available_clients": 2,
    "fraction_fit": 1.0,
    "fraction_evaluate": 1.0
}

# Configurações do servidor
SERVER_CONFIG = {
    "host": "localhost",
    "port": 8080
}

# Desabilitar Weights & Biases
os.environ["WANDB_DISABLED"] = "true"

# Token do HuggingFace (será lido do arquivo se existir)
HF_TOKEN_FILE = "HuggingFaceToken.txt"

def get_hf_token():
    """Carrega o token do HuggingFace se o arquivo existir."""
    try:
        if os.path.exists(HF_TOKEN_FILE):
            with open(HF_TOKEN_FILE, "r") as f:
                return f.read().strip()
    except Exception as e:
        print(f"Aviso: Não foi possível carregar o token do HuggingFace: {e}")
    return None