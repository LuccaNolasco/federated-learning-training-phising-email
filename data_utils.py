# -*- coding: utf-8 -*-
"""
Utilitários para carregamento e processamento de dados para detecção de phishing
com TinyBert adaptado em aprendizado federado.
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Subset
from collections import Counter
import numpy as np
from config import (
    DATASET_NAME, MODEL_NAME, MAX_LENGTH, TEST_SIZE, RANDOM_SEED, get_hf_token
)
from huggingface_hub import HfFolder

def setup_hf_token():
    """Configura o token do HuggingFace se disponível."""
    token = get_hf_token()
    if token:
        HfFolder.save_token(token)
        print("Token do HuggingFace configurado com sucesso.")
    else:
        print("Aviso: Token do HuggingFace não encontrado. Alguns datasets podem não estar acessíveis.")

def encode_label(example):
    """Codifica os labels de texto para valores numéricos."""
    label_text = example["label"].lower().strip()
    if "phishing" in label_text:
        example["label"] = 1
    else:
        example["label"] = 0
    return example

def load_and_preprocess_dataset():
    """Carrega e preprocessa o dataset de phishing."""
    print("Carregando dataset de phishing...")
    
    # Configurar token se disponível
    setup_hf_token()
    
    # Carregar dataset
    dataset = load_dataset(DATASET_NAME)
    dataset = dataset["train"]
    dataset = dataset.remove_columns(['Unnamed: 0'])
    
    # Renomear colunas
    dataset = dataset.rename_column("Email Text", "data")
    dataset = dataset.rename_column("Email Type", "label")
    
    # Codificar labels
    dataset = dataset.map(encode_label)
    
    print(f"Distribuição de labels: {Counter(dataset['label'])}")
    
    return dataset

def tokenize_dataset(dataset, tokenizer):
    """Tokeniza o dataset usando o tokenizer fornecido."""
    def tokenize_batch(batch):
        texts = [str(x) for x in batch["data"]]
        return tokenizer(texts, padding="max_length", truncation=True, max_length=MAX_LENGTH)
    
    print("Tokenizando dataset...")
    tokenized_dataset = dataset.map(tokenize_batch, batched=True)
    
    return tokenized_dataset

def split_dataset(dataset):
    """Divide o dataset em treino e teste de forma estratificada."""
    print("Dividindo dataset em treino e teste...")
    
    # Codificar coluna de label para estratificação
    dataset = dataset.class_encode_column("label")
    
    # Divisão estratificada
    split = dataset.train_test_split(
        test_size=TEST_SIZE,
        shuffle=True,
        seed=RANDOM_SEED,
        stratify_by_column="label"
    )
    
    train_ds = split["train"]
    test_ds = split["test"]
    
    print(f"Treino - Distribuição: {Counter(train_ds['label'])}")
    print(f"Teste - Distribuição: {Counter(test_ds['label'])}")
    
    return train_ds, test_ds

def partition_dataset(dataset, num_clients, client_id):
    """Particiona o dataset para um cliente específico."""
    total_samples = len(dataset)
    samples_per_client = total_samples // num_clients
    
    start_idx = client_id * samples_per_client
    if client_id == num_clients - 1:  # Último cliente pega o resto
        end_idx = total_samples
    else:
        end_idx = start_idx + samples_per_client
    
    # Criar subset para o cliente
    client_indices = list(range(start_idx, end_idx))
    client_dataset = dataset.select(client_indices)
    
    print(f"Cliente {client_id}: {len(client_dataset)} amostras (índices {start_idx}-{end_idx-1})")
    print(f"Cliente {client_id} - Distribuição: {Counter(client_dataset['label'])}")
    
    return client_dataset

def create_dataloaders(train_dataset, test_dataset, batch_size=8):
    """Cria DataLoaders otimizados para treino e teste."""
    # Configurar formato dos datasets para PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # DataLoaders otimizados para CPU (sem pin_memory)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=False,  # Resolver warning em CPU
        num_workers=0      # Evitar overhead de multiprocessing
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=False,  # Resolver warning em CPU
        num_workers=0      # Evitar overhead de multiprocessing
    )
    
    return train_loader, test_loader

def load_tokenizer():
    """Carrega o tokenizer do modelo."""
    print(f"Carregando tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer

def prepare_client_data(client_id, num_clients):
    """Prepara os dados para um cliente específico."""
    print(f"\n=== Preparando dados para Cliente {client_id} ===")
    
    # Carregar e preprocessar dataset
    dataset = load_and_preprocess_dataset()
    
    # Carregar tokenizer
    tokenizer = load_tokenizer()
    
    # Dividir em treino e teste
    train_ds, test_ds = split_dataset(dataset)
    
    # Particionar dados para o cliente
    client_train = partition_dataset(train_ds, num_clients, client_id)
    client_test = partition_dataset(test_ds, num_clients, client_id)
    
    # Tokenizar datasets do cliente
    client_train = tokenize_dataset(client_train, tokenizer)
    client_test = tokenize_dataset(client_test, tokenizer)
    
    print(f"Cliente {client_id} preparado com sucesso!")
    
    return client_train, client_test, tokenizer

def test_phrase(model, tokenizer, text):
    """Testa uma frase específica no modelo treinado."""
    # Preparar tensores
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
    
    # Detectar dispositivo do modelo
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Fazer previsão
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Processar resultado
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs).item()
    label = "Phishing" if pred == 1 else "Legítimo"
    confidence = probs[0][pred].item()
    
    return label, confidence