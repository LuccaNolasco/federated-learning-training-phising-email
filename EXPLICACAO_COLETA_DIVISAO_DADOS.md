# Explicação: Coleta e Divisão de Dados no Sistema de Aprendizado Federado

## Resumo Executivo

Exemplo para 2 clientes
Este documento explica como o sistema de aprendizado federado coleta dados do Hugging Face e os divide entre os clientes. A **barra de progresso mostra 2799 iterações/steps**, que corresponde ao número total de passos de treinamento (não amostras), calculado como: 7.460 amostras ÷ 8 batch_size × 3 épocas = 2.799 steps.

## 1. Processo de Coleta de Dados

### 1.1 Fonte dos Dados
- **Dataset**: `zefang-liu/phishing-email-dataset` do Hugging Face
- **Conteúdo**: Emails classificados como "Phishing" ou "Legítimo"
- **Total de amostras**: 18.650 emails
- **Distribuição original**: 
  - Legítimos (label 0): 11.322 emails
  - Phishing (label 1): 7.328 emails

### 1.2 Pré-processamento
1. **Limpeza**: Remove coluna desnecessária (`Unnamed: 0`)
2. **Renomeação**: 
   - `Email Text` → `data`
   - `Email Type` → `label`
3. **Codificação de labels**: Converte texto para números (0 = Legítimo, 1 = Phishing)

## 2. Divisão Treino/Teste

### 2.1 Configuração da Divisão
- **Proporção**: 80% treino, 20% teste (`TEST_SIZE = 0.2`)
- **Método**: Divisão estratificada (mantém proporção de classes)
- **Seed**: 42 (para reprodutibilidade)

### 2.2 Resultado da Divisão
- **Dataset de Treino**: 14.920 amostras
  - Legítimos: 9.058 amostras
  - Phishing: 5.862 amostras
- **Dataset de Teste**: 3.730 amostras
  - Legítimos: 2.264 amostras
  - Phishing: 1.466 amostras

## 3. Divisão Entre Clientes

### 3.1 Algoritmo de Particionamento

O algoritmo implementado na função `partition_dataset()` funciona da seguinte forma:

```python
def partition_dataset(dataset, num_clients, client_id):
    total_samples = len(dataset)
    samples_per_client = total_samples // num_clients
    
    start_idx = client_id * samples_per_client
    if client_id == num_clients - 1:  # Último cliente pega o resto
        end_idx = total_samples
    else:
        end_idx = start_idx + samples_per_client
```

### 3.2 Explicação da Barra de Progresso (2799)

#### Com 2 Clientes:
- **Total de treino**: 14.920 amostras
- **Amostras por cliente**: 14.920 ÷ 2 = 7.460 amostras
- **Cliente 0**: índices 0-7.459 (7.460 amostras)
- **Cliente 1**: índices 7.460-14.919 (7.460 amostras)

**O número 2799 na barra de progresso representa os STEPS de treinamento, não as amostras!**

**Cálculo dos Steps:**
- **Amostras por cliente**: 7.460
- **Batch size**: 8 (configuração GPU)
- **Épocas**: 3
- **Steps por época**: 7.460 ÷ 8 = 933 steps (arredondado para cima)
- **Total de steps**: 933 × 3 épocas = **2.799 steps**

✅ **2799 = Total de iterações/steps de treinamento, não número de amostras!**

#### Com 1 Cliente:
- **Cliente 0**: recebe TODAS as 14.920 amostras de treino
- **Cliente 0**: recebe TODAS as 3.730 amostras de teste
- Por isso treina com "o dobro" - na verdade, treina com TUDO!

### 3.3 Problema Identificado

O algoritmo atual tem uma **divisão sequencial simples**, que pode causar:
1. **Desbalanceamento de classes** entre clientes
2. **Distribuição não-IID** (dados não independentes e identicamente distribuídos)
3. **Último cliente pode receber mais amostras** devido ao resto da divisão

## 4. Análise do Log Fornecido

Analisando seu log:
```
Cliente 0: 7460 amostras (índices 0-7459)
Cliente 0 - Distribuição: Counter({0: 4609, 1: 2851})
Cliente 0: 1865 amostras (índices 0-1864)
Cliente 0 - Distribuição: Counter({0: 1108, 1: 757})
```

**Interpretação**:
1. **Primeira linha**: Dataset de treino - 7.460 amostras
2. **Segunda linha**: Distribuição de classes no treino
3. **Terceira linha**: Dataset de teste - 1.865 amostras  
4. **Quarta linha**: Distribuição de classes no teste

**Barra de Progresso**:
```
100%|██████████| 2799/2799 [04:03<00:00, 11.48it/s]
```
- **2799**: Total de steps/iterações de treinamento
- **Cálculo**: 7.460 amostras ÷ 8 batch_size × 3 épocas = 2.799 steps
- **11.48it/s**: Velocidade de processamento (iterações por segundo)

## 5. Recomendações de Melhoria

### 5.1 Divisão Estratificada por Cliente
Implementar divisão que mantenha a proporção de classes em cada cliente:

```python
def partition_dataset_stratified(dataset, num_clients, client_id):
    # Separar por classe
    class_0_indices = [i for i, label in enumerate(dataset['label']) if label == 0]
    class_1_indices = [i for i, label in enumerate(dataset['label']) if label == 1]
    
    # Dividir cada classe igualmente
    class_0_per_client = len(class_0_indices) // num_clients
    class_1_per_client = len(class_1_indices) // num_clients
    
    # Selecionar índices para o cliente
    start_0 = client_id * class_0_per_client
    end_0 = start_0 + class_0_per_client
    start_1 = client_id * class_1_per_client
    end_1 = start_1 + class_1_per_client
    
    client_indices = class_0_indices[start_0:end_0] + class_1_indices[start_1:end_1]
    return dataset.select(client_indices)
```

### 5.2 Divisão Aleatória
Implementar shuffle antes da divisão para evitar bias sequencial.

## 6. Configurações Relevantes

### 6.1 Arquivo config.py
- `TEST_SIZE = 0.2`: Define 20% para teste
- `RANDOM_SEED = 42`: Garante reprodutibilidade
- `FLOWER_CONFIG["min_available_clients"] = 2`: Número padrão de clientes

### 6.2 Parâmetros de Treinamento
- **Épocas**: 3
- **Batch Size**: 8 (treino e avaliação)
- **Learning Rate**: 2e-5

## 7. Conclusão

O comportamento observado é **esperado** dado o algoritmo atual:
- **2 clientes**: Cada um recebe metade dos dados
- **1 cliente**: Recebe todos os dados (por isso o "dobro")

A divisão é **sequencial e simples**, mas pode ser melhorada para garantir melhor distribuição de classes e aleatoriedade entre os clientes.

---

**Arquivo gerado automaticamente em**: `r new Date().toLocaleString('pt-BR')`
**Versão do sistema**: Aprendizado Federado com TinyBert
**Dataset**: zefang-liu/phishing-email-dataset