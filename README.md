# Detecção de Phishing com Aprendizado Federado

Sistema de detecção de emails de phishing usando TinyBert adaptado com Flower Framework para aprendizado federado.

## 📋 Visão Geral

Este projeto implementa um sistema de aprendizado federado para detecção de emails de phishing usando:
- **Modelo**: TinyBert (prajjwal1/bert-tiny) do HuggingFace
- **Dataset**: zefang-liu/phishing-email-dataset
- **Framework**: Flower para aprendizado federado
- **Métricas**: Accuracy, F1-Score, Precision, Recall

## 🏗️ Arquitetura

```
📁 PhishingDetection001/
├── 📄 config.py           # Configurações do sistema
├── 📄 data_utils.py       # Utilitários para dados
├── 📄 client.py           # Cliente Flower
├── 📄 server.py           # Servidor Flower
├── 📄 requirements.txt    # Dependências
└── 📄 README.md          # Este arquivo
```

## 🚀 Instalação

### 1. Clonar/Baixar o Projeto
```bash
cd "c:\Aprendizado Federado\PhishingDetection001"
```

### 2. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 3. Configurar Token HuggingFace (Opcional)
Crie um arquivo `HuggingFaceToken.txt` na raiz do projeto com seu token do HuggingFace:
```
hf_seu_token_aqui
```

## 🎯 Como Usar - Passo a Passo Detalhado

### 📋 Pré-requisitos

1. **Instalar dependências:**
   
   **Para GPU Nvidia (recomendado):**
   ```bash
   # Primeiro instale PyTorch com CUDA
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # Depois instale as outras dependências
   pip install -r requirements.txt
   ```
   
   **Para CPU apenas:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Navegar para o diretório**:
   ```bash
   cd "c:\Aprendizado Federado\PhishingDetection001"
   ```

## ⚡ Detecção Automática de Hardware

O sistema detecta automaticamente seu hardware e aplica as configurações ideais:

### 🖥️ Modo GPU (Nvidia CUDA)
- **Configurações**: Idênticas ao Google Colab original
- **Características**: 3 épocas, batch size 8, FP16, pin_memory ativo
- **Tempo estimado**: 2-5 minutos para TODO o ciclo (como no Colab)
- **Ideal para**: Treinamento completo e rápido

### 💻 Modo CPU
- **Configurações**: Otimizadas para velocidade em CPU
- **Características**: 1 época, batch size maior, sem avaliação intermediária
- **Tempo estimado**: 2-5 minutos para TODO o ciclo
- **Ideal para**: Máquinas sem GPU

**O sistema escolhe automaticamente** baseado na detecção de GPU CUDA disponível.

### 🚀 Cenário 1: Teste Básico (2 Clientes)

**IMPORTANTE**: Execute os comandos na ordem exata abaixo!

#### Passo 1: Abrir Terminal do Servidor
1. Abra o **primeiro terminal** (PowerShell/CMD)
2. Navegue para o diretório do projeto:
   ```bash
   cd "c:\Aprendizado Federado\PhishingDetection001"
   ```
3. Execute o servidor:
   ```bash
   python server.py --rounds 3 --min-clients 2
   ```
4. **Aguarde** a mensagem: `INFO flwr 2024-XX-XX XX:XX:XX,XXX | app.py:XXX | Starting Flower server...`
5. **NÃO FECHE** este terminal - deixe rodando!

#### Passo 2: Abrir Terminal do Cliente 0
1. Abra o **segundo terminal** (nova janela)
2. Navegue para o diretório:
   ```bash
   cd "c:\Aprendizado Federado\PhishingDetection001"
   ```
3. Execute o primeiro cliente:
   ```bash
   python client.py --client-id 0 --num-clients 2
   ```
4. Você verá: `[Cliente 0] Conectando ao servidor...`
5. **NÃO FECHE** este terminal!

#### Passo 3: Abrir Terminal do Cliente 1
1. Abra o **terceiro terminal** (nova janela)
2. Navegue para o diretório:
   ```bash
   cd "c:\Aprendizado Federado\PhishingDetection001"
   ```
3. Execute o segundo cliente:
   ```bash
   python client.py --client-id 1 --num-clients 2
   ```
4. Você verá: `[Cliente 1] Conectando ao servidor...`

#### Passo 4: Observar o Treinamento
- O treinamento começará automaticamente quando ambos os clientes se conectarem
- Observe os logs em **todos os 3 terminais**
- O processo levará alguns minutos para completar

### 🔄 Cenário 2: Simulação Avançada (4 Clientes)

#### Passo 1: Servidor (Terminal 1)
```bash
cd "c:\Aprendizado Federado\PhishingDetection001"
python server.py --rounds 5 --min-clients 4 --port 8080
```

#### Passo 2-5: Clientes (Terminais 2, 3, 4, 5)
**Terminal 2:**
```bash
cd "c:\Aprendizado Federado\PhishingDetection001"
python client.py --client-id 0 --num-clients 4
```

**Terminal 3:**
```bash
cd "c:\Aprendizado Federado\PhishingDetection001"
python client.py --client-id 1 --num-clients 4
```

**Terminal 4:**
```bash
cd "c:\Aprendizado Federado\PhishingDetection001"
python client.py --client-id 2 --num-clients 4
```

**Terminal 5:**
```bash
cd "c:\Aprendizado Federado\PhishingDetection001"
python client.py --client-id 3 --num-clients 4
```

### 📺 O Que Esperar Durante a Execução

#### No Terminal do Servidor:
```
INFO flwr 2024-XX-XX XX:XX:XX,XXX | app.py:163 | Starting Flower server, config: ServerConfig(...)
INFO flwr 2024-XX-XX XX:XX:XX,XXX | app.py:176 | Flower ECE: gRPC server running (5 rounds), SSL is disabled
INFO flwr 2024-XX-XX XX:XX:XX,XXX | server.py:89 | Initializing global parameters
INFO flwr 2024-XX-XX XX:XX:XX,XXX | server.py:276 | FL starting

=== Rodada 1 - Agregação de Treinamento ===
Clientes que completaram o treinamento: 2
Loss médio de treinamento: 0.4523

=== Rodada 1 - Agregação de Avaliação ===
Accuracy média: 0.8456
F1-Score médio: 0.8234
```

#### Nos Terminais dos Clientes:
```
[Cliente 0] Conectando ao servidor localhost:8080...
[Cliente 0] Carregando dataset e modelo...
[Cliente 0] Dataset carregado: 1000 amostras de treino, 250 de teste
[Cliente 0] Modelo TinyBert carregado com sucesso
[Cliente 0] Iniciando treinamento - Rodada 1
[Cliente 0] Executando treinamento...
Epoch 1/3: 100%|████████| 125/125 [01:23<00:00,  1.50it/s]
[Cliente 0] Treinamento concluído - Loss: 0.4234
[Cliente 0] Iniciando avaliação
[Cliente 0] Avaliação concluída:
  - Loss: 0.3456
  - Accuracy: 0.8567
  - F1: 0.8234
```

### ⚠️ Dicas Importantes

1. **Ordem de Execução**: SEMPRE inicie o servidor primeiro!
2. **Aguardar Conexão**: Espere cada cliente conectar antes de iniciar o próximo
3. **Não Fechar Terminais**: Mantenha todos os terminais abertos durante o treinamento
4. **Tempo de Execução**: O processo completo leva 5-15 minutos dependendo do hardware
5. **Interrupção**: Para parar, use `Ctrl+C` em cada terminal
6. **Primeira Execução**: Pode demorar mais devido ao download do modelo e dataset
7. **Memória**: Certifique-se de ter pelo menos 4GB de RAM disponível

## ⚙️ Parâmetros de Configuração

### Servidor (`server.py`)
- `--host`: Endereço do servidor (padrão: localhost)
- `--port`: Porta do servidor (padrão: 8080)
- `--rounds`: Número de rodadas de treinamento (padrão: 5)
- `--min-clients`: Número mínimo de clientes (padrão: 2)

### Cliente (`client.py`)
- `--client-id`: ID único do cliente (obrigatório: 0, 1, 2, ...)
- `--num-clients`: Número total de clientes (padrão: 2)
- `--server-address`: Endereço do servidor (padrão: localhost:8080)

## 📊 Configurações do Modelo

As configurações podem ser ajustadas no arquivo `config.py`:

```python
# Configurações de treinamento
TRAINING_CONFIG = {
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 8,
    "num_train_epochs": 3,
    "weight_decay": 0.01,
    # ...
}

# Configurações do Flower
FLOWER_CONFIG = {
    "num_rounds": 5,
    "min_fit_clients": 2,
    "fraction_fit": 1.0,
    # ...
}
```

## 📈 Monitoramento

Durante o treinamento, você verá:

### No Servidor:
```
=== Rodada 1 - Agregação de Treinamento ===
Clientes que completaram o treinamento: 2
Loss médio de treinamento: 0.4523

=== Rodada 1 - Agregação de Avaliação ===
Accuracy média: 0.8456
F1-Score médio: 0.8234
Precision média: 0.8123
Recall médio: 0.8345
```

### Nos Clientes:
```
[Cliente 0] Iniciando treinamento - Rodada 1
[Cliente 0] Executando treinamento...
[Cliente 0] Treinamento concluído - Loss: 0.4234
[Cliente 0] Iniciando avaliação
[Cliente 0] Avaliação concluída:
  - Loss: 0.3456
  - Accuracy: 0.8567
  - F1: 0.8234
```

## 🔍 Estrutura dos Dados

O dataset é automaticamente:
1. **Carregado** do HuggingFace Hub
2. **Preprocessado** (labels codificados: 0=Legítimo, 1=Phishing)
3. **Dividido** em treino/teste (80%/20%)
4. **Particionado** entre os clientes
5. **Tokenizado** com TinyBert tokenizer

### Distribuição por Cliente
- Cada cliente recebe uma partição única dos dados
- Particionamento sequencial (Cliente 0: índices 0-N/k, Cliente 1: N/k-2N/k, etc.)
- Preserva distribuição de classes em cada partição

## 🧪 Testes do Modelo

Após o treinamento, o servidor testa automaticamente o modelo com frases de exemplo:

```python
test_phrases = [
    "Click here to reset your account password now.",  # Phishing
    "Meeting confirmed for 10 AM tomorrow in the office.",  # Legítimo
    "This is your confirmation code. Do not share it with anyone. Type in your credit card information",  # Phishing
    # ...
]
```

## 🛠️ Solução de Problemas

### ✅ Detecção Automática de Hardware
**O sistema agora detecta automaticamente seu hardware:**
- **GPU Nvidia**: Usa configurações originais do Colab (rápidas e completas)
- **CPU**: Usa configurações otimizadas para velocidade
- **Sem configuração manual necessária**

### GPU Não Detectada
**Sintoma**: Sistema usa modo CPU mesmo tendo GPU Nvidia
**Soluções**:
1. **Instale CUDA**: Baixe do site oficial da Nvidia
2. **Instale PyTorch com CUDA**: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
3. **Verifique**: Execute `python -c "import torch; print(torch.cuda.is_available())"`

### Performance Ainda Lenta
**Se mesmo com GPU o sistema estiver lento:**
1. **Verifique drivers**: Atualize drivers da GPU
2. **Memória GPU**: Reduza batch size se houver erro de memória
3. **Outros processos**: Feche programas que usam GPU

### Erro: "Token HuggingFace não encontrado"
- **Solução**: Crie o arquivo `HuggingFaceToken.txt` ou ignore (alguns datasets públicos funcionam sem token)

### Erro: "Conexão recusada"
- **Solução**: Certifique-se de que o servidor está rodando antes dos clientes
- Verifique se a porta não está em uso

### Erro: "CUDA out of memory"
- **Solução**: Reduza `per_device_train_batch_size` e `per_device_eval_batch_size` no `config_performance.py`
- Use o modo "test" para consumir menos memória

### Erro: "Clientes insuficientes"
- **Solução**: Certifique-se de que o número de clientes conectados >= `min_clients`

## 📋 Requisitos do Sistema

- **Python**: 3.8+
- **RAM**: 4GB+ (recomendado: 8GB+)
- **GPU**: Opcional (CUDA compatível)
- **Espaço**: ~2GB para modelo e dados

## 🔧 Personalização

### Alterar Modelo
No `config.py`:
```python
MODEL_NAME = "distilbert-base-uncased"  # Exemplo
NUM_LABELS = 2
```

### Alterar Dataset
No `config.py`:
```python
DATASET_NAME = "seu-dataset-aqui"
```

### Ajustar Estratégia
No `server.py`, modifique a classe `PhishingFedAvg` ou use outras estratégias do Flower.

## 📚 Referências

- [Flower Framework](https://flower.ai/)
- [TinyBert Paper](https://arxiv.org/abs/2110.01518)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [Phishing Email Dataset](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset)

## 📄 Licença

Este projeto é para fins educacionais e de pesquisa.

---

**Desenvolvido para aprendizado federado de detecção de phishing com TinyBert e Flower Framework** 🌸