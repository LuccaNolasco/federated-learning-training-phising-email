# Documentação das Novas Funcionalidades - Sistema de Aprendizado Federado

## Visão Geral

Este documento descreve as novas funcionalidades implementadas no sistema de aprendizado federado para detecção de phishing, incluindo cálculo de loss por round de agregação e geração de matrizes de confusão.

## 1. Cálculo de Loss por Round de Agregação

### Funcionalidade
- **Objetivo**: Monitorar a evolução do loss do modelo agregado após cada round de treinamento
- **Implementação**: Avaliação automática do modelo agregado no conjunto de teste após cada agregação

### Arquivos Modificados
- `server.py`: Classe `PhishingFedAvg`
  - Método `_evaluate_aggregated_model_loss()`: Calcula o loss do modelo agregado
  - Método `_save_loss_to_csv()`: Persiste os dados de loss
  - Método `_generate_loss_chart()`: Gera gráfico de evolução

### Dados Gerados

#### CSV de Loss (`loss_tracking/loss_per_round.csv`)
```csv
round,loss,timestamp
1,0.6234,2024-01-15 10:30:45
2,0.5891,2024-01-15 10:35:12
3,0.5456,2024-01-15 10:39:38
...
```

**Campos:**
- `round`: Número do round de agregação (1, 2, 3, ...)
- `loss`: Valor do loss calculado no conjunto de teste
- `timestamp`: Data e hora da avaliação

#### Gráfico de Evolução (`loss_tracking/loss_evolution_chart.png`)
- **Tipo**: Gráfico de linha
- **Eixo X**: Rounds de agregação
- **Eixo Y**: Valor do loss
- **Formato**: PNG, 300 DPI

## 2. Matrizes de Confusão

### Funcionalidade
- **Objetivo**: Avaliar a performance de classificação do modelo agregado final e de cada cliente individual
- **Implementação**: Geração automática ao final do treinamento

### Arquivos Modificados
- `server.py`: 
  - Método `_generate_confusion_matrices()`: Orquestra a geração
  - Método `_generate_aggregated_model_confusion_matrix()`: Matriz do modelo agregado
  - Método `_generate_client_confusion_matrices()`: Matrizes dos clientes
- `client.py`:
  - Método `_save_predictions()`: Salva predições durante avaliação

### Dados Gerados

#### Matriz do Modelo Agregado
**Arquivos:**
- `confusion_matrices/aggregated_model_confusion_matrix.png`: Visualização
- `confusion_matrices/aggregated_model_classification_report.csv`: Relatório detalhado

#### Matrizes dos Clientes Individuais
**Arquivos por cliente:**
- `confusion_matrices/client_{id}_confusion_matrix.png`: Visualização
- `confusion_matrices/client_{id}_classification_report.csv`: Relatório detalhado
- `client_predictions/client_{id}_predictions.pkl`: Predições brutas (formato pickle)

#### Formato do Relatório de Classificação (CSV)
```csv
,precision,recall,f1-score,support
Legítimo,0.95,0.92,0.93,1000
Phishing,0.91,0.94,0.92,800
accuracy,,,0.93,1800
macro avg,0.93,0.93,0.93,1800
weighted avg,0.93,0.93,0.93,1800
```

## 3. Integração com Sistema Existente

### Compatibilidade
- ✅ **Métricas existentes**: Mantidas integralmente
- ✅ **Coleta de dados**: Compatível com `MetricsCollector`
- ✅ **Visualizações**: Integradas com `GraphGenerator`
- ✅ **Persistência**: Utiliza estrutura de diretórios existente

### Fluxo de Execução
1. **Durante o treinamento**: 
   - Cada round: Cálculo e salvamento do loss
   - Cada avaliação de cliente: Salvamento de predições
2. **Ao final do treinamento**:
   - Geração do gráfico de loss
   - Geração de todas as matrizes de confusão
   - Integração com resumo final existente

## 4. Estrutura de Diretórios

```
PhishingDetection001/
├── loss_tracking/
│   ├── loss_per_round.csv
│   └── loss_evolution_chart.png
├── confusion_matrices/
│   ├── aggregated_model_confusion_matrix.png
│   ├── aggregated_model_classification_report.csv
│   ├── client_1_confusion_matrix.png
│   ├── client_1_classification_report.csv
│   └── ...
├── client_predictions/
│   ├── client_1_predictions.pkl
│   ├── client_2_predictions.pkl
│   └── ...
└── [arquivos existentes...]
```

## 5. Requisitos Técnicos

### Dependências Adicionais
- `pickle`: Para serialização de predições (built-in Python)
- `sklearn.metrics`: Para confusion_matrix e classification_report (já existente)
- `matplotlib.pyplot`: Para visualizações (já existente)
- `seaborn`: Para matrizes de confusão estilizadas (já existente)

### Recursos Computacionais
- **Memória**: +~50MB para armazenamento de predições
- **Processamento**: +~10-15% de tempo por round (avaliação adicional)
- **Armazenamento**: ~5-10MB por experimento (gráficos + dados)

## 6. Configuração e Uso

### Ativação Automática
As novas funcionalidades são ativadas automaticamente quando o servidor é executado. Não requer configuração adicional.

### Personalização
Para modificar comportamentos, edite os seguintes métodos em `server.py`:
- `_evaluate_aggregated_model_loss()`: Frequência de avaliação
- `_generate_loss_chart()`: Estilo do gráfico
- `_generate_confusion_matrices()`: Formato das matrizes

## 7. Tratamento de Erros

### Cenários Cobertos
- **Falha na avaliação de loss**: Continua execução, registra erro
- **Erro na geração de gráficos**: Continua execução, mantém dados CSV
- **Predições de cliente ausentes**: Gera apenas matriz do modelo agregado
- **Problemas de I/O**: Logs detalhados para diagnóstico

### Logs e Debugging
Todas as operações geram logs informativos no console, incluindo:
- Caminhos dos arquivos gerados
- Estatísticas de performance
- Mensagens de erro detalhadas

## 8. Validação e Qualidade

### Precisão Numérica
- **Loss**: Calculado com precisão float32 do PyTorch
- **Métricas**: Utilizando sklearn com configurações padrão
- **Arredondamento**: 4 casas decimais para exibição

### Consistência Visual
- **Paleta de cores**: Azul (Blues) para matrizes de confusão
- **Fontes**: Tamanhos padronizados (12pt labels, 14pt títulos)
- **Resolução**: 300 DPI para todas as imagens
- **Layout**: Tight layout para otimização de espaço

## 9. Manutenção e Extensões Futuras

### Possíveis Melhorias
1. **Métricas adicionais**: AUC-ROC, precisão por classe
2. **Visualizações interativas**: Plotly para gráficos dinâmicos
3. **Exportação**: Formatos adicionais (PDF, SVG)
4. **Análise temporal**: Evolução das métricas por cliente

### Pontos de Extensão
- `_evaluate_aggregated_model_loss()`: Adicionar métricas customizadas
- `_generate_confusion_matrices()`: Novos formatos de visualização
- `_save_predictions()`: Metadados adicionais por cliente