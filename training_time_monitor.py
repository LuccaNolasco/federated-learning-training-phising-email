# -*- coding: utf-8 -*-
"""
Monitor de tempo de treinamento exclusivo para steps de treino.
Exclui tempo de avaliação, salvamento e pausas entre épocas.
"""

import time
from typing import List, Optional
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


class TrainingTimeMonitor(TrainerCallback):
    """
    Callback personalizado para medir apenas o tempo de treinamento efetivo.
    
    Mede exclusivamente:
    - Tempo dos steps de treinamento (forward + backward + optimizer step)
    
    Exclui:
    - Tempo de avaliação ao final das épocas
    - Tempo de salvamento de checkpoints
    - Pausas/travadas entre épocas
    - Tempo de logging extensivo
    """
    
    def __init__(self):
        self.epoch_training_times: List[float] = []
        self.current_epoch_start: Optional[float] = None
        self.step_start_time: Optional[float] = None
        self.total_training_time: float = 0.0
        self.current_epoch_time: float = 0.0
        self.is_training_step: bool = False
        self.is_evaluating: bool = False  # Flag para detectar modo de avaliação
        
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Inicia cronômetro da época."""
        self.current_epoch_start = time.time()
        self.current_epoch_time = 0.0
        self.is_evaluating = False
        print(f"\n{'='*60}")
        print(f"[ÉPOCA {int(state.epoch)}] Iniciando treinamento...")
        print(f"{'='*60}")
        
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Marca início da avaliação."""
        self.is_evaluating = True
        print(f"\n{'─'*40}")
        print("[AVALIAÇÃO] Pausando cronômetro de treinamento")
        print(f"{'─'*40}")
        
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Inicia cronômetro do step de treinamento."""
        # Só medir se não estivermos em modo de avaliação
        if not self.is_evaluating:
            self.step_start_time = time.time()
            self.is_training_step = True
        else:
            self.is_training_step = False
            
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Finaliza cronômetro do step de treinamento."""
        if self.is_training_step and self.step_start_time is not None:
            step_time = time.time() - self.step_start_time
            self.current_epoch_time += step_time
            self.step_start_time = None
            self.is_training_step = False
            
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Finaliza cronômetro da época e armazena o tempo."""
        if self.current_epoch_start is not None:
            # Armazenar apenas o tempo acumulado dos steps de treinamento
            self.epoch_training_times.append(self.current_epoch_time)
            self.total_training_time += self.current_epoch_time
            
            print(f"\n[ÉPOCA {int(state.epoch)} CONCLUÍDA]")
            print(f"  ✓ Tempo de treinamento efetivo: {self.current_epoch_time:.2f}s")
            print(f"  ✓ Tempo total acumulado: {self.total_training_time:.2f}s")
            
            self.current_epoch_start = None
            
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Finaliza o monitoramento e exibe resumo."""
        print(f"\n{'='*60}")
        print(f"🎉 TREINAMENTO CONCLUÍDO COM SUCESSO!")
        print(f"{'='*60}")
        print(f"⏱️  Tempo total de treinamento efetivo: {self.total_training_time:.2f}s")
        print(f"📊 Número de épocas: {len(self.epoch_training_times)}")
        if self.epoch_training_times:
            avg_time = self.total_training_time / len(self.epoch_training_times)
            print(f"⚡ Tempo médio por época: {avg_time:.2f}s")
        print(f"{'='*60}\n")
        print(f"Tempo médio por época: {self.total_training_time / len(self.epoch_training_times):.2f}s")
        print(f"Detalhamento por época:")
        for i, epoch_time in enumerate(self.epoch_training_times, 1):
            print(f"  - Época {i}: {epoch_time:.2f}s")
            
    def get_total_training_time(self) -> float:
        """Retorna o tempo total de treinamento efetivo."""
        return self.total_training_time
        
    def get_epoch_times(self) -> List[float]:
        """Retorna lista com tempos de cada época."""
        return self.epoch_training_times.copy()
        
    def get_average_epoch_time(self) -> float:
        """Retorna tempo médio por época."""
        if not self.epoch_training_times:
            return 0.0
        return self.total_training_time / len(self.epoch_training_times)


class PrecisePerformanceMonitor:
    """
    Monitor de desempenho que usa o TrainingTimeMonitor para tempo preciso.
    Mantém compatibilidade com o PerformanceMonitor original para energia.
    """
    
    def __init__(self):
        self.training_monitor = TrainingTimeMonitor()
        self.start_energy = None
        self.energy_monitor = None
        
        # Importar e inicializar monitoramento de energia do módulo original
        try:
            from graph_utils import PerformanceMonitor
            self.energy_monitor = PerformanceMonitor()
        except ImportError:
            print("Warning: Não foi possível importar PerformanceMonitor para energia")
            
    def get_training_callback(self) -> TrainingTimeMonitor:
        """Retorna o callback para ser usado no Trainer."""
        return self.training_monitor
        
    def start_energy_monitoring(self):
        """Inicia apenas o monitoramento de energia."""
        if self.energy_monitor:
            self.energy_monitor.start_monitoring()
            
    def stop_monitoring(self) -> tuple:
        """
        Para o monitoramento e retorna (energia_consumida, tempo_treinamento_efetivo).
        
        Returns:
            tuple: (energia_consumida_watts, tempo_treinamento_segundos)
        """
        # Obter tempo de treinamento efetivo
        training_time = self.training_monitor.get_total_training_time()
        
        # Obter energia consumida
        energy_consumption = 0.0
        if self.energy_monitor:
            energy_consumption, _ = self.energy_monitor.stop_monitoring()
            # Ignorar o tempo do energy_monitor, usar apenas nossa medição precisa
            
        return energy_consumption, training_time
        
    def get_detailed_metrics(self) -> dict:
        """Retorna métricas detalhadas do treinamento."""
        return {
            'total_training_time': self.training_monitor.get_total_training_time(),
            'epoch_times': self.training_monitor.get_epoch_times(),
            'average_epoch_time': self.training_monitor.get_average_epoch_time(),
            'num_epochs': len(self.training_monitor.get_epoch_times())
        }