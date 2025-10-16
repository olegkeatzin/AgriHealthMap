import torch
import torch.nn as nn

class TabularNDVIPredictor(nn.Module):
    """
    Модель для предсказания NDVI на основе табличных временных рядов.
    Использует двунаправленную LSTM для анализа последовательности векторов признаков.
    """
    def __init__(self, input_features: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3):
        """
        Args:
            input_features (int): Количество признаков в каждом временном шаге (количество колонок).
            hidden_dim (int): Размер скрытого состояния LSTM.
            num_layers (int): Количество слоев LSTM.
            dropout (float): Вероятность dropout для регуляризации.
        """
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # Входной формат: (batch, seq_len, features)
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True # Двунаправленность позволяет смотреть на данные "вперед" и "назад" во времени
        )

        # Полносвязная сеть для финального предсказания
        self.regressor_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), # *2, так как LSTM двунаправленная
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Выход - одно число (предсказанный NDVI)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x имеет форму: [batch_size, time_steps, num_features]
        
        # Прогоняем через LSTM
        # lstm_out содержит выходы для каждого временного шага
        lstm_out, _ = self.lstm(x)
        
        # Нас интересует только выход последнего временного шага для предсказания будущего
        last_time_step_out = lstm_out[:, -1, :]
        
        # Прогоняем через полносвязные слои для получения итогового предсказания
        prediction = self.regressor_head(last_time_step_out)
        
        return prediction