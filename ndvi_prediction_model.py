"""
Модуль для предсказания динамики NDVI
Интеграция с AgriHealthMap
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# === ОПРЕДЕЛЕНИЕ МОДЕЛИ ===

class LSTMPredictor(nn.Module):
    """LSTM модель для прогнозирования NDVI."""

    def __init__(self, input_size: int = 2, hidden_size: int = 64, num_layers: int = 2,
                 forecast_horizon: int = 2, dropout: float = 0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, forecast_horizon)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Используем последний скрытый слой
        last_hidden = h_n[-1]  # (batch, hidden_size)

        # Прогноз
        output = self.fc(last_hidden)  # (batch, forecast_horizon)
        return output


class TransformerPredictor(nn.Module):
    """Transformer модель для прогнозирования NDVI."""

    def __init__(self, input_size: int = 2, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, forecast_horizon: int = 2, dropout: float = 0.2):
        super().__init__()

        self.input_projection = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, forecast_horizon)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        x = self.transformer(x)  # (batch, seq_len, d_model)

        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)

        # Прогноз
        output = self.fc(x)  # (batch, forecast_horizon)
        return output


# === INFERENCE CLASS ===

class NDVIPredictionModel:
    """Wrapper для модели предсказания NDVI"""

    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu',
                 model_type: str = 'lstm', sequence_length: int = 5,
                 forecast_horizon: int = 2):
        """
        Инициализация модели предсказания NDVI

        Args:
            model_path: Путь к файлу модели (.pth)
            device: 'cpu' или 'cuda'
            model_type: 'lstm' или 'transformer'
            sequence_length: Длина входной последовательности
            forecast_horizon: Горизонт прогноза
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon

        # Создаем модель
        if model_type == "lstm":
            self.model = LSTMPredictor(
                input_size=2,
                hidden_size=64,
                num_layers=2,
                forecast_horizon=forecast_horizon,
                dropout=0.2
            )
        elif model_type == "transformer":
            self.model = TransformerPredictor(
                input_size=2,
                d_model=64,
                nhead=4,
                num_layers=2,
                forecast_horizon=forecast_horizon,
                dropout=0.2
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Загружаем веса если указан путь
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"NDVI prediction model loaded from {model_path}")
        else:
            print("Warning: No model weights loaded for NDVI. Using random initialization.")

        self.model.to(self.device)
        self.model.eval()

    def calculate_ndvi_from_bands(self, sentinel_bands: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Вычисление NDVI из каналов Sentinel-2

        Args:
            sentinel_bands: Словарь с каналами B04 (Red) и B08 (NIR)

        Returns:
            NDVI карта [H, W]
        """
        if 'B04' not in sentinel_bands or 'B08' not in sentinel_bands:
            raise ValueError("Требуются каналы B04 (Red) и B08 (NIR) для расчета NDVI")

        red = sentinel_bands['B04'].astype(np.float32)
        nir = sentinel_bands['B08'].astype(np.float32)

        # Если 3D массив, берем первый канал
        if red.ndim == 3:
            red = red[:, :, 0]
        if nir.ndim == 3:
            nir = nir[:, :, 0]

        # NDVI = (NIR - Red) / (NIR + Red)
        denominator = nir + red
        ndvi = np.where(denominator != 0, (nir - red) / denominator, 0)

        # Ограничиваем значения [-1, 1]
        ndvi = np.clip(ndvi, -1, 1)

        return ndvi

    @torch.no_grad()
    def predict_ndvi_timeseries(self, ndvi_sequence: np.ndarray) -> Dict:
        """
        Предсказание будущих значений NDVI на основе временного ряда

        Args:
            ndvi_sequence: Массив временного ряда NDVI [seq_len, H, W] или [seq_len]

        Returns:
            Словарь с результатами предсказания
        """
        # Проверяем размерность
        if ndvi_sequence.ndim == 3:
            # Пространственный NDVI - вычисляем статистики
            seq_len, h, w = ndvi_sequence.shape

            # Вычисляем mean и std для каждого временного шага
            sequence_stats = []
            for t in range(seq_len):
                ndvi_t = ndvi_sequence[t]
                mean_val = np.nanmean(ndvi_t)
                std_val = np.nanstd(ndvi_t)
                sequence_stats.append([mean_val, std_val])

            input_sequence = np.array(sequence_stats)  # [seq_len, 2]
        elif ndvi_sequence.ndim == 2:
            # Уже в формате [seq_len, features]
            input_sequence = ndvi_sequence
        elif ndvi_sequence.ndim == 1:
            # Одномерный временной ряд - добавляем std
            input_sequence = np.stack([ndvi_sequence, np.zeros_like(ndvi_sequence)], axis=1)
        else:
            raise ValueError(f"Unexpected ndvi_sequence shape: {ndvi_sequence.shape}")

        # Проверяем длину последовательности
        if len(input_sequence) < self.sequence_length:
            # Дополняем нулями если последовательность короткая
            padding = np.zeros((self.sequence_length - len(input_sequence), 2))
            input_sequence = np.vstack([padding, input_sequence])
        elif len(input_sequence) > self.sequence_length:
            # Берем последние sequence_length шагов
            input_sequence = input_sequence[-self.sequence_length:]

        # Конвертируем в тензор
        input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(self.device)

        # Предсказание
        prediction = self.model(input_tensor)
        predicted_values = prediction[0].cpu().numpy()

        return {
            'predicted_ndvi': predicted_values,
            'forecast_horizon': self.forecast_horizon,
            'input_sequence': input_sequence
        }

    def _classify_vegetation_health(self, ndvi: np.ndarray) -> Dict[str, float]:
        """
        Классификация здоровья растительности на основе NDVI

        NDVI диапазоны:
        - < 0: Вода, снег, облака
        - 0 - 0.2: Голая почва, скалы
        - 0.2 - 0.4: Редкая растительность
        - 0.4 - 0.6: Умеренная растительность
        - 0.6 - 0.8: Здоровая растительность
        - > 0.8: Очень густая растительность

        Args:
            ndvi: NDVI карта

        Returns:
            Словарь с процентами площади для каждого класса
        """
        total_pixels = ndvi.size

        classification = {
            'water_snow': float(np.sum(ndvi < 0) / total_pixels * 100),
            'bare_soil': float(np.sum((ndvi >= 0) & (ndvi < 0.2)) / total_pixels * 100),
            'sparse_vegetation': float(np.sum((ndvi >= 0.2) & (ndvi < 0.4)) / total_pixels * 100),
            'moderate_vegetation': float(np.sum((ndvi >= 0.4) & (ndvi < 0.6)) / total_pixels * 100),
            'healthy_vegetation': float(np.sum((ndvi >= 0.6) & (ndvi < 0.8)) / total_pixels * 100),
            'very_dense_vegetation': float(np.sum(ndvi >= 0.8) / total_pixels * 100)
        }

        return classification


# === UTILITY FUNCTIONS ===

def visualize_ndvi(ndvi: np.ndarray, colormap: str = 'RdYlGn') -> np.ndarray:
    """
    Визуализация NDVI с цветовой картой

    Args:
        ndvi: NDVI карта [-1, 1]
        colormap: Название colormap (RdYlGn, viridis, etc.)

    Returns:
        RGB изображение [H, W, 3]
    """
    # Нормализуем NDVI к диапазону [0, 1]
    ndvi_normalized = (ndvi + 1) / 2.0
    ndvi_normalized = np.clip(ndvi_normalized, 0, 1)

    # Применяем colormap
    cmap = cm.get_cmap(colormap)
    colored = cmap(ndvi_normalized)

    # Конвертируем в RGB uint8
    rgb_image = (colored[:, :, :3] * 255).astype(np.uint8)

    return rgb_image


def visualize_ndvi_change(ndvi_current: np.ndarray,
                          ndvi_previous: np.ndarray,
                          colormap: str = 'RdBu') -> np.ndarray:
    """
    Визуализация изменения NDVI между двумя временными точками

    Args:
        ndvi_current: Текущий NDVI
        ndvi_previous: Предыдущий NDVI
        colormap: Colormap для визуализации

    Returns:
        RGB изображение изменений [H, W, 3]
    """
    # Вычисляем разницу
    ndvi_diff = ndvi_current - ndvi_previous

    # Нормализуем к [-1, 1] -> [0, 1]
    ndvi_diff_normalized = (ndvi_diff + 1) / 2.0
    ndvi_diff_normalized = np.clip(ndvi_diff_normalized, 0, 1)

    # Применяем colormap
    cmap = cm.get_cmap(colormap)
    colored = cmap(ndvi_diff_normalized)

    # Конвертируем в RGB uint8
    rgb_image = (colored[:, :, :3] * 255).astype(np.uint8)

    return rgb_image


def create_ndvi_overlay(rgb_image: np.ndarray,
                        ndvi: np.ndarray,
                        alpha: float = 0.5,
                        colormap: str = 'RdYlGn') -> np.ndarray:
    """
    Создание overlay NDVI поверх RGB изображения

    Args:
        rgb_image: RGB изображение [H, W, 3]
        ndvi: NDVI карта
        alpha: Прозрачность NDVI (0-1)
        colormap: Colormap для NDVI

    Returns:
        Overlay изображение [H, W, 3]
    """
    # Нормализуем RGB если нужно
    if rgb_image.max() <= 1.0:
        rgb_image = (rgb_image * 255).astype(np.uint8)

    # Визуализируем NDVI
    ndvi_colored = visualize_ndvi(ndvi, colormap)

    # Resize NDVI если размеры не совпадают
    if rgb_image.shape[:2] != ndvi_colored.shape[:2]:
        ndvi_colored = cv2.resize(ndvi_colored,
                                  (rgb_image.shape[1], rgb_image.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)

    # Создаем overlay
    overlay = cv2.addWeighted(rgb_image, 1 - alpha, ndvi_colored, alpha, 0)

    return overlay
