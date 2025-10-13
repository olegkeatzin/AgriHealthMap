"""
Модуль для предсказания NDVI (Normalized Difference Vegetation Index)
и временного анализа растительности
Интеграция с AgriHealthMap
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2


# === ОПРЕДЕЛЕНИЕ МОДЕЛИ ===

class NDVIPredictor(nn.Module):
    """
    Модель для предсказания будущих значений NDVI
    на основе временных рядов спутниковых данных
    """

    def __init__(self, in_channels=10, hidden_dim=128, num_layers=2):
        super().__init__()

        # Encoder для извлечения пространственных признаков
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # LSTM для временной зависимости
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Decoder для восстановления пространственного разрешения
        self.spatial_decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 2, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),  # Выход - одноканальный NDVI
            nn.Tanh()  # NDVI в диапазоне [-1, 1]
        )

    def forward(self, x):
        """
        Args:
            x: [batch, time, channels, H, W]
        Returns:
            ndvi_pred: [batch, 1, H, W] - предсказанный NDVI
        """
        batch, time_steps, C, H, W = x.shape

        # Извлекаем пространственные признаки для каждого временного шага
        spatial_features = []
        for t in range(time_steps):
            feat = self.spatial_encoder(x[:, t])  # [batch, 128, H/2, W/2]
            spatial_features.append(feat)

        # Стекуем временные признаки
        # [batch, time, 128, H/2, W/2]
        spatial_features = torch.stack(spatial_features, dim=1)

        # Reshape для LSTM [batch, time, features]
        B, T, C_feat, H_feat, W_feat = spatial_features.shape
        spatial_features = spatial_features.view(B, T, -1)

        # LSTM
        lstm_out, _ = self.lstm(spatial_features)  # [batch, time, hidden*2]

        # Берем последний временной шаг
        last_hidden = lstm_out[:, -1, :]  # [batch, hidden*2]

        # Reshape обратно в пространственные размеры
        last_hidden = last_hidden.view(B, -1, H_feat, W_feat)

        # Декодер для восстановления разрешения
        ndvi_pred = self.spatial_decoder(last_hidden)  # [batch, 1, H, W]

        return ndvi_pred


# === INFERENCE CLASS ===

class NDVIPredictionModel:
    """Wrapper для модели предсказания NDVI"""

    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        """
        Инициализация модели предсказания NDVI

        Args:
            model_path: Путь к файлу модели (.pth)
            device: 'cpu' или 'cuda'
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')

        # Создаем модель
        self.model = NDVIPredictor(in_channels=10, hidden_dim=128, num_layers=2)

        # Загружаем веса если указан путь
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"NDVI prediction model loaded from {model_path}")
        else:
            print("Warning: No NDVI model weights loaded. Using random initialization.")

        self.model.to(self.device)
        self.model.eval()

    def calculate_ndvi_from_bands(self, sentinel_bands: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Вычисление NDVI из каналов Sentinel-2

        Args:
            sentinel_bands: Словарь с каналами (должен содержать B04 и B08)

        Returns:
            NDVI массив [H, W] в диапазоне [-1, 1]
        """
        # Извлекаем Red и NIR каналы
        red = sentinel_bands.get('B04', sentinel_bands.get('B04', None))
        nir = sentinel_bands.get('B08', sentinel_bands.get('B08', None))

        if red is None or nir is None:
            raise ValueError("Для вычисления NDVI требуются каналы B04 (Red) и B08 (NIR)")

        # Если 3D массив, берем первый канал
        if red.ndim == 3:
            red = red[:, :, 0]
        if nir.ndim == 3:
            nir = nir[:, :, 0]

        # Вычисляем NDVI: (NIR - Red) / (NIR + Red)
        epsilon = 1e-6
        ndvi = (nir - red) / (nir + red + epsilon)

        # Ограничиваем диапазон [-1, 1]
        ndvi = np.clip(ndvi, -1, 1)

        return ndvi.astype(np.float32)

    def predict_future_ndvi(self,
                           historical_data: List[Dict[str, np.ndarray]],
                           days_ahead: int = 7) -> Dict:
        """
        Предсказание NDVI на будущее на основе исторических данных

        Args:
            historical_data: Список словарей с Sentinel-2 каналами за разные даты
                            (минимум 3-5 временных точек)
            days_ahead: Количество дней вперед для предсказания

        Returns:
            Словарь с результатами предсказания
        """
        if len(historical_data) < 3:
            raise ValueError("Требуется минимум 3 временных точки для предсказания")

        # Препроцессинг исторических данных
        processed_images = []
        for data in historical_data:
            img = self._preprocess_sentinel_data(data)
            processed_images.append(img)

        # Стекуем в последовательность [1, time, channels, H, W]
        sequence = torch.stack(processed_images, dim=1)

        # Предсказание
        with torch.no_grad():
            predicted_ndvi = self.model(sequence)  # [1, 1, H, W]
            predicted_ndvi = predicted_ndvi.squeeze().cpu().numpy()  # [H, W]

        # Вычисляем текущий NDVI для сравнения
        current_ndvi = self.calculate_ndvi_from_bands(historical_data[-1])

        # Вычисляем изменение
        ndvi_change = predicted_ndvi - current_ndvi

        # Классификация здоровья растительности
        health_classification = self._classify_vegetation_health(predicted_ndvi)

        # Статистика
        statistics = self._calculate_ndvi_statistics(predicted_ndvi, current_ndvi, ndvi_change)

        return {
            'predicted_ndvi': predicted_ndvi,
            'current_ndvi': current_ndvi,
            'ndvi_change': ndvi_change,
            'health_classification': health_classification,
            'statistics': statistics,
            'days_ahead': days_ahead
        }

    def analyze_ndvi_timeseries(self,
                                historical_data: List[Dict[str, np.ndarray]],
                                dates: Optional[List[str]] = None) -> Dict:
        """
        Анализ временного ряда NDVI

        Args:
            historical_data: Список словарей с Sentinel-2 каналами
            dates: Список дат (опционально)

        Returns:
            Словарь с результатами анализа временного ряда
        """
        # Вычисляем NDVI для каждой временной точки
        ndvi_series = []
        for data in historical_data:
            ndvi = self.calculate_ndvi_from_bands(data)
            ndvi_series.append(ndvi)

        ndvi_array = np.stack(ndvi_series, axis=0)  # [time, H, W]

        # Тренд (линейная регрессия для каждого пикселя)
        trend = self._calculate_trend(ndvi_array)

        # Сезонность (вариация)
        seasonality = self._calculate_seasonality(ndvi_array)

        # Аномалии
        anomalies = self._detect_anomalies(ndvi_array)

        # Средние значения по времени
        mean_ndvi_per_timestep = [float(ndvi.mean()) for ndvi in ndvi_series]

        return {
            'ndvi_timeseries': ndvi_array,
            'mean_values': mean_ndvi_per_timestep,
            'trend': trend,
            'seasonality': seasonality,
            'anomalies': anomalies,
            'dates': dates if dates else [f"t{i}" for i in range(len(historical_data))]
        }

    def _preprocess_sentinel_data(self, sentinel_bands: Dict[str, np.ndarray]) -> torch.Tensor:
        """Препроцессинг Sentinel-2 данных"""
        band_order = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']

        image_stack = []
        for band_name in band_order:
            if band_name in sentinel_bands:
                band = sentinel_bands[band_name]
                if band.ndim == 3:
                    band = band[:, :, 0]
                image_stack.append(band)
            else:
                h, w = list(sentinel_bands.values())[0].shape[:2]
                image_stack.append(np.zeros((h, w), dtype=np.float32))

        image = np.stack(image_stack, axis=0)  # [10, H, W]

        # Нормализация
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        for i in range(image.shape[0]):
            channel = image[i]
            p2, p98 = np.percentile(channel, (2, 98))
            if p98 > p2:
                image[i] = np.clip((channel - p2) / (p98 - p2), 0, 1)

        return torch.from_numpy(image).float().to(self.device)

    def _classify_vegetation_health(self, ndvi: np.ndarray) -> Dict[str, float]:
        """
        Классификация здоровья растительности на основе NDVI

        NDVI ranges:
        < 0: Вода, снег, облака
        0-0.2: Голая почва, асфальт
        0.2-0.4: Разреженная растительность
        0.4-0.6: Умеренная растительность
        0.6-0.8: Здоровая растительность
        > 0.8: Очень здоровая/плотная растительность
        """
        total_pixels = ndvi.size

        health_classes = {
            'water_or_snow': (ndvi < 0).sum() / total_pixels * 100,
            'bare_soil': ((ndvi >= 0) & (ndvi < 0.2)).sum() / total_pixels * 100,
            'sparse_vegetation': ((ndvi >= 0.2) & (ndvi < 0.4)).sum() / total_pixels * 100,
            'moderate_vegetation': ((ndvi >= 0.4) & (ndvi < 0.6)).sum() / total_pixels * 100,
            'healthy_vegetation': ((ndvi >= 0.6) & (ndvi < 0.8)).sum() / total_pixels * 100,
            'very_healthy': (ndvi >= 0.8).sum() / total_pixels * 100
        }

        return {k: round(float(v), 2) for k, v in health_classes.items()}

    def _calculate_ndvi_statistics(self, predicted_ndvi, current_ndvi, change) -> Dict:
        """Вычисление статистики NDVI"""
        return {
            'predicted_mean': float(predicted_ndvi.mean()),
            'predicted_std': float(predicted_ndvi.std()),
            'predicted_min': float(predicted_ndvi.min()),
            'predicted_max': float(predicted_ndvi.max()),
            'current_mean': float(current_ndvi.mean()),
            'change_mean': float(change.mean()),
            'change_std': float(change.std()),
            'improvement_percentage': float((change > 0).sum() / change.size * 100),
            'degradation_percentage': float((change < 0).sum() / change.size * 100)
        }

    def _calculate_trend(self, ndvi_array: np.ndarray) -> np.ndarray:
        """Вычисление тренда (линейная регрессия по времени)"""
        time_steps = ndvi_array.shape[0]
        x = np.arange(time_steps)

        # Линейная регрессия для каждого пикселя
        x_mean = x.mean()
        trend = np.zeros_like(ndvi_array[0])

        for i in range(ndvi_array.shape[1]):
            for j in range(ndvi_array.shape[2]):
                y = ndvi_array[:, i, j]
                if not np.isnan(y).any():
                    slope = ((x - x_mean) * (y - y.mean())).sum() / ((x - x_mean) ** 2).sum()
                    trend[i, j] = slope

        return trend

    def _calculate_seasonality(self, ndvi_array: np.ndarray) -> float:
        """Вычисление сезонной вариации"""
        return float(ndvi_array.std(axis=0).mean())

    def _detect_anomalies(self, ndvi_array: np.ndarray, threshold: float = 2.0) -> np.ndarray:
        """Детекция аномалий (выбросов) во временном ряду"""
        mean = ndvi_array.mean(axis=0)
        std = ndvi_array.std(axis=0)

        # Аномалии - пиксели, где текущее значение отличается от среднего > threshold * std
        anomalies = np.abs(ndvi_array[-1] - mean) > (threshold * std)

        return anomalies.astype(np.uint8)


# === UTILITY FUNCTIONS ===

def visualize_ndvi(ndvi: np.ndarray, colormap: str = 'RdYlGn') -> np.ndarray:
    """
    Визуализация NDVI с цветовой картой

    Args:
        ndvi: NDVI массив [-1, 1]
        colormap: Название colormap ('RdYlGn', 'Greens', 'viridis')

    Returns:
        RGB изображение [H, W, 3]
    """
    import matplotlib.cm as cm

    # Нормализуем в [0, 1]
    ndvi_norm = (ndvi + 1) / 2
    ndvi_norm = np.clip(ndvi_norm, 0, 1)

    # Применяем colormap
    cmap = cm.get_cmap(colormap)
    colored = cmap(ndvi_norm)

    # Конвертируем в RGB uint8
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)

    return rgb


def visualize_ndvi_change(current_ndvi: np.ndarray,
                          predicted_ndvi: np.ndarray,
                          rgb_image: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Визуализация изменения NDVI

    Args:
        current_ndvi: Текущий NDVI
        predicted_ndvi: Предсказанный NDVI
        rgb_image: RGB изображение для overlay (опционально)

    Returns:
        Визуализация изменения
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    change = predicted_ndvi - current_ndvi

    # Используем diverging colormap (красный = ухудшение, зеленый = улучшение)
    cmap = cm.get_cmap('RdYlGn')

    # Нормализуем change в [0, 1]
    change_norm = (change + 1) / 2
    change_norm = np.clip(change_norm, 0, 1)

    colored = cmap(change_norm)
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)

    # Если есть RGB изображение, делаем overlay
    if rgb_image is not None:
        if rgb_image.max() <= 1.0:
            rgb_image = (rgb_image * 255).astype(np.uint8)

        if rgb_image.shape[:2] != rgb.shape[:2]:
            rgb = cv2.resize(rgb, (rgb_image.shape[1], rgb_image.shape[0]))

        rgb = cv2.addWeighted(rgb_image, 0.6, rgb, 0.4, 0)

    return rgb
