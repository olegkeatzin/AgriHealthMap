"""
ConvLSTM модель для предсказания NDVI карт (УЛУЧШЕННАЯ ВЕРСИЯ)
Обученная модель из train_models/ndvi_prediction_training.ipynb

УЛУЧШЕНИЯ:
- LSTM для анализа временной динамики погодных данных (вместо простого Linear)
- Обработка всей последовательности погоды (5 шагов), а не только последнего
- Gradient clipping для стабильности обучения
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# =============================================================================
# КОНФИГУРАЦИЯ МОДЕЛИ
# =============================================================================
class NDVIConvLSTMConfig:
    """Конфигурация для ConvLSTM NDVI модели"""
    # Размеры данных
    MAP_HEIGHT = 500
    MAP_WIDTH = 503
    MAP_CHANNELS = 1  # NDVI - один канал
    INPUT_TIMESTEPS = 5  # Количество входных временных точек

    # Признаки
    WEATHER_FEATURES = 15
    TOPO_FEATURES = 10

    # Архитектура
    HIDDEN_DIM = 64
    KERNEL_SIZE = 3

    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# АРХИТЕКТУРА МОДЕЛИ
# =============================================================================
class ConvLSTMCell(nn.Module):
    """ConvLSTM ячейка для обработки пространственно-временных данных"""
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class NDVIPredictionModel(nn.Module):
    """
    Модель для предсказания NDVI с использованием ConvLSTM и дополнительных признаков
    """
    def __init__(self, config):
        super(NDVIPredictionModel, self).__init__()

        self.config = config
        self.hidden_dim = config.HIDDEN_DIM

        # ConvLSTM для обработки последовательности карт NDVI
        self.convlstm_cell = ConvLSTMCell(
            input_dim=config.MAP_CHANNELS,
            hidden_dim=self.hidden_dim,
            kernel_size=config.KERNEL_SIZE
        )

        # Обработка погодных данных - УЛУЧШЕННАЯ ВЕРСИЯ С LSTM
        # LSTM анализирует временную динамику погодных условий
        self.weather_encoder = nn.LSTM(
            input_size=config.WEATHER_FEATURES,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # Обработка топографических данных
        self.topo_encoder = nn.Sequential(
            nn.Linear(config.TOPO_FEATURES, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # Объединение признаков
        self.feature_fusion = nn.Sequential(
            nn.Linear(32 + 16, 64),  # weather (32) + topo (16)
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Декодер для генерации карты NDVI
        self.decoder = nn.Sequential(
            nn.Conv2d(self.hidden_dim + 1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Tanh()  # NDVI в диапазоне [-1, 1]
        )

    def forward(self, x_maps, x_weather, topo_features):
        """
        Args:
            x_maps: (batch, timesteps, channels, height, width)
            x_weather: (batch, timesteps, weather_features)
            topo_features: (batch, topo_features)
        """
        batch_size, seq_len, C, H, W = x_maps.size()

        # Инициализация скрытых состояний ConvLSTM
        h = torch.zeros(batch_size, self.hidden_dim, H, W).to(x_maps.device)
        c = torch.zeros(batch_size, self.hidden_dim, H, W).to(x_maps.device)

        # Обработка последовательности через ConvLSTM
        for t in range(seq_len):
            h, c = self.convlstm_cell(x_maps[:, t], (h, c))

        # Обработка погодных данных - УЛУЧШЕННАЯ ВЕРСИЯ
        # LSTM обрабатывает всю временную последовательность погоды
        _, (weather_hidden_state, _) = self.weather_encoder(x_weather)
        weather_encoded = weather_hidden_state[-1]  # Берем скрытое состояние последнего слоя (batch, 32)

        # Обработка топографических данных
        topo_encoded = self.topo_encoder(topo_features)  # (batch, 16)

        # Объединение признаков
        combined_features = torch.cat([weather_encoded, topo_encoded], dim=1)
        fused_features = self.feature_fusion(combined_features)  # (batch, 64)

        # Преобразуем fused_features в пространственную форму
        feature_map = fused_features.unsqueeze(-1).unsqueeze(-1)  # (batch, 64, 1, 1)
        feature_map = feature_map.expand(-1, -1, H, W)  # (batch, 64, H, W)
        feature_map_pooled = torch.mean(feature_map, dim=1, keepdim=True)  # (batch, 1, H, W)

        # Объединяем с выходом ConvLSTM
        combined = torch.cat([h, feature_map_pooled], dim=1)  # (batch, hidden_dim+1, H, W)

        # Генерация предсказанной карты NDVI
        output = self.decoder(combined)  # (batch, 1, H, W)

        return output


# =============================================================================
# ОБЕРТКА ДЛЯ ИСПОЛЬЗОВАНИЯ МОДЕЛИ
# =============================================================================
class NDVIConvLSTMPredictor:
    """
    Класс для загрузки и использования обученной ConvLSTM модели
    """
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Args:
            model_path: Путь к файлу .pth с весами модели
            device: 'cpu' или 'cuda'
        """
        self.device = device
        self.config = NDVIConvLSTMConfig()
        self.config.DEVICE = device

        # Создаем модель
        self.model = NDVIPredictionModel(self.config).to(self.device)

        # Загружаем веса
        if model_path:
            self._load_weights(model_path)

        self.model.eval()
        print(f"[OK] NDVIConvLSTMPredictor initialized on {device}")

    def _load_weights(self, model_path: str):
        """Загрузка весов модели"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"[OK] Model weights loaded from {model_path}")
            else:
                self.model.load_state_dict(checkpoint)
                print(f"[OK] Model weights loaded from {model_path}")
        except Exception as e:
            print(f"[WARNING] Failed to load weights: {e}")
            print(f"[WARNING] Using random initialization")

    def predict(
        self,
        ndvi_sequence: np.ndarray,
        weather_sequence: np.ndarray,
        topo_features: np.ndarray,
        return_numpy: bool = True
    ) -> Dict:
        """
        Предсказание будущей карты NDVI

        Args:
            ndvi_sequence: (timesteps, height, width) - последовательность NDVI карт
            weather_sequence: (timesteps, weather_features) - погодные данные
            topo_features: (topo_features,) - топографические характеристики
            return_numpy: вернуть результат как numpy array

        Returns:
            dict с ключами:
                - predicted_ndvi_map: предсказанная карта NDVI
                - statistics: статистика NDVI
                - health_classification: классификация здоровья растительности
        """
        with torch.no_grad():
            # Преобразуем в тензоры
            # x_maps: (1, timesteps, 1, height, width)
            x_maps = torch.FloatTensor(ndvi_sequence).unsqueeze(0).unsqueeze(2).to(self.device)

            # x_weather: (1, timesteps, weather_features)
            x_weather = torch.FloatTensor(weather_sequence).unsqueeze(0).to(self.device)

            # topo: (1, topo_features)
            topo = torch.FloatTensor(topo_features).unsqueeze(0).to(self.device)

            # Предсказание
            output = self.model(x_maps, x_weather, topo)  # (1, 1, H, W)

            # Извлекаем результат
            predicted_ndvi = output.squeeze().cpu().numpy()

            # Статистика
            stats = {
                'mean_ndvi': float(np.nanmean(predicted_ndvi)),
                'std_ndvi': float(np.nanstd(predicted_ndvi)),
                'min_ndvi': float(np.nanmin(predicted_ndvi)),
                'max_ndvi': float(np.nanmax(predicted_ndvi)),
                'median_ndvi': float(np.nanmedian(predicted_ndvi))
            }

            # Классификация здоровья растительности
            health_classification = self._classify_vegetation_health(predicted_ndvi)

            return {
                'predicted_ndvi_map': predicted_ndvi if return_numpy else output,
                'statistics': stats,
                'health_classification': health_classification
            }

    def _classify_vegetation_health(self, ndvi: np.ndarray) -> Dict:
        """Классификация здоровья растительности по NDVI"""
        # Пороги классификации
        thresholds = {
            'no_vegetation': (-1.0, 0.1),
            'unhealthy': (0.1, 0.3),
            'moderate': (0.3, 0.5),
            'healthy': (0.5, 0.7),
            'very_healthy': (0.7, 1.0)
        }

        classification = {}
        total_pixels = ndvi.size

        for category, (low, high) in thresholds.items():
            mask = (ndvi >= low) & (ndvi < high)
            count = np.sum(mask)
            percentage = (count / total_pixels) * 100
            classification[category] = {
                'pixels': int(count),
                'percentage': round(percentage, 2)
            }

        # Доминирующая категория
        dominant = max(classification.items(), key=lambda x: x[1]['percentage'])
        classification['dominant_class'] = dominant[0]

        return classification


# =============================================================================
# УТИЛИТЫ ДЛЯ ВИЗУАЛИЗАЦИИ
# =============================================================================
def visualize_ndvi_prediction(
    predicted_ndvi: np.ndarray,
    current_ndvi: Optional[np.ndarray] = None,
    colormap: str = 'RdYlGn',
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Визуализация предсказанной карты NDVI

    Args:
        predicted_ndvi: предсказанная карта NDVI
        current_ndvi: текущая карта NDVI (опционально)
        colormap: цветовая карта
        save_path: путь для сохранения (опционально)

    Returns:
        RGB изображение
    """
    if current_ndvi is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Текущий NDVI
        im1 = axes[0].imshow(current_ndvi, cmap=colormap, vmin=-1, vmax=1)
        axes[0].set_title('Current NDVI')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046)

        # Предсказанный NDVI
        im2 = axes[1].imshow(predicted_ndvi, cmap=colormap, vmin=-1, vmax=1)
        axes[1].set_title('Predicted NDVI')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046)

        # Разница
        difference = predicted_ndvi - current_ndvi
        im3 = axes[2].imshow(difference, cmap='RdYlGn', vmin=-0.3, vmax=0.3)
        axes[2].set_title('Difference (Predicted - Current)')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046)

    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        im = ax.imshow(predicted_ndvi, cmap=colormap, vmin=-1, vmax=1)
        ax.set_title('Predicted NDVI')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    # Конвертируем в RGB
    cmap = plt.get_cmap(colormap)
    normalized = (predicted_ndvi - (-1)) / (1 - (-1))
    rgb = (cmap(normalized)[:, :, :3] * 255).astype(np.uint8)

    plt.close()

    return rgb


def create_ndvi_colormap() -> LinearSegmentedColormap:
    """Создание кастомной цветовой карты для NDVI"""
    colors = [
        (0.0, '#8B4513'),   # Коричневый (нет растительности)
        (0.1, '#D2691E'),   # Светло-коричневый
        (0.2, '#FFD700'),   # Золотой
        (0.3, '#FFFF00'),   # Желтый (слабая растительность)
        (0.4, '#ADFF2F'),   # Желто-зеленый
        (0.5, '#7FFF00'),   # Светло-зеленый (умеренная)
        (0.6, '#00FF00'),   # Зеленый (здоровая)
        (0.7, '#00CC00'),   # Темно-зеленый
        (0.8, '#009900'),   # Очень темно-зеленый (очень здоровая)
        (1.0, '#006400')    # Темный лесной зеленый
    ]

    return LinearSegmentedColormap.from_list('ndvi_custom', colors)


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================
def safe_float(value, default=0.0):
    """Безопасное преобразование в float"""
    try:
        return float(value)
    except (ValueError, TypeError):
        if isinstance(value, str):
            import re
            match = re.search(r'\d+\.?\d*', value)
            if match:
                return float(match.group())
        return default


if __name__ == "__main__":
    # Пример использования
    print("NDVIConvLSTMPredictor - Модель для предсказания NDVI")
    print("="*60)

    config = NDVIConvLSTMConfig()
    print(f"Config:")
    print(f"  MAP_SIZE: {config.MAP_HEIGHT}x{config.MAP_WIDTH}")
    print(f"  INPUT_TIMESTEPS: {config.INPUT_TIMESTEPS}")
    print(f"  HIDDEN_DIM: {config.HIDDEN_DIM}")
    print(f"  WEATHER_FEATURES: {config.WEATHER_FEATURES}")
    print(f"  TOPO_FEATURES: {config.TOPO_FEATURES}")
