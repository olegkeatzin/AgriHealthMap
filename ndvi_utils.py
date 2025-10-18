"""
Утилиты для работы с NDVI (расчет и визуализация)
"""

import numpy as np
import matplotlib.cm as cm
from typing import Dict


def calculate_ndvi_from_bands(sentinel_bands: Dict[str, np.ndarray]) -> np.ndarray:
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


def classify_vegetation_health(ndvi: np.ndarray) -> Dict[str, float]:
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


def visualize_ndvi(ndvi: np.ndarray, colormap: str = 'RdYlGn', vmin: float = -1, vmax: float = 1) -> np.ndarray:
    """
    Визуализация NDVI с цветовой картой

    Args:
        ndvi: NDVI карта
        colormap: Название colormap (RdYlGn, viridis, RdBu, etc.)
        vmin: Минимальное значение для нормализации
        vmax: Максимальное значение для нормализации

    Returns:
        RGB изображение [H, W, 3]
    """
    # Нормализуем NDVI к диапазону [0, 1]
    ndvi_normalized = (ndvi - vmin) / (vmax - vmin)
    ndvi_normalized = np.clip(ndvi_normalized, 0, 1)

    # Применяем colormap
    cmap = cm.get_cmap(colormap)
    colored = cmap(ndvi_normalized)

    # Конвертируем в RGB uint8
    rgb_image = (colored[:, :, :3] * 255).astype(np.uint8)

    return rgb_image
