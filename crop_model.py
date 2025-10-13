"""
Модуль для инференса модели детекции сельскохозяйственных культур
Интеграция с AgriHealthMap
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import cv2


# === ОПРЕДЕЛЕНИЕ МОДЕЛИ ===

class AttentionBlock(nn.Module):
    """Attention mechanism для U-Net"""
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        att = self.attention(x)
        return x * att


class AttentionUNet(nn.Module):
    """Attention U-Net для сегментации культур"""

    def __init__(self, in_channels=10, num_classes=6):
        super().__init__()

        # Encoder (5 уровней)
        self.enc1 = self._make_layer(in_channels, 64)
        self.enc2 = self._make_layer(64, 128)
        self.enc3 = self._make_layer(128, 256)
        self.enc4 = self._make_layer(256, 512)
        self.enc5 = self._make_layer(512, 1024)

        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = self._make_layer(1024, 2048)
        self.attention_bottleneck = AttentionBlock(2048)

        # Decoder с attention
        self.upconv5 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.attention5 = AttentionBlock(1024)
        self.dec5 = self._make_layer(2048, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.attention4 = AttentionBlock(512)
        self.dec4 = self._make_layer(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.attention3 = AttentionBlock(256)
        self.dec3 = self._make_layer(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.attention2 = AttentionBlock(128)
        self.dec2 = self._make_layer(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.attention1 = AttentionBlock(64)
        self.dec1 = self._make_layer(128, 64)

        # Final classifier
        self.final = nn.Conv2d(64, num_classes, 1)

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc5))
        bottleneck = self.attention_bottleneck(bottleneck)

        # Decoder
        dec5 = self.upconv5(bottleneck)
        dec5 = self.attention5(dec5)
        dec5 = torch.cat([dec5, enc5], dim=1)
        dec5 = self.dec5(dec5)

        dec4 = self.upconv4(dec5)
        dec4 = self.attention4(dec4)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = self.attention3(dec3)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = self.attention2(dec2)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = self.attention1(dec1)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        return self.final(dec1)


# === INFERENCE CLASS ===

class CropDetectionModel:
    """Wrapper для модели детекции культур"""

    # Классы культур
    CLASS_NAMES = [
        'background',      # 0 - фон
        'wheat',          # 1 - пшеница
        'corn',           # 2 - кукуруза
        'sunflower',      # 3 - подсолнечник
        'soybean',        # 4 - соя
        'other_crops'     # 5 - другие культуры
    ]

    CLASS_COLORS = [
        [0, 0, 0],        # background - черный
        [255, 215, 0],    # wheat - золотой
        [255, 255, 0],    # corn - желтый
        [255, 140, 0],    # sunflower - оранжевый
        [0, 255, 0],      # soybean - зеленый
        [144, 238, 144]   # other_crops - светло-зеленый
    ]

    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        """
        Инициализация модели

        Args:
            model_path: Путь к файлу модели (.pth)
            device: 'cpu' или 'cuda'
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')

        # Загружаем веса если указан путь
        if model_path and Path(model_path).exists():
            try:
                # Пытаемся загрузить как TorchScript модель
                self.model = torch.jit.load(model_path, map_location=self.device)
                print(f"TorchScript model loaded from {model_path}")
            except Exception as e:
                # Если не TorchScript, загружаем как обычный checkpoint
                print(f"Not a TorchScript model, loading as checkpoint: {e}")
                self.model = AttentionUNet(in_channels=10, num_classes=6)
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                print(f"Model loaded from {model_path}")
        else:
            print("Warning: No model weights loaded. Using random initialization.")
            self.model = AttentionUNet(in_channels=10, num_classes=6)

        self.model.to(self.device)
        self.model.eval()

    def preprocess_sentinel_data(self, sentinel_bands: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Препроцессинг данных Sentinel-2

        Args:
            sentinel_bands: Словарь с 10 каналами Sentinel-2
                            Ожидаемые ключи: B01-B12, B8A

        Returns:
            Тензор PyTorch [1, 10, H, W]
        """
        # Порядок каналов для модели (10 каналов)
        band_order = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']

        # Собираем каналы в правильном порядке
        image_stack = []
        for band_name in band_order:
            if band_name in sentinel_bands:
                band = sentinel_bands[band_name]
                # Если 3D массив, берем первый канал
                if band.ndim == 3:
                    band = band[:, :, 0]
                image_stack.append(band)
            else:
                # Если канал отсутствует, создаем нулевой
                h, w = list(sentinel_bands.values())[0].shape[:2]
                image_stack.append(np.zeros((h, w), dtype=np.float32))

        # Стекуем каналы: [H, W, 10]
        image = np.stack(image_stack, axis=2)

        # Нормализация по каналам
        image_norm = self._normalize_image(image)

        # Конвертируем в тензор PyTorch: [1, 10, H, W]
        image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0).float()

        return image_tensor.to(self.device)

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Нормализация спектральных каналов (0-1)"""
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        normalized = np.zeros_like(image, dtype=np.float32)
        for i in range(image.shape[2]):
            channel = image[:, :, i]
            p2, p98 = np.percentile(channel, (2, 98))
            if p98 > p2:
                normalized[:, :, i] = np.clip((channel - p2) / (p98 - p2), 0, 1)

        return normalized

    @torch.no_grad()
    def predict(self, sentinel_bands: Dict[str, np.ndarray],
                return_probabilities: bool = False) -> Dict:
        """
        Предсказание типов культур

        Args:
            sentinel_bands: Словарь с каналами Sentinel-2
            return_probabilities: Возвращать ли вероятности классов

        Returns:
            Словарь с результатами:
                - mask: Маска классов [H, W]
                - colored_mask: Цветная маска [H, W, 3]
                - class_distribution: Распределение классов
                - probabilities (опционально): Вероятности [H, W, num_classes]
        """
        # Препроцессинг
        image_tensor = self.preprocess_sentinel_data(sentinel_bands)

        # Инференс
        output = self.model(image_tensor)

        # Получаем предсказания
        probabilities = F.softmax(output, dim=1)[0]  # [num_classes, H, W]
        predicted_mask = output.argmax(dim=1)[0].cpu().numpy()  # [H, W]

        # Цветная маска
        colored_mask = self._create_colored_mask(predicted_mask)

        # Распределение классов
        class_distribution = self._calculate_distribution(predicted_mask)

        result = {
            'mask': predicted_mask.astype(np.uint8),
            'colored_mask': colored_mask,
            'class_distribution': class_distribution,
            'class_names': self.CLASS_NAMES
        }

        if return_probabilities:
            result['probabilities'] = probabilities.cpu().numpy().transpose(1, 2, 0)

        return result

    def _create_colored_mask(self, mask: np.ndarray) -> np.ndarray:
        """Создание цветной маски"""
        colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id, color in enumerate(self.CLASS_COLORS):
            colored[mask == class_id] = color
        return colored

    def _calculate_distribution(self, mask: np.ndarray) -> Dict[str, float]:
        """Подсчет распределения классов"""
        unique, counts = np.unique(mask, return_counts=True)
        total = mask.size

        distribution = {}
        for class_id, class_name in enumerate(self.CLASS_NAMES):
            if class_id in unique:
                idx = np.where(unique == class_id)[0][0]
                percentage = (counts[idx] / total) * 100
            else:
                percentage = 0.0
            distribution[class_name] = round(percentage, 2)

        return distribution

    def predict_from_multiband_array(self, multiband_image: np.ndarray) -> Dict:
        """
        Предсказание из мультиспектрального массива [H, W, 10]

        Args:
            multiband_image: Numpy массив с 10 каналами

        Returns:
            Словарь с результатами
        """
        # Нормализация
        image_norm = self._normalize_image(multiband_image)

        # Конвертируем в тензор
        image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0).float()
        image_tensor = image_tensor.to(self.device)

        # Инференс
        with torch.no_grad():
            output = self.model(image_tensor)
            predicted_mask = output.argmax(dim=1)[0].cpu().numpy()

        # Формируем результат
        colored_mask = self._create_colored_mask(predicted_mask)
        class_distribution = self._calculate_distribution(predicted_mask)

        return {
            'mask': predicted_mask.astype(np.uint8),
            'colored_mask': colored_mask,
            'class_distribution': class_distribution,
            'class_names': self.CLASS_NAMES
        }


# === UTILITY FUNCTIONS ===

def visualize_prediction(rgb_image: np.ndarray,
                         prediction: Dict,
                         alpha: float = 0.5) -> np.ndarray:
    """
    Создание overlay визуализации

    Args:
        rgb_image: RGB изображение [H, W, 3]
        prediction: Результат предсказания
        alpha: Прозрачность маски (0-1)

    Returns:
        Overlay изображение [H, W, 3]
    """
    # Нормализуем RGB если нужно
    if rgb_image.max() <= 1.0:
        rgb_image = (rgb_image * 255).astype(np.uint8)

    colored_mask = prediction['colored_mask']

    # Resize mask если размеры не совпадают
    if rgb_image.shape[:2] != colored_mask.shape[:2]:
        colored_mask = cv2.resize(colored_mask,
                                  (rgb_image.shape[1], rgb_image.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

    # Создаем overlay
    overlay = cv2.addWeighted(rgb_image, 1 - alpha, colored_mask, alpha, 0)

    return overlay
