# AgriHealthMap API Documentation

## Обзор

AgriHealthMap - это FastAPI приложение для мониторинга сельскохозяйственных земель с использованием спутниковых данных Sentinel-2 и моделей машинного обучения для детекции типов культур.

## Установка

### 1. Установка зависимостей

```bash
# Используя uv (рекомендуется)
uv sync

# Или с помощью pip
pip install -e .
```

### 2. Настройка переменных окружения

Создайте файл `.env` в корне проекта:

```env
# Sentinel Hub API credentials (обязательно)
SH_CLIENT_ID=your_sentinel_hub_client_id
SH_CLIENT_SECRET=your_sentinel_hub_client_secret

# Путь к весам модели (опционально)
CROP_MODEL_PATH=/path/to/best_model.pth
```

### 3. Копирование модели

Скопируйте обученную модель в проект:

```bash
# Из основного проекта
cp D:\Hakaton\models\crop_detection\best_model.pth D:\Hakaton\Argo_Health\AgriHealthMap\
```

Обновите `.env`:
```env
CROP_MODEL_PATH=./best_model.pth
```

### 4. Запуск сервера

```bash
# Режим разработки
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Продакшн
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

Откройте в браузере: http://localhost:8000

---

## API Endpoints

### 1. GET `/` - Главная страница

Возвращает HTML интерфейс с картой.

**Response:**
- HTML страница

---

### 2. POST `/get-image` - Получить спутниковое изображение

Загружает изображение Sentinel-2 для указанной области.

**Request Body:**
```json
{
  "bbox": [37.5, 55.7, 37.6, 55.8],
  "layer_type": "true_color",
  "resolution": 10
}
```

**Parameters:**
- `bbox` (list[float]): Bounding box [min_lon, min_lat, max_lon, max_lat]
- `layer_type` (string): Тип слоя - `"true_color"` или `"ndvi"`
- `resolution` (int): Разрешение в метрах (10-500)

**Response:**
```json
{
  "image_base64": "data:image/jpeg;base64,...",
  "bbox": [37.5, 55.7, 37.6, 55.8],
  "center_lat": 55.75,
  "center_lon": 37.55,
  "width": 1000,
  "height": 1000
}
```

**Example (Python):**
```python
import requests

url = "http://localhost:8000/get-image"
data = {
    "bbox": [37.5, 55.7, 37.6, 55.8],
    "layer_type": "true_color",
    "resolution": 10
}

response = requests.post(url, json=data)
result = response.json()
print(f"Image size: {result['width']}x{result['height']}")
```

**Example (JavaScript):**
```javascript
const response = await fetch('http://localhost:8000/get-image', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    bbox: [37.5, 55.7, 37.6, 55.8],
    layer_type: 'true_color',
    resolution: 10
  })
});

const data = await response.json();
console.log('Image:', data.image_base64);
```

---

### 3. POST `/detect-crops` - Детекция культур (NEW!)

**Выполняет детекцию сельскохозяйственных культур с использованием модели глубокого обучения.**

**Request Body:**
```json
{
  "bbox": [37.5, 55.7, 37.6, 55.8],
  "layer_type": "true_color",
  "resolution": 10
}
```

**Parameters:**
- `bbox` (list[float]): Bounding box области
- `layer_type` (string): Не используется, но требуется для совместимости
- `resolution` (int): Разрешение в метрах

**Response:**
```json
{
  "rgb_image": "data:image/jpeg;base64,...",
  "crop_mask": "data:image/jpeg;base64,...",
  "overlay": "data:image/jpeg;base64,...",
  "class_distribution": {
    "background": 65.5,
    "wheat": 12.3,
    "corn": 18.2,
    "sunflower": 0.1,
    "soybean": 0.2,
    "other_crops": 3.7
  },
  "class_names": [
    "background",
    "wheat",
    "corn",
    "sunflower",
    "soybean",
    "other_crops"
  ],
  "bbox": [37.5, 55.7, 37.6, 55.8],
  "center_lat": 55.75,
  "center_lon": 37.55,
  "width": 1000,
  "height": 1000
}
```

**Response Fields:**
- `rgb_image`: Оригинальное RGB изображение (base64)
- `crop_mask`: Цветная маска классов культур (base64)
- `overlay`: RGB с наложенной маской (base64)
- `class_distribution`: Процентное распределение классов
- `class_names`: Названия классов
- `bbox`, `center_lat`, `center_lon`: Геопозиция
- `width`, `height`: Размеры изображения

**Классы культур:**
- **0 - background** (черный): Фон, не сельхозземли
- **1 - wheat** (золотой): Пшеница
- **2 - corn** (желтый): Кукуруза
- **3 - sunflower** (оранжевый): Подсолнечник
- **4 - soybean** (зеленый): Соя
- **5 - other_crops** (светло-зеленый): Другие культуры

**Example (Python):**
```python
import requests
import base64
from PIL import Image
from io import BytesIO

url = "http://localhost:8000/detect-crops"
data = {
    "bbox": [39.0, 45.0, 39.2, 45.2],  # Краснодарский край
    "layer_type": "true_color",
    "resolution": 10
}

response = requests.post(url, json=data)
result = response.json()

# Декодируем изображения
def decode_base64_image(base64_str):
    img_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(img_data))

rgb_img = decode_base64_image(result['rgb_image'])
mask_img = decode_base64_image(result['crop_mask'])
overlay_img = decode_base64_image(result['overlay'])

# Сохраняем
rgb_img.save('rgb.jpg')
mask_img.save('mask.jpg')
overlay_img.save('overlay.jpg')

# Выводим статистику
print("Распределение культур:")
for crop, percentage in result['class_distribution'].items():
    if percentage > 0:
        print(f"  {crop}: {percentage}%")
```

**Example (JavaScript):**
```javascript
async function detectCrops(bbox) {
  const response = await fetch('http://localhost:8000/detect-crops', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      bbox: bbox,
      layer_type: 'true_color',
      resolution: 10
    })
  });

  const data = await response.json();

  // Отображаем изображения
  document.getElementById('rgb').src = `data:image/jpeg;base64,${data.rgb_image}`;
  document.getElementById('mask').src = `data:image/jpeg;base64,${data.crop_mask}`;
  document.getElementById('overlay').src = `data:image/jpeg;base64,${data.overlay}`;

  // Выводим статистику
  console.log('Crop distribution:', data.class_distribution);

  return data;
}

// Использование
detectCrops([39.0, 45.0, 39.2, 45.2]);
```

---

### 4. GET `/model-info` - Информация о модели

Возвращает информацию о загруженной модели.

**Response:**
```json
{
  "model_name": "Attention U-Net",
  "classes": [
    "background",
    "wheat",
    "corn",
    "sunflower",
    "soybean",
    "other_crops"
  ],
  "num_classes": 6,
  "class_colors": [
    [0, 0, 0],
    [255, 215, 0],
    [255, 255, 0],
    [255, 140, 0],
    [0, 255, 0],
    [144, 238, 144]
  ],
  "input_channels": 10,
  "description": "Модель для детекции типов сельскохозяйственных культур на основе мультиспектральных данных Sentinel-2",
  "weights_loaded": true
}
```

**Example:**
```python
import requests

response = requests.get("http://localhost:8000/model-info")
info = response.json()

print(f"Model: {info['model_name']}")
print(f"Classes: {', '.join(info['classes'])}")
print(f"Weights loaded: {info['weights_loaded']}")
```

---

## Примеры использования

### Полный пример интеграции

```python
import requests
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO
import base64

class AgriHealthMapClient:
    """Клиент для работы с AgriHealthMap API"""

    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def get_model_info(self):
        """Получить информацию о модели"""
        response = requests.get(f"{self.base_url}/model-info")
        return response.json()

    def get_satellite_image(self, bbox, layer_type="true_color", resolution=10):
        """Получить спутниковое изображение"""
        data = {
            "bbox": bbox,
            "layer_type": layer_type,
            "resolution": resolution
        }
        response = requests.post(f"{self.base_url}/get-image", json=data)
        return response.json()

    def detect_crops(self, bbox, resolution=10):
        """Выполнить детекцию культур"""
        data = {
            "bbox": bbox,
            "layer_type": "true_color",
            "resolution": resolution
        }
        response = requests.post(f"{self.base_url}/detect-crops", json=data)
        return response.json()

    @staticmethod
    def decode_image(base64_str):
        """Декодировать base64 изображение"""
        img_data = base64.b64decode(base64_str)
        return np.array(Image.open(BytesIO(img_data)))

    def visualize_results(self, result):
        """Визуализировать результаты детекции"""
        rgb = self.decode_image(result['rgb_image'])
        mask = self.decode_image(result['crop_mask'])
        overlay = self.decode_image(result['overlay'])

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(rgb)
        axes[0].set_title('RGB Satellite Image')
        axes[0].axis('off')

        axes[1].imshow(mask)
        axes[1].set_title('Crop Classification')
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

        # Выводим статистику
        print("\nCrop Distribution:")
        for crop, percentage in result['class_distribution'].items():
            if percentage > 0:
                print(f"  {crop:15s}: {percentage:5.2f}%")


# Использование
client = AgriHealthMapClient()

# Проверяем модель
info = client.get_model_info()
print(f"Model loaded: {info['model_name']}")
print(f"Classes: {len(info['classes'])}")

# Детекция культур для области в Краснодарском крае
bbox = [39.0, 45.0, 39.2, 45.2]
result = client.detect_crops(bbox, resolution=10)

# Визуализация
client.visualize_results(result)
```

### Интеграция с веб-фронтендом

```html
<!DOCTYPE html>
<html>
<head>
    <title>AgriHealthMap Crop Detection</title>
</head>
<body>
    <h1>Crop Detection</h1>

    <button onclick="detectCrops()">Detect Crops</button>

    <div id="results">
        <h2>RGB Image</h2>
        <img id="rgb" style="max-width: 500px;">

        <h2>Crop Mask</h2>
        <img id="mask" style="max-width: 500px;">

        <h2>Overlay</h2>
        <img id="overlay" style="max-width: 500px;">

        <h2>Statistics</h2>
        <div id="stats"></div>
    </div>

    <script>
        async function detectCrops() {
            // Координаты области (например, Краснодарский край)
            const bbox = [39.0, 45.0, 39.2, 45.2];

            const response = await fetch('http://localhost:8000/detect-crops', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    bbox: bbox,
                    layer_type: 'true_color',
                    resolution: 10
                })
            });

            const data = await response.json();

            // Отображаем изображения
            document.getElementById('rgb').src = `data:image/jpeg;base64,${data.rgb_image}`;
            document.getElementById('mask').src = `data:image/jpeg;base64,${data.crop_mask}`;
            document.getElementById('overlay').src = `data:image/jpeg;base64,${data.overlay}`;

            // Выводим статистику
            let statsHTML = '<ul>';
            for (const [crop, percentage] of Object.entries(data.class_distribution)) {
                if (percentage > 0) {
                    statsHTML += `<li>${crop}: ${percentage.toFixed(2)}%</li>`;
                }
            }
            statsHTML += '</ul>';
            document.getElementById('stats').innerHTML = statsHTML;
        }
    </script>
</body>
</html>
```

---

## Архитектура модели

### Attention U-Net

**Характеристики:**
- **Вход:** 10 каналов Sentinel-2 (B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12)
- **Выход:** 6 классов (background, wheat, corn, sunflower, soybean, other_crops)
- **Параметры:** ~125M
- **Архитектура:** 5-уровневый encoder-decoder с attention механизмами

**Каналы Sentinel-2:**
1. B02 (490 nm) - Blue
2. B03 (560 nm) - Green
3. B04 (665 nm) - Red
4. B05 (705 nm) - Red Edge 1
5. B06 (740 nm) - Red Edge 2
6. B07 (783 nm) - Red Edge 3
7. B08 (842 nm) - NIR
8. B8A (865 nm) - NIR Narrow
9. B11 (1610 nm) - SWIR 1
10. B12 (2190 nm) - SWIR 2

---

## Troubleshooting

### Ошибка: "Model weights not found"
Убедитесь, что:
1. Файл модели существует по указанному пути
2. Переменная `CROP_MODEL_PATH` правильно настроена в `.env`
3. Формат файла - `.pth` (PyTorch checkpoint)

### Ошибка: "Sentinel Hub authentication failed"
Проверьте:
1. Правильность `SH_CLIENT_ID` и `SH_CLIENT_SECRET`
2. Активность аккаунта Sentinel Hub
3. Наличие квоты для запросов

### Медленная работа модели
Рекомендации:
1. Используйте GPU если доступен (измените `device='cuda'` в инициализации)
2. Уменьшите разрешение (resolution=50 вместо 10)
3. Ограничьте размер области (bbox)

---

## Лицензия

MIT License

## Контакты

Для вопросов и предложений создавайте issue в репозитории.
