# 🌾 AgriHealthMap - Полное руководство по API

## Обзор

AgriHealthMap теперь включает **три ML модели** для комплексного анализа сельскохозяйственных земель:

1. **Детекция культур** - определение типов выращиваемых культур
2. **Сегментация полей** - выделение границ сельскохозяйственных полей
3. **Анализ NDVI** - оценка здоровья растительности

---

## 🚀 Быстрый старт

### Установка

```bash
cd D:\Hakaton\Argo_Health\AgriHealthMap

# Копирование моделей (если есть обученные веса)
cp D:\Hakaton\models\crop_detection\best_model.pth ./crop_model.pth
cp D:\Hakaton\models\field_segmentation\best_model.pth ./field_model.pth
cp D:\Hakaton\models\ndvi_prediction\best_model.pth ./ndvi_model.pth

# Настройка .env
cat > .env << EOF
SH_CLIENT_ID=your_sentinel_hub_id
SH_CLIENT_SECRET=your_sentinel_hub_secret
CROP_MODEL_PATH=./crop_model.pth
FIELD_MODEL_PATH=./field_model.pth
NDVI_MODEL_PATH=./ndvi_model.pth
EOF

# Установка зависимостей
pip install -e .

# Запуск
uvicorn main:app --reload
```

---

## 📡 API Endpoints

### 1. GET `/` - Главная страница

Веб-интерфейс с картой.

---

### 2. POST `/get-image` - Спутниковое изображение

Загружает изображение Sentinel-2.

**Request:**
```json
{
  "bbox": [37.5, 55.7, 37.6, 55.8],
  "layer_type": "true_color",
  "resolution": 10
}
```

**Response:**
```json
{
  "image_base64": "...",
  "bbox": [37.5, 55.7, 37.6, 55.8],
  "center_lat": 55.75,
  "center_lon": 37.55,
  "width": 1000,
  "height": 1000
}
```

---

### 3. POST `/detect-crops` - Детекция культур ⭐

Определяет типы выращиваемых культур.

**Request:**
```json
{
  "bbox": [39.0, 45.0, 39.2, 45.2],
  "layer_type": "true_color",
  "resolution": 10
}
```

**Response:**
```json
{
  "rgb_image": "base64...",
  "crop_mask": "base64...",
  "overlay": "base64...",
  "class_distribution": {
    "background": 65.5,
    "wheat": 12.3,
    "corn": 18.2,
    "sunflower": 0.1,
    "soybean": 0.2,
    "other_crops": 3.7
  },
  "class_names": ["background", "wheat", "corn", "sunflower", "soybean", "other_crops"]
}
```

**Классы культур:**
- 0: background (фон)
- 1: wheat (пшеница)
- 2: corn (кукуруза)
- 3: sunflower (подсолнечник)
- 4: soybean (соя)
- 5: other_crops (другие культуры)

---

### 4. POST `/segment-fields` - Сегментация полей ⭐ NEW!

Выделяет границы сельскохозяйственных полей.

**Request:**
```json
{
  "bbox": [39.0, 45.0, 39.2, 45.2],
  "layer_type": "true_color",
  "resolution": 10
}
```

**Response:**
```json
{
  "rgb_image": "base64...",
  "field_mask": "base64...",
  "overlay": "base64...",
  "class_distribution": {
    "background": 45.2,
    "field": 52.3,
    "field_boundary": 1.5,
    "other": 1.0
  },
  "class_names": ["background", "field", "field_boundary", "other"],
  "field_boundaries": [
    {
      "points": [[x1, y1], [x2, y2], ...],
      "area": 15234.5,
      "perimeter": 450.2
    },
    ...
  ],
  "num_fields": 15
}
```

**Классы:**
- 0: background (фон)
- 1: field (поле)
- 2: field_boundary (граница поля)
- 3: other (другие объекты)

---

### 5. POST `/calculate-ndvi` - Анализ NDVI ⭐ NEW!

Вычисляет NDVI и оценивает здоровье растительности.

**Request:**
```json
{
  "bbox": [39.0, 45.0, 39.2, 45.2],
  "layer_type": "true_color",
  "resolution": 10
}
```

**Response:**
```json
{
  "rgb_image": "base64...",
  "ndvi_image": "base64...",
  "health_classification": {
    "water_or_snow": 5.2,
    "bare_soil": 15.3,
    "sparse_vegetation": 20.1,
    "moderate_vegetation": 25.4,
    "healthy_vegetation": 28.0,
    "very_healthy": 6.0
  },
  "statistics": {
    "mean_ndvi": 0.45,
    "std_ndvi": 0.18,
    "min_ndvi": -0.2,
    "max_ndvi": 0.85
  }
}
```

**Классификация здоровья:**
- `water_or_snow`: NDVI < 0 (вода, снег, облака)
- `bare_soil`: 0 ≤ NDVI < 0.2 (голая почва)
- `sparse_vegetation`: 0.2 ≤ NDVI < 0.4 (разреженная растительность)
- `moderate_vegetation`: 0.4 ≤ NDVI < 0.6 (умеренная)
- `healthy_vegetation`: 0.6 ≤ NDVI < 0.8 (здоровая)
- `very_healthy`: NDVI ≥ 0.8 (очень здоровая)

---

### 6. GET `/model-info` - Информация о моделях

Возвращает информацию о всех загруженных моделях.

**Response:**
```json
{
  "crop_detection": {
    "model_name": "Attention U-Net (Crop Detection)",
    "classes": ["background", "wheat", "corn", "sunflower", "soybean", "other_crops"],
    "num_classes": 6,
    "weights_loaded": true
  },
  "field_segmentation": {
    "model_name": "Attention U-Net (Field Segmentation)",
    "classes": ["background", "field", "field_boundary", "other"],
    "num_classes": 4,
    "weights_loaded": true
  },
  "ndvi_prediction": {
    "model_name": "LSTM-CNN NDVI Predictor",
    "description": "Предсказание будущих значений NDVI",
    "weights_loaded": false
  }
}
```

---

## 💡 Примеры использования

### Python - Полный workflow

```python
import requests
import base64
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

class AgriHealthMapClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def detect_crops(self, bbox, resolution=10):
        """Детекция культур"""
        response = requests.post(f"{self.base_url}/detect-crops", json={
            "bbox": bbox,
            "layer_type": "true_color",
            "resolution": resolution
        })
        return response.json()

    def segment_fields(self, bbox, resolution=10):
        """Сегментация полей"""
        response = requests.post(f"{self.base_url}/segment-fields", json={
            "bbox": bbox,
            "layer_type": "true_color",
            "resolution": resolution
        })
        return response.json()

    def calculate_ndvi(self, bbox, resolution=10):
        """Анализ NDVI"""
        response = requests.post(f"{self.base_url}/calculate-ndvi", json={
            "bbox": bbox,
            "layer_type": "true_color",
            "resolution": resolution
        })
        return response.json()

    @staticmethod
    def decode_image(base64_str):
        """Декодирование base64 изображения"""
        img_data = base64.b64decode(base64_str)
        return Image.open(BytesIO(img_data))


# Использование
client = AgriHealthMapClient()

# Краснодарский край
bbox = [39.0, 45.0, 39.2, 45.2]

# 1. Детекция культур
print("1. Детекция культур...")
crops = client.detect_crops(bbox, resolution=10)
print(f"Найдено культур:")
for crop, pct in crops['class_distribution'].items():
    if pct > 0:
        print(f"  {crop}: {pct}%")

# 2. Сегментация полей
print("\n2. Сегментация полей...")
fields = client.segment_fields(bbox, resolution=10)
print(f"Найдено полей: {fields['num_fields']}")
print(f"Общая площадь полей: {sum(f['area'] for f in fields['field_boundaries']):.0f} пикселей")

# 3. Анализ NDVI
print("\n3. Анализ NDVI...")
ndvi = client.calculate_ndvi(bbox, resolution=10)
print(f"Средний NDVI: {ndvi['statistics']['mean_ndvi']:.3f}")
print(f"Здоровая растительность: {ndvi['health_classification']['healthy_vegetation']}%")

# Визуализация
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Crop detection
axes[0, 0].imshow(client.decode_image(crops['rgb_image']))
axes[0, 0].set_title('RGB')
axes[0, 0].axis('off')

axes[0, 1].imshow(client.decode_image(crops['crop_mask']))
axes[0, 1].set_title('Crop Detection')
axes[0, 1].axis('off')

axes[0, 2].imshow(client.decode_image(crops['overlay']))
axes[0, 2].set_title('Crop Overlay')
axes[0, 2].axis('off')

# Field segmentation
axes[1, 0].imshow(client.decode_image(fields['rgb_image']))
axes[1, 0].set_title('RGB')
axes[1, 0].axis('off')

axes[1, 1].imshow(client.decode_image(fields['field_mask']))
axes[1, 1].set_title('Field Segmentation')
axes[1, 1].axis('off')

axes[1, 2].imshow(client.decode_image(ndvi['ndvi_image']))
axes[1, 2].set_title('NDVI')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('agrihealthmap_results.png', dpi=150)
plt.show()

print("\n✓ Анализ завершен!")
```

### JavaScript - Веб-интеграция

```javascript
class AgriHealthMapClient {
    constructor(baseURL = 'http://localhost:8000') {
        this.baseURL = baseURL;
    }

    async detectCrops(bbox, resolution = 10) {
        const response = await fetch(`${this.baseURL}/detect-crops`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                bbox: bbox,
                layer_type: 'true_color',
                resolution: resolution
            })
        });
        return await response.json();
    }

    async segmentFields(bbox, resolution = 10) {
        const response = await fetch(`${this.baseURL}/segment-fields`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                bbox: bbox,
                layer_type: 'true_color',
                resolution: resolution
            })
        });
        return await response.json();
    }

    async calculateNDVI(bbox, resolution = 10) {
        const response = await fetch(`${this.baseURL}/calculate-ndvi`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                bbox: bbox,
                layer_type: 'true_color',
                resolution: resolution
            })
        });
        return await response.json();
    }
}

// Использование
const client = new AgriHealthMapClient();
const bbox = [39.0, 45.0, 39.2, 45.2];

// Комплексный анализ
async function analyzeArea(bbox) {
    // 1. Детекция культур
    const crops = await client.detectCrops(bbox);
    document.getElementById('crops-overlay').src = `data:image/jpeg;base64,${crops.overlay}`;

    // 2. Сегментация полей
    const fields = await client.segmentFields(bbox);
    document.getElementById('fields-overlay').src = `data:image/jpeg;base64,${fields.overlay}`;
    document.getElementById('num-fields').textContent = fields.num_fields;

    // 3. NDVI
    const ndvi = await client.calculateNDVI(bbox);
    document.getElementById('ndvi-image').src = `data:image/jpeg;base64,${ndvi.ndvi_image}`;
    document.getElementById('mean-ndvi').textContent = ndvi.statistics.mean_ndvi.toFixed(3);

    console.log('Analysis complete!');
}

analyzeArea(bbox);
```

---

## 🔧 Конфигурация

### Переменные окружения (.env)

```env
# Sentinel Hub API (обязательно)
SH_CLIENT_ID=your_client_id
SH_CLIENT_SECRET=your_client_secret

# Пути к моделям (опционально)
CROP_MODEL_PATH=./crop_model.pth
FIELD_MODEL_PATH=./field_model.pth
NDVI_MODEL_PATH=./ndvi_model.pth
```

### Настройки производительности

```python
# В main.py можно изменить device на 'cuda' для GPU
crop_model = CropDetectionModel(model_path=CROP_MODEL_PATH, device='cuda')
field_model = FieldSegmentationModel(model_path=FIELD_MODEL_PATH, device='cuda')
ndvi_model = NDVIPredictionModel(model_path=NDVI_MODEL_PATH, device='cuda')
```

---

## 📊 Архитектура моделей

| Модель | Архитектура | Параметры | Вход | Выход |
|--------|-------------|-----------|------|-------|
| Crop Detection | Attention U-Net (5 уровней) | ~125M | 10 каналов Sentinel-2 | 6 классов культур |
| Field Segmentation | Attention U-Net (4 уровня) | ~80M | 10 каналов Sentinel-2 | 4 класса (поля, границы) |
| NDVI Prediction | LSTM-CNN | ~15M | Временной ряд | Будущий NDVI |

---

## 🎯 Use Cases

### 1. Мониторинг урожая

```python
# Отслеживание типов культур по регионам
crops = client.detect_crops([39.0, 45.0, 39.5, 45.5], resolution=50)
wheat_area = crops['class_distribution']['wheat']
print(f"Пшеница занимает {wheat_area}% площади")
```

### 2. Планирование земель

```python
# Определение границ полей для планирования
fields = client.segment_fields([39.0, 45.0, 39.5, 45.5], resolution=10)
for i, field in enumerate(fields['field_boundaries']):
    print(f"Поле {i+1}: площадь {field['area']:.0f} пикселей")
```

### 3. Оценка здоровья растений

```python
# Мониторинг состояния посевов
ndvi = client.calculate_ndvi([39.0, 45.0, 39.5, 45.5], resolution=10)
if ndvi['statistics']['mean_ndvi'] < 0.4:
    print("⚠️ Низкий NDVI - возможны проблемы с растительностью")
```

---

## 🚢 Deployment

### Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

COPY pyproject.toml ./
COPY *.py ./
COPY templates/ ./templates/
COPY *.pth ./

RUN pip install -e .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t agrihealthmap .
docker run -p 8000:8000 \
  -e SH_CLIENT_ID=... \
  -e SH_CLIENT_SECRET=... \
  -e CROP_MODEL_PATH=./crop_model.pth \
  -e FIELD_MODEL_PATH=./field_model.pth \
  agrihealthmap
```

---

## 📈 Производительность

| Операция | CPU (i7) | GPU (RTX 3080) |
|----------|----------|----------------|
| Crop Detection | 3-5 сек | 0.5-1 сек |
| Field Segmentation | 2-4 сек | 0.3-0.7 сек |
| NDVI Calculation | <1 сек | <0.1 сек |

---

## ❓ FAQ

**Q: Можно ли использовать без весов моделей?**
A: Да, модели будут работать со случайной инициализацией (для тестирования API).

**Q: Как ускорить работу?**
A: Используйте GPU (device='cuda'), уменьшите resolution, ограничьте bbox.

**Q: Поддерживаются ли другие спутники кроме Sentinel-2?**
A: Сейчас только Sentinel-2, но можно расширить.

**Q: Можно ли батч-обработку нескольких областей?**
A: Пока нет, но можно добавить эндпоинт для batch inference.

---

## 📞 Поддержка

- **Документация:** `API_DOCUMENTATION.md`, `INTEGRATION_GUIDE.md`
- **Тесты:** `test_integration.py`
- **Основной проект:** `D:\Hakaton\`

---

## ✅ Checklist

- [ ] Установлены зависимости (`pip install -e .`)
- [ ] Настроен `.env` с Sentinel Hub credentials
- [ ] Скопированы веса моделей (опционально)
- [ ] Запущен сервер (`uvicorn main:app --reload`)
- [ ] Протестированы эндпоинты (`python test_integration.py`)

---

**🎉 Готово к использованию!**

Для детальной документации см. `API_DOCUMENTATION.md` и `INTEGRATION_GUIDE.md`.
