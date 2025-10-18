# ConvLSTM NDVI Модель - Интеграция (УЛУЧШЕННАЯ ВЕРСИЯ)

## Обзор

Успешно интегрирована **улучшенная** обученная ConvLSTM модель для предсказания будущих карт NDVI на основе временных рядов спутниковых данных.

### Ключевые улучшения

✨ **LSTM для погоды** - Вместо простого Linear encoder используется 2-слойный LSTM для анализа временной динамики погодных условий на всех 5 шагах

✨ **Полная последовательность** - Обрабатывается вся временная последовательность погоды, а не только последний шаг

✨ **Стабильность** - Gradient clipping для предотвращения взрывного градиента

## Файлы

### Новые файлы

1. **`ndvi_convlstm_model.py`** - Архитектура модели и класс для предсказания
   - `ConvLSTMCell` - ConvLSTM ячейка
   - `NDVIPredictionModel` - Основная модель
   - `NDVIConvLSTMPredictor` - Обертка для использования
   - Утилиты для визуализации

2. **`models/ndvi_convlstm_best.pth`** - Обученные веса модели (930 KB)
   - Обучена на 622 сэмплах
   - Best validation loss: 0.01385
   - 50 эпох обучения

3. **`test_convlstm.py`** - Тестовый скрипт для проверки

### Модифицированные файлы

1. **`main.py`**
   - Добавлен импорт `NDVIConvLSTMPredictor`
   - Добавлена модель данных `NDVIConvLSTMRequest`
   - Добавлен эндпоинт `/predict-ndvi-convlstm`
   - Инициализация модели при старте

## Параметры модели

```python
MAP_HEIGHT = 500
MAP_WIDTH = 503
MAP_CHANNELS = 1  # NDVI
INPUT_TIMESTEPS = 5  # Временных шагов

WEATHER_FEATURES = 15  # Погодные признаки
TOPO_FEATURES = 10     # Топографические признаки

HIDDEN_DIM = 64        # Скрытая размерность
KERNEL_SIZE = 3        # Размер ядра свертки
```

## API Endpoint

### POST /predict-ndvi-convlstm

Предсказание будущей карты NDVI через ConvLSTM модель.

**Запрос:**

```json
{
    "bbox": [38.9, 45.0, 39.0, 45.1],
    "dates": [
        "2024-09-01",
        "2024-09-15",
        "2024-09-29",
        "2024-10-13",
        "2024-10-27"
    ],
    "resolution": 50
}
```

**Параметры:**

- `bbox` (List[float]): [min_lon, min_lat, max_lon, max_lat]
- `dates` (List[str]): 5 дат в формате YYYY-MM-DD (последовательные временные шаги)
- `resolution` (int): Разрешение в метрах (10-100)

**Ответ:**

```json
{
    "predicted_ndvi": "base64_encoded_image",
    "current_ndvi": "base64_encoded_image",
    "difference_map": "base64_encoded_image",
    "health_classification": {
        "no_vegetation": {"pixels": 1250, "percentage": 5.0},
        "unhealthy": {"pixels": 2500, "percentage": 10.0},
        "moderate": {"pixels": 7500, "percentage": 30.0},
        "healthy": {"pixels": 10000, "percentage": 40.0},
        "very_healthy": {"pixels": 3750, "percentage": 15.0},
        "dominant_class": "healthy"
    },
    "statistics": {
        "mean_ndvi": 0.5234,
        "std_ndvi": 0.1456,
        "min_ndvi": 0.1234,
        "max_ndvi": 0.8234,
        "median_ndvi": 0.5456
    },
    "change_analysis": {
        "improvement_percent": 35.67,
        "degradation_percent": 12.45,
        "stable_percent": 51.88
    },
    "timeline": {
        "dates": ["2024-09-01", ..., "2024-10-27"],
        "ndvi_values": [0.45, 0.48, 0.51, 0.54, 0.56],
        "predicted_value": 0.58
    },
    "input_dates": ["2024-09-01", ..., "2024-10-27"],
    "bbox": [38.9, 45.0, 39.0, 45.1],
    "center_lat": 45.05,
    "center_lon": 38.95,
    "width": 256,
    "height": 256
}
```

## Использование

### Python

```python
import requests

BASE_URL = "http://localhost:8000"

# Запрос
response = requests.post(
    f"{BASE_URL}/predict-ndvi-convlstm",
    json={
        "bbox": [38.9, 45.0, 39.0, 45.1],
        "dates": [
            "2024-09-01",
            "2024-09-15",
            "2024-09-29",
            "2024-10-13",
            "2024-10-27"
        ],
        "resolution": 50
    }
)

result = response.json()

# Статистика
print(f"Mean NDVI: {result['statistics']['mean_ndvi']:.4f}")
print(f"Dominant class: {result['health_classification']['dominant_class']}")

# Изменения
print(f"Improvement: {result['change_analysis']['improvement_percent']:.2f}%")
```

### JavaScript

```javascript
const response = await fetch('http://localhost:8000/predict-ndvi-convlstm', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        bbox: [38.9, 45.0, 39.0, 45.1],
        dates: [
            "2024-09-01",
            "2024-09-15",
            "2024-09-29",
            "2024-10-13",
            "2024-10-27"
        ],
        resolution: 50
    })
});

const result = await response.json();

// Отображение предсказанного NDVI
const img = document.getElementById('predicted-ndvi');
img.src = `data:image/png;base64,${result.predicted_ndvi}`;

// Статистика
console.log('Mean NDVI:', result.statistics.mean_ndvi);
console.log('Health:', result.health_classification.dominant_class);
```

### cURL

```bash
curl -X POST "http://localhost:8000/predict-ndvi-convlstm" \
  -H "Content-Type: application/json" \
  -d '{
    "bbox": [38.9, 45.0, 39.0, 45.1],
    "dates": [
      "2024-09-01",
      "2024-09-15",
      "2024-09-29",
      "2024-10-13",
      "2024-10-27"
    ],
    "resolution": 50
  }'
```

## Тестирование

```bash
# Запустите сервер
python main.py
# или
.\start_server.bat

# В другом терминале запустите тест
python test_convlstm.py
```

## Архитектура модели

### Входные данные

1. **NDVI последовательность** (5, H, W)
   - 5 временных шагов NDVI карт
   - Вычисляется из спутниковых данных Sentinel-2

2. **Погодные данные** (5, 15)
   - 15 признаков для каждого временного шага:
     - temperature, precipitation, humidity
     - wind_speed, pressure, cloud_cover
     - solar_radiation, evapotranspiration
     - dew_point, frost_days
     - growing_degree_days, heat_stress_index
     - drought_index, rainfall_anomaly, temperature_anomaly

3. **Топографические данные** (10,)
   - Статичные признаки области:
     - elevation_mean, elevation_std
     - slope_mean, slope_std, slope_max
     - aspect_mean
     - roughness_mean, roughness_std
     - twi_mean, twi_std

### Архитектура

```
Input NDVI Sequence (5, 1, H, W)
         ↓
    ConvLSTM Cell
         ↓
  Hidden State (64, H, W)
         |
         |    Weather Sequence (5, 15) → 2-Layer LSTM → (32) ✨ УЛУЧШЕНО
         |    Topo (10) → Linear → (16)
         |         ↓
         |    Fusion (48) → Linear → (64)
         |         ↓
         |    Feature Map (1, H, W)
         ↓         ↓
    Concatenate (65, H, W)
         ↓
      Decoder (BatchNorm + Dropout)
         ↓
   Predicted NDVI (1, H, W)
```

**Разница с базовой версией:**
- ❌ Простая: `Weather[last] → Linear → (32)`
- ✅ Улучшенная: `Weather[all 5 steps] → LSTM(2 layers) → (32)`

### Параметры

- **Всего параметров:** ~219,937
- **ConvLSTM:** ~66K параметров
- **Weather encoder:** ~1.1K параметров
- **Topo encoder:** ~544 параметров
- **Decoder:** ~152K параметров

## Производительность

### Метрики обучения

- **Train Loss:** 0.0151 (финальная)
- **Val Loss:** 0.0147 (лучшая)
- **Test Loss:** ~0.015 (примерно)

### Скорость

- **Предсказание:** ~2-5 секунд (зависит от размера области)
- **Загрузка данных:** ~30-60 секунд (5 спутниковых снимков)

### Требования к памяти

- **GPU:** ~1-2 GB VRAM (при использовании GPU)
- **RAM:** ~2-4 GB
- **Модель:** 930 KB

## Примеры использования

### 1. Мониторинг сельскохозяйственных полей

```python
# Предсказание NDVI для поля через 14 дней
dates = generate_dates(num_days=5, interval_days=14)
bbox = [39.5, 45.2, 39.6, 45.3]  # Поле в Краснодарском крае

response = requests.post(
    f"{BASE_URL}/predict-ndvi-convlstm",
    json={"bbox": bbox, "dates": dates, "resolution": 20}
)

result = response.json()
if result['change_analysis']['degradation_percent'] > 20:
    print("⚠️ Прогнозируется ухудшение растительности!")
```

### 2. Сравнение регионов

```python
regions = {
    "Краснодар": [38.9, 45.0, 39.0, 45.1],
    "Ростов": [39.7, 47.2, 39.8, 47.3]
}

for region_name, bbox in regions.items():
    result = predict_ndvi(bbox)
    print(f"{region_name}: {result['statistics']['mean_ndvi']:.4f}")
```

### 3. Временной анализ

```python
# Получаем предсказание
result = predict_ndvi(bbox, dates)

# Строим график временного ряда
import matplotlib.pyplot as plt

timeline = result['timeline']
dates = timeline['dates'] + ['PREDICTION']
values = timeline['ndvi_values'] + [timeline['predicted_value']]

plt.plot(dates, values, marker='o')
plt.axhline(y=0.5, color='r', linestyle='--', label='Healthy threshold')
plt.xlabel('Date')
plt.ylabel('NDVI')
plt.title('NDVI Forecast')
plt.legend()
plt.show()
```

## Troubleshooting

### Ошибка: "Model weights not found"

```bash
# Проверьте наличие файла модели
ls models/ndvi_convlstm_best.pth

# Если отсутствует, скопируйте из результатов обучения
cp D:/Hakaton/ndvi_convlstm_improved/best_model.pth models/ndvi_convlstm_best.pth
```

### Ошибка: "Invalid date format"

Убедитесь, что даты в формате `YYYY-MM-DD`:
```python
dates = ["2024-09-01", "2024-09-15", ...]  # ✓ Правильно
dates = ["01-09-2024", "15/09/2024", ...]  # ✗ Неправильно
```

### Медленное выполнение

- Уменьшите `resolution` (например, с 10 до 50)
- Уменьшите область `bbox`
- Используйте GPU если доступен

### Timeout

```python
# Увеличьте timeout
response = requests.post(url, json=data, timeout=600)  # 10 минут
```

## Дальнейшие улучшения

1. **Добавить кэширование** спутниковых данных
2. **Ускорить загрузку** данных через параллельные запросы
3. **Добавить батч-предсказания** для нескольких областей
4. **Интеграция с фронтендом** для визуализации
5. **Добавить confidence intervals** для предсказаний
6. **Мультимодельный ансамбль** для повышения точности

## Ссылки

- **Обучение модели:** `train_models/ndvi_prediction_training.ipynb`
- **Результаты обучения:** `D:/Hakaton/ndvi_convlstm_improved/`
- **API документация:** http://localhost:8000/docs
- **Swagger UI:** http://localhost:8000/redoc
