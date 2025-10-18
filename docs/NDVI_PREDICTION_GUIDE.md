# Руководство по использованию прогноза NDVI

## Что изменилось?

**Удалены старые модели:**
- ❌ Табличная модель NDVI (требовала ручной ввод погодных данных)
- ❌ Простая LSTM модель
- ❌ Transformer модель

**Добавлена новая модель:**
- ✅ **ConvLSTM модель** - автоматический прогноз NDVI на основе спутниковых данных

## Как пользоваться прогнозом NDVI?

### Через веб-интерфейс (рекомендуется)

1. **Запустите сервер:**
   ```bash
   cd D:\Hakaton\Argo_Health\AgriHealthMap
   python main.py
   ```

2. **Откройте браузер:**
   - Перейдите на http://localhost:8000

3. **Выберите область:**
   - Нажмите на кнопку "Прямоугольник" на карте (□)
   - Выделите интересующую вас область на карте

4. **Запустите прогноз:**
   - В правой панели нажмите кнопку **"Прогноз NDVI"**
   - Система автоматически:
     - Сгенерирует 5 дат (с апреля текущего года с интервалом 30 дней)
     - Загрузит спутниковые снимки для каждой даты
     - Вычислит NDVI для всех снимков
     - Соберет погодные и топографические данные
     - Построит прогноз через ConvLSTM модель

5. **Изучите результаты:**
   - **📈 Предсказанный NDVI** - будущее состояние растительности
   - **📊 Текущий NDVI** - текущее состояние (последний снимок)
   - **🔄 Карта изменений** - где ожидается улучшение/ухудшение
   - **Статистика** - средние значения, min/max
   - **Анализ изменений** - процент улучшения/ухудшения/стабильности
   - **Временная шкала** - NDVI для всех входных дат + предсказание

### Через API (для разработчиков)

#### 1. Автоматическая генерация дат

```bash
curl http://localhost:8000/auto-generate-dates?count=5&interval_days=30&start_month=4
```

**Ответ:**
```json
{
  "dates": ["2025-04-01", "2025-05-01", "2025-05-31", "2025-06-30", "2025-07-30"],
  "count": 5
}
```

**Параметры:**
- `count` - количество дат (по умолчанию 5)
- `interval_days` - интервал между датами в днях (по умолчанию 30)
- `start_month` - начальный месяц (по умолчанию 4 = апрель)

#### 2. Запрос прогноза NDVI

```bash
curl -X POST "http://localhost:8000/predict-ndvi-convlstm" \
  -H "Content-Type: application/json" \
  -d '{
    "bbox": [38.9, 45.0, 39.0, 45.1],
    "dates": ["2025-04-01", "2025-05-01", "2025-05-31", "2025-06-30", "2025-07-30"],
    "resolution": 10
  }'
```

**Параметры запроса:**
- `bbox` - [min_lon, min_lat, max_lon, max_lat]
- `dates` - список из 5 дат в формате YYYY-MM-DD
- `resolution` - разрешение в метрах (10-100, по умолчанию 10)

**Ответ:**
```json
{
  "predicted_ndvi": "base64_image...",
  "current_ndvi": "base64_image...",
  "difference_map": "base64_image...",
  "health_classification": {
    "water_snow": 0.5,
    "bare_soil": 5.2,
    "sparse_vegetation": 15.8,
    "moderate_vegetation": 35.4,
    "healthy_vegetation": 38.1,
    "very_dense_vegetation": 5.0
  },
  "statistics": {
    "mean_ndvi": 0.652,
    "std_ndvi": 0.089,
    "min_ndvi": 0.123,
    "max_ndvi": 0.891
  },
  "change_analysis": {
    "improvement_percent": 23.45,
    "degradation_percent": 12.34,
    "stable_percent": 64.21
  },
  "timeline": {
    "dates": ["2025-04-01", "2025-05-01", "2025-05-31", "2025-06-30", "2025-07-30"],
    "ndvi_values": [0.623, 0.645, 0.658, 0.671, 0.680],
    "predicted_value": 0.652
  }
}
```

## Как работает модель?

### Архитектура ConvLSTM

1. **Входные данные:**
   - 5 карт NDVI (временная последовательность)
   - 15 погодных признаков для каждой даты (температура, осадки, влажность и т.д.)
   - 10 топографических признаков (высота, уклон, экспозиция и т.д.)

2. **Обработка:**
   - **ConvLSTM слои** - обрабатывают пространственно-временную динамику NDVI
   - **LSTM кодировщик** - обрабатывает временную динамику погоды
   - **MLP** - обрабатывает статические топографические данные
   - **Attention механизм** - фокусируется на важных признаках

3. **Выход:**
   - Предсказанная карта NDVI (будущее состояние)
   - Классификация здоровья растительности
   - Статистика и анализ изменений

### Технические характеристики

- **Модель:** Attention U-Net с ConvLSTM
- **Параметры:** ~15M
- **Веса:** D:\Hakaton\Argo_Health\AgriHealthMap\models\ndvi_convlstm_best.pth
- **Val Loss:** 0.01385 (best epoch)
- **Обучающих образцов:** 622
- **Эпох обучения:** 50

## Примеры использования

### Python

```python
import requests
import json
from datetime import datetime, timedelta

# 1. Генерация дат
dates_response = requests.get("http://localhost:8000/auto-generate-dates",
                             params={"count": 5, "interval_days": 30})
dates = dates_response.json()["dates"]

# 2. Прогноз NDVI
bbox = [38.9, 45.0, 39.0, 45.1]  # Пример области

response = requests.post("http://localhost:8000/predict-ndvi-convlstm",
    json={
        "bbox": bbox,
        "dates": dates,
        "resolution": 10
    }
)

result = response.json()

print(f"Средний NDVI: {result['statistics']['mean_ndvi']:.3f}")
print(f"Улучшение: {result['change_analysis']['improvement_percent']:.1f}%")
print(f"Ухудшение: {result['change_analysis']['degradation_percent']:.1f}%")
```

### JavaScript (Fetch API)

```javascript
// 1. Генерация дат
const datesResponse = await fetch('/auto-generate-dates?count=5&interval_days=30');
const datesData = await datesResponse.json();

// 2. Прогноз NDVI
const predictionResponse = await fetch('/predict-ndvi-convlstm', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        bbox: [38.9, 45.0, 39.0, 45.1],
        dates: datesData.dates,
        resolution: 10
    })
});

const result = await predictionResponse.json();

// 3. Отображение изображения
document.getElementById('ndvi-image').src =
    `data:image/jpeg;base64,${result.predicted_ndvi}`;
```

## Ограничения

1. **Размер области:** Рекомендуется не более 1° x 1° для быстрой обработки
2. **Даты:** Требуется ровно 5 дат в хронологическом порядке
3. **Временной диапазон:** Не более 180 дней между первой и последней датой
4. **Разрешение:** 10-100 метров (меньше = медленнее)
5. **Доступность данных:** Зависит от облачности и наличия снимков Sentinel-2

## Устранение неполадок

### Ошибка: "No data available for the specified date"
**Решение:** Выберите другие даты или область. Sentinel-2 может не иметь данных для конкретной даты.

### Ошибка: "Слишком большая область"
**Решение:** Уменьшите размер bbox до 1° x 1° или меньше.

### Модель работает медленно
**Решение:**
- Увеличьте `resolution` (например, с 10 до 50)
- Используйте GPU вместо CPU (укажите `device='cuda'` в main.py)
- Уменьшите размер области

## Дополнительная информация

- **Документация API:** http://localhost:8000/docs
- **Исходный код модели:** D:\Hakaton\Argo_Health\AgriHealthMap\ndvi_convlstm_model.py
- **Notebook с обучением:** D:\Hakaton\Argo_Health\AgriHealthMap\train_models\ndvi_prediction_training.ipynb
- **Архитектура:** IMPROVED версия с LSTM weather encoder (не простая Linear версия)

## Следующие шаги

1. Попробуйте прогноз на вашей области через веб-интерфейс
2. Изучите результаты и статистику
3. Сравните с реальными данными для валидации
4. Настройте параметры (интервал дат, разрешение) под ваши задачи
