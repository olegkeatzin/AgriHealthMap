# 🌾 AgriHealthMap + Crop Detection Model

## ✨ Что нового

Модель детекции сельскохозяйственных культур на основе **Attention U-Net** успешно интегрирована в AgriHealthMap!

### Новые возможности:
- 🔍 **Детекция 6 типов культур** (пшеница, кукуруза, подсолнечник, соя, другие)
- 🗺️ **Автоматическая обработка** спутниковых данных Sentinel-2
- 📊 **Статистика распределения** культур по площади
- 🎨 **Цветные визуализации** с overlay
- 🚀 **REST API** для интеграции с внешними приложениями

---

## 🚀 Быстрый старт

### 1. Копирование модели

```bash
# Скопируйте обученную модель
cp D:\Hakaton\models\crop_detection\best_model.pth \
   D:\Hakaton\Argo_Health\AgriHealthMap\best_model.pth
```

### 2. Настройка .env

Создайте файл `.env`:

```env
SH_CLIENT_ID=your_sentinel_hub_client_id
SH_CLIENT_SECRET=your_sentinel_hub_client_secret
CROP_MODEL_PATH=./best_model.pth
```

### 3. Установка

```bash
cd D:\Hakaton\Argo_Health\AgriHealthMap

# Установка зависимостей
pip install -e .

# Или с помощью uv
uv sync
```

### 4. Запуск

```bash
uvicorn main:app --reload
```

Откройте: **http://localhost:8000**

### 5. Тестирование

```bash
python test_integration.py
```

---

## 📡 Новые API Endpoints

### POST `/detect-crops`

Детекция культур для указанной области.

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
  }
}
```

### GET `/model-info`

Информация о модели.

**Response:**
```json
{
  "model_name": "Attention U-Net",
  "classes": ["background", "wheat", "corn", "sunflower", "soybean", "other_crops"],
  "num_classes": 6,
  "weights_loaded": true
}
```

---

## 📝 Примеры использования

### Python

```python
import requests

# Детекция культур
response = requests.post("http://localhost:8000/detect-crops", json={
    "bbox": [39.0, 45.0, 39.2, 45.2],  # Краснодарский край
    "layer_type": "true_color",
    "resolution": 10
})

result = response.json()

# Статистика
for crop, pct in result['class_distribution'].items():
    if pct > 0:
        print(f"{crop}: {pct}%")
```

### JavaScript

```javascript
fetch('http://localhost:8000/detect-crops', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    bbox: [39.0, 45.0, 39.2, 45.2],
    layer_type: 'true_color',
    resolution: 10
  })
})
.then(res => res.json())
.then(data => {
  console.log('Crops:', data.class_distribution);
  document.getElementById('overlay').src = `data:image/jpeg;base64,${data.overlay}`;
});
```

### cURL

```bash
curl -X POST http://localhost:8000/detect-crops \
  -H "Content-Type: application/json" \
  -d '{
    "bbox": [39.0, 45.0, 39.2, 45.2],
    "layer_type": "true_color",
    "resolution": 10
  }'
```

---

## 🏗️ Архитектура

```
AgriHealthMap/
├── main.py              # FastAPI + новые endpoints
├── sentinel.py          # Sentinel Hub API
├── crop_model.py        # 🆕 Модель детекции культур
├── best_model.pth       # 🆕 Веса модели
├── test_integration.py  # 🆕 Тесты
├── templates/
│   └── index.html
└── docs/
    ├── API_DOCUMENTATION.md      # 🆕 Полная документация API
    └── INTEGRATION_GUIDE.md      # 🆕 Руководство по интеграции
```

---

## 🎯 Классы культур

| ID | Класс | Цвет | Описание |
|----|-------|------|----------|
| 0 | background | ⚫ Черный | Фон (не сельхозземли) |
| 1 | wheat | 🟡 Золотой | Пшеница |
| 2 | corn | 🟨 Желтый | Кукуруза |
| 3 | sunflower | 🟧 Оранжевый | Подсолнечник |
| 4 | soybean | 🟢 Зеленый | Соя |
| 5 | other_crops | 🟩 Светло-зеленый | Другие культуры |

---

## 📚 Документация

- **Полная документация API:** [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
- **Руководство по интеграции:** [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)
- **Основной проект:** `D:\Hakaton\README.md`

---

## 🔧 Troubleshooting

### Модель не загружается?

```bash
# Проверьте путь
ls -lh best_model.pth

# Проверьте .env
cat .env | grep CROP_MODEL_PATH
```

### Ошибка Sentinel Hub?

Проверьте credentials в `.env`:
```env
SH_CLIENT_ID=...
SH_CLIENT_SECRET=...
```

### Медленная работа?

1. Используйте GPU (если доступен):
   ```env
   # В crop_model.py измените device='cpu' на device='cuda'
   ```

2. Уменьшите resolution:
   ```json
   {"resolution": 50}  // вместо 10
   ```

---

## 🚢 Deployment

### Docker

```bash
docker build -t agrihealthmap .
docker run -p 8000:8000 \
  -e SH_CLIENT_ID=... \
  -e SH_CLIENT_SECRET=... \
  -e CROP_MODEL_PATH=./best_model.pth \
  agrihealthmap
```

### Production

```bash
gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

---

## 📊 Статистика модели

- **Архитектура:** Attention U-Net (5 уровней)
- **Параметры:** ~125M
- **Вход:** 10 каналов Sentinel-2
- **Выход:** 6 классов культур
- **Производительность:**
  - CPU: ~3-5 сек / изображение
  - GPU: ~0.5-1 сек / изображение

---

## 🤝 Contributing

Для добавления новых функций:

1. Форкните репозиторий
2. Создайте ветку (`git checkout -b feature/NewFeature`)
3. Коммит (`git commit -m 'Add NewFeature'`)
4. Пуш (`git push origin feature/NewFeature`)
5. Откройте Pull Request

---

## 📄 License

MIT License

---

## ⭐ Credits

- **Модель:** Attention U-Net с Focal + Dice Loss
- **Данные:** Sentinel-2 L2A (ESA Copernicus)
- **API:** SentinelHub
- **Framework:** FastAPI + PyTorch

---

**🎉 Готово к использованию!**

Для вопросов и поддержки см. [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)
