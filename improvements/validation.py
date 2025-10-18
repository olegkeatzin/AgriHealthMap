"""
Валидация входных данных
"""
from fastapi import HTTPException

def validate_bbox(bbox):
    """Валидация bounding box"""
    if len(bbox) != 4:
        raise HTTPException(400, "bbox должен содержать 4 значения: [min_lon, min_lat, max_lon, max_lat]")

    min_lon, min_lat, max_lon, max_lat = bbox

    # Проверка диапазонов
    if not (-180 <= min_lon <= 180) or not (-180 <= max_lon <= 180):
        raise HTTPException(400, "Долгота должна быть в диапазоне [-180, 180]")

    if not (-90 <= min_lat <= 90) or not (-90 <= max_lat <= 90):
        raise HTTPException(400, "Широта должна быть в диапазоне [-90, 90]")

    # Проверка порядка
    if min_lon >= max_lon:
        raise HTTPException(400, "min_lon должен быть меньше max_lon")

    if min_lat >= max_lat:
        raise HTTPException(400, "min_lat должен быть меньше max_lat")

    # Проверка размера (не более 1 градуса)
    if (max_lon - min_lon) > 1.0:
        raise HTTPException(400, "Слишком большая область (max 1° по долготе)")

    if (max_lat - min_lat) > 1.0:
        raise HTTPException(400, "Слишком большая область (max 1° по широте)")

    # Проверка минимального размера
    if (max_lon - min_lon) < 0.001:
        raise HTTPException(400, "Слишком маленькая область (min 0.001°)")

    if (max_lat - min_lat) < 0.001:
        raise HTTPException(400, "Слишком маленькая область (min 0.001°)")

    return True

def validate_dates(dates):
    """Валидация списка дат"""
    from datetime import datetime

    if not dates or len(dates) != 5:
        raise HTTPException(400, "Требуется ровно 5 дат")

    parsed_dates = []
    for date_str in dates:
        try:
            parsed = datetime.strptime(date_str, '%Y-%m-%d')
            parsed_dates.append(parsed)
        except ValueError:
            raise HTTPException(400, f"Неверный формат даты: {date_str}. Используйте YYYY-MM-DD")

    # Проверка хронологического порядка
    for i in range(len(parsed_dates) - 1):
        if parsed_dates[i] >= parsed_dates[i + 1]:
            raise HTTPException(400, "Даты должны быть в хронологическом порядке")

    # Проверка временного диапазона (не более 6 месяцев)
    if (parsed_dates[-1] - parsed_dates[0]).days > 180:
        raise HTTPException(400, "Слишком большой временной диапазон (max 180 дней)")

    return True

# Использование:
"""
from improvements.validation import validate_bbox, validate_dates

@app.post("/predict-ndvi-convlstm")
async def predict_ndvi_convlstm(request: NDVIConvLSTMRequest):
    validate_bbox(request.bbox)
    validate_dates(request.dates)
    # ...
"""
