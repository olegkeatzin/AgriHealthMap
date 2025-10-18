# --- СТАНДАРТНЫЕ И СТОРОННИЕ БИБЛИОТЕКИ ---
import os
import io
import base64
import traceback
from typing import List, Literal, Optional, Annotated
from pathlib import Path
import asyncio

# --- БИБЛИОТЕКИ ДЛЯ РАБОТЫ С ДАННЫМИ И МОДЕЛЯМИ ---
import numpy as np
import torch
from PIL import Image
from dotenv import load_dotenv

# --- БИБЛИОТЕКИ FastAPI ---
from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# --- НАШИ СОБСТВЕННЫЕ МОДУЛИ ---
from sentinel import Sentinel
from crop_model import CropDetectionModel, visualize_prediction
from field_segmentation_model import FieldSegmentationModel, visualize_field_segmentation
from ndvi_utils import calculate_ndvi_from_bands, classify_vegetation_health, visualize_ndvi
from ndvi_convlstm_model import NDVIConvLSTMPredictor, visualize_ndvi_prediction
from database import db_manager, FavoriteFieldCreate
from authentification import router as auth_router, get_current_user, User
from stats import get_all_statistics

# --- ЗАГРУЗКА ПЕРЕМЕННЫХ ОКРУЖЕНИЯ ---
load_dotenv()

# ==============================================================================
# 1. МОДЕЛИ ДАННЫХ ДЛЯ ЗАПРОСОВ (PYDANTIC)
# ==============================================================================

class BboxRequest(BaseModel):
    """Модель для запросов, работающих с гео-областью (Bounding Box)."""
    bbox: List[float]
    layer_type: Literal["true_color", "ndvi"] = "true_color"
    resolution: int = Field(10, ge=10, le=500)

class NDVIConvLSTMRequest(BaseModel):
    """Модель для запроса на предсказание NDVI через ConvLSTM модель."""
    bbox: List[float] = Field(..., description="[min_lon, min_lat, max_lon, max_lat]")
    dates: List[str] = Field(..., min_items=5, max_items=5, description="5 дат в формате YYYY-MM-DD")
    resolution: int = Field(10, ge=10, le=100)

class FavoriteFieldResponse(FavoriteFieldCreate):
    id: str

# ==============================================================================
# 2. ИНИЦИАЛИЗАЦИЯ FastAPI И КОНФИГУРАЦИЯ
# ==============================================================================

app = FastAPI(title="AgriHealthMap API")
templates = Jinja2Templates(directory="templates")

app.include_router(auth_router)

SH_CLIENT_ID = os.getenv("SH_CLIENT_ID")
SH_CLIENT_SECRET = os.getenv("SH_CLIENT_SECRET")
if not SH_CLIENT_ID or not SH_CLIENT_SECRET:
    raise ValueError("Пожалуйста, укажите SH_CLIENT_ID и SH_CLIENT_SECRET в .env файле.")

sentinel = Sentinel(SH_CLIENT_ID, SH_CLIENT_SECRET)
sentinel.set_date('latest')

# ==============================================================================
# 3. ЗАГРУЗКА И ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ
# ==============================================================================

CROP_MODEL_PATH = os.getenv("CROP_MODEL_PATH")
FIELD_MODEL_PATH = os.getenv("FIELD_MODEL_PATH")
NDVI_CONVLSTM_MODEL_PATH = os.getenv("NDVI_CONVLSTM_MODEL_PATH", "models/ndvi_convlstm_best.pth")

crop_model = CropDetectionModel(model_path=CROP_MODEL_PATH, device='cpu')
field_model = FieldSegmentationModel(model_path=FIELD_MODEL_PATH, device='cpu')
ndvi_convlstm_model = NDVIConvLSTMPredictor(model_path=NDVI_CONVLSTM_MODEL_PATH, device='cpu')

print(f"✓ Crop detection model initialized (weights: {CROP_MODEL_PATH or 'random init'})")
print(f"✓ Field segmentation model initialized (weights: {FIELD_MODEL_PATH or 'random init'})")
print(f"✓ NDVI ConvLSTM prediction model initialized (weights: {NDVI_CONVLSTM_MODEL_PATH})")

# ==============================================================================
# 4. API ЭНДПОИНТЫ
# ==============================================================================

def array_to_base64(arr: np.ndarray, format: str = "JPEG") -> str:
    img = Image.fromarray(arr)
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/model-info")
async def model_info():
    return JSONResponse(content={
        "crop_detection": {"model_name": "Attention U-Net (Crop Detection)", "weights_loaded": CROP_MODEL_PATH is not None},
        "field_segmentation": {"model_name": "Attention U-Net (Field Segmentation)", "weights_loaded": FIELD_MODEL_PATH is not None},
        "ndvi_convlstm_prediction": {"model_name": "ConvLSTM NDVI Predictor", "weights_loaded": True}
    })

@app.post("/get-image")
async def get_image(request: BboxRequest):
    try:
        min_lon, min_lat, max_lon, max_lat = request.bbox
        sentinel_bbox = sentinel.set_aoi(min_lon, min_lat, max_lon, max_lat)
        sentinel.set_resolution(request.resolution)
        size = sentinel.bbox_size

        image_dict, capture_date = sentinel.get_data([request.layer_type])
        image_data = image_dict[request.layer_type]
        
        if request.layer_type == 'true_color':
            image_arr = (image_data * 255 * 2.5).astype(np.uint8)
        elif request.layer_type == 'ndvi':
            image_arr = visualize_ndvi(image_data)
        else:
            image_arr = (image_data * 255).astype(np.uint8)
        
        image_base64 = array_to_base64(image_arr)
        
        return JSONResponse(content={
            "image_base64": image_base64,
            "bbox": request.bbox,
            "capture_date": capture_date,
            "center_lat": round(sentinel_bbox.middle[1], 5),
            "center_lon": round(sentinel_bbox.middle[0], 5),
            "width": size[0],  
            "height": size[1]
        })
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-crops")
async def detect_crops(request: BboxRequest):
    try:
        min_lon, min_lat, max_lon, max_lat = request.bbox
        sentinel_bbox = sentinel.set_aoi(min_lon, min_lat, max_lon, max_lat)
        sentinel.set_resolution(request.resolution)
        size = sentinel.bbox_size

        if max(size) > 2500:
            ratio = max(size) / 2500
            size = (int(size[0] / ratio), int(size[1] / ratio))

        required_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
        band_data, capture_date = sentinel.get_data(required_bands + ['true_color'])

        sentinel_bands = {}
        for key, value in band_data.items():
            band_name = key.replace('.tif', '') 
            sentinel_bands[band_name] = value

        prediction = crop_model.predict(sentinel_bands)
        
        # ИСПРАВЛЕНИЕ: Используем `sentinel_bands` вместо `band_data`
        rgb_image = (sentinel_bands['true_color'] * 255 * 2.5).astype(np.uint8)
        overlay = visualize_prediction(rgb_image, prediction, alpha=0.4)

        rgb_base64 = array_to_base64(rgb_image)
        mask_base64 = array_to_base64(prediction['colored_mask'])
        overlay_base64 = array_to_base64(overlay)

        return JSONResponse(content={
            "rgb_image": rgb_base64,
            "crop_mask": mask_base64,
            "overlay": overlay_base64,
            "class_distribution": prediction['class_distribution'],
            "class_names": prediction['class_names'],
            "capture_date": capture_date,
            "bbox": request.bbox,
            "center_lat": round(sentinel_bbox.middle[1], 5),
            "center_lon": round(sentinel_bbox.middle[0], 5),
            "width": size[0],
            "height": size[1]
        })
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/segment-fields")
async def segment_fields(request: BboxRequest):
    try:
        min_lon, min_lat, max_lon, max_lat = request.bbox
        sentinel_bbox = sentinel.set_aoi(min_lon, min_lat, max_lon, max_lat)
        sentinel.set_resolution(request.resolution)
        size = sentinel.bbox_size

        if max(size) > 2500:
            ratio = max(size) / 2500
            size = (int(size[0] / ratio), int(size[1] / ratio))

        required_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
        band_data, capture_date = sentinel.get_data(required_bands + ['true_color'])

        sentinel_bands = {}
        for key, value in band_data.items():
            band_name = key.replace('.tif', '') 
            sentinel_bands[band_name] = value

        prediction = field_model.predict(sentinel_bands)
        
        # ИСПРАВЛЕНИЕ: Используем `sentinel_bands` вместо `band_data`
        rgb_image = (sentinel_bands['true_color'] * 255 * 2.5).astype(np.uint8)
        overlay = visualize_field_segmentation(rgb_image, prediction, alpha=0.2, draw_boundaries=True)

        rgb_base64 = array_to_base64(rgb_image)
        mask_base64 = array_to_base64(prediction['colored_mask'])
        overlay_base64 = array_to_base64(overlay)

        return JSONResponse(content={
            "rgb_image": rgb_base64,
            "field_mask": mask_base64,
            "overlay": overlay_base64,
            "class_distribution": prediction['class_distribution'],
            "class_names": prediction['class_names'],
            "field_boundaries": prediction['field_boundaries'],
            "num_fields": len(prediction['field_boundaries']),
            "capture_date": capture_date,
            "bbox": request.bbox,
            "center_lat": round(sentinel_bbox.middle[1], 5),
            "center_lon": round(sentinel_bbox.middle[0], 5),
            "width": size[0],
            "height": size[1]
        })
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/calculate-ndvi")
async def calculate_ndvi(request: BboxRequest):
    try:
        min_lon, min_lat, max_lon, max_lat = request.bbox
        sentinel_bbox = sentinel.set_aoi(min_lon, min_lat, max_lon, max_lat)
        sentinel.set_resolution(request.resolution)
        size = sentinel.bbox_size

        if max(size) > 2500:
            ratio = max(size) / 2500
            size = (int(size[0] / ratio), int(size[1] / ratio))

        # --- ИЗМЕНЕНИЕ НАЧИНАЕТСЯ ЗДЕСЬ ---
        # Запрашиваем данные Sentinel и внешние статистические данные параллельно
        required_bands = ['B02', 'B03', 'B04', 'B08']
        
        sentinel_task = asyncio.to_thread(sentinel.get_data, required_bands)
        stats_task = get_all_statistics(request.bbox)
        
        (band_data, capture_date), environmental_data = await asyncio.gather(
            sentinel_task, 
            stats_task
        )
        # --- ИЗМЕНЕНИЕ ЗАКАНЧИВАЕТСЯ ЗДЕСЬ ---

        sentinel_bands = {}
        for key, value in band_data.items():
            band_name = key.replace('.tif', '')
            sentinel_bands[band_name] = value

        ndvi = calculate_ndvi_from_bands(sentinel_bands)
        health_classification = classify_vegetation_health(ndvi)
        ndvi_colored = visualize_ndvi(ndvi, colormap='RdYlGn')

        rgb_image = np.stack([
            (sentinel_bands['B04'] * 2.5 * 255),
            (sentinel_bands['B03'] * 2.5 * 255),
            (sentinel_bands['B02'] * 2.5 * 255)
        ], axis=2).astype(np.uint8)
        
        rgb_base64 = array_to_base64(rgb_image)
        ndvi_base64 = array_to_base64(ndvi_colored)

        return JSONResponse(content={
            "rgb_image": rgb_base64,
            "ndvi_image": ndvi_base64,
            "health_classification": health_classification,
            "statistics": {
                "mean_ndvi": float(np.nanmean(ndvi)),
                "std_ndvi": float(np.nanstd(ndvi)),
                "min_ndvi": float(np.nanmin(ndvi)),
                "max_ndvi": float(np.nanmax(ndvi))
            },
            # --- ДОБАВЛЯЕМ НОВЫЙ КЛЮЧ В ОТВЕТ ---
            "environmental_data": environmental_data, 
            "capture_date": capture_date,
            "bbox": request.bbox,
            "center_lat": round(sentinel_bbox.middle[1], 5),
            "center_lon": round(sentinel_bbox.middle[0], 5),
            "width": size[0],
            "height": size[1]
        })
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/auto-generate-dates")
async def auto_generate_dates(count: int = 5, interval_days: int = 30):
    """
    Автоматическая генерация списка дат для ConvLSTM модели.

    Генерирует даты ОТ ТЕКУЩЕЙ ДАТЫ назад (исторические данные)
    и дату предсказания на следующий месяц вперед.

    ОГРАНИЧЕНИЕ: Прогнозы доступны только в сельскохозяйственный сезон (апрель-октябрь).
    Зимой и поздней осенью прогнозы не имеют смысла.

    Args:
        count: Количество входных дат (по умолчанию 5)
        interval_days: Интервал между датами в днях (по умолчанию 30)

    Returns:
        Список входных дат (до сегодня включительно) + дата предсказания (следующий месяц)

    Raises:
        HTTPException: Если текущий месяц вне сезона (ноябрь-март)
    """
    from datetime import datetime, timedelta

    # Текущая дата - это ПОСЛЕДНЯЯ входная дата
    today = datetime.now()
    current_month = today.month

    # Проверка: прогнозы доступны только с апреля (4) по октябрь (10)
    if current_month < 4 or current_month > 10:
        # Определяем следующий доступный месяц
        if current_month < 4:
            next_available = f"апрель {today.year}"
        else:  # current_month > 10
            next_available = f"апрель {today.year + 1}"

        raise HTTPException(
            status_code=400,
            detail={
                "error": "Прогноз недоступен вне сельскохозяйственного сезона",
                "message": f"Прогнозы NDVI доступны только с апреля по октябрь. Сейчас {today.strftime('%B %Y')}.",
                "current_month": current_month,
                "allowed_months": "апрель (4) - октябрь (10)",
                "next_available_date": next_available,
                "reason": "В зимний период и поздней осенью растительность находится в состоянии покоя, прогнозы NDVI не информативны."
            }
        )

    # Генерируем входные даты от прошлого к настоящему
    # count=5: сегодня - 120 дней, сегодня - 90, сегодня - 60, сегодня - 30, сегодня
    input_dates = []
    for i in range(count - 1, -1, -1):  # От 4 до 0 (в обратном порядке)
        date = today - timedelta(days=i * interval_days)
        input_dates.append(date.strftime('%Y-%m-%d'))

    # Дата предсказания = сегодня + interval_days (следующий месяц)
    predicted_date = today + timedelta(days=interval_days)
    predicted_month = predicted_date.month

    # Предупреждение если предсказание выходит за пределы сезона
    warning = None
    if predicted_month > 10:
        warning = f"Прогноз на {predicted_date.strftime('%B %Y')} может быть менее точным - это поздняя осень/зима."

    return {
        "input_dates": input_dates,
        "predicted_date": predicted_date.strftime('%Y-%m-%d'),
        "count": len(input_dates),
        "interval_days": interval_days,
        "today": today.strftime('%Y-%m-%d'),
        "current_month": current_month,
        "season": "active" if 4 <= current_month <= 9 else "late_season",
        "warning": warning
    }

@app.post("/predict-ndvi-convlstm")
async def predict_ndvi_convlstm(request: NDVIConvLSTMRequest):
    """
    Предсказание будущей карты NDVI через ConvLSTM модель.

    Использует временную последовательность из 5 спутниковых снимков
    для предсказания будущего состояния растительности.
    """
    try:
        min_lon, min_lat, max_lon, max_lat = request.bbox
        sentinel_bbox = sentinel.set_aoi(min_lon, min_lat, max_lon, max_lat)
        sentinel.set_resolution(request.resolution)

        size = sentinel.bbox_size
        if max(size) > 2500:
            ratio = max(size) / 2500
            size = (int(size[0] / ratio), int(size[1] / ratio))

        # Собираем данные для всех временных шагов
        required_bands = ['B02', 'B03', 'B04', 'B08']
        ndvi_sequence = []
        weather_sequence = []
        capture_dates = []

        # Получаем данные для каждой даты
        for date_str in request.dates:
            sentinel.set_date(date_str)

            # Получаем спутниковые данные и статистику параллельно
            sentinel_task = asyncio.to_thread(sentinel.get_data, required_bands)
            stats_task = get_all_statistics(request.bbox)

            (band_data, capture_date), environmental_data = await asyncio.gather(
                sentinel_task,
                stats_task
            )

            # Преобразуем bands в dict
            sentinel_bands = {key.replace('.tif', ''): value for key, value in band_data.items()}

            # Вычисляем NDVI для этого временного шага
            ndvi_map = calculate_ndvi_from_bands(sentinel_bands)
            ndvi_sequence.append(ndvi_map)

            # Извлекаем погодные признаки (15 признаков)
            weather_features = [
                environmental_data.get('temperature', 0),
                environmental_data.get('precipitation', 0),
                environmental_data.get('humidity', 0),
                environmental_data.get('wind_speed', 0),
                environmental_data.get('pressure', 0),
                environmental_data.get('cloud_cover', 0),
                environmental_data.get('solar_radiation', 0),
                environmental_data.get('evapotranspiration', 0),
                environmental_data.get('dew_point', 0),
                environmental_data.get('frost_days', 0),
                environmental_data.get('growing_degree_days', 0),
                environmental_data.get('heat_stress_index', 0),
                environmental_data.get('drought_index', 0),
                environmental_data.get('rainfall_anomaly', 0),
                environmental_data.get('temperature_anomaly', 0)
            ]
            weather_sequence.append(weather_features)
            capture_dates.append(capture_date)

        # Топографические признаки (10 признаков) - статичные для всей области
        # Получаем их из последнего environmental_data
        topo_features = np.array([
            environmental_data.get('elevation_mean', 0),
            environmental_data.get('elevation_std', 0),
            environmental_data.get('slope_mean', 0),
            environmental_data.get('slope_std', 0),
            environmental_data.get('slope_max', 0),
            environmental_data.get('aspect_mean', 0),
            environmental_data.get('roughness_mean', 0),
            environmental_data.get('roughness_std', 0),
            environmental_data.get('twi_mean', 0),
            environmental_data.get('twi_std', 0)
        ])

        # Преобразуем в numpy arrays
        ndvi_sequence = np.array(ndvi_sequence)  # (5, H, W)
        weather_sequence = np.array(weather_sequence)  # (5, 15)

        # Предсказание через ConvLSTM
        prediction = ndvi_convlstm_model.predict(
            ndvi_sequence=ndvi_sequence,
            weather_sequence=weather_sequence,
            topo_features=topo_features,
            return_numpy=True
        )

        # Визуализация
        predicted_ndvi_map = prediction['predicted_ndvi_map']
        current_ndvi = ndvi_sequence[-1]  # Последний NDVI из входной последовательности

        # Создаем цветные визуализации
        predicted_colored = visualize_ndvi(predicted_ndvi_map, colormap='RdYlGn')
        current_colored = visualize_ndvi(current_ndvi, colormap='RdYlGn')

        # Карта изменений
        difference_map = predicted_ndvi_map - current_ndvi
        difference_colored = visualize_ndvi(difference_map, colormap='RdBu', vmin=-0.3, vmax=0.3)

        # Анализ изменений
        improvement_area = np.sum(difference_map > 0.05) / difference_map.size * 100
        degradation_area = np.sum(difference_map < -0.05) / difference_map.size * 100
        stable_area = 100 - improvement_area - degradation_area

        # Вычисляем дату предсказания (следующий месяц после последней входной даты)
        from datetime import datetime, timedelta
        last_input_date = datetime.strptime(request.dates[-1], '%Y-%m-%d')
        predicted_date = last_input_date + timedelta(days=30)  # Следующий месяц

        return JSONResponse(content={
            "predicted_ndvi": array_to_base64(predicted_colored),
            "current_ndvi": array_to_base64(current_colored),
            "difference_map": array_to_base64(difference_colored),
            "health_classification": prediction['health_classification'],
            "statistics": prediction['statistics'],
            "change_analysis": {
                "improvement_percent": round(improvement_area, 2),
                "degradation_percent": round(degradation_area, 2),
                "stable_percent": round(stable_area, 2)
            },
            "timeline": {
                "input_dates": capture_dates,
                "ndvi_values": [float(np.nanmean(ndvi)) for ndvi in ndvi_sequence],
                "predicted_date": predicted_date.strftime('%Y-%m-%d'),
                "predicted_value": prediction['statistics']['mean_ndvi']
            },
            "prediction_info": {
                "predicted_for_date": predicted_date.strftime('%Y-%m-%d'),
                "last_input_date": request.dates[-1],
                "forecast_horizon_days": 30
            },
            "bbox": request.bbox,
            "center_lat": round(sentinel_bbox.middle[1], 5),
            "center_lon": round(sentinel_bbox.middle[0], 5),
            "width": size[0],
            "height": size[1]
        })
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/favorites", response_model=FavoriteFieldResponse, tags=["Favorites"])
async def add_favorite_field(field: FavoriteFieldCreate, current_user: Annotated[User, Depends(get_current_user)]):
    new_field = db_manager.create_favorite_for_user(current_user.username, field)
    return FavoriteFieldResponse(**new_field)

@app.get("/favorites", response_model=List[FavoriteFieldResponse], tags=["Favorites"])
async def get_all_favorite_fields(current_user: Annotated[User, Depends(get_current_user)]):
    favorites_list = db_manager.get_favorites_for_user(current_user.username)
    return [FavoriteFieldResponse(**fav) for fav in favorites_list]

@app.delete("/favorites/{field_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Favorites"])
async def delete_favorite_field(field_id: str, current_user: Annotated[User, Depends(get_current_user)]):
    """
    Удаляет поле из избранного по его ID.
    Возвращает 204 No Content в случае успеха.
    Возвращает 404 Not Found, если поле не найдено или не принадлежит текущему пользователю.
    """
    success = db_manager.delete_favorite_for_user(current_user.username, field_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Favorite field not found or you do not have permission to delete it."
        )
    return None # При статусе 204 тело ответа должно быть пустым