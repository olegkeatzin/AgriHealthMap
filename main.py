from sentinel import Sentinel
from crop_model import CropDetectionModel, visualize_prediction
from field_segmentation_model import FieldSegmentationModel, visualize_field_segmentation
from ndvi_prediction_model import NDVIPredictionModel, visualize_ndvi, visualize_ndvi_change
import matplotlib.cm as cm
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse,JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel,Field
from typing import List, Literal, Optional
import io
from PIL import Image
import base64
import os
from dotenv import load_dotenv

load_dotenv()

# --- МОДЕЛЬ ДАННЫХ ДЛЯ ЗАПРОСА ---
# FastAPI будет проверять, что фронтенд присылает именно эти данные
class BboxRequest(BaseModel):
    bbox: List[float]
    layer_type: Optional[Literal["true_color", "ndvi"]] = None 
    # Добавляем поле resolution с проверкой: значение от 10 до 500
    resolution: int = Field(..., ge=10, le=500)


# --- ИНИЦИАЛИЗАЦИЯ FastAPI И ШАБЛОНОВ ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --- КОНФИГУРАЦИЯ SENTINEL HUB ---
# ❗️ ВАЖНО: Замените эти строки на ваши реальные Client ID и Client Secret
SH_CLIENT_ID = os.getenv("SH_CLIENT_ID")
print(SH_CLIENT_ID)
SH_CLIENT_SECRET = os.getenv("SH_CLIENT_SECRET")
print(SH_CLIENT_SECRET)
if not SH_CLIENT_ID or not SH_CLIENT_SECRET:
    raise ValueError("Пожалуйста, укажите SH_CLIENT_ID и SH_CLIENT_SECRET.")


sentinel = Sentinel(SH_CLIENT_ID,SH_CLIENT_SECRET)#sentinel hub
sentinel.set_date('latest')

# --- ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ ---
# Пути к весам моделей (опционально)
CROP_MODEL_PATH = os.getenv("CROP_MODEL_PATH", None)
FIELD_MODEL_PATH = os.getenv("FIELD_MODEL_PATH", None)
NDVI_MODEL_PATH = os.getenv("NDVI_MODEL_PATH", None)

crop_model = CropDetectionModel(model_path=CROP_MODEL_PATH, device='cpu')
field_model = FieldSegmentationModel(model_path=FIELD_MODEL_PATH, device='cpu')
ndvi_model = NDVIPredictionModel(model_path=NDVI_MODEL_PATH, device='cpu')

print(f"✓ Crop detection model initialized (weights: {CROP_MODEL_PATH or 'random init'})")
print(f"✓ Field segmentation model initialized (weights: {FIELD_MODEL_PATH or 'random init'})")
print(f"✓ NDVI prediction model initialized (weights: {NDVI_MODEL_PATH or 'random init'})")


# --- EVALSCRIPTS ---
# Скрипты, которые говорят Sentinel Hub, как обработать снимок


# --- ЭНДПОИНТЫ ---

# 1. Отдаёт главную страницу
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 2. Принимает координаты и возвращает снимок
@app.post("/get-image")
async def get_image(request: BboxRequest):
    try:
        min_lon, min_lat, max_lon, max_lat = request.bbox
        sentinel_bbox = sentinel.set_aoi(min_lon, min_lat, max_lon, max_lat)
        # ИСПОЛЬЗУЕМ РАЗРЕШЕНИЕ ИЗ ЗАПРОСА
        sentinel.set_resolution(request.resolution)

        size = sentinel.bbox_size
        
        MAX_SIZE = 2500
        if size[0] > MAX_SIZE or size[1] > MAX_SIZE:
            ratio = max(size) / MAX_SIZE
            size = (int(size[0] / ratio), int(size[1] / ratio))
            print(f"Размер изображения был уменьшен до {size} для предотвращения ошибки.")
        
        
        image_data = sentinel.get_data([request.layer_type])
        print(image_data)
        if request.layer_type == 'true_color':
            image = Image.fromarray((image_data * 255).astype(np.uint8),mode="RGB")
        if request.layer_type == 'ndvi':
            cmap = cm.get_cmap('RdYlGn')
            colored = cmap(image_data)
            print(colored)
            rgb_array = (colored[:, :, :3] * 255).astype(np.uint8)
            print(rgb_array)
            image = Image.fromarray(rgb_array, mode='RGB')        
        # Конвертируем изображение в байты, чтобы отправить его через API
        buffer = io.BytesIO()
        image.save(buffer, format="jpeg")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        # 4. ФОРМИРУЕМ JSON-ОТВЕТ С ДАННЫМИ
        center_coords = sentinel_bbox.middle
        response_data = {
            "image_base64": image_base64,
            "bbox": request.bbox,
            "center_lat": round(center_coords[1], 5),
            "center_lon": round(center_coords[0], 5),
            "width": size[0],  
            "height": size[1]
        }
        
        return JSONResponse(content=response_data)

    except Exception as e:
        print(f"Произошла ошибка: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 3. Детекция культур на основе спутниковых данных
@app.post("/detect-crops")
async def detect_crops(request: BboxRequest):
    """
    Выполняет детекцию сельскохозяйственных культур для указанной области

    Возвращает:
    - Оригинальное RGB изображение
    - Маску классов культур
    - Цветную визуализацию с overlay
    - Статистику распределения культур
    """
    try:
        min_lon, min_lat, max_lon, max_lat = request.bbox
        sentinel_bbox = sentinel.set_aoi(min_lon, min_lat, max_lon, max_lat)
        sentinel.set_resolution(request.resolution)

        size = sentinel.bbox_size

        MAX_SIZE = 2500
        if size[0] > MAX_SIZE or size[1] > MAX_SIZE:
            ratio = max(size) / MAX_SIZE
            size = (int(size[0] / ratio), int(size[1] / ratio))
            print(f"Размер изображения был уменьшен до {size}")

        # Получаем все необходимые каналы для модели (10 каналов Sentinel-2)
        required_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
        band_data = sentinel.get_data(required_bands)

        # Формируем словарь с данными каналов
        sentinel_bands = {}
        for key, value in band_data.items():
            band_name = key.replace('.tif', '') # Удаляем '.tif'
            sentinel_bands[band_name] = value

        # Запускаем модель детекции
        prediction = crop_model.predict(sentinel_bands, return_probabilities=False)

        # Получаем RGB для визуализации
        rgb_bands = ['B04', 'B03', 'B02']
        rgb_data = sentinel.get_data(rgb_bands)

        # Собираем 3D-массив из словаря каналов, используя np.stack
        # Указываем порядок [Red, Green, Blue] для правильного отображения
        rgb_image = np.stack(
        [rgb_data['B04.tif'], rgb_data['B03.tif'], rgb_data['B02.tif']],
         axis=2
        )

        # Теперь преобразуем готовый массив в формат для изображений (0-255, uint8)
        rgb_image = (rgb_image * 255).astype(np.uint8)

        # Создаем overlay visualization
        overlay = visualize_prediction(rgb_image, prediction, alpha=0.4)

        # Конвертируем изображения в base64
        def array_to_base64(arr):
            img = Image.fromarray(arr)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

        rgb_base64 = array_to_base64(rgb_image)
        mask_base64 = array_to_base64(prediction['colored_mask'])
        overlay_base64 = array_to_base64(overlay)

        # Формируем ответ
        center_coords = sentinel_bbox.middle
        response_data = {
            "rgb_image": rgb_base64,
            "crop_mask": mask_base64,
            "overlay": overlay_base64,
            "class_distribution": prediction['class_distribution'],
            "class_names": prediction['class_names'],
            "bbox": request.bbox,
            "center_lat": round(center_coords[1], 5),
            "center_lon": round(center_coords[0], 5),
            "width": size[0],
            "height": size[1]
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        print(f"Ошибка при детекции культур: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# 4. Получение информации о моделях
@app.get("/model-info")
async def model_info():
    """Возвращает информацию о всех загруженных моделях"""
    return JSONResponse(content={
        "crop_detection": {
            "model_name": "Attention U-Net (Crop Detection)",
            "classes": crop_model.CLASS_NAMES,
            "num_classes": len(crop_model.CLASS_NAMES),
            "class_colors": crop_model.CLASS_COLORS,
            "description": "Детекция типов сельскохозяйственных культур",
            "weights_loaded": CROP_MODEL_PATH is not None
        },
        "field_segmentation": {
            "model_name": "Attention U-Net (Field Segmentation)",
            "classes": field_model.CLASS_NAMES,
            "num_classes": len(field_model.CLASS_NAMES),
            "class_colors": field_model.CLASS_COLORS,
            "description": "Сегментация сельскохозяйственных полей",
            "weights_loaded": FIELD_MODEL_PATH is not None
        },
        "ndvi_prediction": {
            "model_name": "LSTM-CNN NDVI Predictor",
            "description": "Предсказание будущих значений NDVI",
            "weights_loaded": NDVI_MODEL_PATH is not None
        }
    })


# 5. Сегментация полей
@app.post("/segment-fields")
async def segment_fields(request: BboxRequest):
    """
    Выполняет сегментацию сельскохозяйственных полей

    Возвращает:
    - RGB изображение
    - Маску сегментации полей
    - Overlay с контурами
    - Статистику и контуры полей
    """
    try:
        min_lon, min_lat, max_lon, max_lat = request.bbox
        sentinel_bbox = sentinel.set_aoi(min_lon, min_lat, max_lon, max_lat)
        sentinel.set_resolution(request.resolution)

        size = sentinel.bbox_size

        MAX_SIZE = 2500
        if size[0] > MAX_SIZE or size[1] > MAX_SIZE:
            ratio = max(size) / MAX_SIZE
            size = (int(size[0] / ratio), int(size[1] / ratio))
            print(f"Размер изображения был уменьшен до {size}")

        # Получаем все необходимые каналы
        required_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
        band_data = sentinel.get_data(required_bands)

        sentinel_bands = {}
        for key, value in band_data.items():
            band_name = key.replace('.tif', '') # Удаляем '.tif'
            sentinel_bands[band_name] = value

        # Запускаем модель сегментации
        prediction = field_model.predict(sentinel_bands, return_probabilities=False)

        # Получаем RGB для визуализации
        rgb_bands = ['B04', 'B03', 'B02']
        rgb_data = sentinel.get_data(rgb_bands)

# Собираем 3D-массив из словаря каналов, используя np.stack
# Указываем порядок [Red, Green, Blue] для правильного отображения
        rgb_image = np.stack(
        [rgb_data['B04.tif'], rgb_data['B03.tif'], rgb_data['B02.tif']],
        axis=2
        )

# Теперь преобразуем готовый массив в формат для изображений (0-255, uint8)
        rgb_image = (rgb_image * 255).astype(np.uint8)

        # Создаем overlay с границами полей
        overlay = visualize_field_segmentation(rgb_image, prediction, alpha=0.4, draw_boundaries=True)

        # Конвертируем изображения в base64
        def array_to_base64(arr):
            img = Image.fromarray(arr)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

        rgb_base64 = array_to_base64(rgb_image)
        mask_base64 = array_to_base64(prediction['colored_mask'])
        overlay_base64 = array_to_base64(overlay)

        # Формируем ответ
        center_coords = sentinel_bbox.middle
        response_data = {
            "rgb_image": rgb_base64,
            "field_mask": mask_base64,
            "overlay": overlay_base64,
            "class_distribution": prediction['class_distribution'],
            "class_names": prediction['class_names'],
            "field_boundaries": prediction['field_boundaries'],
            "num_fields": len(prediction['field_boundaries']),
            "bbox": request.bbox,
            "center_lat": round(center_coords[1], 5),
            "center_lon": round(center_coords[0], 5),
            "width": size[0],
            "height": size[1]
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        print(f"Ошибка при сегментации полей: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# 6. Вычисление и визуализация NDVI
@app.post("/calculate-ndvi")
async def calculate_ndvi(request: BboxRequest):
    """
    Вычисляет NDVI для указанной области

    Возвращает:
    - RGB изображение
    - NDVI визуализацию
    - Статистику здоровья растительности
    """
    try:
        min_lon, min_lat, max_lon, max_lat = request.bbox
        sentinel_bbox = sentinel.set_aoi(min_lon, min_lat, max_lon, max_lat)
        sentinel.set_resolution(request.resolution)

        size = sentinel.bbox_size

        MAX_SIZE = 2500
        if size[0] > MAX_SIZE or size[1] > MAX_SIZE:
            ratio = max(size) / MAX_SIZE
            size = (int(size[0] / ratio), int(size[1] / ratio))

        # Получаем каналы для NDVI (B04, B08)
        required_bands = ['B02', 'B03', 'B04', 'B08']
        band_data = sentinel.get_data(required_bands)

        # Новый, исправленный код
        sentinel_bands = {}
        for key, value in band_data.items():
            # Удаляем '.tif' из ключа, чтобы получить чистое имя канала
            # Например, 'B02.tif' -> 'B02'
            band_name = key.replace('.tif', '') 
            sentinel_bands[band_name] = value

        # Вычисляем NDVI
        ndvi = ndvi_model.calculate_ndvi_from_bands(sentinel_bands)

        # Классификация здоровья растительности
        health_classification = ndvi_model._classify_vegetation_health(ndvi)

        # Визуализация NDVI
        ndvi_colored = visualize_ndvi(ndvi, colormap='RdYlGn')

        # RGB для сравнения
        rgb_image = np.stack([
            sentinel_bands['B04'],
            sentinel_bands['B03'],
            sentinel_bands['B02']
        ], axis=2)
        rgb_image = (rgb_image * 255).astype(np.uint8)

        # Конвертируем в base64
        def array_to_base64(arr):
            img = Image.fromarray(arr)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

        rgb_base64 = array_to_base64(rgb_image)
        ndvi_base64 = array_to_base64(ndvi_colored)

        # Статистика
        center_coords = sentinel_bbox.middle
        response_data = {
            "rgb_image": rgb_base64,
            "ndvi_image": ndvi_base64,
            "health_classification": health_classification,
            "statistics": {
                "mean_ndvi": float(ndvi.mean()),
                "std_ndvi": float(ndvi.std()),
                "min_ndvi": float(ndvi.min()),
                "max_ndvi": float(ndvi.max())
            },
            "bbox": request.bbox,
            "center_lat": round(center_coords[1], 5),
            "center_lon": round(center_coords[0], 5),
            "width": size[0],
            "height": size[1]
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        print(f"Ошибка при вычислении NDVI: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
# 7. Прогнозирование NDVI
@app.post("/predict-ndvi")
async def predict_ndvi(request: BboxRequest):
    """
    Прогнозирует будущее значение NDVI для указанной области.
    Для этого запрашивается временной ряд снимков.
    """
    try:
        min_lon, min_lat, max_lon, max_lat = request.bbox
        sentinel.set_aoi(min_lon, min_lat, max_lon, max_lat)
        sentinel.set_resolution(request.resolution) # Используем разрешение из запроса

        # Для прогноза нам нужен временной ряд данных.
        # Запросим данные NDVI за последние 180 дней.
        # ПРИМЕЧАНИЕ: Это предполагает, что ваш класс Sentinel
        # имеет метод для получения данных за период.
        # Если его нет, эту логику нужно будет реализовать в sentinel.py
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        # Получаем временную серию NDVI
        # Этот метод может потребовать доработки в вашем sentinel.py
        ndvi_timeseries = sentinel.get_data_for_time_range(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            layer='ndvi'
        )

        # Запускаем модель прогнозирования
        # Модель должна принять серию снимков и вернуть один прогноз
        predicted_ndvi = ndvi_model.predict_ndvi_timeseries(ndvi_timeseries)

        # Визуализируем результат
        predicted_ndvi_colored = visualize_ndvi(predicted_ndvi, colormap='RdYlGn')

        # Конвертируем в base64
        def array_to_base64(arr):
            img = Image.fromarray(arr)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

        predicted_ndvi_base64 = array_to_base64(predicted_ndvi_colored)

        # Формируем ответ
        response_data = {
            "predicted_ndvi_image": predicted_ndvi_base64,
            "statistics": {
                "mean_predicted_ndvi": float(predicted_ndvi.mean()),
                "max_predicted_ndvi": float(predicted_ndvi.max())
            },
            "message": "Прогноз NDVI на следующую доступную дату."
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        print(f"Ошибка при прогнозировании NDVI: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))