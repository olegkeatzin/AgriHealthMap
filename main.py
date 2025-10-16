# --- СТАНДАРТНЫЕ И СТОРОННИЕ БИБЛИОТЕКИ ---
import os
import io
import base64
import traceback
import pickle
from typing import List, Literal, Optional, Annotated
from pathlib import Path

# --- БИБЛИОТЕКИ ДЛЯ РАБОТЫ С ДАННЫМИ И МОДЕЛЯМИ ---
import numpy as np
import pandas as pd
import torch
import matplotlib.cm as cm
from PIL import Image
from dotenv import load_dotenv

# --- БИБЛИОТЕКИ FastAPI ---
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# --- НАШИ СОБСТВЕННЫЕ МОДУЛИ ---
from sentinel import Sentinel
from crop_model import CropDetectionModel, visualize_prediction
from field_segmentation_model import FieldSegmentationModel, visualize_field_segmentation
from ndvi_prediction_model import NDVIPredictionModel, visualize_ndvi
from tabular_model import TabularNDVIPredictor
from database import db_manager, FavoriteFieldCreate
from authentification import router as auth_router, get_current_user, User

# --- ЗАГРУЗКА ПЕРЕМЕННЫХ ОКРУЖЕНИЯ ---
load_dotenv()

# ==============================================================================
# 1. МОДЕЛИ ДАННЫХ ДЛЯ ЗАПРОСОВ (PYDANTIC)
# ==============================================================================

class BboxRequest(BaseModel):
    """Модель для запросов, работающих с гео-областью (Bounding Box)."""
    bbox: List[float]
    layer_type: Literal["true_color", "ndvi"] = "true_color"
    resolution: int = Field(..., ge=10, le=500)

class SingleDayFeatures(BaseModel):
    """Модель для признаков за один временной шаг (один день) для табличной модели."""
    # Укажите здесь ВСЕ признаки, которые использовались при обучении
    temperature: float
    precipitation: float
    soil_moisture: float
    region: str
    soil_texture: str
    # ... и другие ваши признаки

class NDVIPredictionRequest(BaseModel):
    """Модель для запроса на предсказание NDVI по табличным данным."""
    # Модель ожидает ровно 5 шагов, как при обучении
    sequence: List[SingleDayFeatures] = Field(..., min_items=5, max_items=5)

class FavoriteFieldResponse(FavoriteFieldCreate): # Для избранного
    id: str

# ==============================================================================
# 2. ИНИЦИАЛИЗАЦИЯ FastAPI И КОНФИГУРАЦИЯ
# ==============================================================================

app = FastAPI(title="AgriHealthMap API")
templates = Jinja2Templates(directory="templates")

# --- ПОДКЛЮЧАЕМ РОУТЕР АУТЕНТИФИКАЦИИ ---
# Все эндпоинты из authentication.py (`/register`, `/token`) теперь часть нашего приложения
app.include_router(auth_router)

# --- Конфигурация Sentinel Hub ---
SH_CLIENT_ID = os.getenv("SH_CLIENT_ID")
SH_CLIENT_SECRET = os.getenv("SH_CLIENT_SECRET")
if not SH_CLIENT_ID or not SH_CLIENT_SECRET:
    raise ValueError("Пожалуйста, укажите SH_CLIENT_ID и SH_CLIENT_SECRET в .env файле.")

sentinel = Sentinel(SH_CLIENT_ID, SH_CLIENT_SECRET)
sentinel.set_date('latest')

# ==============================================================================
# 3. ЗАГРУЗКА И ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ
# ==============================================================================

# --- Пути к весам моделей из .env ---
CROP_MODEL_PATH = os.getenv("CROP_MODEL_PATH")
FIELD_MODEL_PATH = os.getenv("FIELD_MODEL_PATH")
NDVI_MODEL_PATH = os.getenv("NDVI_MODEL_PATH")
TABULAR_NDVI_ARTIFACTS_DIR = os.getenv("TABULAR_NDVI_ARTIFACTS_DIR")

# --- Модели для обработки снимков ---
crop_model = CropDetectionModel(model_path=CROP_MODEL_PATH, device='cpu')
field_model = FieldSegmentationModel(model_path=FIELD_MODEL_PATH, device='cpu')
ndvi_model = NDVIPredictionModel(model_path=NDVI_MODEL_PATH, device='cpu') # Используется для расчета NDVI по снимкам

print(f"✓ Crop detection model initialized (weights: {CROP_MODEL_PATH or 'random init'})")
print(f"✓ Field segmentation model initialized (weights: {FIELD_MODEL_PATH or 'random init'})")
print(f"✓ NDVI calculation model initialized (weights: {NDVI_MODEL_PATH or 'random init'})")

# --- Класс-обертка для табличной модели предсказания NDVI ---
class TabularNDVIPredictorWrapper:
    """Инкапсулирует логику загрузки, подготовки данных и предсказания для табличной модели."""
    def __init__(self, artifacts_dir: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        artifacts_path = Path(artifacts_dir)
        
        self.model_path = artifacts_path / "best_model.pth"
        self.scaler_path = artifacts_path / "scaler.pkl"
        self.features_path = artifacts_path / "model_features.pkl"

        if not all([self.model_path.exists(), self.scaler_path.exists(), self.features_path.exists()]):
            raise FileNotFoundError("Один или несколько артефактов (модель, скейлер, признаки) для табличной модели NDVI не найдены.")

        with open(self.features_path, 'rb') as f:
            self.feature_cols = pickle.load(f)
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        numerical_cols_df = pd.DataFrame(columns=self.feature_cols)
        self.numerical_cols = [col for col in numerical_cols_df.columns if not col.startswith('region_') and not col.startswith('soil_') and pd.api.types.is_numeric_dtype(numerical_cols_df[col])]


        self.model = TabularNDVIPredictor(input_features=len(self.feature_cols))
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.weights_loaded = True
        print(f"✓ Tabular NDVI prediction model initialized (weights: {self.model_path})")

    def prepare_data(self, sequence_data: list) -> torch.Tensor:
        """Готовит входные данные, полностью повторяя пайплайн из train/predict."""
        df = pd.DataFrame(sequence_data)
        df = pd.get_dummies(df, columns=['region', 'soil_texture'], prefix=['region', 'soil'], drop_first=True)
        df = df.reindex(columns=self.feature_cols, fill_value=0)
        
        # Фильтруем числовые колонки, которые есть в датафрейме
        cols_to_scale = [col for col in self.numerical_cols if col in df.columns]
        if cols_to_scale:
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        
        data_np = df[self.feature_cols].values.astype(np.float32)
        return torch.from_numpy(data_np).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, sequence_data: list) -> float:
        """Принимает список словарей, возвращает одно предсказанное значение NDVI."""
        input_tensor = self.prepare_data(sequence_data)
        prediction = self.model(input_tensor)
        return prediction.cpu().item()

# --- Инициализация табличной модели ---
tabular_ndvi_model = None
try:
    if TABULAR_NDVI_ARTIFACTS_DIR:
        tabular_ndvi_model = TabularNDVIPredictorWrapper(artifacts_dir=TABULAR_NDVI_ARTIFACTS_DIR)
except Exception as e:
    print(f"✗ Failed to initialize Tabular NDVI model: {e}")


# ==============================================================================
# 4. API ЭНДПОИНТЫ
# ==============================================================================

# --- Вспомогательная функция для конвертации изображений ---
def array_to_base64(arr: np.ndarray, format: str = "JPEG") -> str:
    """Конвертирует numpy-массив в строку base64."""
    img = Image.fromarray(arr)
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# --- Главная страница ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- Получение информации о моделях ---
@app.get("/model-info")
async def model_info():
    """Возвращает информацию о всех загруженных моделях."""
    return JSONResponse(content={
        "crop_detection": {
            "model_name": "Attention U-Net (Crop Detection)",
            "description": "Детекция типов сельскохозяйственных культур по спутниковым снимкам.",
            "weights_loaded": CROP_MODEL_PATH is not None
        },
        "field_segmentation": {
            "model_name": "Attention U-Net (Field Segmentation)",
            "description": "Сегментация границ сельскохозяйственных полей.",
            "weights_loaded": FIELD_MODEL_PATH is not None
        },
        "ndvi_calculation": {
            "model_name": "NDVI Calculator",
            "description": "Вычисление индекса NDVI по каналам B04 и B08 Sentinel-2.",
            "weights_loaded": True # Это не модель, а формула, всегда доступна
        },
        "tabular_ndvi_prediction": { # <-- Информация о новой модели
            "model_name": "Tabular Transformer NDVI Predictor",
            "description": "Предсказание будущего значения NDVI на основе временного ряда табличных данных (погода, почва и т.д.).",
            "weights_loaded": tabular_ndvi_model.weights_loaded if tabular_ndvi_model else False,
            "status": "Ready" if tabular_ndvi_model else "Not Initialized"
        }
    })

# --- Получение спутникового снимка ---
@app.post("/get-image")
async def get_image(request: BboxRequest):
    """Принимает Bbox и возвращает спутниковый снимок в указанном слое."""
    try:
        min_lon, min_lat, max_lon, max_lat = request.bbox
        sentinel_bbox = sentinel.set_aoi(min_lon, min_lat, max_lon, max_lat)
        sentinel.set_resolution(request.resolution)
        size = sentinel.bbox_size

        image_data = sentinel.get_data([request.layer_type])[request.layer_type]
        
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
            "center_lat": round(sentinel_bbox.middle[1], 5),
            "center_lon": round(sentinel_bbox.middle[0], 5),
            "width": size[0],  
            "height": size[1]
        })
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# --- Детекция культур ---
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

        # Новый, исправленный код
        sentinel_bands = {}
        for key, value in band_data.items():
            # Удаляем '.tif' из ключа, чтобы получить чистое имя канала
            # Например, 'B02.tif' -> 'B02'
            band_name = key.replace('.tif', '') 
            sentinel_bands[band_name] = value

        # Запускаем модель детекции
        prediction = crop_model.predict(sentinel_bands, return_probabilities=False)

        # Получаем RGB для визуализации
        rgb_data = sentinel.get_data(['true_color'])
        rgb_image = (rgb_data * 255).astype(np.uint8)

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


# --- Сегментация полей ---
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

        # Новый, исправленный код
        sentinel_bands = {}
        for key, value in band_data.items():
            # Удаляем '.tif' из ключа, чтобы получить чистое имя канала
            # Например, 'B02.tif' -> 'B02'
            band_name = key.replace('.tif', '') 
            sentinel_bands[band_name] = value


        # Запускаем модель сегментации
        prediction = field_model.predict(sentinel_bands, return_probabilities=False)

        # Получаем RGB для визуализации
        rgb_data = sentinel.get_data(['true_color'])
        rgb_image = (rgb_data * 255).astype(np.uint8)

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

# --- Вычисление NDVI по снимку ---
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
    


# --- ПРЕДСКАЗАНИЕ NDVI ПО ТАБЛИЧНЫМ ДАННЫМ (НОВЫЙ ЭНДПОИНТ) ---
@app.post("/predict-tabular-ndvi", summary="Предсказать NDVI по временному ряду")
async def predict_tabular_ndvi(request: NDVIPredictionRequest):
    """
    Принимает последовательность из 5 временных шагов с табличными данными
    и возвращает предсказанное значение NDVI на следующий шаг.
    """
    if not tabular_ndvi_model:
        raise HTTPException(status_code=501, detail="Табличная модель предсказания NDVI не инициализирована на сервере.")
    
    try:
        sequence_dicts = [item.model_dump() for item in request.sequence]
        predicted_ndvi = tabular_ndvi_model.predict(sequence_dicts)
        
        return JSONResponse(content={
            "predicted_ndvi": round(predicted_ndvi, 4),
            "input_sequence_length": len(sequence_dicts)
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера при предсказании NDVI: {str(e)}")

# --- ЗАЩИЩЕННЫЕ ЭНДПОИНТЫ ДЛЯ ИЗБРАННЫХ ПОЛЕЙ ---
# Они импортируют зависимость get_current_user из authentication.py

@app.post("/favorites", response_model=FavoriteFieldResponse, tags=["Favorites"])
async def add_favorite_field(
    field: FavoriteFieldCreate,
    current_user: Annotated[User, Depends(get_current_user)]
):
    """Добавляет новое поле в избранное для текущего пользователя."""
    new_field = db_manager.create_favorite_for_user(current_user.username, field)
    return FavoriteFieldResponse(**new_field)

@app.get("/favorites", response_model=List[FavoriteFieldResponse], tags=["Favorites"])
async def get_all_favorite_fields(
    current_user: Annotated[User, Depends(get_current_user)]
):
    """Возвращает список избранных полей ТОЛЬКО для текущего пользователя."""
    favorites_list = db_manager.get_favorites_for_user(current_user.username)
    return [FavoriteFieldResponse(**fav) for fav in favorites_list]