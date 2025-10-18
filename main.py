# --- СТАНДАРТНЫЕ И СТОРОННИЕ БИБЛИОТЕКИ ---
import os
import io
import base64
import traceback
import pickle
from typing import List, Literal, Optional, Annotated
from pathlib import Path
import asyncio

# --- БИБЛИОТЕКИ ДЛЯ РАБОТЫ С ДАННЫМИ И МОДЕЛЯМИ ---
import numpy as np
import pandas as pd
import torch
import matplotlib.cm as cm
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
from ndvi_prediction_model import NDVIPredictionModel, visualize_ndvi
from tabular_model import TabularNDVIPredictor
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

class SingleDayFeatures(BaseModel):
    """Модель для признаков за один временной шаг (один день) для табличной модели."""
    temperature: float
    precipitation: float
    soil_moisture: float
    region: str
    soil_texture: str

class NDVIPredictionRequest(BaseModel):
    """Модель для запроса на предсказание NDVI по табличным данным."""
    sequence: List[SingleDayFeatures] = Field(..., min_items=5, max_items=5)

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
NDVI_MODEL_PATH = os.getenv("NDVI_MODEL_PATH")
TABULAR_NDVI_ARTIFACTS_DIR = os.getenv("TABULAR_NDVI_ARTIFACTS_DIR")

crop_model = CropDetectionModel(model_path=CROP_MODEL_PATH, device='cpu')
field_model = FieldSegmentationModel(model_path=FIELD_MODEL_PATH, device='cpu')
ndvi_model = NDVIPredictionModel(model_path=NDVI_MODEL_PATH, device='cpu')

print(f"✓ Crop detection model initialized (weights: {CROP_MODEL_PATH or 'random init'})")
print(f"✓ Field segmentation model initialized (weights: {FIELD_MODEL_PATH or 'random init'})")
print(f"✓ NDVI calculation model initialized (weights: {NDVI_MODEL_PATH or 'random init'})")

class TabularNDVIPredictorWrapper:
    def __init__(self, artifacts_dir: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        artifacts_path = Path(artifacts_dir)
        
        self.model_path = artifacts_path / "best_model.pth"
        self.scaler_path = artifacts_path / "scaler.pkl"
        self.features_path = artifacts_path / "model_features.pkl"

        if not all([self.model_path.exists(), self.scaler_path.exists(), self.features_path.exists()]):
            raise FileNotFoundError("Один или несколько артефактов для табличной модели NDVI не найдены.")

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
        df = pd.DataFrame(sequence_data)
        df = pd.get_dummies(df, columns=['region', 'soil_texture'], prefix=['region', 'soil'], drop_first=True)
        df = df.reindex(columns=self.feature_cols, fill_value=0)
        
        cols_to_scale = [col for col in self.numerical_cols if col in df.columns]
        if cols_to_scale:
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        
        data_np = df[self.feature_cols].values.astype(np.float32)
        return torch.from_numpy(data_np).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, sequence_data: list) -> float:
        input_tensor = self.prepare_data(sequence_data)
        prediction = self.model(input_tensor)
        return prediction.cpu().item()

tabular_ndvi_model = None
try:
    if TABULAR_NDVI_ARTIFACTS_DIR:
        tabular_ndvi_model = TabularNDVIPredictorWrapper(artifacts_dir=TABULAR_NDVI_ARTIFACTS_DIR)
except Exception as e:
    print(f"✗ Failed to initialize Tabular NDVI model: {e}")

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
        "ndvi_calculation": {"model_name": "NDVI Calculator", "weights_loaded": True},
        "tabular_ndvi_prediction": {"model_name": "Tabular Transformer NDVI Predictor", "weights_loaded": tabular_ndvi_model.weights_loaded if tabular_ndvi_model else False}
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

        ndvi = ndvi_model.calculate_ndvi_from_bands(sentinel_bands)
        health_classification = ndvi_model._classify_vegetation_health(ndvi)
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