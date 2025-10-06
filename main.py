from sentinel import Sentinel
import matplotlib.cm as cm
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse,JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel,Field
from typing import List, Literal
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
    layer_type: Literal["true_color", "ndvi"]
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