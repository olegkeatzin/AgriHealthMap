"""
Batch предсказания для нескольких областей
"""
from typing import List
from pydantic import BaseModel
from fastapi import HTTPException
import asyncio

class BatchNDVIRequest(BaseModel):
    """Запрос на batch предсказание"""
    areas: List[dict]  # [{"bbox": [...], "dates": [...], "resolution": 50}, ...]
    max_concurrent: int = 3  # Максимум параллельных задач

async def predict_single_area(area_data, predictor_func):
    """Предсказание для одной области"""
    try:
        result = await predictor_func(area_data)
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "error": str(e), "area": area_data.get("bbox")}

async def batch_predict(areas: List[dict], predictor_func, max_concurrent=3):
    """Batch предсказание с ограничением параллельности"""

    if len(areas) > 10:
        raise HTTPException(400, "Максимум 10 областей за раз")

    # Создаем semaphore для ограничения параллельности
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_predict(area):
        async with semaphore:
            return await predict_single_area(area, predictor_func)

    # Запускаем все задачи параллельно
    tasks = [limited_predict(area) for area in areas]
    results = await asyncio.gather(*tasks)

    # Подсчет успешных/неудачных
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful

    return {
        "total": len(areas),
        "successful": successful,
        "failed": failed,
        "results": results
    }

# Использование в main.py:
"""
from improvements.batch_predictions import BatchNDVIRequest, batch_predict

@app.post("/batch-predict-ndvi")
async def batch_predict_ndvi(request: BatchNDVIRequest):
    \"\"\"Предсказание NDVI для нескольких областей\"\"\"

    async def predict_wrapper(area_data):
        # Создаем запрос для одной области
        single_request = NDVIConvLSTMRequest(**area_data)
        return await predict_ndvi_convlstm(single_request)

    results = await batch_predict(
        areas=request.areas,
        predictor_func=predict_wrapper,
        max_concurrent=request.max_concurrent
    )

    return results

# Пример запроса:
# POST /batch-predict-ndvi
# {
#     "areas": [
#         {
#             "bbox": [38.9, 45.0, 39.0, 45.1],
#             "dates": ["2024-09-01", ...],
#             "resolution": 50
#         },
#         {
#             "bbox": [39.5, 45.2, 39.6, 45.3],
#             "dates": ["2024-09-01", ...],
#             "resolution": 50
#         }
#     ],
#     "max_concurrent": 2
# }
"""
