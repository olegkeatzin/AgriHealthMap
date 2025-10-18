
import httpx
import asyncio
from typing import List, Dict, Any

# Используем асинхронный HTTP-клиент для параллельных запросов
client = httpx.AsyncClient()

# Словарь для перевода типов почв на русский язык (упрощенный)
SOIL_TYPE_MAP = {
    "Sand": "Песок",
    "Loamy Sand": "Супесь",
    "Sandy Loam": "Песчаный суглинок",
    "Loam": "Суглинок",
    "Silt Loam": "Пылеватый суглинок",
    "Silt": "Пыль",
    "Sandy Clay Loam": "Песчано-глинистый суглинок",
    "Clay Loam": "Глинистый суглинок",
    "Silty Clay Loam": "Пылевато-глинистый суглинок",
    "Sandy Clay": "Песчаная глина",
    "Silty Clay": "Пылеватая глина",
    "Clay": "Глина"
}


async def get_weather_data(lat: float, lon: float) -> Dict[str, Any]:
    """Получает текущие погодные данные из Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": "true"
    }
    try:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json().get("current_weather", {})
        return {
            "temperature": data.get("temperature", "N/A"),
            "windspeed": data.get("windspeed", "N/A"),
            "weathercode": data.get("weathercode", "N/A"),
        }
    except Exception as e:
        print(f"Ошибка получения данных о погоде: {e}")
        return {"error": str(e)}

async def get_soil_data(lat: float, lon: float) -> Dict[str, Any]:
    """Получает данные о почве из Open-Meteo."""
    url = "https://api.open-meteo.com/v1/soil"
    params = {
        "latitude": lat,
        "longitude": lon,
        # Запрашиваем тип почвы по классификации USDA на верхнем слое 0-5см
        "get": "soil_type_0-5cm"
    }
    try:
        response = await client.get(url, params=params)
        response.raise_for_status()
        soil_index = response.json().get("get", {}).get("soil_type_0-5cm", [None])[0]
        soil_type = list(SOIL_TYPE_MAP.keys())[soil_index] if soil_index is not None else "Не определен"
        
        return {
            "type": SOIL_TYPE_MAP.get(soil_type, soil_type),
        }
    except Exception as e:
        print(f"Ошибка получения данных о почве: {e}")
        return {"error": str(e)}

async def get_elevation_data(lat: float, lon: float) -> Dict[str, Any]:
    """Получает данные о высоте над уровнем моря."""
    # Используем другой API специально для рельефа
    url = "https://api.open-elevation.com/api/v1/lookup"
    params = {
        "locations": f"{lat},{lon}"
    }
    try:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json().get("results", [{}])[0]
        return {
            "elevation": data.get("elevation", "N/A"),
        }
    except Exception as e:
        print(f"Ошибка получения данных о рельефе: {e}")
        return {"error": str(e)}

async def get_all_statistics(bbox: List[float]) -> Dict[str, Any]:
    """
    Агрегирует данные о погоде, почве и рельефе для центральной точки bbox.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2

    # Запускаем все запросы параллельно
    weather_task = get_weather_data(center_lat, center_lon)
    soil_task = get_soil_data(center_lat, center_lon)
    elevation_task = get_elevation_data(center_lat, center_lon)

    results = await asyncio.gather(weather_task, soil_task, elevation_task)

    return {
        "weather": results[0],
        "soil": results[1],
        "elevation": results[2]
    }