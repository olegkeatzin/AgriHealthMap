"""
Тестовый скрипт для проверки ConvLSTM NDVI предсказания
"""
import requests
import json
import sys
import io
from datetime import datetime, timedelta

# Для Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_URL = "http://localhost:8000"

def generate_dates(num_days=5, interval_days=14):
    """Генерация последовательности дат"""
    end_date = datetime.now()
    dates = []
    for i in range(num_days):
        date = end_date - timedelta(days=(num_days - 1 - i) * interval_days)
        dates.append(date.strftime('%Y-%m-%d'))
    return dates

def test_predict_ndvi_convlstm():
    """Тест предсказания NDVI через ConvLSTM"""
    print("\n" + "="*60)
    print("ТЕСТ: Предсказание NDVI через ConvLSTM")
    print("="*60)

    # Проверка доступности сервера
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        print(f"\n[OK] Сервер доступен: {BASE_URL}")
    except requests.exceptions.ConnectionError:
        print(f"\n[ERROR] Сервер недоступен: {BASE_URL}")
        print("   Запустите сервер: python main.py или .\\start_server.bat")
        return

    # Генерируем даты
    dates = generate_dates(num_days=5, interval_days=14)
    print(f"\n[INFO] Сгенерированные даты:")
    for i, date in enumerate(dates, 1):
        print(f"   {i}. {date}")

    # Тестовая область (небольшая область в Краснодарском крае)
    bbox = [38.9, 45.0, 39.0, 45.1]  # [min_lon, min_lat, max_lon, max_lat]

    print(f"\n[INFO] Тестовая область:")
    print(f"   BBox: {bbox}")
    print(f"   Разрешение: 50 метров")

    # Запрос
    request_data = {
        "bbox": bbox,
        "dates": dates,
        "resolution": 50
    }

    print(f"\n[INFO] Отправляем запрос к /predict-ndvi-convlstm...")
    print(f"   Это может занять несколько минут (загрузка 5 спутниковых снимков)...")

    try:
        response = requests.post(
            f"{BASE_URL}/predict-ndvi-convlstm",
            json=request_data,
            timeout=300  # 5 минут timeout
        )

        print(f"\n[INFO] Статус ответа: {response.status_code}")

        if response.status_code == 200:
            result = response.json()

            print(f"\n[OK] Предсказание успешно выполнено!")
            print("\n" + "-"*60)
            print("РЕЗУЛЬТАТЫ:")
            print("-"*60)

            # Статистика
            stats = result.get('statistics', {})
            print(f"\nСтатистика предсказанного NDVI:")
            print(f"   Среднее:  {stats.get('mean_ndvi', 0):.4f}")
            print(f"   Медиана:  {stats.get('median_ndvi', 0):.4f}")
            print(f"   Мин:      {stats.get('min_ndvi', 0):.4f}")
            print(f"   Макс:     {stats.get('max_ndvi', 0):.4f}")
            print(f"   Std Dev:  {stats.get('std_ndvi', 0):.4f}")

            # Классификация здоровья
            health = result.get('health_classification', {})
            print(f"\nКлассификация здоровья растительности:")
            for category, data in health.items():
                if category != 'dominant_class' and isinstance(data, dict):
                    print(f"   {category:15s}: {data.get('percentage', 0):6.2f}% ({data.get('pixels', 0)} пикселей)")

            dominant = health.get('dominant_class', 'unknown')
            print(f"\n   Доминирующий класс: {dominant}")

            # Анализ изменений
            changes = result.get('change_analysis', {})
            print(f"\nАнализ изменений (predicted vs current):")
            print(f"   Улучшение:    {changes.get('improvement_percent', 0):.2f}%")
            print(f"   Ухудшение:    {changes.get('degradation_percent', 0):.2f}%")
            print(f"   Стабильно:    {changes.get('stable_percent', 0):.2f}%")

            # Временной ряд
            timeline = result.get('timeline', {})
            print(f"\nВременной ряд NDVI:")
            ndvi_values = timeline.get('ndvi_values', [])
            timeline_dates = timeline.get('dates', dates)
            for date, ndvi in zip(timeline_dates, ndvi_values):
                print(f"   {date}: {ndvi:.4f}")
            print(f"   ПРЕДСКАЗАНИЕ: {timeline.get('predicted_value', 0):.4f}")

            # Изображения
            print(f"\nПолученные изображения:")
            print(f"   predicted_ndvi:  {len(result.get('predicted_ndvi', ''))} байт (base64)")
            print(f"   current_ndvi:    {len(result.get('current_ndvi', ''))} байт (base64)")
            print(f"   difference_map:  {len(result.get('difference_map', ''))} байт (base64)")

            print(f"\nМетаданные:")
            print(f"   Входные даты:  {result.get('input_dates', [])}")
            print(f"   BBox:          {result.get('bbox', [])}")
            print(f"   Центр:         ({result.get('center_lat', 0):.5f}, {result.get('center_lon', 0):.5f})")
            print(f"   Размер:        {result.get('width', 0)}x{result.get('height', 0)}")

            print("\n" + "="*60)
            print("[OK] ТЕСТ УСПЕШНО ЗАВЕРШЕН!")
            print("="*60)

        else:
            print(f"\n[ERROR] Ошибка предсказания:")
            print(f"   Статус: {response.status_code}")
            print(f"   Ответ: {response.text}")

    except requests.exceptions.Timeout:
        print(f"\n[ERROR] Превышено время ожидания (5 минут)")
        print("   Попробуйте уменьшить область или разрешение")
    except Exception as e:
        print(f"\n[ERROR] Ошибка выполнения запроса:")
        print(f"   {str(e)}")

def test_model_info():
    """Проверка информации о модели"""
    print("\n" + "="*60)
    print("ПРОВЕРКА: Информация о модели")
    print("="*60)

    try:
        response = requests.get(f"{BASE_URL}/model-info", timeout=5)

        if response.status_code == 200:
            info = response.json()
            print(f"\n[OK] Модели загружены:")

            models = info.get('models', {})
            for model_name, model_info in models.items():
                print(f"\n{model_name}:")
                print(f"   Статус:      {model_info.get('status', 'unknown')}")
                print(f"   Параметры:   {model_info.get('parameters', 0):,}")
                if 'path' in model_info:
                    print(f"   Путь:        {model_info.get('path', 'N/A')}")

        else:
            print(f"[WARNING] Не удалось получить информацию о моделях")

    except Exception as e:
        print(f"[WARNING] Ошибка получения информации: {e}")

def main():
    print("="*60)
    print("ТЕСТИРОВАНИЕ CONVLSTM NDVI МОДЕЛИ")
    print("="*60)

    test_model_info()
    test_predict_ndvi_convlstm()

    print("\n" + "="*60)
    print("КАК ИСПОЛЬЗОВАТЬ:")
    print("="*60)
    print("""
1. Запрос:
   POST /predict-ndvi-convlstm

2. Тело запроса (JSON):
   {
       "bbox": [min_lon, min_lat, max_lon, max_lat],
       "dates": ["YYYY-MM-DD", "YYYY-MM-DD", ...],  // 5 дат
       "resolution": 50  // метры на пиксель
   }

3. Ответ:
   {
       "predicted_ndvi": "base64_image",
       "current_ndvi": "base64_image",
       "difference_map": "base64_image",
       "statistics": {...},
       "health_classification": {...},
       "change_analysis": {...},
       "timeline": {...}
   }

Документация API: http://localhost:8000/docs
    """)

if __name__ == "__main__":
    main()
