"""
Тестовый скрипт для проверки интеграции модели детекции культур
"""

import requests
import base64
from PIL import Image
from io import BytesIO
import json

# URL API
BASE_URL = "http://localhost:8000"

def test_model_info():
    """Тест 1: Проверка информации о модели"""
    print("=" * 60)
    print("ТЕСТ 1: Информация о модели")
    print("=" * 60)

    try:
        response = requests.get(f"{BASE_URL}/model-info")
        response.raise_for_status()

        info = response.json()
        print(f"✓ Статус: {response.status_code}")
        print(f"  Модель: {info['model_name']}")
        print(f"  Классов: {info['num_classes']}")
        print(f"  Классы: {', '.join(info['classes'])}")
        print(f"  Веса загружены: {'Да' if info['weights_loaded'] else 'Нет (случайная инициализация)'}")
        print("\n✅ Тест пройден!\n")
        return True

    except Exception as e:
        print(f"\n❌ Ошибка: {e}\n")
        return False


def test_get_image():
    """Тест 2: Получение спутникового изображения"""
    print("=" * 60)
    print("ТЕСТ 2: Получение спутникового изображения")
    print("=" * 60)

    # Краснодарский край (тестовая область)
    bbox = [39.0, 45.0, 39.1, 45.1]

    try:
        response = requests.post(f"{BASE_URL}/get-image", json={
            "bbox": bbox,
            "layer_type": "true_color",
            "resolution": 50  # Низкое разрешение для быстроты
        })
        response.raise_for_status()

        result = response.json()
        print(f"✓ Статус: {response.status_code}")
        print(f"  Bbox: {result['bbox']}")
        print(f"  Центр: {result['center_lat']}, {result['center_lon']}")
        print(f"  Размер: {result['width']}x{result['height']}")
        print(f"  Изображение получено: {len(result['image_base64'])} символов base64")
        print("\n✅ Тест пройден!\n")
        return True

    except Exception as e:
        print(f"\n❌ Ошибка: {e}\n")
        return False


def test_detect_crops(save_images=True):
    """Тест 3: Детекция культур"""
    print("=" * 60)
    print("ТЕСТ 3: Детекция культур")
    print("=" * 60)

    # Краснодарский край (область с сельхозземлями)
    bbox = [39.0, 45.0, 39.15, 45.15]

    try:
        print("Отправка запроса...")
        response = requests.post(f"{BASE_URL}/detect-crops", json={
            "bbox": bbox,
            "layer_type": "true_color",
            "resolution": 50  # Низкое разрешение для скорости
        })
        response.raise_for_status()

        result = response.json()
        print(f"✓ Статус: {response.status_code}")
        print(f"  Bbox: {result['bbox']}")
        print(f"  Размер: {result['width']}x{result['height']}")

        # Статистика по классам
        print("\n  Распределение культур:")
        for class_name in result['class_names']:
            percentage = result['class_distribution'][class_name]
            if percentage > 0:
                bar = "█" * int(percentage / 5)
                print(f"    {class_name:15s}: {percentage:6.2f}% {bar}")

        # Сохраняем изображения
        if save_images:
            print("\n  Сохранение изображений...")

            def save_base64_image(base64_str, filename):
                img_data = base64.b64decode(base64_str)
                img = Image.open(BytesIO(img_data))
                img.save(filename)
                print(f"    ✓ {filename}")

            save_base64_image(result['rgb_image'], 'test_rgb.jpg')
            save_base64_image(result['crop_mask'], 'test_mask.jpg')
            save_base64_image(result['overlay'], 'test_overlay.jpg')

        print("\n✅ Тест пройден!\n")
        return True

    except Exception as e:
        print(f"\n❌ Ошибка: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_full_workflow():
    """Полный рабочий процесс"""
    print("=" * 60)
    print("ПОЛНЫЙ ТЕСТ WORKFLOW")
    print("=" * 60)
    print("\nПроверяем работу всех компонентов...\n")

    results = {
        "Информация о модели": test_model_info(),
        "Получение изображения": test_get_image(),
        "Детекция культур": test_detect_crops(save_images=True)
    }

    # Итоги
    print("=" * 60)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}  {test_name}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        print("\nИнтеграция работает корректно.")
        print("Модель готова к использованию.")
    else:
        print("⚠️  НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОШЛИ")
        print("\nПроверьте:")
        print("  1. Сервер запущен (uvicorn main:app --reload)")
        print("  2. Настроен .env с SH_CLIENT_ID и SH_CLIENT_SECRET")
        print("  3. Установлены все зависимости (pip install -e .)")

    print("=" * 60)


if __name__ == "__main__":
    import sys

    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ИНТЕГРАЦИИ МОДЕЛИ ДЕТЕКЦИИ КУЛЬТУР")
    print("=" * 60)
    print(f"API URL: {BASE_URL}")
    print("=" * 60 + "\n")

    # Проверка доступности сервера
    try:
        response = requests.get(BASE_URL, timeout=5)
        print("✓ Сервер доступен\n")
    except requests.exceptions.ConnectionError:
        print("❌ Ошибка: Сервер недоступен!")
        print("\nЗапустите сервер:")
        print("  uvicorn main:app --reload")
        print("\nИли укажите другой URL:")
        print("  python test_integration.py http://your-server:8000")
        sys.exit(1)

    # Если указан другой URL
    if len(sys.argv) > 1:
        BASE_URL = sys.argv[1]
        print(f"Используется URL: {BASE_URL}\n")

    # Запуск тестов
    test_full_workflow()

    print("\n✨ Готово!\n")
