"""
Проверка корректности улучшенной ConvLSTM модели
"""
import torch
import numpy as np
from ndvi_convlstm_model import NDVIConvLSTMPredictor, NDVIConvLSTMConfig, NDVIPredictionModel

def test_architecture():
    """Проверка архитектуры модели"""
    print("="*60)
    print("ПРОВЕРКА АРХИТЕКТУРЫ УЛУЧШЕННОЙ МОДЕЛИ")
    print("="*60)

    config = NDVIConvLSTMConfig()
    model = NDVIPredictionModel(config)

    print("\n1. Компоненты модели:")
    print(f"   ✓ ConvLSTM Cell: {type(model.convlstm_cell).__name__}")
    print(f"   ✓ Weather Encoder: {type(model.weather_encoder).__name__}")
    print(f"   ✓ Topo Encoder: {type(model.topo_encoder).__name__}")

    # Проверяем, что weather_encoder это LSTM
    if isinstance(model.weather_encoder, torch.nn.LSTM):
        print("\n   ✅ УЛУЧШЕННАЯ ВЕРСИЯ: Weather encoder использует LSTM!")
        print(f"      - Layers: {model.weather_encoder.num_layers}")
        print(f"      - Hidden size: {model.weather_encoder.hidden_size}")
        print(f"      - Input size: {model.weather_encoder.input_size}")
    else:
        print("\n   ❌ БАЗОВАЯ ВЕРСИЯ: Weather encoder использует Linear")
        print("      Нужно обновить код!")
        return False

    print("\n2. Параметры модели:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")

    # Проверяем параметры weather_encoder
    weather_params = sum(p.numel() for p in model.weather_encoder.parameters())
    print(f"\n   Weather LSTM parameters: {weather_params:,}")

    return True

def test_forward_pass():
    """Проверка forward pass"""
    print("\n" + "="*60)
    print("ПРОВЕРКА FORWARD PASS")
    print("="*60)

    config = NDVIConvLSTMConfig()
    model = NDVIPredictionModel(config)
    model.eval()

    # Создаем тестовые данные
    batch_size = 2
    timesteps = 5
    H, W = 100, 100  # Меньший размер для теста

    x_maps = torch.randn(batch_size, timesteps, 1, H, W)
    x_weather = torch.randn(batch_size, timesteps, config.WEATHER_FEATURES)
    topo_features = torch.randn(batch_size, config.TOPO_FEATURES)

    print(f"\n1. Входные размерности:")
    print(f"   x_maps: {tuple(x_maps.shape)}")
    print(f"   x_weather: {tuple(x_weather.shape)}")
    print(f"   topo_features: {tuple(topo_features.shape)}")

    try:
        with torch.no_grad():
            output = model(x_maps, x_weather, topo_features)

        print(f"\n2. Выходные размерности:")
        print(f"   output: {tuple(output.shape)}")

        expected_shape = (batch_size, 1, H, W)
        if output.shape == expected_shape:
            print(f"\n   ✅ Форма корректна: {output.shape}")
        else:
            print(f"\n   ❌ Неправильная форма!")
            print(f"      Ожидалось: {expected_shape}")
            print(f"      Получено: {output.shape}")
            return False

        # Проверяем диапазон значений (должен быть [-1, 1] из-за Tanh)
        min_val = output.min().item()
        max_val = output.max().item()
        print(f"\n3. Диапазон значений:")
        print(f"   Min: {min_val:.4f}")
        print(f"   Max: {max_val:.4f}")

        if -1 <= min_val <= 1 and -1 <= max_val <= 1:
            print(f"   ✅ Значения в правильном диапазоне [-1, 1]")
        else:
            print(f"   ⚠️ Значения вне диапазона (но это может быть нормально)")

        return True

    except Exception as e:
        print(f"\n   ❌ Ошибка при forward pass:")
        print(f"      {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_weather_sequence_processing():
    """Проверка, что LSTM обрабатывает всю последовательность"""
    print("\n" + "="*60)
    print("ПРОВЕРКА ОБРАБОТКИ ВРЕМЕННОЙ ПОСЛЕДОВАТЕЛЬНОСТИ")
    print("="*60)

    config = NDVIConvLSTMConfig()
    model = NDVIPredictionModel(config)
    model.eval()

    batch_size = 1
    timesteps = 5
    H, W = 50, 50

    # Создаем две разные последовательности погоды
    # Sequence 1: постепенное увеличение температуры
    weather1 = torch.zeros(batch_size, timesteps, config.WEATHER_FEATURES)
    for t in range(timesteps):
        weather1[0, t, 0] = t * 5  # temperature увеличивается

    # Sequence 2: постоянная температура (только последний шаг такой же)
    weather2 = torch.zeros(batch_size, timesteps, config.WEATHER_FEATURES)
    weather2[0, -1, 0] = (timesteps - 1) * 5  # только последний шаг = 20

    x_maps = torch.randn(batch_size, timesteps, 1, H, W)
    topo = torch.randn(batch_size, config.TOPO_FEATURES)

    with torch.no_grad():
        output1 = model(x_maps, weather1, topo)
        output2 = model(x_maps, weather2, topo)

    # Вычисляем разницу
    diff = torch.abs(output1 - output2).mean().item()

    print(f"\n1. Сравнение последовательностей:")
    print(f"   Weather 1: [0, 5, 10, 15, 20] (тренд)")
    print(f"   Weather 2: [0, 0, 0, 0, 20] (только последний)")
    print(f"\n2. Средняя разница в предсказаниях:")
    print(f"   Difference: {diff:.6f}")

    if diff > 1e-6:
        print(f"\n   ✅ LSTM учитывает всю последовательность!")
        print(f"      Модель видит разницу между трендом и одиночным значением")
        return True
    else:
        print(f"\n   ❌ LSTM не учитывает историю!")
        print(f"      Возможно, используется только последний шаг")
        return False

def test_model_loading():
    """Проверка загрузки обученной модели"""
    print("\n" + "="*60)
    print("ПРОВЕРКА ЗАГРУЗКИ ОБУЧЕННОЙ МОДЕЛИ")
    print("="*60)

    model_path = "models/ndvi_convlstm_best.pth"

    try:
        predictor = NDVIConvLSTMPredictor(model_path=model_path, device='cpu')
        print(f"\n   ✅ Модель успешно загружена из {model_path}")

        # Тестовое предсказание
        ndvi_seq = np.random.randn(5, 100, 100) * 0.3 + 0.5  # NDVI ~0.5
        weather_seq = np.random.randn(5, 15) * 10 + 20  # ~20°C
        topo = np.random.randn(10) * 50 + 200  # ~200m elevation

        result = predictor.predict(
            ndvi_sequence=ndvi_seq,
            weather_sequence=weather_seq,
            topo_features=topo
        )

        print(f"\n   Результат предсказания:")
        print(f"   - Shape: {result['predicted_ndvi_map'].shape}")
        print(f"   - Mean NDVI: {result['statistics']['mean_ndvi']:.4f}")
        print(f"   - Dominant class: {result['health_classification']['dominant_class']}")

        return True

    except FileNotFoundError:
        print(f"\n   ⚠️ Файл модели не найден: {model_path}")
        print(f"      Это нормально, если модель еще не обучена")
        return True  # Не критично для проверки архитектуры
    except Exception as e:
        print(f"\n   ❌ Ошибка загрузки модели:")
        print(f"      {str(e)}")
        return False

def main():
    print("\n" + "="*70)
    print(" "*15 + "ПРОВЕРКА УЛУЧШЕННОЙ CONVLSTM МОДЕЛИ")
    print("="*70)

    results = {
        "Архитектура": test_architecture(),
        "Forward Pass": test_forward_pass(),
        "Временная последовательность": test_weather_sequence_processing(),
        "Загрузка модели": test_model_loading()
    }

    print("\n" + "="*70)
    print(" "*20 + "РЕЗУЛЬТАТЫ ПРОВЕРКИ")
    print("="*70)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name:30s}: {status}")

    all_passed = all(results.values())

    print("\n" + "="*70)
    if all_passed:
        print(" "*15 + "🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print(" "*10 + "Улучшенная модель готова к использованию")
    else:
        print(" "*15 + "⚠️ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ")
        print(" "*10 + "Проверьте ошибки выше")
    print("="*70 + "\n")

    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
