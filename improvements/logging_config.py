"""
Настройка логирования
"""
import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logging(log_dir="logs", level=logging.INFO):
    """Настройка логирования"""

    # Создаем директорию для логов
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Формат логов
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # Создаем форматтер
    formatter = logging.Formatter(log_format, date_format)

    # Файл для всех логов
    file_handler = logging.FileHandler(
        log_path / f"app_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    # Файл только для ошибок
    error_handler = logging.FileHandler(
        log_path / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
    )
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)

    # Вывод в консоль
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    # Настройка root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)

    return root_logger

# Использование в main.py:
"""
from improvements.logging_config import setup_logging

# В начале файла
logger = setup_logging()

# В коде
@app.post("/predict-ndvi-convlstm")
async def predict_ndvi_convlstm(request):
    logger.info(f"Received prediction request for bbox: {request.bbox}")
    try:
        result = ...
        logger.info(f"Prediction successful, mean NDVI: {result['statistics']['mean_ndvi']}")
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise
"""
