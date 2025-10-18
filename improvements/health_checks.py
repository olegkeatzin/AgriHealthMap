"""
Расширенные health checks
"""
import os
import psutil
import torch
from datetime import datetime

def get_system_health():
    """Проверка состояния системы"""

    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)

    # Memory
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    memory_available_gb = memory.available / (1024**3)

    # Disk
    disk = psutil.disk_usage('/')
    disk_percent = disk.percent
    disk_free_gb = disk.free / (1024**3)

    # GPU
    gpu_available = torch.cuda.is_available()
    gpu_info = {}
    if gpu_available:
        gpu_info = {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0)
        }
    else:
        gpu_info = {"available": False}

    return {
        "status": "healthy" if cpu_percent < 90 and memory_percent < 90 else "degraded",
        "timestamp": datetime.now().isoformat(),
        "cpu": {
            "percent": round(cpu_percent, 2),
            "count": psutil.cpu_count()
        },
        "memory": {
            "percent": round(memory_percent, 2),
            "available_gb": round(memory_available_gb, 2),
            "total_gb": round(memory.total / (1024**3), 2)
        },
        "disk": {
            "percent": round(disk_percent, 2),
            "free_gb": round(disk_free_gb, 2),
            "total_gb": round(disk.total / (1024**3), 2)
        },
        "gpu": gpu_info
    }

def check_models_loaded(models: dict):
    """Проверка загрузки моделей"""
    model_status = {}

    for name, model in models.items():
        try:
            # Проверяем, что модель в eval mode и на правильном device
            is_loaded = model is not None
            model_status[name] = {
                "loaded": is_loaded,
                "status": "ok" if is_loaded else "not_loaded"
            }
        except Exception as e:
            model_status[name] = {
                "loaded": False,
                "status": "error",
                "error": str(e)
            }

    return model_status

def check_env_variables():
    """Проверка переменных окружения"""
    required_vars = [
        "SH_CLIENT_ID",
        "SH_CLIENT_SECRET",
        "SECRET_KEY"
    ]

    env_status = {}
    for var in required_vars:
        is_set = bool(os.getenv(var))
        env_status[var] = {
            "set": is_set,
            "status": "ok" if is_set else "missing"
        }

    return env_status

# Использование в main.py:
"""
from improvements.health_checks import get_system_health, check_models_loaded, check_env_variables

@app.get("/health/detailed")
async def detailed_health():
    \"\"\"Детальная проверка здоровья системы\"\"\"

    # Проверка моделей
    models = {
        "crop_model": crop_model,
        "field_model": field_model,
        "ndvi_model": ndvi_model,
        "ndvi_convlstm_model": ndvi_convlstm_model
    }

    return {
        "system": get_system_health(),
        "models": check_models_loaded(models),
        "environment": check_env_variables()
    }
"""
