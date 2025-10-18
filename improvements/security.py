"""
Безопасность и ограничения
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

def setup_cors(app: FastAPI):
    """Настройка CORS"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # В production укажите конкретные домены
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

def setup_rate_limiting(app: FastAPI):
    """Настройка rate limiting"""
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    return limiter

# Использование в main.py:
"""
from improvements.security import setup_cors, setup_rate_limiting

app = FastAPI()

# Настройка безопасности
setup_cors(app)
limiter = setup_rate_limiting(app)

# Использование лимитов
@app.post("/predict-ndvi-convlstm")
@limiter.limit("10/minute")  # Макс 10 запросов в минуту
async def predict_ndvi_convlstm(request: Request, data: NDVIConvLSTMRequest):
    # ...
"""

# ВАЖНО: Установите библиотеку:
# pip install slowapi
