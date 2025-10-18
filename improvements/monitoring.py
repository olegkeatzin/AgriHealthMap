"""
Простой мониторинг и метрики
"""
import time
from functools import wraps
from collections import defaultdict
import json
from datetime import datetime

class SimpleMonitor:
    """Простой мониторинг запросов"""

    def __init__(self):
        self.requests = defaultdict(int)
        self.errors = defaultdict(int)
        self.response_times = defaultdict(list)
        self.start_time = time.time()

    def track_request(self, endpoint):
        """Отслеживание запроса"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                self.requests[endpoint] += 1
                start = time.time()

                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start
                    self.response_times[endpoint].append(duration)
                    return result
                except Exception as e:
                    self.errors[endpoint] += 1
                    raise

            return wrapper
        return decorator

    def get_stats(self):
        """Получить статистику"""
        uptime = time.time() - self.start_time

        stats = {
            "uptime_seconds": uptime,
            "total_requests": sum(self.requests.values()),
            "total_errors": sum(self.errors.values()),
            "endpoints": {}
        }

        for endpoint in self.requests:
            times = self.response_times.get(endpoint, [])
            stats["endpoints"][endpoint] = {
                "requests": self.requests[endpoint],
                "errors": self.errors[endpoint],
                "avg_response_time": sum(times) / len(times) if times else 0,
                "max_response_time": max(times) if times else 0,
                "min_response_time": min(times) if times else 0
            }

        return stats

monitor = SimpleMonitor()

# Использование в main.py:
"""
from improvements.monitoring import monitor

@app.get("/stats")
async def get_stats():
    return monitor.get_stats()

@app.post("/predict-ndvi-convlstm")
@monitor.track_request("/predict-ndvi-convlstm")
async def predict_ndvi_convlstm(request):
    # ...
"""
