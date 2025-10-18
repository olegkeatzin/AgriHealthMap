"""
Простое кэширование для спутниковых данных
"""
import hashlib
import pickle
from pathlib import Path
from datetime import datetime, timedelta

class SentinelCache:
    """Кэш для данных Sentinel Hub"""

    def __init__(self, cache_dir="cache/sentinel", ttl_hours=24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)

    def _get_cache_key(self, bbox, date, bands):
        """Генерация ключа кэша"""
        key_str = f"{bbox}_{date}_{'_'.join(sorted(bands))}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, bbox, date, bands):
        """Получить из кэша"""
        key = self._get_cache_key(bbox, date, bands)
        cache_file = self.cache_dir / f"{key}.pkl"

        if cache_file.exists():
            # Проверка TTL
            file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - file_time < self.ttl:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

        return None

    def set(self, bbox, date, bands, data):
        """Сохранить в кэш"""
        key = self._get_cache_key(bbox, date, bands)
        cache_file = self.cache_dir / f"{key}.pkl"

        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

    def clear_old(self):
        """Очистка старого кэша"""
        now = datetime.now()
        for cache_file in self.cache_dir.glob("*.pkl"):
            file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if now - file_time > self.ttl:
                cache_file.unlink()

# Использование в main.py:
"""
from improvements.caching import SentinelCache

sentinel_cache = SentinelCache()

# В эндпоинте:
cached_data = sentinel_cache.get(bbox, date, bands)
if cached_data:
    return cached_data
else:
    data = sentinel.get_data(bands)
    sentinel_cache.set(bbox, date, bands, data)
    return data
"""
