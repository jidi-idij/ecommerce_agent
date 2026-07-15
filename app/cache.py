"""缓存层 - Redis + 本地兜底"""
import json
import pickle
import hashlib
import time
from typing import List, Optional, Dict, Any

from config.settings import redis_config


class CacheManager:
    """Redis缓存管理器（线程安全单例）"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._redis = None
            cls._instance._ok = True
            cls._instance._local = {}
            cls._instance._ttl = {}
        return cls._instance

    def _redis_client(self):
        if not self._ok or self._redis is not None:
            return self._redis
        try:
            import redis as redis_lib
            r = redis_lib.Redis(
                host=redis_config.host, port=redis_config.port, db=redis_config.db,
                password=redis_config.password,
                socket_connect_timeout=redis_config.socket_connect_timeout,
                socket_timeout=redis_config.socket_timeout, decode_responses=False,
            )
            r.ping()
            self._redis = r
        except Exception:
            self._ok = False
        return self._redis

    def _key(self, prefix: str, query: str) -> str:
        return f"{prefix}{hashlib.md5(query.encode()).hexdigest()[:12]}"

    def _local_get(self, key: str):
        if key in self._local and time.time() < self._ttl.get(key, 0):
            return self._local[key]
        return None

    def _local_set(self, key: str, value, ttl: int = None):
        self._local[key] = value
        self._ttl[key] = time.time() + (ttl or redis_config.cache_ttl)

    def get(self, prefix: str, query: str):
        key = self._key(prefix, query)
        r = self._redis_client()
        if r:
            try:
                data = r.get(key)
                if data:
                    return pickle.loads(data)
            except Exception:
                pass
        return self._local_get(key)

    def set(self, prefix: str, query: str, value, ttl: int = None):
        key = self._key(prefix, query)
        ttl = ttl or redis_config.cache_ttl
        r = self._redis_client()
        if r:
            try:
                r.setex(key, ttl, pickle.dumps(value))
            except Exception:
                pass
        self._local_set(key, value, ttl)

    def stats(self) -> dict:
        return {"redis_connected": self._redis_client() is not None, "local_keys": len(self._local)}


cache = CacheManager()
