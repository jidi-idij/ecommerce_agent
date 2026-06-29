"""
Redis缓存层 - 商品向量缓存、热门商品缓存
"""
import json
import pickle
import hashlib
import time
from typing import List, Optional, Dict, Any

from config.settings import redis_config


class CacheManager:
    """Redis缓存管理器"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._redis = None
            cls._instance._local_cache = {}
            cls._instance._local_ttl = {}
            cls._instance._redis_tried = False  # 新增：是否已尝试过连接
        return cls._instance
    
    @property
    def redis(self):
        """惰性连接Redis，只尝试一次"""
        if self._redis is None and not self._redis_tried:
            self._redis_tried = True  # 标记已尝试
            try:
                import redis as redis_lib
                self._redis = redis_lib.Redis(
                    host=redis_config.host,
                    port=redis_config.port,
                    db=redis_config.db,
                    password=redis_config.password,
                    socket_connect_timeout=redis_config.socket_connect_timeout,
                    socket_timeout=redis_config.socket_timeout,
                    decode_responses=False,
                )
                self._redis.ping()
                print("[Cache] Redis连接成功")
            except Exception as e:
                print(f"[Cache] Redis连接失败({e})，使用本地缓存兜底，后续不再重试")
                self._redis = None
        return self._redis
    
    def _make_key(self, prefix: str, query: str) -> str:
        """生成缓存key"""
        hash_val = hashlib.md5(query.encode()).hexdigest()[:12]
        return f"{prefix}{hash_val}"
    
    def _local_get(self, key: str) -> Any:
        """本地缓存读取"""
        if key in self._local_cache:
            if time.time() < self._local_ttl.get(key, 0):
                return self._local_cache[key]
            else:
                del self._local_cache[key]
                del self._local_ttl[key]
        return None
    
    def _local_set(self, key: str, value: Any, ttl: int = None):
        """本地缓存写入"""
        self._local_cache[key] = value
        self._local_ttl[key] = time.time() + (ttl or redis_config.cache_ttl)
    
    def get_vector_result(self, query: str) -> Optional[List[str]]:
        """
        获取向量搜索缓存结果（返回商品ID列表）
        
        Args:
            query: 查询文本
            
        Returns:
            商品ID列表，无缓存返回None
        """
        key = self._make_key("vec:", query)
        
        # 先查Redis
        if self.redis:
            try:
                data = self.redis.get(key)
                if data:
                    return pickle.loads(data)
            except Exception as e:
                print(f"[Cache] Redis读取失败: {e}")
        
        # 再查本地缓存
        return self._local_get(key)
    
    def set_vector_result(self, query: str, product_ids: List[str], ttl: int = None):
        """
        缓存向量搜索结果
        
        Args:
            query: 查询文本
            product_ids: 商品ID列表
            ttl: 过期时间（秒）
        """
        key = self._make_key("vec:", query)
        ttl = ttl or redis_config.cache_ttl
        
        if self.redis:
            try:
                self._redis.setex(key, ttl, pickle.dumps(product_ids))
                print(f"[Cache] 向量搜索结果已缓存: {query[:20]}...")
            except Exception as e:
                print(f"[Cache] Redis写入失败: {e}")
        
        # 同时写入本地缓存
        self._local_set(key, product_ids, ttl)
    
    def get_keyword_result(self, query: str) -> Optional[List[str]]:
        """获取关键词搜索缓存结果"""
        key = self._make_key("kw:", query)
        
        if self.redis:
            try:
                data = self.redis.get(key)
                if data:
                    return pickle.loads(data)
            except Exception:
                pass
        
        return self._local_get(key)
    
    def set_keyword_result(self, query: str, product_ids: List[str], ttl: int = None):
        """缓存关键词搜索结果"""
        key = self._make_key("kw:", query)
        ttl = ttl or redis_config.cache_ttl
        
        if self.redis:
            try:
                self._redis.setex(key, ttl, pickle.dumps(product_ids))
            except Exception:
                pass
        
        self._local_set(key, product_ids, ttl)
    
    def get_hot_products(self, category: str = "all", limit: int = 10) -> Optional[List[Dict]]:
        """
        获取热门商品缓存
        
        Args:
            category: 商品分类
            limit: 数量限制
        """
        key = f"hot:{category}:{limit}"
        
        if self.redis:
            try:
                data = self.redis.get(key)
                if data:
                    return json.loads(data)
            except Exception:
                pass
        
        return self._local_get(key)
    
    def set_hot_products(self, category: str, products: List[Dict], limit: int = 10, ttl: int = 1800):
        """缓存热门商品"""
        key = f"hot:{category}:{limit}"
        
        if self.redis:
            try:
                self._redis.setex(key, ttl, json.dumps(products, ensure_ascii=False).encode())
                print(f"[Cache] 热门商品已缓存: {category}")
            except Exception as e:
                print(f"[Cache] 热门商品缓存失败: {e}")
        
        self._local_set(key, products, ttl)
    
    def get_price_history(self, product_id: str) -> Optional[List[Dict]]:
        """获取商品价格历史"""
        key = f"price:{product_id}"
        
        if self.redis:
            try:
                data = self.redis.get(key)
                if data:
                    return json.loads(data)
            except Exception:
                pass
        
        return self._local_get(key)
    
    def record_price(self, product_id: str, price: float):
        """记录商品价格"""
        key = f"price:{product_id}"
        record = {"price": price, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        
        if self.redis:
            try:
                # 使用列表存储最近30条价格记录
                pipe = self.redis.pipeline()
                pipe.lpush(key, json.dumps(record))
                pipe.ltrim(key, 0, 29)
                pipe.expire(key, 86400 * 30)  # 30天
                pipe.execute()
            except Exception:
                pass
        
        # 本地缓存
        history = self._local_get(key) or []
        history.insert(0, record)
        history = history[:30]
        self._local_set(key, history, 86400 * 30)
    
    def clear_expired(self):
        """清理过期本地缓存"""
        now = time.time()
        expired = [k for k, v in self._local_ttl.items() if v < now]
        for k in expired:
            self._local_cache.pop(k, None)
            self._local_ttl.pop(k, None)
    
    def stats(self) -> Dict[str, Any]:
        """缓存统计"""
        self.clear_expired()
        redis_info = {}
        if self.redis:
            try:
                info = self.redis.info()
                redis_info = {
                    "connected": True,
                    "used_memory_human": info.get("used_memory_human", "N/A"),
                    "connected_clients": info.get("connected_clients", 0),
                    "total_keys": self.redis.dbsize(),
                }
            except Exception as e:
                redis_info = {"connected": False, "error": str(e)}
        else:
            redis_info = {"connected": False}
        
        return {
            "redis": redis_info,
            "local_cache_keys": len(self._local_cache),
        }


# 全局缓存实例
cache = CacheManager()
