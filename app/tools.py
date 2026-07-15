"""MCP Tools层 - 商品搜索、价格监控、库存查询"""
from __future__ import annotations

import re
import math
import time
import random
from collections import Counter
from typing import List, Optional, Dict, Any

from app.models import Product, MOCK_PRODUCTS, get_product
from app.cache import cache


def _to_vec(text: str) -> Counter:
    words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+', text.lower())
    ngrams = []
    for w in words:
        if re.match(r'^[\u4e00-\u9fff]+$', w) and len(w) >= 2:
            for i in range(len(w) - 1):
                ngrams.append(w[i:i + 2])
    return Counter(words + ngrams)


def _cosine(v1: Counter, v2: Counter) -> float:
    inter = set(v1.keys()) & set(v2.keys())
    if not inter:
        return 0.0
    num = sum(v1[x] * v2[x] for x in inter)
    den = (sum(v ** 2 for v in v1.values()) * sum(v ** 2 for v in v2.values())) ** 0.5
    return num / den if den else 0.0


class ProductSearchTool:
    """商品搜索工具"""

    @staticmethod
    def search(query: str = "", category: str = None, brand: str = None,
               min_price: float = None, max_price: float = None, limit: int = 20) -> List[Dict]:
        cache_key = f"search:{query}:{category}:{brand}:{min_price}:{max_price}"
        cached = cache.get("kw", cache_key)
        if cached is not None:
            return cached

        terms = [t for t in query.split() if len(t) >= 2] if query else []
        results = []
        for p in MOCK_PRODUCTS:
            text = f"{p.title} {p.description} {p.brand} {' '.join(p.tags)}".lower()
            if not terms or any(t in text for t in terms):
                results.append(p)

        if category:
            results = [p for p in results if category in p.category.value or category in p.title]
        if brand:
            results = [p for p in results if brand.lower() in p.brand.lower()]
        if min_price is not None:
            results = [p for p in results if p.price >= min_price]
        if max_price is not None:
            results = [p for p in results if p.price <= max_price]

        out = [p.to_dict() for p in results[:limit]]
        cache.set("kw", cache_key, out, ttl=1800)
        return out

    @staticmethod
    def semantic_search(query: str, category: str = None, limit: int = 20) -> List[Dict]:
        candidates = [p for p in MOCK_PRODUCTS if not category or category in p.category.value or category in p.title]
        if not candidates:
            return []
        qv = _to_vec(query)
        scored = []
        for p in candidates:
            txt = f"{p.title} {p.description} {p.brand} {' '.join(p.tags)}"
            sim = _cosine(qv, _to_vec(txt))
            if sim > 0.01:
                scored.append((sim, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p.to_dict() for _, p in scored[:limit]]

    @staticmethod
    def detail(product_id: str) -> Optional[Dict]:
        p = get_product(product_id)
        return p.to_dict() if p else None

    @staticmethod
    def hot(limit: int = 10) -> List[Dict]:
        key = f"hot:all:{limit}"
        cached = cache.get("hot", key)
        if cached is not None:
            return cached
        out = [p.to_dict() for p in sorted(MOCK_PRODUCTS, key=lambda x: x.sales_count, reverse=True)[:limit]]
        cache.set("hot", key, out, ttl=3600)
        return out

    @staticmethod
    def recommend(product_id: str, limit: int = 5) -> List[Dict]:
        product = get_product(product_id)
        if not product:
            return []
        cands = [p for p in MOCK_PRODUCTS if p.id != product_id]

        def score(p: Product) -> float:
            s = 0
            if p.category == product.category:
                s += 5
            if p.brand == product.brand:
                s += 3
            s += max(0, 2 - abs(p.price - product.price) / max(product.price, 1))
            return s

        cands.sort(key=score, reverse=True)
        return [p.to_dict() for p in cands[:limit]]


class PriceMonitorTool:
    """价格监控工具"""

    @staticmethod
    def compare(product_ids: List[str]) -> List[Dict]:
        results = []
        for pid in product_ids:
            p = get_product(pid)
            if p:
                results.append({
                    "id": p.id, "title": p.title, "price": p.price,
                    "original_price": p.original_price, "discount": f"{p.discount_rate * 100:.0f}%",
                    "brand": p.brand, "rating": p.rating,
                })
        return results

    @staticmethod
    def best_deals(limit: int = 10) -> List[Dict]:
        deals = []
        for p in MOCK_PRODUCTS:
            if p.original_price > p.price:
                rate = (p.original_price - p.price) / p.original_price
                deals.append({"product": p.to_dict(), "drop_rate": round(rate * 100, 1),
                              "drop_amount": round(p.original_price - p.price, 2)})
        deals.sort(key=lambda x: x["drop_rate"], reverse=True)
        return deals[:limit]


class InventoryTool:
    """库存查询工具"""

    @staticmethod
    def check(product_id: str) -> Dict:
        p = get_product(product_id)
        if not p:
            return {"error": "商品不存在"}
        status = "充足" if p.stock > 200 else "紧张" if p.stock > 50 else "告急"
        return {"product_id": product_id, "title": p.title, "stock": p.stock, "status": status, "can_buy": p.stock > 0}


class ReviewTool:
    """评价分析工具"""

    _MOCK_REVIEWS = {
        "P001": [{"rating": 5, "content": "拍照效果惊艳，钛金属质感很好", "tag": "拍照"},
                  {"rating": 5, "content": "A17 Pro性能很强，游戏不卡顿", "tag": "性能"},
                  {"rating": 4, "content": "价格偏高，但物有所值", "tag": "价格"}],
        "P004": [{"rating": 5, "content": "卫星通话功能很实用，信号很好", "tag": "信号"},
                  {"rating": 5, "content": "支持国产，体验不输苹果", "tag": "国产"},
                  {"rating": 4, "content": "续航还可以，充电速度快", "tag": "续航"}],
    }

    @staticmethod
    def reviews(product_id: str, limit: int = 5) -> List[Dict]:
        p = get_product(product_id)
        if not p:
            return []
        revs = ReviewTool._MOCK_REVIEWS.get(product_id, [
            {"rating": 5, "content": "质量很好，物流快", "tag": "综合"},
            {"rating": 4, "content": "性价比不错，推荐购买", "tag": "性价比"},
            {"rating": 5, "content": "使用体验超出预期", "tag": "体验"},
        ])
        return revs[:limit]

    @staticmethod
    def pros_cons(product_id: str) -> Dict[str, List[str]]:
        p = get_product(product_id)
        if not p:
            return {"pros": [], "cons": []}
        pros, cons = [], []
        if p.rating >= 4.7:
            pros.append("用户评分高，口碑优秀")
        if p.sales_count > 100000:
            pros.append("销量火爆，大众认可")
        if p.discount_rate < 0.9:
            pros.append("当前有优惠，性价比高")
        if p.price > 5000:
            cons.append("价格较高，预算需充足")
        if p.stock < 200:
            cons.append("库存紧张，建议尽快下单")
        return {"pros": pros or ["用户反馈良好"], "cons": cons or ["暂无明显缺点"]}
