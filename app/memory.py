"""长期记忆模块 - 用户偏好抽取与持久化（独立于业务代码）

从每轮对话中沉淀用户偏好：
  - 常用品类 / 品牌（计数制，取高频）
  - 预算区间（取历史出现过的最严约束）
  - 价格敏感度（性价比/便宜/划算等信号）
  - 人群场景（学生/老人/送礼对象等）

存储：JSON 文件（默认 data/user_memory.json），按 sid 隔离。
注入：summary_text() 生成简短中文摘要，供意图/生成节点放入 prompt。

用法：
    from app import memory
    memory.update(sid, query, intent_attrs)
    text = memory.summary_text(sid)
    profile = memory.get(sid)
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any, Dict, Optional

from app.models import MOCK_PRODUCTS

_VALID_CATS = sorted(set(p.category.value for p in MOCK_PRODUCTS))
_ALL_BRANDS = sorted(set(p.brand for p in MOCK_PRODUCTS))
_BRAND_ALIAS = {"苹果": "Apple", "iphone": "Apple", "ipad": "Apple", "airpods": "Apple",
                "macbook": "Apple", "红米": "Redmi"}

_PRICE_SENSITIVE = {"便宜", "性价比", "划算", "实惠", "平价", "学生价", "优惠", "打折"}
_PERSONAS = {"学生": "学生党", "老人": "送老人", "父母": "送父母", "女朋友": "送女朋友",
             "男朋友": "送男朋友", "孩子": "送孩子", "女生": "女生向", "男生": "男生向",
             "办公": "办公场景", "游戏": "游戏场景", "拍照": "拍照场景"}

_MAX_PROFILES = 500
_lock = threading.Lock()
_path = os.path.join("data", "user_memory.json")
_profiles: Dict[str, dict] = {}
_loaded = False


def _load():
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        if os.path.exists(_path):
            with open(_path, "r", encoding="utf-8") as f:
                _profiles.update(json.load(f))
    except Exception:
        pass


def _save():
    try:
        os.makedirs(os.path.dirname(_path), exist_ok=True)
        with open(_path, "w", encoding="utf-8") as f:
            json.dump(_profiles, f, ensure_ascii=False, indent=1)
    except Exception:
        pass


def _num(v) -> Optional[float]:
    try:
        return float(str(v).replace("元", "").replace(",", ""))
    except (TypeError, ValueError):
        return None


def update(sid: str, query: str, attrs: Optional[dict] = None):
    """每轮对话后调用，沉淀偏好。attrs 为意图节点抽取的 attributes（更可靠的信号源）"""
    if not sid:
        return
    attrs = attrs or {}
    with _lock:
        _load()
        p = _profiles.setdefault(sid, {
            "categories": {}, "brands": {}, "personas": {},
            "budget_min": None, "budget_max": None,
            "price_sensitive": False, "turns": 0, "updated": 0})

        # 品类：意图结果优先，query 直提兜底
        cat = attrs.get("category")
        if cat in _VALID_CATS:
            p["categories"][cat] = p["categories"].get(cat, 0) + 1
        else:
            for c in _VALID_CATS:
                if c in query:
                    p["categories"][c] = p["categories"].get(c, 0) + 1

        # 品牌：意图结果 + query 提及（含别名）
        brand = attrs.get("brand")
        if brand and str(brand).lower() not in ("null", "none", ""):
            p["brands"][str(brand)] = p["brands"].get(str(brand), 0) + 1
        ql = query.lower()
        for b in _ALL_BRANDS:
            if b.lower() in ql:
                p["brands"][b] = p["brands"].get(b, 0) + 1
        for a, b in _BRAND_ALIAS.items():
            if a in ql:
                p["brands"][b] = p["brands"].get(b, 0) + 1

        # 预算：取历史最严约束（用户说过的最低价上限/最高价下限）
        max_p, min_p = _num(attrs.get("max_price")), _num(attrs.get("min_price"))
        if max_p is not None:
            p["budget_max"] = max_p if p["budget_max"] is None else min(p["budget_max"], max_p)
        if min_p is not None:
            p["budget_min"] = min_p if p["budget_min"] is None else max(p["budget_min"], min_p)

        # 价格敏感度 & 人群场景
        if any(w in query for w in _PRICE_SENSITIVE):
            p["price_sensitive"] = True
        for k, label in _PERSONAS.items():
            if k in query:
                p["personas"][label] = p["personas"].get(label, 0) + 1

        p["turns"] += 1
        p["updated"] = time.time()

        if len(_profiles) > _MAX_PROFILES:
            for k in sorted(_profiles, key=lambda x: _profiles[x].get("updated", 0))[:100]:
                del _profiles[k]
        _save()


def get(sid: str) -> dict:
    with _lock:
        _load()
        return dict(_profiles.get(sid, {}))


def list_all(limit: int = 20) -> Dict[str, dict]:
    with _lock:
        _load()
        items = sorted(_profiles.items(), key=lambda x: x[1].get("updated", 0), reverse=True)
        return dict(items[:limit])


def clear(sid: str):
    with _lock:
        _load()
        _profiles.pop(sid, None)
        _save()


def summary_text(sid: str) -> str:
    """生成供 prompt 注入的偏好摘要（无偏好返回空串）"""
    p = get(sid)
    if not p or p.get("turns", 0) == 0:
        return ""
    parts = []
    cats = sorted(p["categories"], key=p["categories"].get, reverse=True)[:2]
    if cats:
        parts.append(f"关注品类:{'、'.join(cats)}")
    brands = sorted(p["brands"], key=p["brands"].get, reverse=True)[:3]
    if brands:
        parts.append(f"偏好品牌:{'、'.join(brands)}")
    if p.get("budget_max") or p.get("budget_min"):
        lo = f"{p['budget_min']:.0f}" if p.get("budget_min") else "不限"
        hi = f"{p['budget_max']:.0f}" if p.get("budget_max") else "不限"
        parts.append(f"历史预算:{lo}-{hi}元")
    if p.get("price_sensitive"):
        parts.append("价格敏感(关注性价比)")
    personas = sorted(p["personas"], key=p["personas"].get, reverse=True)[:2]
    if personas:
        parts.append(f"场景:{'、'.join(personas)}")
    return "；".join(parts)
