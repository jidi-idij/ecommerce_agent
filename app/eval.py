"""评估模块 - 独立文件，不改动现有业务代码

RAG 层：
  - 规则反推伪标准答案（如「预算2000」→ price<=2000 的商品集合）
  - Recall@K / MRR 量化检索质量
  - 语义扩展类 case（伪标注集为空或过小时判定）用 LLM-as-a-Judge 补充

Agent 层：
  - 结构化字段（意图 / 品类 / 价格）规则直接校验准确率
  - 生成内容沿用 monitor.diagnose 的 LLM 定性诊断
  - 在线指标：impression/click/conversion 事件埋点 → CTR / CVR

用法：
    from app import eval as evalmod
    result = evalmod.evaluate_batch(graph_invoke_fn, cases, use_judge=True)
    evalmod.track_event("click", sid="s1", product_id="P001")
    summary = evalmod.online_summary()
"""
from __future__ import annotations

import re
import json
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.models import MOCK_PRODUCTS, AgentState

# ============================================================
# 规则反推：query → 期望约束（伪标准答案的依据）
# ============================================================

_VALID_CATS = sorted(set(p.category.value for p in MOCK_PRODUCTS))
_ALL_BRANDS = sorted(set(p.brand for p in MOCK_PRODUCTS))

_CAT_ALIAS = {
    "智能手机": "手机", "电脑": "笔记本", "手提电脑": "笔记本", "手提": "笔记本",
    "mac": "笔记本", "ipad": "平板", "pad": "平板",
    "耳麦": "耳机", "耳塞": "耳机", "蓝牙耳机": "耳机",
    "手表": "智能手表", "手环": "智能手表",
    "摄像机": "相机", "微单": "相机", "单反": "相机",
    "掌机": "游戏机", "ps5": "游戏机", "switch": "游戏机",
}


@dataclass
class Expected:
    """规则反推出的期望约束"""
    intent_type: str = ""
    category: Optional[str] = None
    brand: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    price_approx: bool = False  # 「X元左右」类：结构化字段允许留空，改为校验回复商品价格区间
    semantic: bool = False  # 语义扩展类（伪标注集太小，规则评估不可靠）

    def to_dict(self) -> dict:
        return {"intent_type": self.intent_type, "category": self.category,
                "brand": self.brand, "min_price": self.min_price,
                "max_price": self.max_price, "price_approx": self.price_approx,
                "semantic": self.semantic}


def _expected_intent(query: str) -> str:
    if re.search(r"哪个好|对比|比价|哪个更|区别", query):
        return "price_compare"
    if re.search(r"退货|换货|维修|保修|售后|客服|价保", query):
        return "after_sales"
    if re.fullmatch(r"(你好|在吗|哈喽|hi|hello|请问有人吗)[？?~！!]*", query.strip()):
        return "general"
    if re.search(r"推荐|适合|预算", query):
        return "recommendation"
    return "product_search"


def _expected_category(query: str) -> Optional[str]:
    for c in _VALID_CATS:
        if c in query:
            return c
    ql = query.lower()
    for alias, cat in _CAT_ALIAS.items():
        if alias in ql or alias in query:
            return cat
    return None


def _expected_brand(query: str) -> Optional[str]:
    for b in _ALL_BRANDS:
        if b.lower() in query.lower():
            return b
    alias = {"苹果": "Apple", "iphone": "Apple", "airpods": "Apple", "macbook": "Apple"}
    for a, b in alias.items():
        if a in query.lower():
            return b
    return None


def _expected_price(query: str) -> tuple[Optional[float], Optional[float]]:
    """返回 (min_price, max_price)"""
    m = re.search(r"(\d{3,6})\s*[-~到]\s*(\d{3,6})\s*元?", query)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"(?:预算|价格|价位)?.{0,4}?(\d{3,6})\s*元?\s*(?:以下|以内|之内|不超|低于)", query)
    if m:
        return None, float(m.group(1))
    m = re.search(r"(?:预算)(\d{3,6})", query)
    if m:  # 「预算2000」按 2000 以下处理
        return None, float(m.group(1))
    m = re.search(r"(?:超过|高于|以上)\s*(\d{3,6})\s*元?", query)
    if m:
        return float(m.group(1)), None
    m = re.search(r"(\d{3,6})\s*元?\s*左右", query)
    if m:
        v = float(m.group(1))
        return v * 0.8, v * 1.2
    return None, None


def build_expected(query: str) -> Expected:
    """从 query 文本规则反推期望约束"""
    min_p, max_p = _expected_price(query)
    return Expected(intent_type=_expected_intent(query),
                    category=_expected_category(query),
                    brand=_expected_brand(query),
                    min_price=min_p, max_price=max_p,
                    price_approx="左右" in query)


def pseudo_ground_truth(exp: Expected) -> List[str]:
    """按期望约束过滤商品库，得到伪标准答案商品 id 列表"""
    ids = []
    for p in MOCK_PRODUCTS:
        if exp.category and exp.category not in p.category.value and exp.category not in p.title:
            continue
        if exp.brand and exp.brand.lower() not in p.brand.lower():
            continue
        if exp.min_price is not None and p.price < exp.min_price:
            continue
        if exp.max_price is not None and p.price > exp.max_price:
            continue
        ids.append(p.id)
    return ids


# ============================================================
# RAG 指标：Recall@K / MRR
# ============================================================

def recall_at_k(ranked_ids: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    hit = len(set(ranked_ids[:k]) & set(relevant))
    return round(hit / len(relevant), 4)


def mrr(ranked_ids: List[str], relevant: List[str], k: int = 20) -> float:
    rel = set(relevant)
    for i, pid in enumerate(ranked_ids[:k]):
        if pid in rel:
            return round(1.0 / (i + 1), 4)
    return 0.0


# ============================================================
# Agent 结构化字段校验
# ============================================================

def check_structured(final: dict, exp: Expected) -> Dict[str, Any]:
    """意图 / 品类 / 价格 三字段规则校验"""
    checks: Dict[str, Any] = {}
    intent_raw = final.get("intent")
    actual_intent = ""
    attrs: dict = {}
    if intent_raw is not None:
        if hasattr(intent_raw, "intent_type"):
            actual_intent = intent_raw.intent_type.value
            attrs = getattr(intent_raw, "attributes", {}) or {}
        elif isinstance(intent_raw, dict):
            actual_intent = intent_raw.get("intent_type", "")
            attrs = intent_raw.get("attributes", {}) or {}

    # 意图：recommendation 与 product_search 视为同族（系统路由相同）
    ok_intent = (actual_intent == exp.intent_type or
                 {actual_intent, exp.intent_type} <= {"recommendation", "product_search"})
    checks["intent"] = {"expected": exp.intent_type, "actual": actual_intent, "ok": ok_intent}

    # 品类：仅当期望有品类时校验
    if exp.category:
        actual_cat = attrs.get("category") or ""
        checks["category"] = {"expected": exp.category, "actual": actual_cat,
                              "ok": exp.category in str(actual_cat)}

    # 价格：校验 max_price（最常见）与 min_price
    # 「X元左右」类：意图规则本就要求留空，不做结构化校验，改为在 evaluate_case 中校验回复商品价格区间
    if exp.price_approx:
        return checks

    def _num(v):
        try:
            return float(str(v).replace("元", "").replace(",", ""))
        except (TypeError, ValueError):
            return None

    if exp.max_price is not None:
        actual_max = _num(attrs.get("max_price"))
        checks["max_price"] = {"expected": exp.max_price, "actual": actual_max,
                               "ok": actual_max is not None and abs(actual_max - exp.max_price) / exp.max_price <= 0.25}
    if exp.min_price is not None:
        actual_min = _num(attrs.get("min_price"))
        checks["min_price"] = {"expected": exp.min_price, "actual": actual_min,
                               "ok": actual_min is not None and abs(actual_min - exp.min_price) / max(exp.min_price, 1) <= 0.25}
    return checks


def check_price_range(final: dict, exp: Expected) -> Optional[Dict[str, Any]]:
    """「X元左右」类：校验推荐商品价格是否落在期望区间内（至少一款命中即通过）"""
    if not exp.price_approx or (exp.min_price is None and exp.max_price is None):
        return None
    recs = final.get("recommended_products") or []
    prices = []
    for r in recs:
        try:
            prices.append(float(r.get("price", r["price"]) if isinstance(r, dict) else r.price))
        except (KeyError, TypeError, ValueError):
            continue
    if not prices:
        return {"expected": [exp.min_price, exp.max_price], "actual": None, "ok": False}
    hit = [v for v in prices
           if (exp.min_price is None or v >= exp.min_price)
           and (exp.max_price is None or v <= exp.max_price)]
    return {"expected": [exp.min_price, exp.max_price], "actual": prices[:5], "ok": len(hit) > 0}


# ============================================================
# LLM-as-a-Judge（语义扩展类补充评估）
# ============================================================

_JUDGE_PROMPT = """你是电商导购质量评估员。给定用户查询、系统回复和商品库事实，打分 0/1/2：
2=完全满足需求（商品相关、理由合理、格式清晰）
1=部分满足（有相关商品但理由牵强，或格式混乱）
0=未满足（答非所问/商品与需求无关）
注意：以【商品库事实】为准——库中不存在的商品/价格不要臆测；若库中确实无满足条件的商品，系统如实说明并给出替代推荐不应判0分。
只输出 JSON：{{"score": 0|1|2, "reason": "一句话理由"}}

查询: {query}
商品库事实: {context}
回复: {response}"""


def _judge_context(exp: Expected, relevant: List[str]) -> str:
    """给 Judge 的商品库事实：满足约束的商品（前8条）+ 全库价格区间"""
    by_id = {p.id: p for p in MOCK_PRODUCTS}
    lines = [f"{by_id[i].title} ¥{by_id[i].price:.0f}" for i in relevant[:8] if i in by_id]
    fact = f"满足约束的商品共{len(relevant)}款" + ("：" + "；".join(lines) if lines else "（无）")
    prices = [p.price for p in MOCK_PRODUCTS]
    cats = "、".join(f"{c}({sum(1 for p in MOCK_PRODUCTS if p.category.value == c)}款)"
                     for c in _VALID_CATS)
    return f"{fact}。全库{len(MOCK_PRODUCTS)}款，价格区间¥{min(prices):.0f}-¥{max(prices):.0f}，分类：{cats}"


def llm_judge(query: str, response: str, context: str = "") -> Dict[str, Any]:
    """语义扩展类 case 的 LLM 评分，失败返回 None"""
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        from config.settings import llm_config
        client = ChatOpenAI(model=llm_config.model, api_key=llm_config.api_key,
                            base_url=llm_config.base_url, temperature=0.1, max_tokens=200)
        r = client.invoke([HumanMessage(content=_JUDGE_PROMPT.format(
            query=query, context=context or "（未提供）", response=(response or "")[:600]))])
        text = (r.content or "").strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text.strip())
        return {"score": int(data.get("score", 0)), "reason": str(data.get("reason", ""))[:120]}
    except Exception as e:
        return {"score": None, "reason": f"judge_failed: {e}"}


# ============================================================
# 单条 / 批量评估
# ============================================================

def evaluate_case(query: str, final: dict, use_judge: bool = False) -> Dict[str, Any]:
    """评估单条 graph 执行结果"""
    exp = build_expected(query)
    relevant = pseudo_ground_truth(exp)
    exp.semantic = len(relevant) < 3  # 伪标注集过小 → 判定为语义扩展类

    # 排序结果作为系统输出排序列表
    prods = final.get("sorted_products") or final.get("search_results") or []
    ranked_ids = [p.id if hasattr(p, "id") else p.get("id", "") for p in prods]

    rec: Dict[str, Any] = {"query": query, "expected": exp.to_dict(),
                           "relevant_cnt": len(relevant),
                           "response": final.get("response", "")}
    # 无约束查询：伪标注集=全库，Recall 口径失真，不计 RAG 指标
    unconstrained = len(relevant) >= len(MOCK_PRODUCTS)
    rec["unconstrained"] = unconstrained
    # RAG 指标仅对检索类意图计算（售后/闲聊/比价不走检索链路）
    if (not exp.semantic and not unconstrained
            and exp.intent_type in ("product_search", "recommendation")):
        rec["rag"] = {"recall@5": recall_at_k(ranked_ids, relevant, 5),
                      "recall@10": recall_at_k(ranked_ids, relevant, 10),
                      "recall@20": recall_at_k(ranked_ids, relevant, 20),
                      "mrr": mrr(ranked_ids, relevant)}
    rec["agent"] = check_structured(final, exp)
    pr = check_price_range(final, exp)
    if pr is not None:
        rec["agent"]["price_range"] = pr
    if use_judge and (exp.semantic or unconstrained or exp.intent_type in ("recommendation", "general")):
        rec["judge"] = llm_judge(query, final.get("response", ""),
                                 _judge_context(exp, relevant))
    return rec


def evaluate_batch(graph_invoke_fn, cases: List[str], use_judge: bool = False,
                   sleep_s: float = 0.3) -> Dict[str, Any]:
    """批量评估：逐条跑 graph + 评估，汇总指标"""
    details = []
    for i, q in enumerate(cases):
        try:
            final = graph_invoke_fn(AgentState(query=q))
            details.append(evaluate_case(q, final, use_judge))
        except Exception as e:
            details.append({"query": q, "error": str(e)})
        if i < len(cases) - 1:
            time.sleep(sleep_s)

    valid = [d for d in details if not d.get("error")]
    rag_valid = [d["rag"] for d in valid if "rag" in d]
    judged = [d["judge"] for d in valid if d.get("judge", {}).get("score") is not None]

    def _acc(key: str) -> Optional[float]:
        vals = [d["agent"][key]["ok"] for d in valid if key in d.get("agent", {})]
        return round(sum(vals) / len(vals), 4) if vals else None

    def _avg(items, key):
        return round(sum(x[key] for x in items) / len(items), 4) if items else None

    return {
        "total": len(cases), "evaluated": len(valid),
        "rag": {"cases": len(rag_valid),
                "recall@5": _avg(rag_valid, "recall@5"),
                "recall@10": _avg(rag_valid, "recall@10"),
                "recall@20": _avg(rag_valid, "recall@20"),
                "mrr": _avg(rag_valid, "mrr")},
        "agent": {"intent_acc": _acc("intent"),
                  "category_acc": _acc("category"),
                  "max_price_acc": _acc("max_price"),
                  "min_price_acc": _acc("min_price"),
                  "price_range_acc": _acc("price_range")},
        "judge": {"cases": len(judged),
                  "avg_score": round(sum(j["score"] for j in judged) / len(judged), 3) if judged else None,
                  "pass_rate(score>=1)": round(sum(1 for j in judged if j["score"] >= 1) / len(judged), 4) if judged else None},
        "details": details,
    }


# ============================================================
# 在线指标：impression / click / conversion → CTR / CVR
# ============================================================

_events: List[dict] = []
_events_lock = threading.Lock()
_MAX_EVENTS = 10000


def track_event(event_type: str, sid: str = "", product_id: str = "", meta: dict | None = None):
    """埋点：event_type ∈ impression / click / conversion"""
    if event_type not in ("impression", "click", "conversion"):
        return {"ok": False, "err": "event_type must be impression/click/conversion"}
    with _events_lock:
        _events.append({"t": time.time(), "type": event_type, "sid": sid,
                        "pid": product_id, "meta": meta or {}})
        if len(_events) > _MAX_EVENTS:
            del _events[:_MAX_EVENTS // 2]
    return {"ok": True}


def online_summary() -> dict:
    with _events_lock:
        imp = sum(1 for e in _events if e["type"] == "impression")
        clk = sum(1 for e in _events if e["type"] == "click")
        cnv = sum(1 for e in _events if e["type"] == "conversion")
    return {"impressions": imp, "clicks": clk, "conversions": cnv,
            "CTR": round(clk / imp, 4) if imp else None,
            "CVR": round(cnv / clk, 4) if clk else None}