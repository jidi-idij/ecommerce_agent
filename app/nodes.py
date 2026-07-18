"""LangGraph 节点 - 标准化电商导购 Agent"""
from __future__ import annotations

import json
import math
import time
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.models import AgentState, UserIntent, IntentType, MOCK_PRODUCTS, Product
from app.tools import ProductSearchTool, PriceMonitorTool, InventoryTool, ReviewTool
from app import monitor
from config.settings import llm_config


_CATEGORY_MAP = {
    "电脑": "笔记本", "手提": "笔记本", "手提电脑": "笔记本", "pc": "笔记本", "mac": "笔记本",
    "智能机": "手机", "电话": "手机",
    "ipad": "平板", "pad": "平板",
    "耳麦": "耳机", "耳塞": "耳机", "蓝牙耳机": "耳机",
    "手表": "智能手表", "手环": "智能手表", "watch": "智能手表",
    "摄像机": "相机", "微单": "相机", "单反": "相机",
    "掌机": "游戏机", "ps5": "游戏机", "switch": "游戏机", "xbox": "游戏机",
}
_VALID_CATS = sorted(set(p.category.value for p in MOCK_PRODUCTS))


def _llm(t: float = None):
    return ChatOpenAI(
        model=llm_config.model, api_key=llm_config.api_key,
        base_url=llm_config.base_url,
        temperature=t or llm_config.temperature, max_tokens=llm_config.max_tokens,
    )


def _parse_price(val) -> float | None:
    if val is None or str(val).lower() in ("null", "none", "", "nan"):
        return None
    try:
        if isinstance(val, str):
            val = val.replace("元", "").replace("块", "").replace(",", "").replace("¥", "").replace("$", "").strip()
        return float(val)
    except (ValueError, TypeError):
        return None


def _resolve_cat(raw: str) -> tuple[str | None, str, bool]:
    if not raw or str(raw).lower() in ("null", "none", ""):
        return None, "", False
    rl = str(raw).lower().strip()
    if raw in _VALID_CATS:
        return raw, "", False
    mapped = _CATEGORY_MAP.get(rl)
    if mapped and mapped in _VALID_CATS:
        return mapped, "", False
    for v in _VALID_CATS:
        if raw in v or v in raw:
            return v, "", False
    hint = f"抱歉，暂时没有【{raw}】分类。支持：{'、'.join(_VALID_CATS)}。"
    return None, hint, True


def intent_node(state: AgentState) -> dict: 
    m = monitor.node_start("intent_node", state.query[:80])
    print(f"\n[Node] 意图识别 | \"{state.query}\"")

    system = """你是电商导购意图识别专家。分析用户查询，输出JSON：
{"intent_type":"product_search/price_compare/after_sales/recommendation/general",
 "keywords":"核心搜索词","attributes":{"category":"分类","brand":null,
 "min_price":null,"max_price":null,"sort_by":"default"},"confidence":0.9}
价格规则：超过X元→min_price=X；X元以下→max_price=X；X元左右→不填。"""

    intent = None
    try:
        r = _llm(0.1).invoke([SystemMessage(content=system), HumanMessage(content=f"查询: {state.query}")])
        c = r.content.strip()
        if "```" in c:
            c = c.split("```")[1]
            if c.startswith("json"):
                c = c[4:]
        result = json.loads(c.strip())
        kw = result.get("keywords", "")
        keywords = kw.split() if isinstance(kw, str) else kw
        intent = UserIntent(
            intent_type=IntentType(result.get("intent_type", "general")),
            keywords=keywords, attributes=result.get("attributes", {}),
            raw_query=state.query, confidence=result.get("confidence", 0.5)
        )
        out = f"{intent.intent_type.value} kw={intent.keywords}"
        monitor.node_end(m, output=out, details={"attrs": intent.attributes})
        print(f"[Node] 意图: {out} | 置信度: {intent.confidence}")
    except Exception as e:
        monitor.node_error(m, e)
        intent = UserIntent(intent_type=IntentType.PRODUCT_SEARCH,
                           keywords=[state.query], raw_query=state.query, confidence=0.3)
    return {"intent": intent}


def retrieval_node(state: AgentState) -> dict:
    intent = state.intent
    q = " ".join(intent.keywords) if intent else state.query
    raw_cat = intent.attributes.get("category") if intent else None
    resolved, hint_msg, blocked = _resolve_cat(raw_cat)
    if blocked:
        m = monitor.node_start("retrieval_node", f"blocked={raw_cat}")
        monitor.node_end(m, f"blocked:{raw_cat}")
        return {"search_results": [], "sorted_products": [], "response": hint_msg,
                "is_finished": True, "recommended_products": []}

    cat = resolved
    min_p = _parse_price(intent.attributes.get("min_price")) if intent else None
    max_p = _parse_price(intent.attributes.get("max_price")) if intent else None
    brand = intent.attributes.get("brand") if intent else None
    if brand and str(brand).lower() in ("null", "none", ""):
        brand = None
    if min_p and max_p and min_p > max_p:
        max_p = None

    m = monitor.node_start("retrieval_node", f"kw={q} cat={cat} price={min_p}-{max_p}")
    print(f"\n[Node] 商品检索 | kw=\"{q}\" cat={cat or '全库'} price={min_p or '不限'}-{max_p or '不限'}")

    t0 = time.time()
    kw_results = ProductSearchTool.search(query=q, category=cat, brand=brand,
                                          min_price=min_p, max_price=max_p, limit=50)
    kw_ms = (time.time() - t0) * 1000
    semantic = ProductSearchTool.semantic_search(query=q, category=cat, limit=50)
    sem_ms = (time.time() - t0) * 1000 - kw_ms

    # ===== 融合排序（Fusion）=====
    # 1. 构建排名映射（越靠前排名越小 → 分数越高）
    kw_rank = {item["id"]: i for i, item in enumerate(kw_results)}
    sem_rank = {item["id"]: i for i, item in enumerate(semantic)}

    # 2. 融合分数：关键词通道权重 0.6，语义通道权重 0.4
    FUSION_ALPHA = 0.6
    all_pids = set(kw_rank.keys()) | set(sem_rank.keys())
    products_by_id = {p.id: p for p in MOCK_PRODUCTS}

    def fusion_score(pid: str) -> float:
        kw_score = 1.0 / (kw_rank.get(pid, 999) + 1)
        sem_score = 1.0 / (sem_rank.get(pid, 999) + 1)
        return FUSION_ALPHA * kw_score + (1 - FUSION_ALPHA) * sem_score

    scored = []
    for pid in all_pids:
        prod = products_by_id.get(pid)
        if prod:
            scored.append((fusion_score(pid), prod))
    scored.sort(key=lambda x: x[0], reverse=True)
    merged = [p for _, p in scored]

    # 兜底：双通道都为空时取全库
    if not merged:
        merged = [p for p in MOCK_PRODUCTS if not cat or cat in p.category.value or cat in p.title]

    # 3. 属性过滤
    filtered = [p for p in merged
                if (min_p is None or p.price >= min_p)
                and (max_p is None or p.price <= max_p)
                and (not brand or str(brand).lower() in p.brand.lower())]

    if not filtered:
        msg = _no_product_hint(state, cat, min_p, max_p, brand)
        monitor.node_end(m, "empty", {"hint": msg[:60]})
        return {"search_results": [], "sorted_products": [], "response": msg,
                "is_finished": True, "recommended_products": []}

    # 打印融合排序 Top5
    top5 = [(p.id, fusion_score(p.id)) for p in filtered[:5]]
    monitor.node_end(m, f"{len(filtered)} products",
                     {"kw_ms": round(kw_ms, 1), "sem_ms": round(sem_ms, 1),
                      "fusion_top5": top5})
    return {"search_results": filtered}


def sort_node(state: AgentState) -> dict:
    if state.is_finished or getattr(state, "response", None):
        return {"sorted_products": []}

    m = monitor.node_start("sort_node", f"candidates={len(state.search_results)}")
    if not state.search_results:
        monitor.node_end(m, "empty")
        return {"sorted_products": []}

    sb = state.intent.attributes.get("sort_by", "default") if state.intent else "default"
    prods = state.search_results

    if sb == "price_desc":
        sorted_prods = sorted(prods, key=lambda p: p.price, reverse=True)[:20]
        monitor.node_end(m, f"price_desc {len(sorted_prods)}")
        return {"sorted_products": sorted_prods}
    elif sb == "price_asc":
        sorted_prods = sorted(prods, key=lambda p: p.price)[:20]
        monitor.node_end(m, f"price_asc {len(sorted_prods)}")
        return {"sorted_products": sorted_prods}
    elif sb == "rating":
        sorted_prods = sorted(prods, key=lambda p: p.rating, reverse=True)[:20]
        monitor.node_end(m, f"rating {len(sorted_prods)}")
        return {"sorted_products": sorted_prods}
    elif sb == "sales":
        sorted_prods = sorted(prods, key=lambda p: p.sales_count, reverse=True)[:20]
        monitor.node_end(m, f"sales {len(sorted_prods)}")
        return {"sorted_products": sorted_prods}

    w = {"price": 0.2, "rating": 0.35, "sales": 0.3, "match": 0.15}
    pri = state.intent.attributes.get("priority", "") if state.intent else ""
    if "价格" in pri or "便宜" in pri or "性价比" in pri:
        w = {"price": 0.4, "rating": 0.25, "sales": 0.25, "match": 0.1}

    prices = [p.price for p in prods]
    ratings = [p.rating for p in prods]
    sales = [p.sales_count for p in prods]

    def norm(v, mn, mx):
        return 1.0 if mx == mn else (v - mn) / (mx - mn)

    def score(p: Product) -> float:
        ps = 1 - norm(p.price, min(prices), max(prices))
        rs = norm(p.rating, min(ratings), max(ratings))
        ss = norm(math.log10(max(p.sales_count, 1)),
                  math.log10(max(min(sales), 1)),
                  math.log10(max(sales)))
        ms = 0.5
        if state.intent and state.intent.keywords:
            hits = sum(1 for kw in state.intent.keywords
                       if kw.lower() in f"{p.title} {p.description} {' '.join(p.tags)}".lower())
            ms = min(hits / len(state.intent.keywords), 1.0)
        return w["price"] * ps + w["rating"] * rs + w["sales"] * ss + w["match"] * ms

    scored = sorted(((score(p), p) for p in prods), key=lambda x: x[0], reverse=True)
    sorted_prods = [p for _, p in scored[:20]]
    monitor.node_end(m, f"weighted {len(sorted_prods)}",
                     {"top": [f"{p.id}:{s:.3f}" for s, p in scored[:3]]})
    return {"sorted_products": sorted_prods}


def generate_node(state: AgentState) -> dict:
    if state.is_finished or getattr(state, "response", None):
        return {"response": state.response, "recommended_products": state.recommended_products, "is_finished": True}

    m = monitor.node_start("generate_node", f"products={len(state.sorted_products)}")
    print(f"\n{'='*50}\n[Node] 生成Agent\n{'='*50}")

    if not state.sorted_products:
        monitor.node_end(m, "no_products")
        return {"response": "抱歉，没有找到符合条件的商品。请换个说法试试？", "is_finished": True}

    products_text = "\n".join(
        f"{i+1}. {p.title} ¥{p.price} 评分{p.rating} 销量{p.sales_count}"
        for i, p in enumerate(state.sorted_products[:6])
    )

    system = """你是专业电商导购。按以下格式输出（语气亲切，尽量100字以内。）：

【一句话总结用户需求】
【推荐1-3个商品（商品名 - 价格 - 一句话优势，每个商品后换行）】
【最终推荐XXX，推荐理由】

示例输出：
以下是一些性价比高的拍照手机：

1. 小米14（¥3999）：徕卡影像，骁龙8Gen3，配置均衡
2. Redmi K70 Pro（¥3299）：2K屏+120W快充，性价比之王

最推荐：Redmi K70 Pro，三千价位配置最全。"""
    user = f"查询: {state.query}\n\n商品:\n{products_text}\n\n请推荐。"

    try:
        r = _llm(0.7).invoke([SystemMessage(content=system), HumanMessage(content=user)])
        response = (r.content or "").strip()
        if not response:
            raise ValueError("LLM返回空内容")
        recs = [p.to_dict() for p in state.sorted_products[:5]]
        monitor.node_end(m, f"{len(response)}chars", {"products": len(recs)})
        print(f"[Node] 生成完成: {len(response)}字")
    except Exception as e:
        monitor.node_error(m, e)
        response = f"为您找到{len(state.sorted_products)}款商品：\n" + "\n".join(
            f"{i+1}. {p.title} ¥{p.price}" for i, p in enumerate(state.sorted_products[:3]))
        recs = [p.to_dict() for p in state.sorted_products[:5]]

    return {"response": response, "recommended_products": recs, "is_finished": True}

def price_compare_node(state: AgentState) -> dict:
    if state.is_finished or getattr(state, "response", None):
        return {"response": state.response, "recommended_products": [], "is_finished": True}

    m = monitor.node_start("price_compare_node", state.query[:80])
    found = [p for p in MOCK_PRODUCTS
             if any(kw in p.title.lower() for kw in state.intent.keywords if len(kw) > 1)]
    if len(found) < 2:
        found = MOCK_PRODUCTS[:5]

    comp = PriceMonitorTool.compare([p.id for p in found[:5]])
    lines = [f"📊 **{state.query} 比价**\n\n| 商品 | 价格 | 原价 | 折扣 | 品牌 | 评分 |\n|------|------|------|------|------|------|"]
    for item in comp:
        lines.append(f"| {item['title'][:20]}... | ¥{item['price']} | ¥{item['original_price']} | {item['discount']} | {item['brand']} | ⭐{item['rating']} |")
    bv = min(comp, key=lambda x: x['price'])
    lines.append(f"\n💡 **最划算**: {bv['title'][:25]}...（¥{bv['price']}）")
    recs = [p.to_dict() for p in found[:5]]
    monitor.node_end(m, f"compare {len(comp)}")
    return {"response": "\n".join(lines), "recommended_products": recs, "is_finished": True}


def after_sales_node(state: AgentState) -> dict:
    if state.is_finished or getattr(state, "response", None):
        return {"response": state.response, "is_finished": True}

    m = monitor.node_start("after_sales_node", state.query[:80])
    response = """您好！售后客服为您服务：

- 退货退款 - 7天无理由，商品未拆封
- 换货 - 15天内质量问题免费换
- 保修维修 - 1年官方保修
- 价保申请 - 30天内降价退差

请描述您的具体情况，我会为您详细解答流程。"""
    monitor.node_end(m, "after_sales_reply")
    return {"response": response, "is_finished": True}


def general_chat_node(state: AgentState) -> dict:
    if state.is_finished or getattr(state, "response", None):
        return {"response": state.response, "is_finished": True}

    m = monitor.node_start("general_chat_node", state.query[:80])
    try:
        r = _llm(0.8).invoke([SystemMessage(content="你是友好的电商助手。"),
                               HumanMessage(content=state.query)])
        response = (r.content or "").strip()
        if not response:
            raise ValueError("LLM返回空内容")
    except Exception:
        response = "您好！我可以帮您搜索商品、比价、推荐产品。请告诉我您想购买什么？📱💻🎧"
    monitor.node_end(m, f"{len(response)}chars")
    return {"response": response, "is_finished": True}


def _no_product_hint(state: AgentState, cat, min_p, max_p, brand) -> str:
    filters = []
    if cat:
        filters.append(f"【{cat}】")
    if brand:
        filters.append(f"品牌【{brand}】")
    if min_p:
        filters.append(f"¥{min_p:.0f}元以上")
    if max_p:
        filters.append(f"¥{max_p:.0f}元以下")
    reason = f"抱歉，没有找到符合{' + '.join(filters)}的商品。" if filters else "抱歉，没有找到符合条件的商品。"
    cands = sorted([p for p in MOCK_PRODUCTS if not cat or cat in p.category.value or cat in p.title],
                   key=lambda p: p.sales_count, reverse=True)[:3]
    lines = [reason]
    if cands:
        lines.append("\n💡 为您推荐以下热门商品：")
        for i, p in enumerate(cands, 1):
            lines.append(f"{i}. {p.title} ¥{p.price:.0f} ⭐{p.rating}")
    lines.append("\n您可以换个说法或告诉我更多偏好，我帮您重新找！")
    return "\n".join(lines)