"""LangGraph 节点 - 标准化电商导购 Agent"""
from __future__ import annotations

import json
import math
import re
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
    """初始化 LLM 客户端，不设置超时和重试，等待响应即可"""
    return ChatOpenAI(
        model=llm_config.model,
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
        temperature=t or llm_config.temperature,
        max_tokens=llm_config.max_tokens,
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


_GENERIC_TERMS = {"礼物", "礼品", "产品", "商品", "东西", "数码", "电子产品", "通用"}


def _resolve_cat(raw: str) -> tuple[str | None, str, bool]:
    if not raw or str(raw).lower() in ("null", "none", ""):
        return None, "", False
    rl = str(raw).lower().strip()
    if raw in _GENERIC_TERMS or rl in _GENERIC_TERMS:
        return None, "", False  # 泛化词不拦截，按全库检索
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


def _history_text(state: AgentState, max_turns: int = 6) -> str:
    """格式化最近对话历史"""
    hist = getattr(state, "chat_history", None) or []
    lines = [f"{'用户' if h.get('role') == 'user' else '助手'}: {h.get('content', '')[:150]}"
             for h in hist[-max_turns * 2:] if h.get("content")]
    return "\n".join(lines)


def intent_node(state: AgentState) -> dict:
    m = monitor.node_start("intent_node", state.query[:80])
    print(f"\n[Node] 意图识别 | \"{state.query}\"")

    system = """你是电商导购意图识别专家。分析用户查询，输出JSON：
{"intent_type":"product_search/price_compare/after_sales/recommendation/general",
 "keywords":"核心搜索词（中文词之间用空格分隔，如：拍照 5000元）",
 "attributes":{"category":"分类","brand":null,
 "min_price":null,"max_price":null,"sort_by":"default"},"confidence":0.9}
注意：keywords 中不要包含 category 已经识别的品类词（如已识别 category 为手机，则 keywords 中不要再写"手机"）。
价格规则：超过X元→min_price=X；X元以下→max_price=X；X元左右→不填。
促销规则：查询优惠/打折/满减/优惠券/活动等促销信息→product_search，keywords填促销类型（如：优惠 打折），category填null。
送礼规则：适合送XX的礼物/礼品→recommendation，category填null（不要填"礼物"）。
上下文规则：若提供对话历史且当前查询依赖上文（如"便宜点的""第二个怎么样""那款有货吗""换个品牌"），
必须结合历史补全意图——继承上文的 category/brand/价格约束，并把指代词还原成具体商品或品类写入 keywords。"""

    hist_text = _history_text(state)
    if hist_text:
        user_msg = f"对话历史:\n{hist_text}\n\n当前查询: {state.query}"
    else:
        user_msg = f"查询: {state.query}"
    if getattr(state, "memory", ""):
        user_msg = f"用户长期偏好（参考，当前查询优先）: {state.memory}\n\n{user_msg}"

    intent = None
    try:
        r = _llm(0.1).invoke([SystemMessage(content=system), HumanMessage(content=user_msg)])
        content = (r.content or "").strip()
        if not content:
            raise ValueError("LLM返回空内容")

        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content.strip())
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

    cat_val = intent.attributes.get("category") if intent else None
    kw_filtered = [k for k in intent.keywords if k != cat_val] if intent else []
    q = " ".join(kw_filtered) if kw_filtered else state.query

    m = monitor.node_start("retrieval_node", f"kw={q} cat={cat} price={min_p or '不限'}-{max_p or '不限'}")
    print(f"\n[Node] 商品检索 | kw=\"{q}\" cat={cat or '全库'} price={min_p or '不限'}-{max_p or '不限'}")

    t0 = time.time()
    kw_results = ProductSearchTool.search(query=q, category=cat, brand=brand,
                                          min_price=min_p, max_price=max_p, limit=50)
    kw_ms = (time.time() - t0) * 1000
    
    semantic = ProductSearchTool.semantic_search(query=q, category=cat, 
                                                  min_price=min_p, max_price=max_p,
                                                  limit=50)
    sem_ms = (time.time() - t0) * 1000 - kw_ms

    kw_rank = {item["id"]: i for i, item in enumerate(kw_results)}
    sem_rank = {item["id"]: i for i, item in enumerate(semantic)}

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

    if not merged:
        merged = [p for p in MOCK_PRODUCTS if not cat or cat in p.category.value or cat in p.title]

    filtered = [p for p in merged
                if (min_p is None or p.price >= min_p)
                and (max_p is None or p.price <= max_p)
                and (not brand or str(brand).lower() in p.brand.lower())]

    if not filtered and (min_p is not None or max_p is not None or cat or brand):
        filtered = [p for p in MOCK_PRODUCTS
                    if (not cat or cat in p.category.value or cat in p.title)
                    and (min_p is None or p.price >= min_p)
                    and (max_p is None or p.price <= max_p)
                    and (not brand or str(brand).lower() in p.brand.lower())]

    if not filtered:
        msg = _no_product_hint(state, cat, min_p, max_p, brand)
        monitor.node_end(m, "empty", {"hint": msg[:60]})
        return {"search_results": [], "sorted_products": [], "response": msg,
                "is_finished": True, "recommended_products": []}

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
        return {"sorted_prods": sorted_prods}
    elif sb == "sales":
        sorted_prods = sorted(prods, key=lambda p: p.sales_count, reverse=True)[:20]
        monitor.node_end(m, f"sales {len(sorted_prods)}")
        return {"sorted_prods": sorted_prods}

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
    hist_text = _history_text(state)
    user = f"查询: {state.query}\n\n商品:\n{products_text}\n\n请推荐。"
    if hist_text:
        user = f"对话历史:\n{hist_text}\n\n{user}\n（若用户是在上文基础上追问，请承接上文语境回答。）"
    if getattr(state, "memory", ""):
        user = f"用户长期偏好: {state.memory}\n\n{user}\n（可适当贴合用户偏好推荐，但不得违背当前查询的明确约束。）"

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


def _norm_text(s: str) -> str:
    """归一化：去空格、转小写，让「华为Mate60」能匹配「华为 Mate 60 Pro」"""
    return re.sub(r"\s+", "", (s or "").lower())


def _match_for_compare(state: AgentState) -> list:
    """比价商品匹配：关键词直接命中优先，仅品牌命中兜底"""
    terms = [_norm_text(kw) for kw in state.intent.keywords if len(kw) > 1]
    q = _norm_text(state.query)

    strong, weak = [], []
    for p in MOCK_PRODUCTS:
        t = _norm_text(p.title) + _norm_text(p.brand)
        if any(term in t for term in terms):
            strong.append(p)
        elif _norm_text(p.brand) in q:
            weak.append(p)

    found = strong
    if len(found) < 2:
        # 补足：query 中提到的品牌各取销量最高的 1 款
        seen = {p.brand for p in found}
        for p in sorted(weak, key=lambda x: x.sales_count, reverse=True):
            if p.brand not in seen:
                found.append(p)
                seen.add(p.brand)
    return found


def price_compare_node(state: AgentState) -> dict:
    if state.is_finished or getattr(state, "response", None):
        return {"response": state.response, "recommended_products": [], "is_finished": True}

    m = monitor.node_start("price_compare_node", state.query[:80])
    found = _match_for_compare(state)
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

- 🔄 **退货退款** - 7天无理由，商品未拆封
- 🔄 **换货** - 15天内质量问题免费换
- 🔧 **保修维修** - 1年官方保修
- 💰 **价保申请** - 30天内降价退差

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
    
    # 逐级放宽价格区间（±20% → ±50% → 同品类畅销），保证尽量给出替代推荐
    cands = []
    for lo, hi in ((0.8, 1.2), (0.5, 1.5), (0.0, 999)):
        cands = sorted([p for p in MOCK_PRODUCTS
                        if (not cat or cat in p.category.value or cat in p.title)
                        and (max_p is None or p.price <= max_p * hi)
                        and (min_p is None or p.price >= min_p * lo)],
                       key=lambda p: p.sales_count, reverse=True)[:3]
        if cands or not (min_p or max_p):
            break
    
    lines = [reason]
    if cands:
        lines.append("\n💡 为您推荐以下热门商品：")
        for i, p in enumerate(cands, 1):
            lines.append(f"{i}. {p.title} ¥{p.price:.0f} ⭐{p.rating}")
    lines.append("\n您可以换个说法或告诉我更多偏好，我帮您重新找！")
    return "\n".join(lines)