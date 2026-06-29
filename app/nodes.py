"""
LangGraph Agent节点层
包含四个核心Agent节点：
1. 意图识别Agent (intent_node)
2. 商品检索Agent (retrieval_node)  
3. 排序Agent (sort_node)
4. 生成Agent (generate_node)
"""
import json
import time
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.models import AgentState, UserIntent, IntentType, MOCK_PRODUCTS, Product
from app.models import search_by_keyword
from app.tools import ProductSearchTool, PriceMonitorTool, InventoryTool, ReviewTool
from app.cache import cache
from config.settings import llm_config, app_config


# ==================== LLM 初始化 ====================

def get_llm(temperature: float = None):
    """获取大模型实例"""
    return ChatOpenAI(
        model=llm_config.model,
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
        temperature=temperature or llm_config.temperature,
        max_tokens=llm_config.max_tokens,
    )


# ==================== 1. 意图识别Agent ====================

def intent_node(state: AgentState) -> AgentState:
    """
    意图识别Agent节点
    分析用户输入，识别购物意图类型、提取关键词和属性
    """
    print(f"\n{'='*50}")
    print(f"[Node] 意图识别Agent - 输入: \"{state.query}\"")
    print(f"{'='*50}")
    
    state.log("intent", "开始意图识别", {"query": state.query})
    
    # 定义系统提示词 - 意图识别专家
    system_prompt = """你是电商导购系统的意图识别专家。请分析用户的购物查询，精准提取搜索参数。

## 意图类型
1. product_search - 商品搜索
2. price_compare - 比价
3. after_sales - 售后咨询
4. recommendation - 求推荐
5. general - 通用闲聊

## 核心规则
- category: 商品分类。用户没说具体分类时填null（如"适合孩子的商品"→null，全库搜）
- brand: 品牌偏好，没有则null
- min_price/max_price: 价格范围。提到"最贵的"→min_price设高值（如8000）；"便宜的"→max_price设低值（如2000）
- keywords: 按空格分隔的搜索关键词，只保留核心词（如"适合给孩子买的商品"→"孩子 儿童 安全 耐用"）
- sort_by: 排序意图。可选：price_desc(最贵优先)/price_asc(最便宜)/rating(评分最高)/sales(销量最高)/default(默认)
- priority: 综合优先级描述

## 输出格式（仅JSON）
{
    "intent_type": "意图类型",
    "keywords": "关键词 用空格分隔",
    "attributes": {
        "category": "分类或null",
        "brand": "品牌或null",
        "min_price": 数字或null,
        "max_price": 数字或null,
        "sort_by": "price_desc/price_asc/rating/sales/default",
        "priority": "优先级描述"
    },
    "confidence": 0.95
}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"用户查询: {state.query}")
    ]
    
    try:
        llm = get_llm(temperature=0.1)  # 意图识别用低温度
        response = llm.invoke(messages)
        
        # 解析JSON
        content = response.content.strip()
        # 处理markdown代码块
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        result = json.loads(content.strip())
        
        # 构建意图对象
        intent = UserIntent(
            intent_type=IntentType(result.get("intent_type", "general")),
            keywords=result.get("keywords", []),
            attributes=result.get("attributes", {}),
            raw_query=state.query,
            confidence=result.get("confidence", 0.5)
        )
        
        state.intent = intent
        state.log("intent", "意图识别完成", {
            "intent_type": intent.intent_type.value,
            "keywords": intent.keywords,
            "attributes": intent.attributes,
            "confidence": intent.confidence
        })
        
        print(f"[Node] 意图: {intent.intent_type.value}, 关键词: {intent.keywords}, 置信度: {intent.confidence}")
        
    except Exception as e:
        print(f"[Node] 意图识别失败: {e}")
        # 兜底：降级为通用搜索
        state.intent = UserIntent(
            intent_type=IntentType.PRODUCT_SEARCH,
            keywords=[state.query],
            raw_query=state.query,
            confidence=0.3
        )
        state.log("intent", "意图识别失败，使用兜底", {"error": str(e)})
    
    return state


# ==================== 2. 商品检索Agent ====================

def retrieval_node(state: AgentState) -> AgentState:
    """
    商品检索Agent节点
    混合召回：关键词搜索(模拟ES) + 向量搜索(模拟语义匹配)
    返回Top50候选商品
    """
    print(f"\n{'='*50}")
    print(f"[Node] 商品检索Agent")
    print(f"{'='*50}")
    
    intent = state.intent
    if not intent:
        state.error = "意图未识别"
        return state
    
    state.log("retrieval", "开始商品检索", {"keywords": intent.keywords, "attributes": intent.attributes})
    
    # 构造查询字符串
    query = " ".join(intent.keywords) if intent.keywords else state.query
    category = intent.attributes.get("category", "")
    brand = intent.attributes.get("brand", "")
    min_price = intent.attributes.get("min_price")
    max_price = intent.attributes.get("max_price")
    
    all_results = []
    
    # ===== 通道1: 关键词搜索（模拟ES）=====
    # category 为大模型识别的null时，全库搜索不做分类过滤
    print(f"[Node] 通道1 - 关键词搜索: \"{query}\" category={category or '全库'}")
    kw_start = time.time()
    
    kw_results = ProductSearchTool.search(
        query=query,
        category=category if category != "null" else None,
        brand=brand if brand != "null" else None,
        min_price=min_price,
        max_price=max_price,
        limit=app_config.top_k_keyword
    )
    
    kw_time = (time.time() - kw_start) * 1000
    print(f"[Node] 关键词搜索: {len(kw_results)}条, {kw_time:.1f}ms")
    
    for item in kw_results:
        all_results.append({
            "product_id": item["id"],
            "score": 1.0,  # 关键词匹配基础分
            "source": "keyword"
        })
    
    # ===== 通道2: 向量搜索（模拟语义召回）=====
    print(f"[Node] 通道2 - 向量语义搜索")
    vec_start = time.time()
    
    # 检查缓存
    cached_ids = cache.get_vector_result(query)
    
    if cached_ids:
        # 使用缓存
        print(f"[Node] 向量缓存命中: {len(cached_ids)}条")
        for pid in cached_ids:
            existing = next((r for r in all_results if r["product_id"] == pid), None)
            if existing:
                existing["score"] += 0.8  # 向量匹配加分
                existing["source"] = "hybrid"
            else:
                all_results.append({
                    "product_id": pid,
                    "score": 0.8,
                    "source": "vector"
                })
    else:
        # 使用大模型做语义相似度排序（模拟向量搜索）
        try:
            llm = get_llm(temperature=0.0)
            
            # 构建商品候选集（关键词搜索的结果 + 同分类商品）
            candidate_ids = set(r["product_id"] for r in all_results)
            
            # 补充同分类商品
            if category:
                for p in MOCK_PRODUCTS:
                    if category in p.category.value and p.id not in candidate_ids:
                        candidate_ids.add(p.id)
            
            # 语义相关性评分
            candidates_text = "\n".join([
                f"{p.id}: {p.title} | {p.description[:60]}"
                for p in MOCK_PRODUCTS if p.id in candidate_ids
            ])
            
            relevance_prompt = f"""请评估以下商品与用户查询的相关性，返回相关商品ID列表。

用户查询: {query}

商品列表:
{candidates_text}

请输出与查询最相关的商品ID（最多{app_config.top_k_vector}个），按相关度排序，格式为JSON数组:
["ID1", "ID2", ...]"""
            
            response = llm.invoke([
                SystemMessage(content="你是商品语义搜索专家，只输出JSON数组格式。"),
                HumanMessage(content=relevance_prompt)
            ])
            
            content = response.content.strip()
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            relevant_ids = json.loads(content.strip())
            
            # 保存到缓存
            cache.set_vector_result(query, relevant_ids)
            
            for idx, pid in enumerate(relevant_ids):
                existing = next((r for r in all_results if r["product_id"] == pid), None)
                relevance_score = 1.0 - (idx * 0.02)  # 排名越靠前分越高
                if existing:
                    existing["score"] += relevance_score
                    existing["source"] = "hybrid"
                else:
                    all_results.append({
                        "product_id": pid,
                        "score": relevance_score,
                        "source": "vector"
                    })
            
            print(f"[Node] 向量召回: {len(relevant_ids)}条")
            
        except Exception as e:
            print(f"[Node] 向量搜索失败: {e}")
            # 兜底：使用关键词搜索结果的分数降序
            pass
    
    vec_time = (time.time() - vec_start) * 1000
    print(f"[Node] 向量搜索耗时: {vec_time:.1f}ms")
    
    # 合并去重，按分数排序
    seen_ids = set()
    merged_results = []
    for r in sorted(all_results, key=lambda x: x["score"], reverse=True):
        if r["product_id"] not in seen_ids:
            seen_ids.add(r["product_id"])
            product = next((p for p in MOCK_PRODUCTS if p.id == r["product_id"]), None)
            if product:
                merged_results.append(product)
    
    state.search_results = merged_results[:app_config.top_k_keyword]
    state.log("retrieval", "检索完成", {
        "total_candidates": len(all_results),
        "unique_results": len(merged_results),
        "final_results": len(state.search_results),
        "kw_time_ms": kw_time,
        "vec_time_ms": vec_time,
    })
    
    print(f"[Node] 检索完成: 去重后{len(merged_results)}条，取Top{len(state.search_results)}")
    
    return state


# ==================== 3. 排序Agent ====================

def sort_node(state: AgentState) -> AgentState:
    """
    排序Agent节点
    多因子加权排序：价格/评分/销量/匹配度
    返回精排后的Top20
    """
    print(f"\n{'='*50}")
    print(f"[Node] 排序Agent")
    print(f"{'='*50}")
    
    if not state.search_results:
        print("[Node] 无检索结果，跳过排序")
        return state
    
    state.log("sort", "开始排序", {"candidates": len(state.search_results)})
    
    sort_by = state.intent.attributes.get("sort_by", "default") if state.intent else "default"
    
    # 根据用户明确的排序意图，选择排序策略
    if sort_by == "price_desc":
        # 最贵优先（如"最贵的手机"）
        print(f"[Node] 排序策略: 价格从高到低")
        state.sorted_products = sorted(state.search_results, key=lambda p: p.price, reverse=True)[:app_config.top_k_rerank]
        state.log("sort", "价格降序完成", {"sort_by": sort_by, "count": len(state.sorted_products)})
        return state
    elif sort_by == "price_asc":
        # 最便宜优先
        print(f"[Node] 排序策略: 价格从低到高")
        state.sorted_products = sorted(state.search_results, key=lambda p: p.price)[:app_config.top_k_rerank]
        state.log("sort", "价格升序完成", {"sort_by": sort_by, "count": len(state.sorted_products)})
        return state
    elif sort_by == "rating":
        # 评分最高
        print(f"[Node] 排序策略: 评分最高")
        state.sorted_products = sorted(state.search_results, key=lambda p: p.rating, reverse=True)[:app_config.top_k_rerank]
        state.log("sort", "评分降序完成", {"sort_by": sort_by, "count": len(state.sorted_products)})
        return state
    elif sort_by == "sales":
        # 销量最高
        print(f"[Node] 排序策略: 销量最高")
        state.sorted_products = sorted(state.search_results, key=lambda p: p.sales_count, reverse=True)[:app_config.top_k_rerank]
        state.log("sort", "销量降序完成", {"sort_by": sort_by, "count": len(state.sorted_products)})
        return state
    
    # ===== 默认：多因子加权排序 =====
    priority = state.intent.attributes.get("priority", "") if state.intent else ""
    
    weights = {
        "price": app_config.sort_weight_price,
        "rating": app_config.sort_weight_rating,
        "sales": app_config.sort_weight_sales,
        "match": app_config.sort_weight_match,
    }
    
    if "价格" in priority or "便宜" in priority or "性价比" in priority:
        weights["price"] = 0.4
        weights["match"] = 0.1
    elif "性能" in priority or "配置" in priority or "拍照" in priority:
        weights["rating"] = 0.45
        weights["match"] = 0.2
    elif "销量" in priority or "热门" in priority:
        weights["sales"] = 0.45
        weights["price"] = 0.1
    
    print(f"[Node] 排序权重: 价格{weights['price']}, 评分{weights['rating']}, 销量{weights['sales']}, 匹配度{weights['match']}")
    
    prices = [p.price for p in state.search_results]
    ratings = [p.rating for p in state.search_results]
    sales_counts = [p.sales_count for p in state.search_results]
    
    min_price, max_price = min(prices), max(prices)
    min_rating, max_rating = min(ratings), max(ratings)
    min_sales, max_sales = min(sales_counts), max(sales_counts)
    
    def normalize(value, min_val, max_val):
        if max_val == min_val:
            return 1.0
        return (value - min_val) / (max_val - min_val)
    
    scored_products = []
    for product in state.search_results:
        price_score = 1 - normalize(product.price, min_price, max_price)
        rating_score = normalize(product.rating, min_rating, max_rating)
        import math
        sales_log = math.log10(max(product.sales_count, 1))
        sales_score = normalize(sales_log, math.log10(max(min_sales, 1)), math.log10(max_sales))
        
        match_score = 0.5
        if state.intent and state.intent.keywords:
            match_count = 0
            search_text = f"{product.title} {product.description} {' '.join(product.tags)}"
            for kw in state.intent.keywords:
                if kw.lower() in search_text.lower():
                    match_count += 1
            match_score = min(match_count / len(state.intent.keywords), 1.0)
        
        final_score = (
            weights["price"] * price_score +
            weights["rating"] * rating_score +
            weights["sales"] * sales_score +
            weights["match"] * match_score
        )
        
        scored_products.append({
            "product": product,
            "final_score": round(final_score, 4),
            "details": {
                "price_score": round(price_score, 3),
                "rating_score": round(rating_score, 3),
                "sales_score": round(sales_score, 3),
                "match_score": round(match_score, 3),
            }
        })
    
    scored_products.sort(key=lambda x: x["final_score"], reverse=True)
    
    # 取TopK
    top_k = app_config.top_k_rerank
    state.sorted_products = [item["product"] for item in scored_products[:top_k]]
    
    state.log("sort", "排序完成", {
        "input_count": len(state.search_results),
        "output_count": len(state.sorted_products),
        "weights": weights,
        "top_scores": [{
            "id": item["product"].id,
            "title": item["product"].title[:20],
            "score": item["final_score"],
            "details": item["details"]
        } for item in scored_products[:5]]
    })
    
    print(f"[Node] 排序完成: Top{len(state.sorted_products)}")
    for item in scored_products[:5]:
        print(f"  #{item['product'].id} {item['product'].title[:25]}... 得分: {item['final_score']}")
    
    return state


# ==================== 4. 生成Agent ====================

def generate_node(state: AgentState) -> AgentState:
    """
    生成Agent节点
    调用大模型生成带购买建议的自然语言回复
    """
    print(f"\n{'='*50}")
    print(f"[Node] 生成Agent")
    print(f"{'='*50}")
    
    intent = state.intent
    products = state.sorted_products
    
    if not products:
        state.response = "抱歉，没有找到符合条件的商品。请尝试调整您的搜索条件，或告诉我更多需求细节。"
        state.is_finished = True
        return state
    
    state.log("generate", "开始生成回复", {"product_count": len(products)})
    
    # 构建商品信息
    product_info_list = []
    for i, p in enumerate(products[:8], 1):  # 取前8个用于生成
        info = f"""{i}. {p.title}
   - 价格: ¥{p.price}（原价¥{p.original_price}，{p.discount_rate*100:.0f}折）
   - 评分: {p.rating}/5.0（销量{p.sales_count}件）
   - 品牌: {p.brand} | 分类: {p.category.value}
   - 库存: {p.stock}件 | 标签: {', '.join(p.tags)}
   - 亮点: {p.description[:80]}"""
        product_info_list.append(info)
    
    product_text = "\n\n".join(product_info_list)
    
    # 获取用户评价摘要
    review_summaries = []
    for p in products[:3]:
        analysis = ReviewTool.analyze_pros_cons(p.id)
        review_summaries.append(f"\n{p.title[:20]}:\n  优点: {', '.join(analysis['pros'][:2])}\n  注意: {', '.join(analysis['cons'][:2])}")
    
    review_text = "".join(review_summaries)
    
    # 构建系统提示词
    system_prompt = """你是专业的电商导购助手，擅长根据用户需求推荐最合适的商品。

## 回复要求
1. 一句话总结用户需求
2. 推荐2-3个商品，每款用一句话说清核心优势和价格
3. 给一个明确的最终推荐
4. 语气亲切像朋友，**总字数严格控制在200字以内**

## 输出格式
不要输出JSON，直接输出自然语言文本。"""

    user_prompt = f"""用户查询: {state.query}
用户意图: {intent.intent_type.value if intent else "unknown"}
关键词: {intent.keywords if intent else []}
属性偏好: {intent.attributes if intent else {}}

候选商品信息:
{product_text}

评价分析:
{review_text}

请为用户生成个性化的导购回复。"""

    try:
        llm = get_llm(temperature=0.7)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        state.response = response.content
        
        # 构造推荐商品列表（用于前端展示）
        state.recommended_products = [p.to_dict() for p in products[:5]]
        
        state.log("generate", "生成完成", {
            "response_length": len(state.response),
            "recommended_count": len(state.recommended_products)
        })
        
        print(f"[Node] 生成完成: {len(state.response)}字")
        
    except Exception as e:
        print(f"[Node] 生成失败: {e}")
        # 兜底回复
        state.response = f"为您找到{len(products)}款相关商品，推荐关注：\n\n"
        for i, p in enumerate(products[:3], 1):
            state.response += f"{i}. **{p.title}** - ¥{p.price}（{p.discount_rate*100:.0f}折）\n   {p.description[:60]}...\n\n"
        state.recommended_products = [p.to_dict() for p in products[:5]]
        state.log("generate", "使用兜底回复", {"error": str(e)})
    
    state.is_finished = True
    return state


# ==================== 特殊意图处理节点 ====================

def price_compare_node(state: AgentState) -> AgentState:
    """比价专用节点"""
    print(f"\n{'='*50}")
    print(f"[Node] 比价Agent")
    print(f"{'='*50}")
    
    query = state.query
    
    # 从查询中提取商品
    found_products = []
    for p in MOCK_PRODUCTS:
        if any(kw in p.title.lower() or kw in p.description.lower() 
               for kw in state.intent.keywords if len(kw) > 1):
            found_products.append(p)
    
    if not found_products:
        # 兜底搜索
        found_products = search_by_keyword(query)[:6]
    
    if len(found_products) < 2:
        state.response = "需要至少两款商品才能比价，请告诉我您想对比的具体商品型号。"
        state.is_finished = True
        return state
    
    # 取Top5进行比价
    compare_products = found_products[:5]
    comparison = PriceMonitorTool.compare_prices([p.id for p in compare_products])
    
    # 构建比价回复
    lines = [f" **{query} 比价结果**\n"]
    lines.append("| 商品 | 价格 | 原价 | 折扣 | 品牌 | 评分 |")
    lines.append("|------|------|------|------|------|------|")
    
    for item in comparison:
        lines.append(f"| {item['title'][:20]}... | ¥{item['price']} | ¥{item['original_price']} | {item['discount']} | {item['brand']} | ⭐{item['rating']} |")
    
    lines.append("\n**💡 购买建议：**")
    
    # 找出最优选择
    best_value = min(comparison, key=lambda x: x['price'])
    best_rated = max(comparison, key=lambda x: x['rating'])
    
    lines.append(f"\n- **最划算**: {best_value['title'][:25]}...（¥{best_value['price']}）")
    if best_rated['id'] != best_value['id']:
        lines.append(f"- **口碑最好**: {best_rated['title'][:25]}...（⭐{best_rated['rating']}）")
    
    state.response = "\n".join(lines)
    state.recommended_products = [p.to_dict() for p in compare_products]
    state.is_finished = True
    
    state.log("price_compare", "比价完成", {"compared_count": len(comparison)})
    return state


def recommendation_node(state: AgentState) -> AgentState:
    """求推荐专用节点"""
    print(f"\n{'='*50}")
    print(f"[Node] 推荐Agent")
    print(f"{'='*50}")
    
    # 先走正常检索流程
    state = retrieval_node(state)
    state = sort_node(state)
    state = generate_node(state)
    
    # 在回复前加上推荐标记
    state.response = "🎯 **专属推荐**\n\n" + state.response
    
    return state


def after_sales_node(state: AgentState) -> AgentState:
    """售后咨询节点"""
    print(f"\n{'='*50}")
    print(f"[Node] 售后Agent")
    print(f"{'='*50}")
    
    system_prompt = """你是电商售后服务助手，帮助用户解决退换货、保修、维修等问题。

## 售后政策
1. 7天无理由退货（未拆封/未激活）
2. 15天质量问题换货
3. 1年官方保修（凭发票）
4. 30天价保服务

## 回复要求
1. 先确认用户的问题类型（退货/换货/保修/其他）
2. 说明具体流程和所需材料
3. 提供预计处理时间
4. 语气耐心友好"""

    try:
        llm = get_llm(temperature=0.5)
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"用户售后咨询: {state.query}")
        ])
        state.response = response.content
    except Exception:
        state.response = """您好！我是售后客服助手。

请问您遇到什么问题？我可以帮您处理：
-  **退货退款** - 7天无理由，需商品未拆封
-  **换货** - 15天内质量问题免费换
-  **保修维修** - 1年官方保修
-  **价保申请** - 30天内降价退差

请描述您的具体情况，我会为您详细解答流程。"""
    
    state.is_finished = True
    state.log("after_sales", "售后回复完成", {})
    return state


def general_chat_node(state: AgentState) -> AgentState:
    """通用闲聊节点"""
    print(f"\n{'='*50}")
    print(f"[Node] 闲聊Agent")
    print(f"{'='*50}")
    
    try:
        llm = get_llm(temperature=0.8)
        response = llm.invoke([
            SystemMessage(content="你是友好的电商助手，可以进行日常对话。如果用户有购物需求，引导其描述需求。"),
            HumanMessage(content=state.query)
        ])
        state.response = response.content
    except Exception:
        greetings = {
            "你好": "您好！我是您的智能导购助手，可以帮您挑选手机、电脑、耳机等数码产品。请问有什么可以帮您的？",
            "谢谢": "不客气！有其他需要随时找我~",
            "再见": "再见！祝您购物愉快~",
        }
        for key, val in greetings.items():
            if key in state.query:
                state.response = val
                break
        else:
            state.response = "您好！我可以帮您搜索商品、比价、推荐产品。请告诉我您想购买什么？📱💻🎧"
    
    state.is_finished = True
    state.log("general", "闲聊回复完成", {})
    return state
