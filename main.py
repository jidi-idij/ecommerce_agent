"""
智能电商导购Agent - FastAPI主服务
提供API接口和前端页面
"""
import os
import sys
import json
import time
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 加载环境变量配置
from config.settings import load_from_env
load_from_env()

from app.models import AgentState, MOCK_PRODUCTS, Product, generate_mock_products
from app.graph import get_shopping_graph
from app.cache import cache
from app.tools import ProductSearchTool, PriceMonitorTool, InventoryTool


# ==================== FastAPI应用 ====================

app = FastAPI(
    title="智能电商导购Agent",
    description="基于LangGraph的多Agent电商导购系统 - 意图识别→商品检索→排序→生成",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 数据模型 ====================

class ChatRequest(BaseModel):
    """对话请求"""
    message: str = Field(..., description="用户消息")
    thread_id: Optional[str] = Field(default=None, description="会话ID（用于多轮对话）")
    stream: bool = Field(default=True, description="是否流式输出")


class ChatResponse(BaseModel):
    """对话响应"""
    response: str = ""
    products: List[Dict[str, Any]] = Field(default_factory=list)
    intent: Optional[str] = None
    execution_time: float = 0.0
    execution_log: List[Dict] = Field(default_factory=list)


class ProductSearchRequest(BaseModel):
    """商品搜索请求"""
    query: str
    category: Optional[str] = None
    brand: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    limit: int = 20


# ==================== API路由 ====================

@app.get("/", response_class=HTMLResponse)
async def index():
    """首页 - 返回前端页面"""
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    对话接口（非流式）
    完整执行多Agent导购流程，返回最终结果
    """
    start_time = time.time()
    
    # 创建初始状态
    state = AgentState(query=request.message)
    
    # 生成会话ID
    thread_id = request.thread_id or f"thread_{int(time.time() * 1000)}"
    
    print(f"\n{'#'*60}")
    print(f"[API] 新对话 | thread={thread_id} | query=\"{request.message}\"")
    print(f"{'#'*60}")
    
    try:
        # 获取图并执行
        graph = get_shopping_graph()
        
        final_state = await asyncio.to_thread(
            lambda: graph.invoke(
                state,
                config={"configurable": {"thread_id": thread_id}}
            )
        )
        
        execution_time = time.time() - start_time
        
        print(f"[API] 执行完成 | 耗时: {execution_time:.2f}s")
        
        # 兼容 intent 是 UserIntent 对象或 dict 的情况
        intent_raw = final_state.get("intent")
        if intent_raw is None:
            intent_val = None
        elif hasattr(intent_raw, "intent_type"):
            intent_val = intent_raw.intent_type.value
        elif isinstance(intent_raw, dict):
            intent_val = intent_raw.get("intent_type")
        else:
            intent_val = None

        return ChatResponse(
            response=final_state.get("response") or "",
            products=final_state.get("recommended_products") or [],
            intent=intent_val,
            execution_time=round(execution_time, 3),
            execution_log=final_state.get("execution_log") or [],
        )
        
    except Exception as e:
        print(f"[API] 执行失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    对话接口（流式SSE）
    实时返回各节点执行状态和最终结果
    """
    async def event_stream():
        start_time = time.time()
        
        state = AgentState(query=request.message)
        thread_id = request.thread_id or f"thread_{int(time.time() * 1000)}"
        
        # 发送开始事件
        yield f"data: {json.dumps({'type': 'start', 'message': '开始处理...', 'thread_id': thread_id}, ensure_ascii=False)}\n\n"
        
        try:
            graph = get_shopping_graph()
            
            # 流式执行
            for chunk in graph.stream(
                state,
                config={"configurable": {"thread_id": thread_id}},
                stream_mode="values"
            ):
                # 检查是否有新日志
                execution_log = chunk.get("execution_log") or []
                if execution_log:
                    latest_log = execution_log[-1]
                    yield f"data: {json.dumps({'type': 'log', 'node': latest_log.get('node', ''), 'message': latest_log.get('message', '')}, ensure_ascii=False)}\n\n"
                
                # 如果已完成，发送结果
                if chunk.get("is_finished") and chunk.get("response"):
                    yield f"data: {json.dumps({'type': 'complete', 'response': chunk.response, 'products': chunk.recommended_products}, ensure_ascii=False)}\n\n"
            
            execution_time = time.time() - start_time
            yield f"data: {json.dumps({'type': 'done', 'execution_time': round(execution_time, 2)}, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            print(f"[API] 流式执行失败: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/api/search")
async def search_products(request: ProductSearchRequest):
    """商品搜索API"""
    results = ProductSearchTool.search(
        query=request.query,
        category=request.category,
        brand=request.brand,
        min_price=request.min_price,
        max_price=request.max_price,
        limit=request.limit
    )
    return {
        "total": len(results),
        "query": request.query,
        "results": results
    }


@app.get("/api/products/hot")
async def hot_products(limit: int = 10):
    """热门商品"""
    return {
        "products": ProductSearchTool.get_hot_products(limit)
    }


@app.get("/api/products/{product_id}")
async def product_detail(product_id: str):
    """商品详情"""
    detail = ProductSearchTool.get_product_detail(product_id)
    if not detail:
        raise HTTPException(status_code=404, detail="商品不存在")
    return detail


@app.get("/api/products/{product_id}/stock")
async def product_stock(product_id: str):
    """商品库存"""
    return InventoryTool.check_stock(product_id)


@app.get("/api/deals")
async def best_deals(limit: int = 10):
    """最佳优惠"""
    return {
        "deals": PriceMonitorTool.get_best_deals(limit)
    }


@app.get("/api/categories")
async def categories():
    """商品分类"""
    from app.models import ProductCategory
    return {
        "categories": [c.value for c in ProductCategory]
    }


@app.get("/api/cache/stats")
async def cache_stats():
    """缓存统计"""
    return cache.stats()


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "products_count": len(MOCK_PRODUCTS),
        "cache": cache.stats(),
    }


# ==================== 启动 ====================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    print(f"""
{'='*60}
  🛒 智能电商导购Agent 启动中...
{'='*60}
  📍 API文档: http://127.0.0.1:{port}/docs
  🖥️  前端页面: http://127.0.0.1:{port}/
  📊 商品数据: {len(MOCK_PRODUCTS)} 款
  🧠 大模型: {os.getenv("DASHSCOPE_MODEL", "qwen-plus")}
  ⚡ Redis: {'已连接' if cache.redis else '本地缓存'}
{'='*60}
""")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
