"""智能电商导购Agent v4.0 - FastAPI 入口"""
import os
import time
import json
import asyncio
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.settings import load_from_env
load_from_env()

from app.models import AgentState, ProductCategory
from app.graph import get_graph
from app.cache import cache
from app.tools import ProductSearchTool, PriceMonitorTool, InventoryTool
import app.monitor as mon

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="智能电商导购Agent v4.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ============ 请求模型 ============

class ChatReq(BaseModel):
    message: str
    thread_id: Optional[str] = None


class BatchReq(BaseModel):
    count: int = Field(default=10, ge=1, le=30)
    custom_prompt: str = ""


class ModeReq(BaseModel):
    enable: bool


# ============ 页面 ============

@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(BASE_DIR, "frontend", "index.html"), "r", encoding="utf-8") as f:
        return f.read()


@app.get("/dev", response_class=HTMLResponse)
async def dev_page():
    with open(os.path.join(BASE_DIR, "frontend", "index.html"), "r", encoding="utf-8") as f:
        return f.read()


# ============ 模式切换 ============

@app.post("/api/dev/mode")
async def set_mode(req: ModeReq):
    mon.set_dev_mode(req.enable)
    return {"dev_mode": req.enable}


@app.get("/api/dev/mode")
async def get_mode():
    return {"dev_mode": mon.is_dev_mode()}


# ============ 对话 ============

@app.post("/api/chat")
async def chat(req: ChatReq):
    t0 = time.time()
    sid = req.thread_id or f"s{int(time.time() * 1000)}"
    state = AgentState(query=req.message)
    mon.start_session(sid, req.message)

    try:
        graph = get_graph()
        final = await asyncio.to_thread(
            lambda: graph.invoke(state, config={"configurable": {"thread_id": sid}})
        )
    except Exception as e:
        mon.finish_session()
        raise HTTPException(500, str(e))

    intent_raw = final.get("intent") if isinstance(final, dict) else getattr(final, "intent", None)
    intent_str = ""
    if intent_raw and hasattr(intent_raw, "intent_type"):
        intent_str = intent_raw.intent_type.value

    full_resp = final.get("response", "") if isinstance(final, dict) else getattr(final, "response", "")
    if not full_resp:
        full_resp = "抱歉，系统暂时无法生成回复，请再试一次。"
    report = mon.finish_session(intent_type=intent_str, response=full_resp)

    diagnosis = ""
    if mon.is_dev_mode() and report:
        diagnosis = await asyncio.to_thread(mon.diagnose, report)

    recs = final.get("recommended_products", []) if isinstance(final, dict) else getattr(final, "recommended_products", [])
    return {
        "response": full_resp,
        "products": recs,
        "intent": intent_str,
        "sid": sid,
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
        "diagnosis": diagnosis,
    }


@app.post("/api/chat/stream")
async def chat_stream(req: ChatReq):
    async def gen():
        sid = req.thread_id or f"s{int(time.time() * 1000)}"
        state = AgentState(query=req.message)
        mon.start_session(sid, req.message)
        yield f"data: {json.dumps({'t': 'session', 'sid': sid, 'query': req.message}, ensure_ascii=False)}\n\n"

        try:
            graph = get_graph()
            final = await asyncio.to_thread(
                lambda: graph.invoke(state, config={"configurable": {"thread_id": sid}})
            )
            intent_raw = final.get("intent") if isinstance(final, dict) else getattr(final, "intent", None)
            intent_str = intent_raw.intent_type.value if intent_raw and hasattr(intent_raw, "intent_type") else ""
            full_resp = final.get("response", "") if isinstance(final, dict) else getattr(final, "response", "")
            report = mon.finish_session(intent_type=intent_str, response=full_resp)
            recs = final.get("recommended_products", []) if isinstance(final, dict) else getattr(final, "recommended_products", [])
            yield f"data: {json.dumps({'t': 'report', 'report': report}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'t': 'done', 'response': full_resp, 'products': recs}, ensure_ascii=False)}\n\n"
        except Exception as e:
            mon.finish_session()
            yield f"data: {json.dumps({'t': 'error', 'err': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# ============ 商品接口 ============

@app.get("/api/products/hot")
async def hot_products(limit: int = 10):
    return {"products": ProductSearchTool.hot(limit)}


@app.get("/api/products/{product_id}")
async def product_detail(product_id: str):
    d = ProductSearchTool.detail(product_id)
    if not d:
        raise HTTPException(404, "商品不存在")
    return d


@app.get("/api/deals")
async def best_deals(limit: int = 10):
    return {"deals": PriceMonitorTool.best_deals(limit)}


@app.get("/api/categories")
async def categories():
    return {"categories": [c.value for c in ProductCategory]}


@app.get("/api/cache/stats")
async def cache_stats():
    return cache.stats()


@app.get("/api/health")
async def health():
    return {"status": "ok", "time": datetime.now().isoformat()}


# ============ 监控流 ============

@app.get("/api/monitor/stream")
async def monitor_stream():
    q = mon.subscribe()
    async def gen():
        while True:
            try:
                event = await asyncio.to_thread(q.get, timeout=30)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception:
                yield f"data: {json.dumps({'t': 'ping'}, ensure_ascii=False)}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


# ============ 批量测试 ============

@app.post("/api/dev/batch-test")
async def batch_test(req: BatchReq):
    t0 = time.time()

    # 生成测试用例
    cases = mon.generate_test_cases(count=req.count, custom_prompt=req.custom_prompt)

    # 获取 graph 调用函数
    graph = get_graph()
    def invoke(state):
        return graph.invoke(state, config={"configurable": {"thread_id": f"batch_{int(time.time() * 1000)}"}})

    # 批量执行
    reports = mon.run_batch_test(invoke, cases)

    # 综合分析
    analysis = ""
    if reports:
        analysis = await asyncio.to_thread(mon.analyze_all_reports, reports)

    ok_c = sum(1 for r in reports if not r.get("error"))
    return {
        "test_cases": cases, "reports": reports, "analysis": analysis,
        "total": len(cases), "ok": ok_c, "err": len(cases) - ok_c,
        "avg_ms": round(sum(r.get("total_ms", 0) for r in reports) / max(len(reports), 1), 1),
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
    }


@app.get("/api/dev/batch-test/reports")
async def batch_reports(limit: int = 50):
    return {"reports": mon.get_recent(limit)}


@app.get("/api/dev/batch-test/report/{sid}")
async def single_report(sid: str):
    for r in mon.get_recent(200):
        if r.get("sid") == sid:
            return {"sid": sid, "query": r.get("query"), "intent": r.get("intent_type"),
                    "response": r.get("response_full", r.get("response", "")),
                    "total_ms": r.get("total_ms"), "nodes": r.get("nodes", [])}
    raise HTTPException(404, f"未找到会话: {sid}")


@app.get("/api/dev/batch-test/responses")
async def all_responses(limit: int = 20):
    items = [{"sid": r.get("sid"), "query": r.get("query"),
              "intent": r.get("intent_type"),
              "response": r.get("response_full", r.get("response", ""))[:800],
              "ms": r.get("total_ms")} for r in mon.get_recent(limit)]
    return {"count": len(items), "responses": items}


# ============ 启动 ============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)