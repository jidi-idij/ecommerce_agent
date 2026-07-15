"""Monitor v4.0 精简版 - 单文件，核心功能完整

功能：
  - 会话追踪（开始/结束/节点耗时/异常）
  - SSE 实时监控广播
  - 批量测试（YAML模板生成 → 批量执行 → LLM综合分析）
  - LLM 单条诊断
  - 内存存储 + 统计查询

用法：
    from app import monitor
    monitor.start_session("s001", "推荐手机")
    node = monitor.node_start("intent_node")
    ...
    monitor.node_end(node, "output")
    report = monitor.finish_session("recommendation", "回复")
    
    # 开发者模式
    monitor.set_dev_mode(True)
    cases = monitor.generate_test_cases(10)
    reports = monitor.run_batch_test(graph_invoke_fn, cases)
    analysis = monitor.analyze_all_reports(reports)
"""
from __future__ import annotations

import os
import time
import json
import yaml
import queue
import random
import hashlib
import threading
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

# ============================================================
# 数据模型
# ============================================================

class NodeStatus(str, Enum):
    RUNNING = "running"
    OK = "ok"
    SLOW = "slow"
    ERROR = "error"


@dataclass
class NodeExec:
    name: str
    start: float = 0.0
    end: float = 0.0
    dur_ms: float = 0.0
    status: NodeStatus = NodeStatus.RUNNING
    input_snap: str = ""
    output_snap: str = ""
    error: str = ""
    stack: str = ""
    details: dict = field(default_factory=dict)

    def finish(self, output: str = "", details: dict | None = None):
        self.end = time.time()
        self.dur_ms = (self.end - self.start) * 1000
        self.status = NodeStatus.OK
        self.output_snap = output
        if details:
            self.details = details

    def fail(self, exc: Exception):
        import traceback
        self.end = time.time()
        self.dur_ms = (self.end - self.start) * 1000
        self.status = NodeStatus.ERROR
        self.error = str(exc)
        self.stack = traceback.format_exc()

    def to_dict(self) -> dict:
        return {"n": self.name, "ms": round(self.dur_ms, 1),
                "s": self.status.value, "in": self.input_snap[:60],
                "out": self.output_snap[:60], "err": self.error[:80]}


@dataclass
class Session:
    sid: str
    query: str
    start: float = field(default_factory=time.time)
    response: str = ""
    intent_type: str = ""
    nodes: List[NodeExec] = field(default_factory=list)

    def begin_node(self, name: str, inp: str = "") -> NodeExec:
        n = NodeExec(name=name, start=time.time(), input_snap=inp)
        self.nodes.append(n)
        return n

    def end(self, intent_type: str = "", response: str = "") -> dict:
        total = (time.time() - self.start) * 1000
        self.response = response
        self.intent_type = intent_type
        return {"sid": self.sid, "query": self.query, "total_ms": round(total, 1),
                "intent_type": intent_type, "response": response[:500],
                "response_full": response, "node_cnt": len(self.nodes),
                "status": "success" if not any(n.status == NodeStatus.ERROR for n in self.nodes) else "partial",
                "nodes": [n.to_dict() for n in self.nodes]}


# ============================================================
# SSE 事件
# ============================================================

class SSE:
    @staticmethod
    def node_start(name: str, inp: str) -> dict:
        return {"t": "node_start", "name": name, "input": inp[:60]}

    @staticmethod
    def node_end(name: str, ms: float, status: NodeStatus, out: str) -> dict:
        return {"t": "node_end", "name": name, "ms": round(ms, 1), "status": status.value, "output": out[:80]}

    @staticmethod
    def node_error(name: str, err: str) -> dict:
        return {"t": "node_err", "name": name, "err": err[:120]}

    @staticmethod
    def session(sid: str, query: str) -> dict:
        return {"t": "session", "sid": sid, "query": query}

    @staticmethod
    def report(r: dict) -> dict:
        return {"t": "report", "report": r}

    @staticmethod
    def batch_start(total: int) -> dict:
        return {"t": "batch_start", "total": total}

    @staticmethod
    def batch_progress(current: int, total: int, query: str, status: str,
                       ms: float = 0, resp_preview: str = "", intent: str = "", has_products: bool = False) -> dict:
        return {"t": "batch_progress", "current": current, "total": total,
                "query": query, "status": status, "ms": ms,
                "response_preview": resp_preview[:300], "intent": intent, "has_products": has_products}

    @staticmethod
    def batch_done(total: int, ok: int, err: int) -> dict:
        return {"t": "batch_done", "total": total, "ok": ok, "err": err}

    @staticmethod
    def diagnose(text: str) -> dict:
        return {"t": "diagnose", "text": text}

    @staticmethod
    def batch_analysis(text: str) -> dict:
        return {"t": "batch_analysis", "text": text}

    @staticmethod
    def mode(dev: bool) -> dict:
        return {"t": "mode", "dev": dev}


# ============================================================
# Monitor 核心类
# ============================================================

class Monitor:
    """监控中心 - 精简版"""

    def __init__(self):
        self._dev_mode = False
        self._active: Optional[Session] = None
        self._history: List[dict] = []
        self._errors: List[dict] = []
        self._queues: List[queue.Queue] = []
        self._seen_hashes: set = set()
        self._lock = threading.Lock()
        self._max_hist = 200
        self._trim_hist = 100
        self._max_err = 100
        self._trim_err = 50
        self._slow_ms = 3000
        self._sleep_between = 0.3
        self._templates: List[dict] = []
        self._load_templates()

    # -- 模板加载 --

    def _load_templates(self):
        """从 YAML 加载测试模板"""
        paths = ["test_templates.yaml", "config/test_templates.yaml", "data/test_templates.yaml"]
        for p in paths:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                    if data and "templates" in data:
                        self._templates = data["templates"]
                        print(f"[Monitor] 加载 {len(self._templates)} 个测试模板")
                        return
                except Exception:
                    continue
        # 内置模板兜底
        self._templates = [
            {"template_id": "rec_budget", "intent_type": "recommendation", "category": "手机",
             "structure": "推荐{adj}{category}，预算{price}",
             "slots": {"category": ["手机", "智能手机"], "adj": ["学生党", "性价比高的", "适合老人的"]},
             "free_slots": ["price"]},
            {"template_id": "compare", "intent_type": "price_compare", "category": "手机",
             "structure": "{brand1}和{brand2}哪个好",
             "slots": {"brand1": ["iPhone 15", "华为Mate60", "小米14"], "brand2": ["华为Mate60", "小米14", "OPPO Find X7"]}},
            {"template_id": "search_extreme", "intent_type": "product_search", "category": "通用",
             "structure": "你们这最{extreme}的{category}",
             "slots": {"extreme": ["贵", "便宜", "受欢迎"], "category": ["产品", "耳机", "手机"]}},
            {"template_id": "scene_gift", "intent_type": "recommendation", "category": "通用",
             "structure": "适合送{recipient}的{category}",
             "slots": {"recipient": ["女朋友", "男朋友", "父母", "孩子"], "category": ["礼物", "手机", "耳机"]}},
            {"template_id": "after_sales", "intent_type": "after_sales", "category": "通用",
             "structure": "怎么{action}",
             "slots": {"action": ["退货", "换货", "维修", "申请售后"]}},
            {"template_id": "spec_search", "intent_type": "product_search", "category": "手机",
             "structure": "{feature}好的{category}{price}",
             "slots": {"feature": ["拍照", "续航", "游戏"], "category": ["手机", "笔记本"],
                       "price": ["5000左右", "2000元以下"]}},
            {"template_id": "chat_greeting", "intent_type": "general", "category": "通用",
             "structure": "{greeting}",
             "slots": {"greeting": ["你好", "在吗", "哈喽"]}},
            {"template_id": "promo", "intent_type": "product_search", "category": "通用",
             "structure": "有什么{promo_type}",
             "slots": {"promo_type": ["优惠活动", "打折", "满减", "优惠券"]}},
        ]

    # -- 模式管理 --

    def set_dev_mode(self, on: bool):
        self._dev_mode = on
        self._broadcast(SSE.mode(on))

    def is_dev_mode(self) -> bool:
        return self._dev_mode

    # -- 会话管理 --

    def start_session(self, sid: str, query: str):
        self._active = Session(sid=sid, query=query)
        self._broadcast(SSE.session(sid, query))

    def node_start(self, name: str, inp: str = "") -> NodeExec | None:
        if self._active is None:
            return None
        n = self._active.begin_node(name, inp)
        self._broadcast(SSE.node_start(name, inp))
        return n

    def node_end(self, node: NodeExec | None, output: str = "", details: dict | None = None):
        if node is None:
            return
        node.finish(output, details)
        self._broadcast(SSE.node_end(node.name, node.dur_ms, node.status, output))

    def node_error(self, node: NodeExec | None, exc: Exception):
        if node is None:
            return
        node.fail(exc)
        self._broadcast(SSE.node_error(node.name, node.error))
        self._errors.append({"time": datetime.now().isoformat(), "err": str(exc), "stack": node.stack[:500]})
        if len(self._errors) > self._max_err:
            self._errors = self._errors[-self._trim_err:]

    def finish_session(self, intent_type: str = "", response: str = "") -> dict:
        if self._active is None:
            return {}
        report = self._active.end(intent_type, response)
        self._active = None
        self._history.append(report)
        if len(self._history) > self._max_hist:
            self._history = self._history[-self._trim_hist:]
        self._broadcast(SSE.report(report))
        return report

    # -- SSE 广播 --

    def _broadcast(self, event: dict):
        dead = []
        for q in self._queues:
            try:
                q.put_nowait(event)
            except queue.Full:
                dead.append(q)
        for d in dead:
            if d in self._queues:
                self._queues.remove(d)

    def subscribe(self) -> queue.Queue:
        q = queue.Queue(maxsize=500)
        self._queues.append(q)
        return q

    # -- 诊断 --

    def diagnose(self, report: dict) -> str:
        nodes = report.get("nodes", [])
        if not nodes:
            return "无执行记录"
        slow = [n for n in nodes if n.get("s") == "slow"]
        errs = [n for n in nodes if n.get("s") == "error"]
        resp = report.get("response_full", report.get("response", ""))
        prompt = f"""你是系统诊断专家。分析以下Agent执行报告，一句话总结核心问题，再列出具体建议。
查询: {report['query']}
总耗时: {report['total_ms']}ms | 节点: {len(nodes)}个 | 慢节点: {len(slow)}个 | 错误: {len(errs)}个
Agent回复长度: {len(resp)}字
节点:
"""
        for n in nodes:
            prompt += f"- {n['n']}: {n['ms']}ms 状态:{n['s']}\n"
        prompt += f"\nAgent回复:\n{resp[:300]}\n\n请用中文: 1)核心问题 2)优化建议"
        return self._call_llm(prompt)

    def analyze_all_reports(self, reports: List[dict]) -> str:
        if not reports:
            return "无测试数据"
        total = len(reports)
        ok = sum(1 for r in reports if not r.get("error"))
        err = total - ok
        avg_ms = sum(r.get("total_ms", 0) for r in reports) / max(total, 1)
        empty_resp = sum(1 for r in reports if not r.get("response_full", "").strip())
        has_products = sum(1 for r in reports if len(r.get("recommended_products", [])) > 0)

        # 节点统计
        ns: Dict[str, dict] = {}
        for r in reports:
            for n in r.get("nodes", []):
                name = n.get("n", "?")
                if name not in ns:
                    ns[name] = {"c": 0, "ms": 0, "err": 0}
                ns[name]["c"] += 1
                ns[name]["ms"] += n.get("ms", 0)
                if n.get("s") == "error":
                    ns[name]["err"] += 1

        node_lines = ""
        for name, d in sorted(ns.items(), key=lambda x: x[1]["ms"]/max(x[1]["c"],1), reverse=True):
            node_lines += f"- {name}: {d['c']}次 平均{d['ms']/max(d['c'],1):.0f}ms 错误{d['err']}次\n"

        # 简化prompt避免超长导致LLM失败
        prompt = f"""分析以下批量测试结果，输出5部分报告：
数据：{total}条 | 成功{ok} | 失败{err} | 平均{avg_ms:.0f}ms | 空回复{empty_resp} | 有商品{has_products}
节点：
{node_lines}
请输出：1)整体评价 2)核心问题 3)瓶颈 4)优化建议 5)覆盖度"""

        result = self._call_llm(prompt)
        # LLM失败或返回空时，使用默认报告（确保永远有内容输出）
        if not result or result.startswith("LLM调用失败") or len(result) < 10:
            result = f"""## 批量测试报告（{total}条）

### 1. 整体评价
- 成功率: {ok}/{total} ({ok/max(total,1)*100:.1f}%)
- 平均耗时: {avg_ms:.0f}ms
- 空回复: {empty_resp}条
- 含推荐商品: {has_products}条

### 2. 核心问题
{(empty_resp > 0) * "- 存在空回复，需检查生成节点\n"}{(err > 0) * "- 存在执行错误，需检查异常报告\n"}- 暂无其他严重问题

### 3. 瓶颈分析
{node_lines}

### 4. 优化建议
- 关注耗时最长的节点进行优化
- 检查空回复和错误用例的具体原因

### 5. 测试覆盖度
- 覆盖{len(ns)}个节点类型
"""
        return result

    def _call_llm(self, prompt: str) -> str:
        """调用 LLM（从 config.settings 获取配置）"""
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage
            from config.settings import llm_config
            client = ChatOpenAI(
                model=llm_config.model,
                api_key=llm_config.api_key,
                base_url=llm_config.base_url,
                temperature=0.3,
                max_tokens=800,
            )
            r = client.invoke([HumanMessage(content=prompt)])
            result = (r.content or "").strip()
            if not result:
                result = "LLM调用失败: 返回空内容"
            self._broadcast(SSE.diagnose(result) if "诊断" in prompt else SSE.batch_analysis(result))
            return result
        except Exception as e:
            return f"LLM调用失败: {e}"

    # -- 测试用例生成（YAML模板驱动，不依赖LLM） --

    def generate_test_cases(self, count: int = 10, custom_prompt: str = "") -> List[str]:
        """从 YAML 模板生成测试用例（纯规则引擎，零 LLM 依赖）"""
        if not self._templates:
            return []
        results = []
        max_attempts = count * 5
        attempts = 0
        weights = [t.get("priority", 1) for t in self._templates]

        while len(results) < count and attempts < max_attempts:
            attempts += 1
            tmpl = random.choices(self._templates, weights=weights, k=1)[0]
            text = self._render_template(tmpl)
            if text and self._validate(text):
                results.append(text)

        return results[:count]

    def _render_template(self, tmpl: dict) -> str | None:
        text = tmpl["structure"]
        slots = tmpl.get("slots", {})
        free_slots = tmpl.get("free_slots", [])

        # 填充枚举槽位
        values = {}
        for key, candidates in slots.items():
            values[key] = random.choice(candidates)

        # 填充自由槽位（简单规则）
        for fs in free_slots:
            if fs == "price":
                values[fs] = random.choice(["2000元左右", "3000元以内", "1500元以下",
                                             "5000元左右", "1000元以下", "8000元以内"])
            else:
                values[fs] = random.choice(["好的", "推荐", "适合"])

        try:
            text = text.format(**values)
        except (KeyError, IndexError):
            return None

        # 避免 brand1 == brand2
        if "brand1" in values and "brand2" in values and values["brand1"] == values["brand2"]:
            return None

        return text

    def _validate(self, text: str) -> bool:
        if not text or len(text) < 3 or len(text) > 100:
            return False
        sensitive = {"色情", "淫秽", "赌博", "毒品", "枪支", "洗钱", "套现", "刷单"}
        if any(w in text for w in sensitive):
            return False
        h = hashlib.md5(text.strip().encode()).hexdigest()
        if h in self._seen_hashes:
            return False
        self._seen_hashes.add(h)
        return True

    # -- 批量测试执行 --

    def run_batch_test(self, graph_invoke_fn, cases: List[str]) -> List[dict]:
        """批量执行测试用例"""
        reports: List[dict] = []
        total = len(cases)
        self._broadcast(SSE.batch_start(total))

        for i, query in enumerate(cases):
            sid = f"batch_{int(time.time() * 1000)}_{i}"
            self.start_session(sid, query)

            try:
                # 构造 mock state 调用 graph
                from app.models import AgentState
                state = AgentState(query=query)
                final = graph_invoke_fn(state)

                intent_str = ""
                intent_raw = final.get("intent")
                if intent_raw:
                    intent_str = (intent_raw.intent_type.value
                                  if hasattr(intent_raw, "intent_type")
                                  else intent_raw.get("intent_type", ""))

                full_resp = final.get("response", "")
                recs = final.get("recommended_products", [])

                report = self.finish_session(intent_type=intent_str, response=full_resp)
                report["recommended_products"] = recs

                self._broadcast(SSE.batch_progress(
                    i + 1, total, query, "ok", report.get("total_ms", 0),
                    full_resp[:300], intent_str, len(recs) > 0
                ))
                reports.append(report)

            except Exception as e:
                report = {"sid": sid, "query": query, "error": str(e),
                          "total_ms": 0, "nodes": [], "response": "",
                          "response_full": "", "recommended_products": []}
                self._broadcast(SSE.batch_progress(i + 1, total, query, "error", 0))
                reports.append(report)
                self._errors.append({"time": datetime.now().isoformat(), "err": str(e), "stack": str(e)[:500]})

            if i < total - 1:
                time.sleep(self._sleep_between)

        ok_c = sum(1 for r in reports if not r.get("error"))
        self._broadcast(SSE.batch_done(total, ok_c, total - ok_c))
        return reports

    # -- 查询接口 --

    def get_recent(self, limit: int = 20) -> List[dict]:
        with self._lock:
            return self._history[-limit:][::-1]

    def get_errors(self, limit: int = 20) -> List[dict]:
        with self._lock:
            return self._errors[-limit:][::-1]

    def stats(self) -> dict:
        with self._lock:
            total = len(self._history)
            err_s = sum(1 for h in self._history if any(n.get("s") == "error" for n in h.get("nodes", [])))
            avg_ms = sum(h.get("total_ms", 0) for h in self._history) / max(total, 1)
            ns: Dict[str, dict] = {}
            for h in self._history:
                for n in h.get("nodes", []):
                    name = n.get("n", "?")
                    if name not in ns:
                        ns[name] = {"c": 0, "ms": 0, "err": 0}
                    ns[name]["c"] += 1
                    ns[name]["ms"] += n.get("ms", 0)
                    if n.get("s") == "error":
                        ns[name]["err"] += 1
            for v in ns.values():
                v["avg"] = round(v["ms"] / max(v["c"], 1), 1)
            return {"total": total, "err_sessions": err_s,
                    "err_rate": f"{err_s / max(total, 1) * 100:.1f}%",
                    "avg_ms": round(avg_ms, 1), "dev_mode": self._dev_mode,
                    "node_stats": ns}


# ============================================================
# 全局单例（延迟初始化）
# ============================================================

_monitor: Optional[Monitor] = None


def _get() -> Monitor:
    global _monitor
    if _monitor is None:
        _monitor = Monitor()
    return _monitor


# -- 兼容 v3.0 的模块级 API --

def set_dev_mode(on: bool):
    _get().set_dev_mode(on)


def is_dev_mode() -> bool:
    return _get().is_dev_mode()


def start_session(sid: str, query: str):
    _get().start_session(sid, query)


def node_start(name: str, inp: str = "") -> NodeExec | None:
    return _get().node_start(name, inp)


def node_end(node: NodeExec | None, output: str = "", details: dict | None = None):
    _get().node_end(node, output, details)


def node_error(node: NodeExec | None, exc: Exception):
    _get().node_error(node, exc)


def finish_session(intent_type: str = "", response: str = "") -> dict:
    return _get().finish_session(intent_type, response)


def diagnose(report: dict) -> str:
    return _get().diagnose(report)


def generate_test_cases(count: int = 10, custom_prompt: str = "") -> List[str]:
    return _get().generate_test_cases(count, custom_prompt)


def run_batch_test(graph_invoke_fn, cases: List[str]) -> List[dict]:
    return _get().run_batch_test(graph_invoke_fn, cases)


def analyze_all_reports(reports: List[dict]) -> str:
    return _get().analyze_all_reports(reports)


def subscribe() -> queue.Queue:
    return _get().subscribe()


def get_recent(limit: int = 20) -> List[dict]:
    return _get().get_recent(limit)


def get_errors(limit: int = 20) -> List[dict]:
    return _get().get_errors(limit)


def stats() -> dict:
    return _get().stats()