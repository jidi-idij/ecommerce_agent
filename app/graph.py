"""
LangGraph 工作流编排
定义多Agent协作的状态机流程
"""
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.models import AgentState, IntentType
from app.nodes import (
    intent_node,
    retrieval_node,
    sort_node,
    generate_node,
    price_compare_node,
    recommendation_node,
    after_sales_node,
    general_chat_node,
)


def create_shopping_graph() -> StateGraph:
    """
    创建电商导购Agent工作流
    
    流程:
    1. intent_node → 识别用户意图
    2. 根据意图路由到不同处理节点:
       - product_search → retrieval → sort → generate
       - price_compare → price_compare_node → END
       - recommendation → retrieval → sort → generate
       - after_sales → after_sales_node → END
       - general → general_chat_node → END
    3. 生成最终回复
    """
    
    # 创建工作流
    workflow = StateGraph(AgentState)
    
    # 注册所有节点
    workflow.add_node("intent", intent_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("sort", sort_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("price_compare", price_compare_node)
    workflow.add_node("recommendation", recommendation_node)
    workflow.add_node("after_sales", after_sales_node)
    workflow.add_node("general_chat", general_chat_node)
    
    # 定义入口点
    workflow.set_entry_point("intent")
    
    # 意图路由函数
    def route_by_intent(state: AgentState) -> Literal[
        "retrieval", "price_compare", "recommendation", "after_sales", "general_chat", "generate"
    ]:
        """根据意图类型路由到对应节点"""
        if state.error:
            return "generate"  # 出错时直接生成兜底回复
        
        intent_type = state.intent.intent_type if state.intent else IntentType.GENERAL
        
        print(f"[Router] 意图路由: {intent_type.value}")
        
        if intent_type == IntentType.PRICE_COMPARE:
            return "price_compare"
        elif intent_type == IntentType.RECOMMENDATION:
            return "recommendation"  # 推荐走完整流程
        elif intent_type == IntentType.AFTER_SALES:
            return "after_sales"
        elif intent_type == IntentType.GENERAL:
            return "general_chat"
        else:
            # PRODUCT_SEARCH 走标准检索流程
            return "retrieval"
    
    # 意图节点后路由
    workflow.add_conditional_edges(
        "intent",
        route_by_intent,
    )
    
    # 标准检索流程: retrieval → sort → generate
    workflow.add_edge("retrieval", "sort")
    workflow.add_edge("sort", "generate")
    
    # 终止节点
    workflow.add_edge("generate", END)
    workflow.add_edge("price_compare", END)
    workflow.add_edge("after_sales", END)
    workflow.add_edge("general_chat", END)
    
    return workflow


def create_graph_with_memory() -> StateGraph:
    """创建带记忆的工作流"""
    workflow = create_shopping_graph()
    
    # 添加内存检查点（用于多轮对话记忆）
    memory = MemorySaver()
    
    # 编译图
    graph = workflow.compile(checkpointer=memory)
    
    return graph


# 全局图实例
_shopping_graph = None

def get_shopping_graph():
    """获取全局图实例（单例）"""
    global _shopping_graph
    if _shopping_graph is None:
        _shopping_graph = create_graph_with_memory()
    return _shopping_graph
