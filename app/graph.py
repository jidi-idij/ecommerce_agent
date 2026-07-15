"""LangGraph 工作流编排"""
from __future__ import annotations

from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.models import AgentState, IntentType
from app.nodes import (
    intent_node, retrieval_node, sort_node, generate_node,
    price_compare_node, after_sales_node, general_chat_node,
)


def create_graph() -> StateGraph:
    workflow = StateGraph(AgentState)
    workflow.add_node("intent", intent_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("sort", sort_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("price_compare", price_compare_node)
    workflow.add_node("after_sales", after_sales_node)
    workflow.add_node("general_chat", general_chat_node)
    workflow.set_entry_point("intent")

    def route(state: AgentState) -> Literal["retrieval", "price_compare", "after_sales", "general_chat", "generate"]:
        if state.error:
            return "generate"
        it = state.intent.intent_type if state.intent else IntentType.GENERAL
        print(f"[Router] → {it.value}")
        if it == IntentType.PRICE_COMPARE:
            return "price_compare"
        if it == IntentType.AFTER_SALES:
            return "after_sales"
        if it == IntentType.GENERAL:
            return "general_chat"
        return "retrieval"

    workflow.add_conditional_edges("intent", route)
    workflow.add_edge("retrieval", "sort")
    workflow.add_edge("sort", "generate")
    workflow.add_edge("generate", END)
    workflow.add_edge("price_compare", END)
    workflow.add_edge("after_sales", END)
    workflow.add_edge("general_chat", END)
    return workflow


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = create_graph().compile(checkpointer=MemorySaver())
    return _graph
