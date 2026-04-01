"""
单股分析系统 - LangGraph 工作流构建
构建完整的单股分析流程图：
  START → 4位分析师(并行) → Bull⇄Bear辩论 → 研究经理 → 交易员 → 风险管理经理 → END
"""
import logging
from typing import Dict, Any, List, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import ToolMessage

from agents.utils.agent_states import AgentState, InvestDebateState, RiskDebateState
from agents.analysts.market_analyst import create_market_analyst, get_stock_market_data, get_market_info_tool
from agents.analysts.fundamentals_analyst import create_fundamentals_analyst, get_fundamentals_data, get_company_info
from agents.analysts.news_analyst import create_news_analyst, get_stock_news
from agents.analysts.social_media_analyst import create_social_media_analyst, get_stock_sentiment
from agents.researchers.bull_researcher import create_bull_researcher
from agents.researchers.bear_researcher import create_bear_researcher
from agents.managers.research_manager import create_research_manager
from agents.managers.risk_manager import create_risk_manager
from agents.trader.trader import create_trader

logger = logging.getLogger("stock_analyzer.graph")


class ConditionalLogic:
    """条件逻辑控制器 - 控制图的流向"""

    def __init__(self, max_debate_rounds: int = 1, max_risk_discuss_rounds: int = 1):
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds

    # ---- 分析师条件路由 ----

    def should_continue_market(self, state: AgentState) -> str:
        """判断市场分析是否继续（还需要工具调用 vs 完成）"""
        messages = state.get("messages", [])
        if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
            tool_calls = state.get("market_tool_calls", 0)
            if tool_calls < 5 and not state.get("market_report"):
                return "tools_market"
        return "next"

    def should_continue_fundamentals(self, state: AgentState) -> str:
        """判断基本面分析是否继续"""
        messages = state.get("messages", [])
        if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
            tool_calls = state.get("fundamentals_tool_calls", 0)
            if tool_calls < 5 and not state.get("fundamentals_report"):
                return "tools_fundamentals"
        return "next"

    def should_continue_news(self, state: AgentState) -> str:
        """判断新闻分析是否继续"""
        messages = state.get("messages", [])
        if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
            tool_calls = state.get("news_tool_calls", 0)
            if tool_calls < 5 and not state.get("news_report"):
                return "tools_news"
        return "next"

    def should_continue_social(self, state: AgentState) -> str:
        """判断情绪分析是否继续"""
        messages = state.get("messages", [])
        if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
            tool_calls = state.get("social_tool_calls", 0)
            if tool_calls < 5 and not state.get("sentiment_report"):
                return "tools_social"
        return "next"

    # ---- 辩论条件路由 ----

    def should_continue_debate(self, state: AgentState) -> str:
        """控制多空辩论流向"""
        debate_state = state.get("investment_debate_state", {})
        count = debate_state.get("count", 0)

        if count < self.max_debate_rounds * 2:
            # 根据当前轮次决定下一个发言者
            history = debate_state.get("history", [])
            if not history or "看涨研究员" in history[-1]:
                return "Bear Researcher"
            else:
                return "Bull Researcher"
        else:
            return "Research Manager"

    # ---- 风险分析条件路由 ----

    def should_continue_risk(self, state: AgentState) -> str:
        """控制风险分析流向（精简版直接到Risk Manager）"""
        return "Risk Manager"


def build_analysis_graph(llm, selected_analysts: List[str] = None):
    """
    构建单股分析的 LangGraph 工作流

    流程图:
      START
        ↓
      ┌──────────────────────────┐
      │  4位分析师 (串行执行)      │
      │  Market → Fundamentals   │
      │  → News → Social         │
      └──────────────────────────┘
        ↓
      ┌──────────────────────────┐
      │  多空辩论                 │
      │  Bull ⇄ Bear (多轮)      │
      └──────────────────────────┘
        ↓
      Research Manager (投资决策)
        ↓
      Trader (交易计划)
        ↓
      Risk Manager (风险评估)
        ↓
      END

    Args:
        llm: LangChain LLM 实例
        selected_analysts: 选择的分析师列表

    Returns:
        编译后的 LangGraph
    """
    if selected_analysts is None:
        selected_analysts = ["market", "fundamentals", "news", "social"]

    # 创建条件逻辑控制器
    conditional = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)

    # 创建分析师节点
    analyst_nodes = {}
    if "market" in selected_analysts:
        analyst_nodes["market"] = create_market_analyst(llm)
    if "fundamentals" in selected_analysts:
        analyst_nodes["fundamentals"] = create_fundamentals_analyst(llm)
    if "news" in selected_analysts:
        analyst_nodes["news"] = create_news_analyst(llm)
    if "social" in selected_analysts:
        analyst_nodes["social"] = create_social_media_analyst(llm)

    # 创建工具节点
    tool_nodes = {
        "market": ToolNode([get_stock_market_data, get_market_info_tool]),
        "fundamentals": ToolNode([get_fundamentals_data, get_company_info]),
        "news": ToolNode([get_stock_news]),
        "social": ToolNode([get_stock_sentiment]),
    }

    # 创建研究员和管理者节点
    bull_researcher = create_bull_researcher(llm)
    bear_researcher = create_bear_researcher(llm)
    research_manager = create_research_manager(llm)
    trader = create_trader(llm)
    risk_manager = create_risk_manager(llm)

    # ---- 构建工作流图 ----
    workflow = StateGraph(AgentState)

    # 添加分析师节点
    for analyst_type, node in analyst_nodes.items():
        workflow.add_node(f"{analyst_type.capitalize()} Analyst", node)
        workflow.add_node(f"tools_{analyst_type}", tool_nodes[analyst_type])

    # 添加辩论和决策节点
    workflow.add_node("Bull Researcher", bull_researcher)
    workflow.add_node("Bear Researcher", bear_researcher)
    workflow.add_node("Research Manager", research_manager)
    workflow.add_node("Trader", trader)
    workflow.add_node("Risk Manager", risk_manager)

    # ---- 设置边（连接关系）----

    analyst_list = list(analyst_nodes.keys())

    # START → 第一个分析师
    workflow.add_edge(START, f"{analyst_list[0].capitalize()} Analyst")

    # 分析师之间的串行连接 + 工具调用路由
    for i, analyst_type in enumerate(analyst_list):
        node_name = f"{analyst_type.capitalize()} Analyst"
        tool_name = f"tools_{analyst_type}"

        # 条件路由：需要工具调用 → 工具节点；完成 → 下一个
        if i < len(analyst_list) - 1:
            next_analyst = f"{analyst_list[i + 1].capitalize()} Analyst"
        else:
            next_analyst = "Bull Researcher"

        # 从工具节点返回到同一个分析师
        workflow.add_edge(tool_name, node_name)

        # 添加条件边
        def make_should_continue(atype, t_name):
            def fn(state):
                if atype == "market":
                    result = conditional.should_continue_market(state)
                elif atype == "fundamentals":
                    result = conditional.should_continue_fundamentals(state)
                elif atype == "news":
                    result = conditional.should_continue_news(state)
                elif atype == "social":
                    result = conditional.should_continue_social(state)
                else:
                    result = "next"
                return t_name if result == f"tools_{atype}" else "next"
            return fn

        workflow.add_conditional_edges(
            node_name,
            make_should_continue(analyst_type, tool_name),
            {tool_name: tool_name, "next": next_analyst}
        )

    # 多空辩论 → 研究经理
    workflow.add_conditional_edges(
        "Bull Researcher",
        conditional.should_continue_debate,
        {"Bear Researcher": "Bear Researcher", "Research Manager": "Research Manager"}
    )
    workflow.add_conditional_edges(
        "Bear Researcher",
        conditional.should_continue_debate,
        {"Bull Researcher": "Bull Researcher", "Research Manager": "Research Manager"}
    )

    # 研究经理 → 交易员 → 风险管理 → END
    workflow.add_edge("Research Manager", "Trader")
    workflow.add_edge("Trader", "Risk Manager")
    workflow.add_edge("Risk Manager", END)

    # 编译图
    graph = workflow.compile()

    graph.get_graph().draw_mermaid_png(output_file_path="langgraph.png")

    logger.info("✅ 单股分析工作流图构建完成")
    logger.info(f"   分析师: {[f'{a.capitalize()} Analyst' for a in analyst_list]}")
    logger.info(f"   流程: 分析师 → 多空辩论 → 研究经理 → 交易员 → 风险管理")

    return graph
