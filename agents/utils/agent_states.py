"""
单股分析系统 - Agent 状态定义
基于 LangGraph 的状态管理
"""
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langgraph.graph.message import MessagesState, add_messages


class InvestDebateState(TypedDict):
    """投资辩论状态 - 管理多空研究员之间的辩论"""
    bull_history: List[str]
    bear_history: List[str]
    history: List[str]
    current_response: str
    judge_decision: str
    count: int


class RiskDebateState(TypedDict):
    """风险辩论状态 - 管理风险分析团队之间的讨论"""
    risky_history: List[str]
    safe_history: List[str]
    neutral_history: List[str]
    history: List[str]
    latest_speaker: str
    current_risky_response: str
    current_safe_response: str
    current_neutral_response: str
    judge_decision: str
    count: int


class AgentState(MessagesState):
    """主代理状态 - 整个单股分析流程的状态管理"""
    # 基本信息
    company_of_interest: str
    trade_date: str
    sender: str

    # 四位分析师的报告
    market_report: str
    sentiment_report: str
    news_report: str
    fundamentals_report: str

    # 工具调用计数器（防止死循环）
    market_tool_calls: int
    news_tool_calls: int
    social_tool_calls: int
    fundamentals_tool_calls: int

    # 辩论状态
    investment_debate_state: InvestDebateState
    risk_debate_state: RiskDebateState

    # 计划与决策
    investment_plan: str
    trader_investment_plan: str
    final_trade_decision: str
