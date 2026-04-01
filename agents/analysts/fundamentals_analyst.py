"""
基本面分析师 - 财务数据分析
分析公司财务报表、估值指标、行业地位等
"""
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

from dataflows.tools import get_stock_fundamentals, get_market_info

logger = logging.getLogger("stock_analyzer.analysts.fundamentals")


@tool
def get_fundamentals_data(stock_code: str) -> str:
    """获取股票的基本面数据，包括财务指标、估值数据、公司信息等。
    
    Args:
        stock_code: 股票代码，如 600519（贵州茅台）、AAPL（苹果）
    """
    return get_stock_fundamentals(stock_code)


@tool
def get_company_info(stock_code: str) -> str:
    """获取公司的基本信息，包括行业、市值、等。
    
    Args:
        stock_code: 股票代码
    """
    info = get_market_info(stock_code)
    return f"市场: {info['market']}, 货币: {info['currency']}, 代码: {info['symbol']}"


def create_fundamentals_analyst(llm):
    """创建基本面分析师节点"""

    tools = [get_fundamentals_data, get_company_info]
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = """你是一位专业的**基本面分析师**，专注于公司财务分析和估值。你的任务是深入分析公司的基本面状况。

## 分析要求

1. **盈利能力分析**：
   - 营收和利润增长趋势
   - 毛利率、净利率变化
   - ROE（净资产收益率）、ROA（总资产收益率）

2. **估值分析**：
   - PE（市盈率）与行业/历史对比
   - PB（市净率）是否合理
   - PEG 估值
   - 与同行业公司对比

3. **财务健康度**：
   - 资产负债率是否合理
   - 流动比率、速动比率
   - 现金流状况（经营/投资/筹资）
   - 是否有债务风险

4. **成长性分析**：
   - 营收增速和利润增速
   - 行业增长空间
   - 竞争优势和护城河

5. **投资建议**：
   - 综合基本面给出明确评级
   - 提供合理估值区间

## 输出要求
- 使用中文输出
- 基于数据的事实分析
- 对比行业平均水平
- 最后给出综合基本面评级"""

    def fundamentals_analyst_node(state: dict) -> dict:
        company = state.get("company_of_interest", "")
        trade_date = state.get("trade_date", "")

        logger.info(f"💼 [基本面分析师] 开始分析: {company}, 日期: {trade_date}")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"请分析 {company} 的基本面情况。分析日期: {trade_date}。请先获取该股票的基本面数据，然后进行深度分析。")
        ]

        # 提取与当前分析师相关的历史消息（工具调用和结果）
        tool_names = [t.name for t in tools]
        for msg in state.get("messages", []):
            if msg.type == "ai" and getattr(msg, "tool_calls", None):
                if any(tc["name"] in tool_names for tc in msg.tool_calls):
                    messages.append(msg)
            elif msg.type == "tool" and getattr(msg, "name", "") in tool_names:
                messages.append(msg)

        response = llm_with_tools.invoke(messages)

        if response.tool_calls:
            logger.info(f"💼 [基本面分析师] 请求工具调用: {[tc['name'] for tc in response.tool_calls]}")
            return {
                "messages": [response],
                "sender": "Fundamentals Analyst",
                "fundamentals_tool_calls": state.get("fundamentals_tool_calls", 0) + 1,
            }

        logger.info(f"💼 [基本面分析师] 分析完成")
        return {
            "messages": [response],
            "sender": "Fundamentals Analyst",
            "fundamentals_report": response.content,
        }

    return fundamentals_analyst_node
