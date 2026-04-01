"""
市场分析师 - 技术分析
分析股票价格走势、技术指标（MA/MACD/RSI/布林带等）
"""
import logging
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from dataflows.tools import get_stock_price_data, get_market_info

logger = logging.getLogger("stock_analyzer.analysts.market")


# 定义市场分析工具
@tool
def get_stock_market_data(stock_code: str, period: str = "3mo") -> str:
    """获取股票的历史价格数据，包括开盘价、收盘价、最高价、最低价、成交量等。
    
    Args:
        stock_code: 股票代码，如 600519（贵州茅台）、AAPL（苹果）、00700（腾讯）
        period: 数据周期，默认3个月
    """
    return get_stock_price_data(stock_code, period)


@tool
def get_market_info_tool(stock_code: str) -> str:
    """获取股票的市场信息，判断属于A股、港股还是美股。
    
    Args:
        stock_code: 股票代码
    """
    info = get_market_info(stock_code)
    return f"市场: {info['market']}, 货币: {info['currency']}, 代码: {info['symbol']}"


def create_market_analyst(llm):
    """
    创建市场分析师节点

    Args:
        llm: LangChain LLM 实例

    Returns:
        市场分析师函数节点
    """
    tools = [get_stock_market_data, get_market_info_tool]
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = """你是一位专业的**市场分析师**，专注于技术分析。你的任务是分析股票的价格走势和技术指标。

## 分析要求

1. **价格趋势分析**：
   - 短期趋势（5日、10日均线方向）
   - 中期趋势（20日、60日均线方向）
   - 价格与均线的关系（多头/空头排列）

2. **技术指标分析**：
   - MACD：金叉/死叉、红绿柱变化、DIF与DEA关系
   - RSI：超买/超卖区间判断（>70超买，<30超卖）
   - 布林带：价格在布林带中的位置、带宽变化
   - 成交量：量价配合关系

3. **支撑与阻力位**：
   - 识别关键支撑位和阻力位
   - 分析价格突破/跌破的可能性

4. **投资建议**：
   - 综合技术面给出明确的短期评级（买入/持有/卖出）
   - 提供目标价位区间

## 输出要求
- 使用中文输出
- 数据驱动的客观分析
- 明确标注关键价位
- 最后给出综合技术面评级"""

    def market_analyst_node(state: dict) -> dict:
        company = state.get("company_of_interest", "")
        trade_date = state.get("trade_date", "")

        logger.info(f"📊 [市场分析师] 开始分析: {company}, 日期: {trade_date}")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"请分析 {company} 的市场技术面情况。分析日期: {trade_date}。请先获取该股票的历史价格数据，然后进行技术分析。")
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

        # 如果 LLM 请求调用工具，返回工具调用信息
        if response.tool_calls:
            logger.info(f"📊 [市场分析师] 请求工具调用: {[tc['name'] for tc in response.tool_calls]}")
            return {
                "messages": [response],
                "sender": "Market Analyst",
                "market_tool_calls": state.get("market_tool_calls", 0) + 1,
            }

        # 没有工具调用，说明分析已完成
        logger.info(f"📊 [市场分析师] 分析完成")
        return {
            "messages": [response],
            "sender": "Market Analyst",
            "market_report": response.content,
        }

    return market_analyst_node
