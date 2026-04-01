"""
新闻分析师 - 新闻事件分析
分析股票相关新闻、事件影响、舆情走向
"""
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

from dataflows.tools import search_stock_news

logger = logging.getLogger("stock_analyzer.analysts.news")


@tool
def get_stock_news(stock_code: str) -> str:
    """搜索并获取股票的最新相关新闻资讯。
    
    Args:
        stock_code: 股票代码，如 600519（贵州茅台）、AAPL（苹果）
    """
    return search_stock_news(stock_code)


def create_news_analyst(llm):
    """创建新闻分析师节点"""

    tools = [get_stock_news]
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = """你是一位专业的**新闻分析师**，专注于分析财经新闻对股票价格的影响。你的任务是评估新闻事件对目标公司的影响。

## 分析要求

1. **新闻梳理**：
   - 收集近期相关重要新闻
   - 按影响程度排序（利好/利空/中性）
   - 识别关键事件节点

2. **影响评估**：
   - 分析每条新闻对公司的影响方向和程度
   - 评估新闻的持续影响还是短期效应
   - 识别潜在的连锁反应

3. **舆情趋势**：
   - 当前整体舆论偏向（正面/负面/中性）
   - 近期舆情变化趋势
   - 市场预期是否已充分反映

4. **风险评估**：
   - 是否存在重大负面事件
   - 监管风险、法律风险
   - 竞争格局变化

5. **综合判断**：
   - 新闻面整体利好/利空/中性
   - 关键催化剂事件
   - 需要持续关注的事项

## 输出要求
- 使用中文输出
- 区分事实和观点
- 评估影响程度（高/中/低）
- 最后给出新闻面综合评级"""

    def news_analyst_node(state: dict) -> dict:
        company = state.get("company_of_interest", "")
        trade_date = state.get("trade_date", "")

        logger.info(f"📰 [新闻分析师] 开始分析: {company}, 日期: {trade_date}")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"请分析 {company} 的相关新闻和事件。分析日期: {trade_date}。请先搜索该股票的最新新闻，然后进行影响分析。")
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
            logger.info(f"📰 [新闻分析师] 请求工具调用: {[tc['name'] for tc in response.tool_calls]}")
            return {
                "messages": [response],
                "sender": "News Analyst",
                "news_tool_calls": state.get("news_tool_calls", 0) + 1,
            }

        logger.info(f"📰 [新闻分析师] 分析完成")
        return {
            "messages": [response],
            "sender": "News Analyst",
            "news_report": response.content,
        }

    return news_analyst_node
