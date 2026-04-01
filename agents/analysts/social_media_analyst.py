"""
社交媒体分析师 - 情绪分析
分析社交媒体上的投资者情绪、讨论热度、市场预期
"""
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

from dataflows.tools import search_stock_sentiment

logger = logging.getLogger("stock_analyzer.analysts.social")


@tool
def get_stock_sentiment(stock_code: str) -> str:
    """搜索并获取股票在社交媒体上的投资者情绪和讨论内容。
    
    Args:
        stock_code: 股票代码，如 600519（贵州茅台）、AAPL（苹果）
    """
    return search_stock_sentiment(stock_code)


def create_social_media_analyst(llm):
    """创建社交媒体分析师节点"""

    tools = [get_stock_sentiment]
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = """你是一位专业的**社交媒体分析师**，专注于分析投资者情绪和市场预期。你的任务是评估市场参与者对目标公司的看法。

## 分析要求

1. **情绪分析**：
   - 整体投资者情绪（乐观/悲观/中性）
   - 情绪强度和变化趋势
   - 机构投资者 vs 散户的观点差异

2. **讨论热度**：
   - 市场关注度水平
   - 热度变化趋势
   - 关键讨论话题

3. **市场预期**：
   - 投资者对未来的预期
   - 与实际基本面的偏差
   - 可能的预期修正方向

4. **反向思考**：
   - 是否存在过度乐观或悲观
   - 散户情绪是否极端
   - 潜在的反转信号

5. **综合判断**：
   - 社交媒体面整体评估
   - 需要警惕的信号
   - 情绪面的投资启示

## 输出要求
- 使用中文输出
- 区分理性和非理性情绪
- 注意社交媒体信息的局限性
- 最后给出情绪面综合评级"""

    def social_media_analyst_node(state: dict) -> dict:
        company = state.get("company_of_interest", "")
        trade_date = state.get("trade_date", "")

        logger.info(f"💬 [社交媒体分析师] 开始分析: {company}, 日期: {trade_date}")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"请分析 {company} 在社交媒体和投资者社区中的情绪和讨论。分析日期: {trade_date}。请先搜索该股票的社交媒体讨论，然后进行情绪分析。")
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
            logger.info(f"💬 [社交媒体分析师] 请求工具调用: {[tc['name'] for tc in response.tool_calls]}")
            return {
                "messages": [response],
                "sender": "Social Media Analyst",
                "social_tool_calls": state.get("social_tool_calls", 0) + 1,
            }

        logger.info(f"💬 [社交媒体分析师] 分析完成")
        return {
            "messages": [response],
            "sender": "Social Media Analyst",
            "sentiment_report": response.content,
        }

    return social_media_analyst_node
