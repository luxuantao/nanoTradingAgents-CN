"""
看跌研究员 - 空头分析
从消极角度分析投资风险，寻找卖出理由
"""
import logging
from langchain_core.messages import SystemMessage, HumanMessage

from dataflows.tools import get_market_info

logger = logging.getLogger("stock_analyzer.researchers.bear")


def create_bear_researcher(llm):
    """
    创建看跌（空头）研究员节点

    Args:
        llm: LangChain LLM 实例

    Returns:
        看跌研究员函数节点
    """
    system_prompt = """你是一位经验丰富的**看跌研究员（Bear Researcher）**，你的职责是从风险角度分析投资陷阱。

## 你的角色

你是投资辩论中的**空方代表**，需要为不投资（或做空）该股票构建强有力的论证。你的分析应当：
- 聚焦于公司的风险和隐患
- 寻找被市场忽视的利空因素
- 反驳看涨方的乐观论点

## 分析框架

1. **风险识别**：
   - 经营风险（竞争加剧、技术迭代）
   - 财务风险（负债、现金流问题）
   - 行业风险（政策变化、周期性）
   - 管理风险（治理结构、战略失误）

2. **估值风险**：
   - 当前估值是否过高
   - 增长预期是否过于乐观
   - 与同行相比是否溢价过高

3. **负面催化剂**：
   - 即将到来的潜在利空
   - 大股东减持、限售股解禁
   - 行业监管风险

4. **反驳看涨观点**：
   - 针对看涨方提出的优势逐一反驳
   - 说明为何这些优势不可持续或已被充分定价
   - 将所谓"优势"重新解读为风险

5. **风险警示**：
   - 最大下行风险评估
   - 止损建议
   - 替代投资标的

## 输出要求
- 使用中文输出
- 语气审慎但论证有力
- 每个风险都有数据支撑
- 如果是第一轮发言，直接给出看跌分析
- 如果是后续轮次，需要回应看涨方的论点"""

    def bear_researcher_node(state: dict) -> dict:
        company = state.get("company_of_interest", "")
        trade_date = state.get("trade_date", "")

        info = get_market_info(company)
        currency = info["currency"]

        market_report = state.get("market_report", "暂无市场分析报告")
        fundamentals_report = state.get("fundamentals_report", "暂无基本面分析报告")
        news_report = state.get("news_report", "暂无新闻分析报告")
        sentiment_report = state.get("sentiment_report", "暂无情绪分析报告")

        debate_state = state.get("investment_debate_state", {})
        bull_history = debate_state.get("bull_history", [])
        bear_history = debate_state.get("bear_history", [])
        debate_count = debate_state.get("count", 0)

        logger.info(f"🐻 [看跌研究员] 开始分析: {company}, 辩论轮次: {debate_count + 1}")

        context = f"""请为 {company}（货币: {currency}）构建看跌风险分析。

## 各分析师报告摘要：
### 市场技术分析报告：
{market_report}

### 基本面分析报告：
{fundamentals_report}

### 新闻分析报告：
{news_report}

### 社交媒体情绪报告：
{sentiment_report}
"""

        if bull_history:
            context += f"\n## 看涨方最新观点（请逐一反驳）：\n{bull_history[-1]}\n"

        if debate_count > 0:
            context += f"\n注意：这是第 {debate_count + 1} 轮辩论，请针对对方的论点进行有力的回应。"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=context)
        ]

        response = llm.invoke(messages)

        new_bear_history = bear_history + [response.content]
        new_history = debate_state.get("history", []) + [
            f"看跌研究员: {response.content[:200]}..."
        ]

        logger.info(f"🐻 [看跌研究员] 分析完成")
        return {
            "messages": [response],
            "sender": "Bear Researcher",
            "investment_debate_state": {
                "bull_history": bull_history,
                "bear_history": new_bear_history,
                "history": new_history,
                "current_response": response.content,
                "judge_decision": debate_state.get("judge_decision", ""),
                "count": debate_count + 1,
            }
        }

    return bear_researcher_node
