"""
看涨研究员 - 多头分析
从积极角度构建投资论证，寻找买入理由
"""
import logging
from langchain_core.messages import SystemMessage, HumanMessage

from dataflows.tools import get_market_info

logger = logging.getLogger("stock_analyzer.researchers.bull")


def create_bull_researcher(llm):
    """
    创建看涨（多头）研究员节点

    Args:
        llm: LangChain LLM 实例

    Returns:
        看涨研究员函数节点
    """
    system_prompt = """你是一位经验丰富的**看涨研究员（Bull Researcher）**，你的职责是从积极的角度分析投资机会。

## 你的角色

你是投资辩论中的**多方代表**，需要为投资该股票构建强有力的论证。你的分析应当：
- 聚焦于公司的优势和增长潜力
- 寻找被市场忽视的价值
- 反驳看跌方的悲观论点

## 分析框架

1. **竞争优势**：
   - 公司的核心竞争力和护城河
   - 行业地位和市场份额
   - 品牌价值和技术壁垒

2. **增长催化剂**：
   - 即将到来的增长催化剂
   - 新产品/新市场/新业务
   - 管理层的战略执行力

3. **估值吸引力**：
   - 当前估值是否被低估
   - 与历史估值和行业平均对比
   - 潜在的估值修复空间

4. **反驳看跌观点**：
   - 针对看跌方提出的风险逐一反驳
   - 说明为何这些风险已被定价或可被克服
   - 将风险转化为机遇

5. **投资论证**：
   - 明确的多头逻辑链
   - 合理的价格目标区间
   - 建议的持仓比例

## 输出要求
- 使用中文输出
- 语气专业但充满信心
- 数据支撑每个论点
- 如果是第一轮发言，直接给出看涨分析
- 如果是后续轮次，需要回应看跌方的论点"""

    def bull_researcher_node(state: dict) -> dict:
        company = state.get("company_of_interest", "")
        trade_date = state.get("trade_date", "")

        # 获取市场信息
        info = get_market_info(company)
        currency = info["currency"]

        # 获取各方报告
        market_report = state.get("market_report", "暂无市场分析报告")
        fundamentals_report = state.get("fundamentals_report", "暂无基本面分析报告")
        news_report = state.get("news_report", "暂无新闻分析报告")
        sentiment_report = state.get("sentiment_report", "暂无情绪分析报告")

        # 获取辩论状态
        debate_state = state.get("investment_debate_state", {})
        bull_history = debate_state.get("bull_history", [])
        bear_history = debate_state.get("bear_history", [])
        debate_count = debate_state.get("count", 0)

        logger.info(f"🐂 [看涨研究员] 开始分析: {company}, 辩论轮次: {debate_count + 1}")

        # 构建消息
        context = f"""请为 {company}（货币: {currency}）构建看涨投资论证。

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

        if bear_history:
            context += f"\n## 看跌方最新观点（请逐一反驳）：\n{bear_history[-1]}\n"

        if debate_count > 0:
            context += f"\n注意：这是第 {debate_count + 1} 轮辩论，请针对对方的论点进行有力的回应。"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=context)
        ]

        response = llm.invoke(messages)

        # 更新辩论状态
        new_bull_history = bull_history + [response.content]
        new_history = debate_state.get("history", []) + [
            f"看涨研究员: {response.content[:200]}..."
        ]

        logger.info(f"🐂 [看涨研究员] 分析完成")
        return {
            "messages": [response],
            "sender": "Bull Researcher",
            "investment_debate_state": {
                "bull_history": new_bull_history,
                "bear_history": bear_history,
                "history": new_history,
                "current_response": response.content,
                "judge_decision": debate_state.get("judge_decision", ""),
                "count": debate_count + 1,
            }
        }

    return bull_researcher_node
