"""
研究经理 - 投资决策
综合多空研究员的辩论，做出最终投资决策和计划
"""
import logging
from langchain_core.messages import SystemMessage, HumanMessage

from dataflows.tools import get_market_info

logger = logging.getLogger("stock_analyzer.managers.research")


def create_research_manager(llm):
    """
    创建研究经理节点

    Args:
        llm: LangChain LLM 实例

    Returns:
        研究经理函数节点
    """
    system_prompt = """你是一位资深的**研究经理（Research Manager）**，负责主持投资辩论并做出最终投资决策。

## 你的角色

你是一位公正的投资决策者，需要综合多空双方的论点，做出理性的投资判断。

## 决策框架

1. **论点评估**：
   - 客观评估多空双方的论点强度
   - 识别哪些论点更有说服力
   - 判断哪些风险已被充分定价

2. **综合判断**：
   - 综合技术面、基本面、新闻面、情绪面
   - 评估当前时点的投资性价比
   - 考虑市场整体环境

3. **投资决策**：
   - 明确的投资建议（强烈推荐买入/买入/持有/卖出/强烈推荐卖出）
   - 目标价格区间
   - 建议的仓位配置
   - 止损位和止盈位

4. **风险提示**：
   - 关键风险因素
   - 需要持续关注的指标
   - 可能改变判断的触发条件

## 输出格式要求

请按以下格式输出你的投资计划：

```
## 投资决策

**投资建议**: [强烈推荐买入/买入/持有/卖出/强烈推荐卖出]
**目标价格**: [价格区间]
**止损价格**: [价格]
**建议仓位**: [百分比]

## 决策理由

[详细的决策理由，引用多空双方的论点]

## 关键风险

[列出关键风险因素]

## 后续关注

[需要持续关注的事项和触发条件]
```"""

    def research_manager_node(state: dict) -> dict:
        company = state.get("company_of_interest", "")
        trade_date = state.get("trade_date", "")

        info = get_market_info(company)
        currency = info["currency"]

        market_report = state.get("market_report", "暂无")
        fundamentals_report = state.get("fundamentals_report", "暂无")
        news_report = state.get("news_report", "暂无")
        sentiment_report = state.get("sentiment_report", "暂无")

        debate_state = state.get("investment_debate_state", {})
        bull_history = debate_state.get("bull_history", [])
        bear_history = debate_state.get("bear_history", [])

        logger.info(f"👔 [研究经理] 开始决策: {company}")

        context = f"""请基于以下信息做出投资决策：

## 目标公司: {company}（货币: {currency}）
## 分析日期: {trade_date}

## 各分析师报告：

### 市场技术分析：
{market_report}

### 基本面分析：
{fundamentals_report}

### 新闻分析：
{news_report}

### 社交媒体情绪：
{sentiment_report}

## 投资辩论记录：

### 看涨方论点：
{chr(10).join([f'[第{i+1}轮] {h}' for i, h in enumerate(bull_history)]) if bull_history else '无'}

### 看跌方论点：
{chr(10).join([f'[第{i+1}轮] {h}' for i, h in enumerate(bear_history)]) if bear_history else '无'}

请综合以上所有信息，做出你的投资决策。"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=context)
        ]

        response = llm.invoke(messages)

        logger.info(f"👔 [研究经理] 决策完成")
        return {
            "messages": [response],
            "sender": "Research Manager",
            "investment_debate_state": {
                **debate_state,
                "judge_decision": response.content,
            },
            "investment_plan": response.content,
        }

    return research_manager_node
