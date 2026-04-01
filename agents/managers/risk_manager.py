"""
风险管理经理 - 风险评估
综合激进、保守、中性三方风险分析，做出风险评级
"""
import logging
from langchain_core.messages import SystemMessage, HumanMessage

from dataflows.tools import get_market_info

logger = logging.getLogger("stock_analyzer.managers.risk")


def _create_risk_debater(role: str, style: str):
    """创建风险讨论者的系统提示"""
    prompts = {
        "aggressive": """你是一位**激进型风险分析师**，你倾向于承担更高风险以追求更高收益。

你的分析角度：
- 风险中蕴含的机会
- 最大回撤的可接受性
- 仓位可以适当放大
- 关注潜在的超额收益

请基于提供的投资计划，从激进角度评估风险。""",
        "conservative": """你是一位**保守型风险分析师**，你注重资本安全和风险控制。

你的分析角度：
- 可能的最大损失
- 下行风险的严重性
- 严格的风险控制措施
- 止损的必要性

请基于提供的投资计划，从保守角度评估风险。""",
        "neutral": """你是一位**中性风险分析师**，你追求风险和收益的平衡。

你的分析角度：
- 风险收益比的合理性
- 波动率的可接受范围
- 适度的仓位配置
- 动态调整策略

请基于提供的投资计划，从中性角度评估风险。""",
    }
    return prompts.get(style, prompts["neutral"])


def create_risk_manager(llm):
    """
    创建风险管理经理节点（包含三位风险分析师 + 风险经理决策）

    由于精简流程，这里将三方讨论合并为一次LLM调用中的综合分析。
    """
    system_prompt = """你是一位资深的**风险管理经理（Risk Manager）**，负责对投资决策进行风险评估。

## 你的职责

你需要在收到交易员的交易计划后，组织风险评估讨论，然后做出最终的风险决策。

## 风险评估框架

### 1. 激进视角（追求收益）
- 当前风险水平是否可以接受？
- 是否有放大仓位的理由？
- 可能的超额收益空间？

### 2. 保守视角（控制风险）
- 最大可能损失是多少？
- 止损是否合理设置？
- 是否需要降低仓位？

### 3. 中性视角（平衡分析）
- 风险收益比是否合理？
- 仓位配置是否适中？
- 是否需要动态调整？

## 最终决策

综合三方意见后，你需要给出：
- **风险等级**: 低/中低/中/中高/高
- **仓位调整建议**: 建议的实际仓位（可能调整交易员的建议）
- **止损调整建议**: 建议的止损位
- **风控措施**: 需要执行的额外风控措施
- **总体建议**: 批准/有条件批准/拒绝

## 输出格式

```
## 风险评估结论

**风险等级**: [低/中低/中/中高/高]
**仓位建议**: [调整后的仓位百分比]
**止损建议**: [调整后的止损价格]
**总体建议**: [批准/有条件批准/拒绝]

## 风险讨论摘要

### 激进分析师观点：
[摘要]

### 保守分析师观点：
[摘要]

### 中性分析师观点：
[摘要]

## 风控措施

[具体的风控措施和条件]
```"""

    def risk_manager_node(state: dict) -> dict:
        company = state.get("company_of_interest", "")
        trade_date = state.get("trade_date", "")

        info = get_market_info(company)
        currency = info["currency"]

        investment_plan = state.get("investment_plan", "暂无投资计划")
        trader_plan = state.get("trader_investment_plan", "暂无交易员计划")

        market_report = state.get("market_report", "暂无")
        fundamentals_report = state.get("fundamentals_report", "暂无")

        logger.info(f"🎯 [风险管理经理] 开始风险评估: {company}")

        context = f"""请对以下投资计划进行风险评估：

## 目标公司: {company}（货币: {currency}）
## 分析日期: {trade_date}

## 研究经理的投资计划：
{investment_plan}

## 交易员的交易计划：
{trader_plan}

## 参考信息：
### 市场分析：
{market_report[:500] if market_report else '暂无'}

### 基本面分析：
{fundamentals_report[:500] if fundamentals_report else '暂无'}

请从激进、保守、中性三个角度分别评估风险，然后给出最终的风险管理决策。"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=context)
        ]

        response = llm.invoke(messages)

        logger.info(f"🎯 [风险管理经理] 风险评估完成")

        risk_debate_state = state.get("risk_debate_state", {
            "risky_history": [], "safe_history": [], "neutral_history": [],
            "history": [], "latest_speaker": "Risk Manager",
            "current_risky_response": "", "current_safe_response": "",
            "current_neutral_response": "", "judge_decision": "",
            "count": 0,
        })

        return {
            "messages": [response],
            "sender": "Risk Manager",
            "risk_debate_state": {
                **risk_debate_state,
                "judge_decision": response.content,
                "count": 1,
            },
            "final_trade_decision": response.content,
        }

    return risk_manager_node
