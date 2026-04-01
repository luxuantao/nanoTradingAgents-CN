"""
交易员 - 执行交易计划
基于研究经理的投资计划，制定具体的交易执行方案
"""
import logging
from langchain_core.messages import SystemMessage, HumanMessage

from dataflows.tools import get_market_info

logger = logging.getLogger("stock_analyzer.trader")


def create_trader(llm):
    """
    创建交易员节点

    Args:
        llm: LangChain LLM 实例

    Returns:
        交易员函数节点
    """
    system_prompt = """你是一位专业的**交易员（Trader）**，负责将投资计划转化为具体的交易执行方案。

## 你的角色

你是投资决策的最后执行者，需要将研究经理的投资建议细化为可执行的交易计划。

## 交易计划制定框架

1. **入场策略**：
   - 入场时机（立即/等待回调/分批建仓）
   - 入场价格区间
   - 分批建仓的具体方案

2. **仓位管理**：
   - 严格按照风险管理的仓位建议
   - 分批买入/卖出的节奏
   - 最大持仓限制

3. **止损策略**：
   - 止损位设置（技术面/基本面）
   - 止损方式（价格止损/时间止损）
   - 止损后的应对方案

4. **止盈策略**：
   - 目标价位和止盈区间
   - 分批止盈方案
   - 持有期限建议

5. **异常应对**：
   - 跳空低开/高开的应对
   - 突发利空的应对
   - 市场整体下跌的应对

## 输出格式

```
## 交易执行计划

**最终交易建议**: **[买入/持有/卖出]**
**目标价格**: [价格区间]
**止损价格**: [价格]
**建议仓位**: [占总资金百分比]
**风险评分**: [1-10分，10为最高风险]
**置信度**: [高/中/低]

## 执行方案

[详细的入场、持仓、退出方案]

## 注意事项

[需要特别注意的风险和机会]
```"""

    def trader_node(state: dict) -> dict:
        company = state.get("company_of_interest", "")
        trade_date = state.get("trade_date", "")

        info = get_market_info(company)
        currency = info["currency"]

        investment_plan = state.get("investment_plan", "暂无投资计划")
        market_report = state.get("market_report", "暂无")
        fundamentals_report = state.get("fundamentals_report", "暂无")
        news_report = state.get("news_report", "暂无")
        sentiment_report = state.get("sentiment_report", "暂无")

        logger.info(f"💼 [交易员] 开始制定交易计划: {company}")

        context = f"""请基于以下投资计划制定交易执行方案：

## 目标公司: {company}（货币: {currency}）
## 分析日期: {trade_date}

## 研究经理的投资计划：
{investment_plan}

## 参考分析报告：

### 市场技术分析：
{market_report[:500] if market_report else '暂无'}

### 基本面分析：
{fundamentals_report[:500] if fundamentals_report else '暂无'}

### 新闻分析：
{news_report[:300] if news_report else '暂无'}

### 社交媒体情绪：
{sentiment_report[:300] if sentiment_report else '暂无'}

请制定详细的交易执行计划。注意：
1. 交易建议必须明确（买入/持有/卖出）
2. 必须提供具体的目标价位和止损价位
3. 必须给出置信度评分"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=context)
        ]

        response = llm.invoke(messages)

        logger.info(f"💼 [交易员] 交易计划制定完成")
        return {
            "messages": [response],
            "sender": "Trader",
            "trader_investment_plan": response.content,
        }

    return trader_node
