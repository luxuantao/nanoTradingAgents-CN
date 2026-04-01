"""
单股分析系统 - 核心入口
精简自 TradingAgents-CN，仅保留单股分析流程
"""
import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from langchain_openai import ChatOpenAI

from config import DEFAULT_CONFIG
from graph.builder import build_analysis_graph

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("stock_analyzer")


class StockAnalyzer:
    """
    单股分析器 - 基于多智能体LLM的股票分析系统

    分析流程:
      1. 四位分析师（市场/基本面/新闻/社交）并行获取数据并分析
      2. 多空研究员进行投资辩论
      3. 研究经理综合决策
      4. 交易员制定执行计划
      5. 风险管理经理评估风险

    用法:
        analyzer = StockAnalyzer()
        result = analyzer.analyze("AAPL", "2025-03-31")
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化分析器

        Args:
            config: 配置字典，None则使用默认配置
        """
        self.config = config or DEFAULT_CONFIG
        self._setup_logging()
        self._init_llm()
        self._build_graph()
        logger.info("🎉 单股分析器初始化完成")

    def _setup_logging(self):
        """配置日志级别"""
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        logging.getLogger().setLevel(log_level)

    def _init_llm(self):
        """初始化 LLM 实例"""
        provider = self.config.get("llm_provider", "openai")
        llm_model = self.config.get("llm_model", "gpt-4o")
        backend_url = self.config.get("backend_url", "https://api.openai.com/v1")
        temperature = self.config.get("temperature", 0.7)
        max_tokens = self.config.get("max_tokens", 4000)

        logger.info(f"🤖 LLM配置: provider={provider}, model={llm_model}")

        # 统一使用一个 LLM 实例（简化版）
        self.llm = ChatOpenAI(
            model=llm_model,
            base_url=backend_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=180,
        )

        logger.info(f"✅ LLM 初始化完成: {provider}/{llm_model}")

    def _build_graph(self):
        """构建分析工作流图"""
        selected_analysts = ["market", "fundamentals", "news", "social"]
        self.graph = build_analysis_graph(self.llm, selected_analysts)

    def analyze(
        self,
        stock_code: str,
        trade_date: str = None,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """
        分析一只股票

        Args:
            stock_code: 股票代码（如 600519, AAPL, 00700）
            trade_date: 分析日期（默认今天）
            progress_callback: 进度回调函数

        Returns:
            dict: 包含分析结果和决策的字典
        """
        if trade_date is None:
            trade_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"{'='*60}")
        logger.info(f"📊 开始分析: {stock_code}")
        logger.info(f"📅 分析日期: {trade_date}")
        logger.info(f"{'='*60}")

        # 构建初始状态
        initial_state = {
            "company_of_interest": stock_code,
            "trade_date": trade_date,
            "sender": "system",
            "messages": [],
            # 报告（初始为空）
            "market_report": "",
            "sentiment_report": "",
            "news_report": "",
            "fundamentals_report": "",
            # 工具调用计数器
            "market_tool_calls": 0,
            "news_tool_calls": 0,
            "social_tool_calls": 0,
            "fundamentals_tool_calls": 0,
            # 辩论状态
            "investment_debate_state": {
                "bull_history": [],
                "bear_history": [],
                "history": [],
                "current_response": "",
                "judge_decision": "",
                "count": 0,
            },
            "risk_debate_state": {
                "risky_history": [],
                "safe_history": [],
                "neutral_history": [],
                "history": [],
                "latest_speaker": "",
                "current_risky_response": "",
                "current_safe_response": "",
                "current_neutral_response": "",
                "judge_decision": "",
                "count": 0,
            },
            # 决策
            "investment_plan": "",
            "trader_investment_plan": "",
            "final_trade_decision": "",
        }

        # 执行图
        start_time = time.time()
        node_timings = {}
        current_node = None
        current_node_start = None

        final_state = None
        for chunk in self.graph.stream(initial_state):
            for node_name in chunk.keys():
                if not node_name.startswith("__"):
                    # 记录节点计时
                    if current_node and current_node_start:
                        elapsed = time.time() - current_node_start
                        node_timings[current_node] = elapsed
                        logger.info(f"⏱️ [{current_node}] 耗时: {elapsed:.2f}秒")

                    current_node = node_name
                    current_node_start = time.time()

                    # 发送进度
                    if progress_callback:
                        progress_map = {
                            "Market Analyst": "📊 市场分析师 - 技术分析中...",
                            "Fundamentals Analyst": "💼 基本面分析师 - 财务分析中...",
                            "News Analyst": "📰 新闻分析师 - 新闻分析中...",
                            "Social Media Analyst": "💬 社交媒体分析师 - 情绪分析中...",
                            "Bull Researcher": "🐂 看涨研究员 - 多方论证中...",
                            "Bear Researcher": "🐻 看跌研究员 - 空方论证中...",
                            "Research Manager": "👔 研究经理 - 综合决策中...",
                            "Trader": "💼 交易员 - 制定交易计划中...",
                            "Risk Manager": "🎯 风险管理经理 - 风险评估中...",
                        }
                        msg = progress_map.get(node_name, f"🔍 {node_name}")
                        progress_callback(msg)

            # 累积状态
            if final_state is None:
                final_state = initial_state.copy()
            for node_name, node_update in chunk.items():
                if not node_name.startswith("__"):
                    final_state.update(node_update)

        # 记录最后一个节点
        if current_node and current_node_start:
            elapsed = time.time() - current_node_start
            node_timings[current_node] = elapsed
            logger.info(f"⏱️ [{current_node}] 耗时: {elapsed:.2f}秒")

        total_time = time.time() - start_time

        # 构建结果
        result = self._build_result(final_state, stock_code, trade_date, node_timings, total_time)

        # 保存结果
        self._save_result(result, stock_code)

        logger.info(f"{'='*60}")
        logger.info(f"✅ 分析完成: {stock_code}")
        logger.info(f"⏱️ 总耗时: {total_time:.2f}秒")
        logger.info(f"{'='*60}")

        return result

    def _build_result(
        self,
        state: dict,
        stock_code: str,
        trade_date: str,
        node_timings: dict,
        total_time: float,
    ) -> Dict[str, Any]:
        """构建分析结果"""
        return {
            "stock_code": stock_code,
            "trade_date": trade_date,
            "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_time_seconds": round(total_time, 2),
            "node_timings": {k: round(v, 2) for k, v in node_timings.items()},
            # 四份分析报告
            "reports": {
                "market": state.get("market_report", ""),
                "fundamentals": state.get("fundamentals_report", ""),
                "news": state.get("news_report", ""),
                "sentiment": state.get("sentiment_report", ""),
            },
            # 投资辩论
            "debate": {
                "bull_arguments": state.get("investment_debate_state", {}).get("bull_history", []),
                "bear_arguments": state.get("investment_debate_state", {}).get("bear_history", []),
            },
            # 决策结果
            "investment_plan": state.get("investment_plan", ""),
            "trader_plan": state.get("trader_investment_plan", ""),
            "final_decision": state.get("final_trade_decision", ""),
        }

    def _save_result(self, result: Dict[str, Any], stock_code: str):
        """保存分析结果到文件"""
        results_dir = Path(self.config.get("results_dir", "./results"))
        results_dir.mkdir(parents=True, exist_ok=True)

        # 保存完整 JSON
        output_file = results_dir / f"{stock_code}_{result['trade_date']}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"📁 结果已保存: {output_file}")

        # 保存可读的 Markdown 报告
        md_file = results_dir / f"{stock_code}_{result['trade_date']}.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(self._generate_markdown_report(result))
        logger.info(f"📄 报告已保存: {md_file}")

    def _generate_markdown_report(self, result: Dict[str, Any]) -> str:
        """生成 Markdown 格式的分析报告"""
        md = f"""# 📊 {result['stock_code']} 单股分析报告

> 分析日期: {result['trade_date']}  
> 生成时间: {result['analysis_time']}  
> 总耗时: {result['total_time_seconds']}秒

---

## 一、市场技术分析

{result['reports']['market'] or '暂无数据'}

---

## 二、基本面分析

{result['reports']['fundamentals'] or '暂无数据'}

---

## 三、新闻分析

{result['reports']['news'] or '暂无数据'}

---

## 四、社交媒体情绪分析

{result['reports']['sentiment'] or '暂无数据'}

---

## 五、投资辩论

### 🐂 看涨方观点

{chr(10).join([f'**第{i+1}轮:** {arg}' for i, arg in enumerate(result['debate']['bull_arguments'])]) or '无'}

### 🐻 看跌方观点

{chr(10).join([f'**第{i+1}轮:** {arg}' for i, arg in enumerate(result['debate']['bear_arguments'])]) or '无'}

---

## 六、投资决策

{result['investment_plan'] or '暂无'}

---

## 七、交易执行计划

{result['trader_plan'] or '暂无'}

---

## 八、风险评估

{result['final_decision'] or '暂无'}

---

*本报告由多智能体LLM系统自动生成，仅供参考，不构成投资建议。*
"""
        return md

    def print_summary(self, result: Dict[str, Any]):
        """打印分析结果摘要"""
        print(f"\n{'='*60}")
        print(f"📊 {result['stock_code']} 分析报告摘要")
        print(f"📅 日期: {result['trade_date']}")
        print(f"⏱️ 耗时: {result['total_time_seconds']}秒")
        print(f"{'='*60}")

        # 投资决策
        if result.get("investment_plan"):
            print(f"\n👔 【研究经理决策】")
            # 提取关键信息的前几行
            lines = result["investment_plan"].split("\n")[:10]
            for line in lines:
                print(f"  {line}")

        # 交易计划
        if result.get("trader_plan"):
            print(f"\n💼 【交易员计划】")
            lines = result["trader_plan"].split("\n")[:10]
            for line in lines:
                print(f"  {line}")

        # 风险评估
        if result.get("final_decision"):
            print(f"\n🎯 【风险评估】")
            lines = result["final_decision"].split("\n")[:10]
            for line in lines:
                print(f"  {line}")

        print(f"\n📁 详细报告已保存到 ./results/ 目录")
        print(f"{'='*60}\n")
