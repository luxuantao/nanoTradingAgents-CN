"""
单股分析系统 - 默认配置
基于 TradingAgents-CN 精简而来，仅保留单股分析流程
"""
import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_CONFIG = {
    # ========== 目录路径 ==========
    "project_dir": os.path.abspath(os.path.dirname(__file__)),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),

    # ========== LLM 配置 ==========
    "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
    "llm_model": os.getenv("LLM_MODEL", "gpt-4o"),
    "backend_url": os.getenv("BACKEND_URL", "https://api.openai.com/v1"),
    "temperature": float(os.getenv("TEMPERATURE", "0.7")),
    "max_tokens": int(os.getenv("MAX_TOKENS", "4000")),

    # ========== 辩论配置 ==========
    "max_debate_rounds": int(os.getenv("MAX_DEBATE_ROUNDS", "1")),
    "max_risk_discuss_rounds": int(os.getenv("MAX_RISK_DISCUSS_ROUNDS", "1")),
}
