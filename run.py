#!/usr/bin/env python3
"""
单股分析系统 - 命令行入口
精简自 TradingAgents-CN

用法:
    python run.py AAPL                  # 分析苹果公司
    python run.py 600519 2025-03-31     # 分析贵州茅台，指定日期
    python run.py NVDA --date 2025-03-31  # 分析英伟达
"""
import argparse
import sys
import os
from dotenv import load_dotenv

# 将当前目录添加到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyzer import StockAnalyzer


def main():
    # 加载 .env 文件中的环境变量
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="📊 单股分析系统 - 基于多智能体LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run.py AAPL                  # 分析苹果（美股）
  python run.py 600519                # 分析贵州茅台（A股）
  python run.py 00700                 # 分析腾讯（港股）
  python run.py NVDA --date 2025-03-31
        """
    )
    parser.add_argument("stock", help="股票代码（如 AAPL, 600519, 00700）")
    parser.add_argument("--date", "-d", default=None, help="分析日期（YYYY-MM-DD），默认今天")
    parser.add_argument("--debug", action="store_true", help="启用调试日志")

    args = parser.parse_args()

    # 设置环境变量
    if args.debug:
        os.environ["LOG_LEVEL"] = "DEBUG"

    # 检查 API Key
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  未设置 OPENAI_API_KEY 环境变量")
        print("   请通过以下方式之一设置:")
        print("   1. 在 .env 文件中设置 OPENAI_API_KEY='your-key' (推荐)")
        print("   2. export OPENAI_API_KEY='your-key'")
        print()
        sys.exit(1)

    # 创建分析器
    config = None
    if os.environ.get("LLM_PROVIDER") or os.environ.get("LLM_MODEL") or os.environ.get("BACKEND_URL"):
        from config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG.copy()
        if os.environ.get("LLM_PROVIDER"):
            config["llm_provider"] = os.environ.get("LLM_PROVIDER")
        if os.environ.get("LLM_MODEL"):
            config["llm_model"] = os.environ.get("LLM_MODEL")
        if os.environ.get("BACKEND_URL"):
            config["backend_url"] = os.environ.get("BACKEND_URL")

    analyzer = StockAnalyzer(config=config)

    # 进度回调
    def progress(msg):
        print(f"\n  {msg}")

    # 执行分析
    try:
        result = analyzer.analyze(
            stock_code=args.stock,
            trade_date=args.date,
            progress_callback=progress,
        )

        # 打印摘要
        analyzer.print_summary(result)

    except KeyboardInterrupt:
        print("\n\n⚠️  分析被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
