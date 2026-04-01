"""
单股分析系统 - 数据获取工具
提供股票行情、基本面、新闻、社交数据的获取能力
支持 A 股、港股、美股
"""
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger("stock_analyzer.tools")

# 尝试导入可选的数据源库
try:
    import baostock as bs
    import pandas as pd
    HAS_BAOSTOCK = True
except ImportError:
    HAS_BAOSTOCK = False
    logger.warning("baostock 未安装，A股数据获取功能将受限。安装: pip install baostock pandas")

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    logger.warning("yfinance 未安装，美股/港股数据获取功能将受限。安装: pip install yfinance")

try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False
    logger.warning("duckduckgo-search 未安装，新闻搜索功能将受限。安装: pip install duckduckgo-search")


def get_market_info(stock_code: str) -> dict:
    """
    判断股票所属市场并返回市场信息

    Args:
        stock_code: 股票代码

    Returns:
        dict: 包含 market, currency, symbol 信息
    """
    code = stock_code.upper().strip()

    # A股判断: 6开头上海，0/3开头深圳
    if code.isdigit() and (code.startswith('6') or code.startswith('0') or code.startswith('3')):
        market = "A股"
        currency = "人民币¥"
        symbol = code
        if code.startswith('6'):
            symbol = f"{code}.SS"
        else:
            symbol = f"{code}.SZ"
    # 港股判断: 0开头5位数字
    elif code.isdigit() and code.startswith('0') and len(code) == 5:
        market = "港股"
        currency = "港币HK$"
        symbol = f"{code}.HK"
    # 美股判断: 字母代码
    elif code.isalpha():
        market = "美股"
        currency = "美元$"
        symbol = code
    else:
        market = "未知"
        currency = ""
        symbol = code

    return {"market": market, "currency": currency, "symbol": symbol}


def get_stock_price_data(stock_code: str, period: str = "3mo") -> str:
    """
    获取股票价格数据

    Args:
        stock_code: 股票代码
        period: 时间周期 (1mo, 3mo, 6mo, 1y, 2y, 5y)

    Returns:
        str: 格式化的价格数据文本
    """
    info = get_market_info(stock_code)
    symbol = info["symbol"]
    market = info["market"]

    try:
        if market == "A股" and HAS_BAOSTOCK:
            return _get_a_stock_price(stock_code, period)
        elif HAS_YFINANCE:
            return _get_yf_price(symbol, period)
        else:
            return f"无法获取 {stock_code} 的价格数据，请安装相应的数据源库。"
    except Exception as e:
        logger.error(f"获取价格数据失败: {e}")
        return f"获取价格数据时出错: {str(e)}"


def _get_a_stock_price(code: str, period: str) -> str:
    """获取A股价格数据（通过 baostock）"""
    try:
        bs.login()
        bs_code = f"sh.{code}" if code.startswith(('6', '9')) else f"sz.{code}"
        
        # 估算最近半年的日期
        start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,close,volume",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2"  # 2: 前复权
        )
        
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
            
        bs.logout()
        
        if not data_list:
            return f"未找到 {code} 的行情数据"
            
        df = pd.DataFrame(data_list, columns=rs.fields)
        df.rename(columns={'date': '日期', 'close': '收盘', 'volume': '成交量'}, inplace=True)

        recent = df.tail(60) if len(df) > 60 else df
        result = f"股票代码: {code}\n"
        result += f"数据范围: {recent.iloc[0]['日期']} ~ {recent.iloc[-1]['日期']}\n"
        result += f"最新收盘价: {recent.iloc[-1]['收盘']}\n"
        result += f"最新成交量: {recent.iloc[-1]['成交量']}\n\n"
        result += recent.tail(20).to_string(index=False)
        return result
    except Exception as e:
        bs.logout()
        return f"A股数据获取失败: {str(e)}"


def _get_yf_price(symbol: str, period: str) -> str:
    """获取美股/港股价格数据（通过 yfinance）"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)

        if df is None or df.empty:
            return f"未找到 {symbol} 的行情数据"

        result = f"股票代码: {symbol}\n"
        result += f"数据范围: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}\n"
        result += f"最新收盘价: {df.iloc[-1]['Close']:.2f}\n"
        result += f"最新成交量: {df.iloc[-1]['Volume']:,}\n\n"
        result += df.tail(20).to_string()
        return result
    except Exception as e:
        return f"价格数据获取失败: {str(e)}"


def get_stock_fundamentals(stock_code: str) -> str:
    """
    获取股票基本面数据

    Args:
        stock_code: 股票代码

    Returns:
        str: 格式化的基本面数据文本
    """
    info = get_market_info(stock_code)
    symbol = info["symbol"]
    market = info["market"]

    try:
        if market == "A股" and HAS_BAOSTOCK:
            return _get_a_stock_fundamentals(stock_code)
        elif HAS_YFINANCE:
            return _get_yf_fundamentals(symbol)
        else:
            return f"无法获取 {stock_code} 的基本面数据。"
    except Exception as e:
        logger.error(f"获取基本面数据失败: {e}")
        return f"获取基本面数据时出错: {str(e)}"


def _get_a_stock_fundamentals(code: str) -> str:
    """获取A股基本面数据（通过 baostock）"""
    try:
        bs.login()
        bs_code = f"sh.{code}" if code.startswith(('6', '9')) else f"sz.{code}"
        
        # 1. 获取基础信息
        rs_basic = bs.query_stock_basic(code=bs_code)
        basic_data = rs_basic.get_row_data() if (rs_basic.error_code == '0' and rs_basic.next()) else None
        
        # 2. 获取最新估值指标 (通过日k线数据)
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        rs_k = bs.query_history_k_data_plus(
            bs_code,
            "date,close,peTTM,pbMRQ,psTTM,pcfNcfTTM,turn",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2"
        )
        
        k_data_list = []
        while (rs_k.error_code == '0') & rs_k.next():
            k_data_list.append(rs_k.get_row_data())
            
        # 3. 获取最新的财务数据 (EPS, ROE, 总股本等)
        current_year = datetime.now().year
        profit_dict = {}
        for year in [current_year, current_year - 1, current_year - 2]:
            for quarter in [4, 3, 2, 1]:
                rs_profit = bs.query_profit_data(code=bs_code, year=year, quarter=quarter)
                if rs_profit.error_code == '0' and rs_profit.next():
                    profit_dict = dict(zip(rs_profit.fields, rs_profit.get_row_data()))
                    break
            if profit_dict:
                break

        bs.logout()
        
        if not basic_data and not k_data_list:
             return f"未找到 {code} 的基本面数据"
             
        result = f"股票代码: {code}\n基本面数据:\n"
        
        if basic_data:
            fields = rs_basic.fields
            for field, value in zip(fields, basic_data):
                field_names = {
                    "code_name": "股票名称",
                    "ipoDate": "上市日期",
                    "outDate": "退市日期",
                    "type": "股票类型",
                    "status": "上市状态"
                }
                zh_field = field_names.get(field, field)
                if field != "code":
                    result += f"  {zh_field}: {value}\n"
                
        if k_data_list:
            df = pd.DataFrame(k_data_list, columns=rs_k.fields)
            latest = df.iloc[-1]
            close_price = float(latest['close']) if latest['close'] else 0.0
            
            result += "\n最新估值指标:\n"
            result += f"  日期: {latest['date']}\n"
            result += f"  收盘价: {latest['close']}\n"
            result += f"  滚动市盈率(PE TTM): {latest['peTTM']}\n"
            result += f"  市净率(PB MRQ): {latest['pbMRQ']}\n"
            result += f"  滚动市销率(PS TTM): {latest['psTTM']}\n"
            result += f"  滚动市现率(PCF): {latest['pcfNcfTTM']}\n"
            result += f"  换手率: {latest['turn']}%\n"
            
            if profit_dict:
                result += f"\n最新财务指标 (财报截止: {profit_dict.get('statDate', '')}):\n"
                eps = profit_dict.get('epsTTM', '')
                roe = profit_dict.get('roeAvg', '')
                net_profit = profit_dict.get('netProfit', '')
                total_share = profit_dict.get('totalShare', '')
                
                if eps:
                    result += f"  每股收益(EPS TTM): {eps}\n"
                if roe:
                    try:
                        roe_pct = float(roe) * 100
                        result += f"  净资产收益率(ROE): {roe_pct:.2f}%\n"
                    except:
                        result += f"  净资产收益率(ROE): {roe}\n"
                if net_profit:
                    try:
                        np_val = float(net_profit)
                        if np_val > 100000000:
                            result += f"  归母净利润: {np_val / 100000000:.2f} 亿\n"
                        elif np_val > 10000:
                            result += f"  归母净利润: {np_val / 10000:.2f} 万\n"
                        else:
                            result += f"  归母净利润: {np_val}\n"
                    except:
                        result += f"  归母净利润: {net_profit}\n"
                        
                if total_share and close_price:
                    try:
                        market_cap = float(total_share) * close_price
                        if market_cap > 100000000:
                            result += f"  总市值: {market_cap / 100000000:.2f} 亿\n"
                        elif market_cap > 10000:
                            result += f"  总市值: {market_cap / 10000:.2f} 万\n"
                    except:
                        pass

        return result
    except Exception as e:
        bs.logout()
        return f"A股基本面数据获取失败: {str(e)}"


def _get_yf_fundamentals(symbol: str) -> str:
    """获取美股/港股基本面数据"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        result = f"股票代码: {symbol}\n基本面数据:\n"
        field_map = {
            "shortName": "公司名称",
            "sector": "所属板块",
            "industry": "所属行业",
            "marketCap": "总市值",
            "trailingPE": "滚动市盈率(PE TTM)",
            "forwardPE": "预测市盈率(Forward PE)",
            "priceToBook": "市净率(PB)",
            "priceToSalesTrailing12Months": "市销率(PS)",
            "trailingEps": "每股收益(EPS)",
            "forwardEps": "预测每股收益",
            "dividendYield": "股息率",
            "returnOnEquity": "净资产收益率(ROE)",
            "returnOnAssets": "总资产收益率(ROA)",
            "debtToEquity": "资产负债率",
            "currentRatio": "流动比率",
            "grossMargins": "毛利率",
            "operatingMargins": "营业利润率",
            "profitMargins": "净利率",
            "revenueGrowth": "营收增长率",
            "earningsGrowth": "净利润增长率",
            "totalRevenue": "总营收",
            "totalDebt": "总负债",
            "totalCash": "总现金",
            "freeCashflow": "自由现金流",
            "fiftyTwoWeekHigh": "52周最高价",
            "fiftyTwoWeekLow": "52周最低价",
            "averageVolume": "平均成交量",
            "analystTargetPrice": "分析师目标价",
            "recommendationKey": "分析师评级"
        }
        
        for field, zh_name in field_map.items():
            if field in info and info[field] is not None:
                value = info[field]
                if field in ["marketCap", "totalRevenue", "totalDebt", "totalCash", "freeCashflow"]:
                    if isinstance(value, (int, float)) and abs(value) > 100000000:
                        value = f"{value / 100000000:,.2f} 亿"
                    elif isinstance(value, (int, float)) and abs(value) > 10000:
                        value = f"{value / 10000:,.2f} 万"
                elif field in ["dividendYield", "returnOnEquity", "returnOnAssets", "grossMargins", "operatingMargins", "profitMargins", "revenueGrowth", "earningsGrowth"]:
                    if isinstance(value, float):
                        value = f"{value * 100:.2f}%"
                elif isinstance(value, float):
                    value = f"{value:,.2f}"
                elif isinstance(value, int) and abs(value) > 1000000:
                    value = f"{value / 1000000:,.2f}M"
                result += f"  {zh_name} ({field}): {value}\n"

        return result
    except Exception as e:
        return f"基本面数据获取失败: {str(e)}"


def search_stock_news(stock_code: str, max_results: int = 10) -> str:
    """
    搜索股票相关新闻

    Args:
        stock_code: 股票代码
        max_results: 最大结果数

    Returns:
        str: 格式化的新闻文本
    """
    info = get_market_info(stock_code)

    if HAS_YFINANCE and info["market"] in ["美股", "港股"]:
        try:
            ticker = yf.Ticker(info["symbol"])
            news = ticker.news
            if news:
                result = f"股票代码: {stock_code} 相关新闻\n\n"
                for i, item in enumerate(news[:max_results]):
                    result += f"{i+1}. {item.get('title', 'N/A')}\n"
                    result += f"   来源: {item.get('publisher', 'N/A')}\n"
                    ts = item.get('providerPublishTime', 0)
                    if ts:
                        dt = datetime.fromtimestamp(ts)
                        result += f"   时间: {dt.strftime('%Y-%m-%d %H:%M')}\n"
                    result += f"   链接: {item.get('link', 'N/A')}\n\n"
                return result
        except Exception as e:
            logger.warning(f"yfinance 新闻获取失败: {e}")

    if HAS_DDGS:
        try:
            with DDGS() as ddgs:
                query = f"{stock_code} 股票 最新消息"
                results = list(ddgs.news(query, max_results=max_results))
                if results:
                    result = f"股票代码: {stock_code} 相关新闻\n\n"
                    for i, item in enumerate(results):
                        result += f"{i+1}. {item.get('title', 'N/A')}\n"
                        result += f"   来源: {item.get('source', 'N/A')}\n"
                        result += f"   时间: {item.get('date', 'N/A')}\n"
                        result += f"   摘要: {item.get('body', 'N/A')}\n\n"
                    return result
        except Exception as e:
            logger.warning(f"DDG 新闻搜索失败: {e}")

    return f"无法获取 {stock_code} 的新闻数据，请安装 yfinance 或 duckduckgo-search。"


def search_stock_sentiment(stock_code: str) -> str:
    """
    搜索股票社交媒体情绪数据

    Args:
        stock_code: 股票代码

    Returns:
        str: 格式化的情绪数据文本
    """
    if HAS_DDGS:
        try:
            with DDGS() as ddgs:
                query = f"{stock_code} 股票 讨论 评价 观点"
                results = list(ddgs.text(query, max_results=8))
                if results:
                    result = f"股票代码: {stock_code} 社交媒体讨论\n\n"
                    for i, item in enumerate(results):
                        result += f"{i+1}. {item.get('title', 'N/A')}\n"
                        result += f"   摘要: {item.get('body', 'N/A')}\n"
                        result += f"   来源: {item.get('href', 'N/A')}\n\n"
                    return result
        except Exception as e:
            logger.warning(f"情绪搜索失败: {e}")

    return f"无法获取 {stock_code} 的社交媒体情绪数据，请安装 duckduckgo-search。"
