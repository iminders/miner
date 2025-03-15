import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import os
import yfinance as yf
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.portfolio.portfolio_optimizer import PortfolioOptimizer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 设置资产列表
    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'JNJ', 'V', 'PG']
    
    # 下载历史数据
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)  # 3年数据
    
    logger.info(f"下载{len(tickers)}个资产的历史数据，时间范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
    
    # 使用yfinance下载数据
    data = yf.download(tickers, start=start_date, end=end_date)
    
    # 提取调整后的收盘价
    prices = data['Adj Close']
    
    # 计算日收益率
    returns = prices.pct_change().dropna()
    
    logger.info(f"数据下载完成，共{len(returns)}个交易日")
    
    # 创建投资组合优化器
    optimizer = PortfolioOptimizer()
    
    # 生成有效前沿
    logger.info("生成有效前沿...")
    ef_results = optimizer.generate_efficient_frontier(returns, num_portfolios=1000)
    
    # 绘制有效前沿
    optimizer.plot_efficient_frontier(ef_results)
    
    # 使用不同方法优化投资组合
    logger.info("使用不同方法优化投资组合...")
    
    # 最小方差投资组合
    min_var_result = optimizer.optimize_