import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import os
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.factor_model.factor_model import FactorModel
from src.backtest.backtest_engine import BacktestEngine
from src.data.data_loader import generate_synthetic_data
from src.factors.time_series_factors import TimeSeriesFactors

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def factor_strategy(factor, data, top_pct=0.3, bottom_pct=0.3):
    """
    基于因子值的简单策略
    
    参数:
    factor (pd.Series): 因子值
    data (pd.DataFrame): 当前交易日数据
    top_pct (float): 买入的顶部百分比
    bottom_pct (float): 卖出的底部百分比
    
    返回:
    dict: 目标仓位 {symbol: quantity}
    """
    # 对因子值进行排序
    sorted_factor = factor.sort_values(ascending=False)
    
    # 确定买入和卖出的阈值
    n_assets = len(sorted_factor)
    top_n = int(n_assets * top_pct)
    bottom_n = int(n_assets * bottom_pct)
    
    # 获取买入和卖出的资产
    buy_assets = sorted_factor.iloc[:top_n].index
    sell_assets = sorted_factor.iloc[-bottom_n:].index
    
    # 创建目标仓位
    target_positions = {}
    
    # 假设我们有100万资金，平均分配给买入的资产
    capital_per_asset = 1000000 / len(buy_assets) if len(buy_assets) > 0 else 0
    
    for asset in buy_assets:
        # 计算可以买入的数量
        price = data[asset]['close']
        quantity = int(capital_per_asset / price)
        target_positions[asset] = quantity
    
    for asset in sell_assets:
        # 卖出资产
        target_positions[asset] = 0
    
    return target_positions

def main():
    # 生成合成数据
    logger.info("生成合成股票数据...")
    
    # 生成10只股票的数据
    stocks = {}
    for i in range(10):
        symbol = f'STOCK_{i+1}'
        stocks[symbol] = generate_synthetic_data(n_samples=1000, start_date='2010-01-01', freq='D', seed=42+i)
    
    # 合并数据
    all_data = {}
    for symbol, data in stocks.items():
        all_data[symbol] = data
    
    # 计算因子
    logger.info("计算时间序列因子...")
    ts_factors = TimeSeriesFactors()
    
    # 为每只股票计算因子
    factors = {}
    for symbol, data in stocks.items():
        # 计算动量因子
        factors[f'{symbol}_momentum_10'] = ts_factors.calculate_momentum(data['close'], window=10)
        
        # 计算均值回归因子
        factors[f'{symbol}_mean_reversion_5'] = ts_factors.calculate_mean_reversion(data['close'], window=5)
        
        # 计算RSI因子
        factors[f'{symbol}_rsi_14'] = ts_factors.calculate_rsi(data['close'], window=14)
    
    # 创建因子DataFrame
    factors_df = pd.DataFrame(factors)
    
    # 创建收益率DataFrame
    returns_df = pd.DataFrame({symbol: data['close'].pct_change() for symbol, data in stocks.items()})
    
    # 创建因子模型
    logger.info("创建因子模型...")
    model = FactorModel()
    model.load_data(factors_df, returns_df)
    
    # 评估因子
    logger.info("评估因子...")
    evaluation_results = model.evaluate_factors()
    
    # 选择因子
    logger.info("选择因子...")
    selected_factors = model.select_factors(method='combined', ic_threshold=0.02, correlation_threshold=0.7)
    print(f"选中的因子: {selected_factors}")
    
    # 优化因子权重
    logger.info("优化因子权重...")
    weights = model.optimize_weights(method='regression')
    
    # 生成组合因子
    logger.info("生成组合因子...")
    combined_factor = model.generate_combined_factor()
    
    # 创建回测引擎
    logger.info("创建回测引擎...")
    backtest = BacktestEngine(initial_capital=1000000.0)
    
    # 准备回测数据
    backtest_data = {}
    for date in combined_factor.index:
        backtest_data[date] = {}
        for symbol in stocks.keys():
            if date in stocks[symbol].index:
                backtest_data[date][symbol] = stocks[symbol].loc[date]
    
    # 转换为DataFrame
    backtest_data = pd.DataFrame(backtest_data).T
    
    # 准备因子数据
    factor_data = pd.DataFrame()
    for date in combined_factor.index:
        factor_values = {}
        for symbol in stocks.keys():
            # 使用组合因子中包含该股票的因子
            symbol_factors = [f for f in combined_factor.index if symbol in f]
            if symbol_factors:
                # 计算该股票的平均因子值
                factor_values[symbol] = combined_factor.loc[symbol_factors].mean()
        
        if factor_values:
            factor_data[date] = pd.Series(factor_values)
    
    # 转置因子数据
    factor_data = factor_data.T
    
    # 运行回测
    logger.info("运行回测...")
    results = backtest.run_backtest(
        backtest_data,
        factor_data,
        factor_strategy,
        rebalance_freq='M'  # 每月再平衡
    )
    
    # 绘制回测结果
    logger.info("绘制回测结果...")
    backtest.plot_results(results)
    
    logger.info("因子策略回测示例完成!")

if __name__ == "__main__":
    main()