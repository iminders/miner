import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.factor_model.factor_model import FactorModel
from src.data.data_loader import load_stock_data
from src.factors.time_series_factors import TimeSeriesFactors

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 加载股票数据
    logger.info("加载股票数据...")
    stock_data = load_stock_data('path/to/stock_data.csv')  # 替换为实际数据路径
    
    # 计算收益率
    stock_data['returns'] = stock_data['close'].pct_change()
    
    # 计算因子
    logger.info("计算时间序列因子...")
    ts_factors = TimeSeriesFactors()
    
    # 计算各种因子
    factors_df = pd.DataFrame(index=stock_data.index)
    
    # 动量因子
    factors_df['momentum_5'] = ts_factors.calculate_momentum(stock_data['close'], window=5)
    factors_df['momentum_10'] = ts_factors.calculate_momentum(stock_data['close'], window=10)
    factors_df['momentum_20'] = ts_factors.calculate_momentum(stock_data['close'], window=20)
    
    # 均值回归因子
    factors_df['mean_reversion_5'] = ts_factors.calculate_mean_reversion(stock_data['close'], window=5)
    factors_df['mean_reversion_10'] = ts_factors.calculate_mean_reversion(stock_data['close'], window=10)
    
    # 波动率因子
    factors_df['volatility_10'] = ts_factors.calculate_volatility(stock_data['returns'], window=10)
    factors_df['volatility_20'] = ts_factors.calculate_volatility(stock_data['returns'], window=20)
    
    # RSI因子
    factors_df['rsi_14'] = ts_factors.calculate_rsi(stock_data['close'], window=14)
    
    # MACD因子
    macd, signal, hist = ts_factors.calculate_macd(stock_data['close'])
    factors_df['macd'] = macd
    factors_df['macd_signal'] = signal
    factors_df['macd_hist'] = hist
    
    # 成交量因子
    factors_df['volume_momentum_5'] = ts_factors.calculate_volume_momentum(stock_data['volume'], window=5)
    factors_df['volume_momentum_10'] = ts_factors.calculate_volume_momentum(stock_data['volume'], window=10)
    
    # 去除NaN值
    factors_df = factors_df.dropna()
    returns = stock_data['returns'].loc[factors_df.index]
    
    # 创建因子模型
    logger.info("创建因子模型...")
    model = FactorModel()
    model.load_data(factors_df, returns)
    
    # 评估因子
    logger.info("评估因子...")
    evaluation_results = model.evaluate_factors()
    
    # 比较因子性能
    logger.info("比较因子性能...")
    comparison = model.compare_factors(metric='ic_ir')
    print("因子IC IR排名:")
    print(comparison)
    
    # 选择因子
    logger.info("选择因子...")
    selected_factors = model.select_factors(method='combined', ic_threshold=0.05, correlation_threshold=0.7)
    print(f"选中的因子: {selected_factors}")
    
    # 绘制因子相关性
    logger.info("绘制因子相关性...")
    model.plot_factor_correlation()
    
    # 优化因子权重
    logger.info("优化因子权重...")
    weights = model.optimize_weights(method='regression')
    print("因子权重:")
    print(weights)
    
    # 生成组合因子
    logger.info("生成组合因子...")
    combined_factor = model.generate_combined_factor()
    
    # 评估组合因子
    logger.info("评估组合因子...")
    evaluation = model.evaluate_combined_factor(test_size=0.3, model_type='linear')
    print(f"组合因子测试集R²: {evaluation['test_r2']:.6f}")
    print(f"组合因子测试集方向准确率: {evaluation['test_direction_accuracy']:.4f}")
    
    # 交叉验证
    logger.info("进行交叉验证...")
    cv_results = model.cross_validate_combined_factor(n_splits=5, model_type='linear')
    print(f"交叉验证平均测试集R²: {cv_results['avg_test_r2']:.6f} ± {cv_results['std_test_r2']:.6f}")
    print(f"交叉验证平均测试集方向准确率: {cv_results['avg_test_direction_accuracy']:.4f} ± {cv_results['std_test_direction_accuracy']:.4f}")
    
    # 绘制组合因子性能
    logger.info("绘制组合因子性能...")
    model.plot_combined_factor_performance()
    
    # 保存模型
    logger.info("保存模型...")
    model.save_model('/Users/liuwen/Documents/work/repos/github.com/iminders/miner/models/factor_model.pkl')
    
    logger.info("示例完成!")

if __name__ == "__main__":
    main()