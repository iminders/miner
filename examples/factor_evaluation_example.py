# -*- coding: utf-8 -*-
"""
因子评估示例
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.basic_features import BasicFeatureExtractor
from src.features.microstructure_factors import MicrostructureFactorExtractor
from src.features.time_series_factors import TimeSeriesFactorExtractor
from src.evaluation.factor_evaluation import FactorEvaluator

def create_sample_data(n_samples=1000, levels=5, seed=42):
    """
    创建样本数据用于测试

    参数:
    n_samples (int): 样本数量
    levels (int): 订单簿深度级别
    seed (int): 随机种子

    返回:
    tuple: (orderbook, trades, future_returns) 样本订单簿数据、成交数据和未来收益率
    """
    np.random.seed(seed)

    # 创建时间索引
    timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='S')

    # 初始化订单簿数据
    orderbook = pd.DataFrame(index=timestamps)

    # 生成基础价格 - 添加一些趋势和周期性
    base_price = 100.0
    trend = np.linspace(0, 0.5, n_samples)  # 上升趋势
    cycle = 0.2 * np.sin(np.linspace(0, 10 * np.pi, n_samples))  # 周期性波动
    noise = np.random.normal(0, 0.01, n_samples)  # 随机噪声

    # 添加一些价格跳跃
    jumps = np.zeros(n_samples)
    jump_points = np.random.choice(range(100, n_samples-100), size=5, replace=False)
    for jp in jump_points:
        jumps[jp] = np.random.choice([-0.2, 0.2])  # 随机上涨或下跌跳跃

    # 组合价格序列
    price_trend = base_price + np.cumsum(trend + cycle + noise + jumps)

    # 为每个级别生成价格和数量
    for i in range(1, levels + 1):
        # 买盘价格（递减）
        bid_price = price_trend - (i - 1) * 0.01
        orderbook[f'bid_price_{i}'] = bid_price

        # 卖盘价格（递增）
        ask_price = price_trend + i * 0.01
        orderbook[f'ask_price_{i}'] = ask_price

        # 买盘数量
        bid_volume = np.random.lognormal(mean=5, sigma=0.5, size=n_samples) * (1 + (i - 1) * 0.5)
        orderbook[f'bid_volume_{i}'] = bid_volume

        # 卖盘数量
        ask_volume = np.random.lognormal(mean=5, sigma=0.5, size=n_samples) * (1 + (i - 1) * 0.5)
        orderbook[f'ask_volume_{i}'] = ask_volume

    # 创建成交数据
    trades = pd.DataFrame(index=timestamps)

    # 生成成交价格（在买一卖一之间随机）
    trades['price'] = (orderbook['bid_price_1'] + orderbook['ask_price_1']) / 2 + np.random.normal(0, 0.005, n_samples)

    # 生成成交量
    trades['volume'] = np.random.lognormal(mean=4, sigma=0.8, size=n_samples)

    # 生成成交方向（1为买，-1为卖）
    trades['direction'] = np.random.choice([1, -1], size=n_samples)

    # 计算中间价
    mid_price = (orderbook['bid_price_1'] + orderbook['ask_price_1']) / 2

    # 生成未来收益率（60秒后）
    future_price = mid_price.shift(-60)  # 60秒后的价格
    future_returns = (future_price - mid_price) / mid_price  # 60秒后的收益率

    return orderbook, trades, future_returns

def main():
    """
    主函数
    """
    print("创建样本数据...")
    orderbook, trades, future_returns = create_sample_data(n_samples=1000, levels=5)

    print("订单簿数据示例:")
    print(orderbook.head())

    print("\n成交数据示例:")
    print(trades.head())

    print("\n未来收益率示例:")
    print(future_returns.head())

    print("\n提取因子...")
    # 提取基本特征
    basic_extractor = BasicFeatureExtractor()
    basic_features = basic_extractor.extract_all_basic_features(orderbook)

    # 提取微观结构因子
    micro_extractor = MicrostructureFactorExtractor()
    micro_factors = micro_extractor.extract_all_microstructure_factors(orderbook, trades)

    # 提取时序特征因子
    ts_extractor = TimeSeriesFactorExtractor()
    ts_factors = ts_extractor.extract_all_time_series_factors(orderbook)

    # 选择一些因子进行评估
    selected_factors = {
        'bid_ask_spread': basic_features['bid_ask_spread'],
        'mid_price_volatility': basic_features['mid_price_volatility'],
        'order_imbalance': basic_features['order_imbalance'],
        'effective_spread': micro_factors['effective_spread'],
        'market_depth': micro_factors['market_depth'],
        'momentum_60s': ts_factors['momentum_60s'],
        'volatility_60s': ts_factors['volatility_60s']
    }

    # 创建因子DataFrame
    factors_df = pd.DataFrame(selected_factors)

    print("\n因子数据示例:")
    print(factors_df.head())

    print("\n初始化因子评估器...")
    evaluator = FactorEvaluator()

    # 评估每个因子
    print("\n评估因子...")
    evaluation_results = {}

    for factor_name, factor_values in selected_factors.items():
        print(f"\n评估因子: {factor_name}")
        # 评估因子
        factor_results = evaluator.evaluate_factor(factor_values, future_returns, factors_df.drop(factor_name, axis=1))
        evaluation_results[factor_name] = factor_results

        # 打印主要评估指标
        print(f"信息系数(IC): {factor_results['IC']:.4f}")
        if 'neutral_IC' in factor_results:
            print(f"中性化后信息系数: {factor_results['neutral_IC']:.4f}")

        # 可视化评估结果
        evaluator.plot_factor_evaluation(factor_results, factor_name)

    # 比较所有因子的IC
    ic_values = {factor_name: results['IC'] for factor_name, results in evaluation_results.items()}
    ic_df = pd.Series(ic_values).sort_values(ascending=False)

    print("\n所有因子的信息系数(IC):")
    print(ic_df)

    # 可视化所有因子的IC
    plt.figure(figsize=(10, 6))
    ic_df.plot(kind='bar')
    plt.title('因子信息系数(IC)比较')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('factor_ic_comparison.png')
    plt.show()

    # 比较所有因子的衰减特性
    plt.figure(figsize=(12, 8))

    for factor_name, results in evaluation_results.items():
        if 'decay' in results:
            plt.plot(results['decay'].index, results['decay']['IC'], marker='o', label=factor_name)

    plt.title('因子衰减特性比较')
    plt.xlabel('预测周期')
    plt.ylabel('信息系数(IC)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('factor_decay_comparison.png')
    plt.show()

    # 分析因子之间的相关性
    factor_corr = factors_df.corr()

    plt.figure(figsize=(10, 8))
    plt.matshow(factor_corr, fignum=1)
    plt.colorbar()
    plt.xticks(range(len(factor_corr.columns)), factor_corr.columns, rotation=90)
    plt.yticks(range(len(factor_corr.columns)), factor_corr.columns)
    plt.title('因子相关性矩阵')
    plt.tight_layout()
    plt.savefig('factor_correlation_matrix.png')
    plt.show()

    # 构建多因子组合
    print("\n构建多因子组合...")

    # 根据IC值对因子进行加权
    weights = ic_df.abs() / ic_df.abs().sum()

    # 标准化每个因子
    normalized_factors = pd.DataFrame()
    for factor_name in weights.index:
        factor = factors_df[factor_name]
        normalized_factors[factor_name] = (factor - factor.mean()) / factor.std()

    # 计算加权组合因子
    combined_factor = pd.Series(0, index=normalized_factors.index)
    for factor_name, weight in weights.items():
        # 根据IC符号确定因子方向
        direction = np.sign(ic_df[factor_name])
        combined_factor += direction * weight * normalized_factors[factor_name]

    # 评估组合因子
    print("\n评估组合因子...")
    combined_results = evaluator.evaluate_factor(combined_factor, future_returns)

    print(f"组合因子信息系数(IC): {combined_results['IC']:.4f}")

    # 可视化组合因子评估结果
    evaluator.plot_factor_evaluation(combined_results, 'Combined_Factor')

    # 比较组合因子与单因子的IC
    all_ic = ic_df.copy()
    all_ic['Combined_Factor'] = combined_results['IC']
    all_ic = all_ic.sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    all_ic.plot(kind='bar')
    plt.title('组合因子与单因子信息系数(IC)比较')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('combined_factor_comparison.png')
    plt.show()

    print("\n示例完成。")

if __name__ == "__main__":
    main()