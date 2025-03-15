# -*- coding: utf-8 -*-
"""
时序特征因子提取示例
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.basic_features import BasicFeatureExtractor
from src.features.time_series_factors import TimeSeriesFactorExtractor

def create_sample_orderbook(n_samples=1000, levels=5, seed=42):
    """
    创建样本订单簿数据用于测试

    参数:
    n_samples (int): 样本数量
    levels (int): 订单簿深度级别
    seed (int): 随机种子

    返回:
    pd.DataFrame: 样本订单簿数据
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

    return orderbook

def main():
    """
    主函数
    """
    print("创建样本订单簿数据...")
    orderbook = create_sample_orderbook(n_samples=1000, levels=5)

    print("订单簿数据示例:")
    print(orderbook.head())

    print("\n初始化时序特征因子提取器...")
    extractor = TimeSeriesFactorExtractor()

    print("\n提取时序特征因子...")
    factors = extractor.extract_all_time_series_factors(orderbook)

    print("\n提取的因子列表:")
    for factor_name, factor_data in factors.items():
        print(f"{factor_name}: Series with length {len(factor_data)}")

    # 可视化中间价
    basic_extractor = BasicFeatureExtractor()
    mid_price = basic_extractor.extract_mid_price(orderbook)

    plt.figure(figsize=(15, 5))
    plt.plot(mid_price)
    plt.title('中间价')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('mid_price.png')
    plt.show()

    # 可视化一些关键因子
    # 选择一些代表性因子进行可视化
    key_factors = [
        'momentum_60s',
        'volatility_60s',
        'price_deviation_60s',
        'z_score_60s',
        'jump_indicator_30s',
        'jump_intensity_30s',
        'ma_cross_120s',
        'direction_consistency_120s'
    ]

    # 创建一个包含选定因子的DataFrame
    selected_factors = pd.DataFrame({k: factors[k] for k in key_factors if k in factors})

    # 可视化选定的因子
    plt.figure(figsize=(15, 20))

    for i, factor_name in enumerate(selected_factors.columns):
        plt.subplot(len(selected_factors.columns), 1, i+1)
        plt.plot(selected_factors[factor_name])
        plt.title(factor_name)
        plt.xticks(rotation=45)
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('time_series_factors.png')
    plt.show()

    # 分析因子之间的相关性
    print("\n因子相关性分析...")

    # 计算相关性矩阵
    corr_matrix = selected_factors.corr()

    # 可视化相关性矩阵
    plt.figure(figsize=(12, 10))
    plt.matshow(corr_matrix, fignum=1)
    plt.colorbar()
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.title('时序特征因子相关性矩阵')
    plt.savefig('time_series_factor_correlation.png')
    plt.show()

    # 分析因子与未来价格变动的关系
    print("\n因子预测能力分析...")

    # 计算未来收益率（1分钟后）
    future_returns = mid_price.pct_change().shift(-60)  # 60秒后的收益率

    # 计算因子与未来收益率的相关性
    factor_predictive_power = {}
    for factor_name, factor_values in factors.items():
        correlation = factor_values.corr(future_returns)
        if not np.isnan(correlation):
            factor_predictive_power[factor_name] = correlation

    # 按相关性绝对值排序
    sorted_factors = sorted(factor_predictive_power.items(), key=lambda x: abs(x[1]), reverse=True)

    # 可视化前15个最具预测力的因子
    top_factors = sorted_factors[:15]

    plt.figure(figsize=(12, 8))
    factor_names = [f[0] for f in top_factors]
    correlations = [f[1] for f in top_factors]

    plt.barh(factor_names, correlations)
    plt.xlabel('与未来收益率的相关性')
    plt.title('因子预测能力排名（前15名）')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('factor_predictive_power.png')
    plt.show()

    print("\n示例完成。")

if __name__ == "__main__":
    main()