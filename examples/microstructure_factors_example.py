# -*- coding: utf-8 -*-
"""
微观结构因子提取示例
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

def create_sample_data(n_samples=1000, levels=5, seed=42):
    """
    创建样本数据用于测试

    参数:
    n_samples (int): 样本数量
    levels (int): 订单簿深度级别
    seed (int): 随机种子

    返回:
    tuple: (orderbook, trades) 样本订单簿数据和成交数据
    """
    np.random.seed(seed)

    # 创建时间索引
    timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='S')

    # 初始化订单簿数据
    orderbook = pd.DataFrame(index=timestamps)

    # 生成基础价格
    base_price = 100.0
    price_trend = np.cumsum(np.random.normal(0, 0.01, n_samples))

    # 为每个级别生成价格和数量
    for i in range(1, levels + 1):
        # 买盘价格（递减）
        bid_price = base_price + price_trend - (i - 1) * 0.01
        orderbook[f'bid_price_{i}'] = bid_price

        # 卖盘价格（递增）
        ask_price = base_price + price_trend + i * 0.01
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

    return orderbook, trades

def main():
    """
    主函数
    """
    print("创建样本数据...")
    orderbook, trades = create_sample_data(n_samples=1000, levels=5)

    print("订单簿数据示例:")
    print(orderbook.head())

    print("\n成交数据示例:")
    print(trades.head())

    print("\n初始化微观结构因子提取器...")
    extractor = MicrostructureFactorExtractor(levels=5)

    print("\n提取微观结构因子...")
    factors = extractor.extract_all_microstructure_factors(orderbook, trades)

    print("\n提取的因子列表:")
    for factor_name, factor_data in factors.items():
        if isinstance(factor_data, pd.DataFrame):
            print(f"{factor_name}: DataFrame with shape {factor_data.shape}")
        else:
            print(f"{factor_name}: Series with length {len(factor_data)}")

    # 可视化一些关键因子
    plt.figure(figsize=(15, 12))

    # 绘制订单簿压力因子
    plt.subplot(3, 2, 1)
    plt.plot(factors['bid_pressure'], 'g-', label='买盘压力')
    plt.plot(factors['ask_pressure'], 'r-', label='卖盘压力')
    plt.plot(factors['net_pressure'], 'b-', label='净压力')
    plt.title('订单簿压力因子')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)

    # 绘制有效价差
    plt.subplot(3, 2, 2)
    plt.plot(factors['effective_spread'])
    plt.title('有效价差')
    plt.xticks(rotation=45)
    plt.grid(True)

    # 绘制市场深度因子
    plt.subplot(3, 2, 3)
    plt.plot(factors['market_depth'])
    plt.title('市场深度因子')
    plt.xticks(rotation=45)
    plt.grid(True)

    # 绘制订单流毒性
    plt.subplot(3, 2, 4)
    plt.plot(factors['order_flow_toxicity'])
    plt.title('订单流毒性')
    plt.xticks(rotation=45)
    plt.grid(True)

    # 绘制Kyle's Lambda
    plt.subplot(3, 2, 5)
    plt.plot(factors['kyle_lambda'])
    plt.title('Kyle\'s Lambda (价格影响因子)')
    plt.xticks(rotation=45)
    plt.grid(True)

    # 绘制订单簿斜率比
    plt.subplot(3, 2, 6)
    plt.plot(factors['slope_ratio'])
    plt.title('订单簿斜率比')
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('microstructure_factors.png')
    plt.show()

    # 分析因子之间的相关性
    print("\n因子相关性分析...")
    # 将Series类型的因子合并为一个DataFrame进行相关性分析
    series_factors = {k: v for k, v in factors.items() if isinstance(v, pd.Series)}
    factor_df = pd.DataFrame(series_factors)

    # 计算相关性矩阵
    corr_matrix = factor_df.corr()

    # 可视化相关性矩阵
    plt.figure(figsize=(12, 10))
    plt.matshow(corr_matrix, fignum=1)
    plt.colorbar()
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.title('微观结构因子相关性矩阵')
    plt.savefig('microstructure_factor_correlation.png')
    plt.show()

    # 分析因子与价格变动的关系
    print("\n因子与价格变动关系分析...")

    # 计算中间价收益率
    basic_extractor = BasicFeatureExtractor(levels=5)
    mid_price = basic_extractor.extract_mid_price(orderbook)
    returns = mid_price.pct_change()

    # 将收益率添加到因子DataFrame
    factor_df['returns'] = returns

    # 计算因子与收益率的相关性
    returns_corr = factor_df.corr()['returns'].drop('returns')

    # 可视化因子与收益率的相关性
    plt.figure(figsize=(10, 6))
    returns_corr.sort_values().plot(kind='bar')
    plt.title('因子与价格收益率的相关性')
    plt.grid(True, alpha=0.3)
    plt.savefig('factor_returns_correlation.png')
    plt.show()

    # 分析因子对未来收益率的预测能力
    print("\n因子预测能力分析...")

    # 计算未来1分钟收益率
    future_returns = returns.shift(-60)  # 60秒后的收益率

    # 将未来收益率添加到因子DataFrame
    factor_df['future_returns'] = future_returns

    # 计算因子与未来收益率的相关性
    future_corr = factor_df.corr()['future_returns'].drop(['returns', 'future_returns'])

    # 可视化因子与未来收益率的相关性
    plt.figure(figsize=(10, 6))
    future_corr.sort_values().plot(kind='bar')
    plt.title('因子与未来收益率的相关性')
    plt.grid(True, alpha=0.3)
    plt.savefig('factor_future_returns_correlation.png')
    plt.show()

    print("\n示例完成。")

if __name__ == "__main__":
    main()