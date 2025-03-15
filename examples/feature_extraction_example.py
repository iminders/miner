# -*- coding: utf-8 -*-
"""
基础特征提取示例
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.basic_features import BasicFeatureExtractor

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

    return orderbook

def main():
    """
    主函数
    """
    print("创建样本订单簿数据...")
    orderbook = create_sample_orderbook(n_samples=1000, levels=5)

    print("订单簿数据示例:")
    print(orderbook.head())

    print("\n初始化特征提取器...")
    extractor = BasicFeatureExtractor(levels=5)

    print("\n提取基本特征...")
    features = extractor.extract_all_basic_features(orderbook)

    print("\n提取的特征列表:")
    for feature_name, feature_data in features.items():
        if isinstance(feature_data, pd.DataFrame):
            print(f"{feature_name}: DataFrame with shape {feature_data.shape}")
        else:
            print(f"{feature_name}: Series with length {len(feature_data)}")

    # 可视化一些关键特征
    plt.figure(figsize=(15, 10))

    # 绘制中间价
    plt.subplot(3, 2, 1)
    plt.plot(features['mid_price'])
    plt.title('中间价')
    plt.xticks(rotation=45)
    plt.grid(True)

    # 绘制买卖价差
    plt.subplot(3, 2, 2)
    plt.plot(features['spread'])
    plt.title('买卖价差')
    plt.xticks(rotation=45)
    plt.grid(True)

    # 绘制订单簿不平衡指标
    plt.subplot(3, 2, 3)
    plt.plot(features['order_book_imbalance'])
    plt.title('订单簿不平衡指标')
    plt.xticks(rotation=45)
    plt.grid(True)

    # 绘制买卖盘深度
    plt.subplot(3, 2, 4)
    plt.plot(features['bid_depth'], label='买盘深度')
    plt.plot(features['ask_depth'], label='卖盘深度')
    plt.title('订单簿深度')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)

    # 绘制相对价差
    plt.subplot(3, 2, 5)
    plt.plot(features['relative_spread'])
    plt.title('相对买卖价差')
    plt.xticks(rotation=45)
    plt.grid(True)

    # 绘制加权中间价与普通中间价的对比
    plt.subplot(3, 2, 6)
    plt.plot(features['mid_price'], label='中间价')
    plt.plot(features['weighted_mid_price'], label='加权中间价')
    plt.title('中间价对比')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('basic_features.png')
    plt.show()

    print("\n特征相关性分析...")
    # 将Series类型的特征合并为一个DataFrame进行相关性分析
    series_features = {k: v for k, v in features.items() if isinstance(v, pd.Series)}
    feature_df = pd.DataFrame(series_features)

    # 计算相关性矩阵
    corr_matrix = feature_df.corr()

    # 可视化相关性矩阵
    plt.figure(figsize=(10, 8))
    plt.matshow(corr_matrix, fignum=1)
    plt.colorbar()
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.title('特征相关性矩阵')
    plt.savefig('feature_correlation.png')
    plt.show()

    print("\n示例完成。")

if __name__ == "__main__":
    main()