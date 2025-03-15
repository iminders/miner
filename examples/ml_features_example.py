# -*- coding: utf-8 -*-
"""
机器学习特征工程示例
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.basic_features import BasicFeatureExtractor
from src.features.ml_features import MLFeatureEngineer

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

    # 生成未来收益率（60秒后）
    # 使用一些特征的组合来模拟，使其具有一定的可预测性
    mid_price = (orderbook['bid_price_1'] + orderbook['ask_price_1']) / 2
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

    print("\n初始化机器学习特征工程器...")
    engineer = MLFeatureEngineer()

    print("\n执行特征工程...")
    engineered_features = engineer.engineer_all_features(orderbook, trades, future_returns)

    print("\n工程化特征示例:")
    print(engineered_features.head())

    print(f"\n工程化特征形状: {engineered_features.shape}")

    # 移除NaN值
    clean_features = engineered_features.dropna()
    clean_target = future_returns.loc[clean_features.index].dropna()
    common_index = clean_features.index.intersection(clean_target.index)

    X = clean_features.loc[common_index]
    y = clean_target.loc[common_index]

    print(f"\n清洗后的数据形状: X={X.shape}, y={y.shape}")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False  # 时间序列数据不打乱
    )

    print(f"\n训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"测试集形状: X_test={X_test.shape}, y_test={y_test.shape}")

    # 训练随机森林模型
    print("\n训练随机森林模型...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # 预测
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # 评估模型
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    print(f"\n训练集MSE: {train_mse:.6f}")
    print(f"测试集MSE: {test_mse:.6f}")
    print(f"训练集R²: {train_r2:.6f}")
    print(f"测试集R²: {test_r2:.6f}")

    # 特征重要性分析
    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    top_features = feature_importance.sort_values(ascending=False).head(15)

    print("\n最重要的15个特征:")
    for feature, importance in top_features.items():
        print(f"{feature}: {importance:.6f}")

    # 可视化特征重要性
    plt.figure(figsize=(12, 8))
    top_features.plot(kind='barh')
    plt.title('特征重要性排名（前15名）')
    plt.xlabel('重要性')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()

    # 可视化预测结果
    plt.figure(figsize=(15, 6))

    # 训练集预测
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_pred_train, alpha=0.5)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title('训练集预测结果')

    # 测试集预测
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title('测试集预测结果')

    plt.tight_layout()
    plt.savefig('prediction_results.png')
    plt.show()

    # 可视化预测时间序列
    plt.figure(figsize=(15, 6))

    # 测试集时间序列预测
    plt.plot(y_test.index, y_test.values, 'b-', label='实际收益率')
    plt.plot(y_test.index, y_pred_test, 'r-', label='预测收益率')
    plt.xlabel('时间')
    plt.ylabel('收益率')
    plt.title('未来收益率预测')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('time_series_prediction.png')
    plt.show()

    print("\n示例完成。")

if __name__ == "__main__":
    main()