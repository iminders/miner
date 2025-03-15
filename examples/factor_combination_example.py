# -*- coding: utf-8 -*-
"""
多因子组合示例
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
from src.evaluation.factor_combination import FactorCombiner

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
        'volatility_60s': ts_factors['volatility_60s'],
        'mean_reversion': ts_factors['mean_reversion'],
        'price_impact': micro_factors['price_impact'],
        'order_flow_toxicity': micro_factors['order_flow_toxicity']
    }

    # 创建因子DataFrame
    factors_df = pd.DataFrame(selected_factors)

    print("\n因子数据示例:")
    print(factors_df.head())

    # 初始化因子评估器和组合器
    print("\n初始化因子评估器和组合器...")
    evaluator = FactorEvaluator()
    combiner = FactorCombiner()

    # 计算每个因子的IC
    print("\n计算每个因子的IC...")
    ic_values = {}
    ic_series = {}

    for factor_name, factor_values in selected_factors.items():
        # 计算IC
        ic = evaluator.calculate_ic(factor_values, future_returns)
        ic_values[factor_name] = ic

        # 计算IC时间序列
        ic_ts = evaluator.calculate_ic_series(factor_values, future_returns, window=60)
        ic_series[factor_name] = ic_ts

    # 创建IC值的Series
    ic_df = pd.Series(ic_values).sort_values(ascending=False)

    print("\n各因子IC值:")
    print(ic_df)

    # 分析因子相关性
    print("\n分析因子相关性...")
    corr_matrix = combiner.analyze_factor_correlation(factors_df)

    print("\n因子相关性矩阵:")
    print(corr_matrix)

    # 可视化相关性热图
    print("\n绘制相关性热图...")
    combiner.plot_correlation_heatmap(corr_matrix)

    # 对因子进行聚类
    print("\n对因子进行聚类...")
    cluster_result = combiner.cluster_factors(factors_df, n_clusters=4)

    print("\n聚类结果:")
    for cluster_id, factors in cluster_result['cluster_dict'].items():
        print(f"聚类 {cluster_id}: {', '.join(factors)}")

    # 可视化聚类树状图
    print("\n绘制聚类树状图...")
    combiner.plot_dendrogram(cluster_result)

    # 从每个聚类中选择代表性因子
    print("\n从每个聚类中选择代表性因子...")
    representative_factors = combiner.select_representative_factors(factors_df, future_returns, cluster_result)

    print("\n选择的代表性因子:")
    print(representative_factors.columns.tolist())

    # 使用不同方法优化权重
    print("\n使用不同方法优化权重...")

    # 1. 等权重
    equal_weights = combiner.optimize_weights(representative_factors, future_returns, method='equal')
    print("\n等权重:")
    print(equal_weights)

    # 2. 基于IC的权重
    ic_weights = combiner.optimize_weights(representative_factors, future_returns, method='ic', ic_series=ic_series)
    print("\n基于IC的权重:")
    print(ic_weights)

    # 3. 基于回归的权重
    regression_weights = combiner.optimize_weights(representative_factors, future_returns, method='regression')
    print("\n基于回归的权重:")
    print(regression_weights)

    # 4. 均值-方差优化权重
    try:
        mv_weights = combiner.optimize_weights(representative_factors, future_returns, method='mean_variance')
        print("\n均值-方差优化权重:")
        print(mv_weights)
    except Exception as e:
        print(f"\n均值-方差优化失败: {e}")
        mv_weights = equal_weights

    # 组合因子
    print("\n组合因子...")

    # 1. 等权重组合
    equal_combined = combiner.combine_factors(representative_factors, equal_weights)

    # 2. IC权重组合
    ic_combined = combiner.combine_factors(representative_factors, ic_weights)

    # 3. 回归权重组合
    regression_combined = combiner.combine_factors(representative_factors, regression_weights)

    # 4. 均值-方差权重组合
    mv_combined = combiner.combine_factors(representative_factors, mv_weights)

    # 评估组合因子
    print("\n评估组合因子...")

    # 1. 等权重组合评估
    equal_eval = combiner.evaluate_combined_factor(equal_combined, future_returns, evaluator)

    # 2. IC权重组合评估
    ic_eval = combiner.evaluate_combined_factor(ic_combined, future_returns, evaluator)

    # 3. 回归权重组合评估
    regression_eval = combiner.evaluate_combined_factor(regression_combined, future_returns, evaluator)

    # 4. 均值-方差权重组合评估
    mv_eval = combiner.evaluate_combined_factor(mv_combined, future_returns, evaluator)

    # 比较不同组合方法的IC
    combined_ic = {
        '等权重组合': equal_eval.get('IC', 0),
        'IC权重组合': ic_eval.get('IC', 0),
        '回归权重组合': regression_eval.get('IC', 0),
        '均值-方差权重组合': mv_eval.get('IC', 0)
    }

    combined_ic_df = pd.Series(combined_ic).sort_values(ascending=False)

    print("\n不同组合方法的IC值:")
    print(combined_ic_df)

    # 可视化不同组合方法的IC
    plt.figure(figsize=(10, 6))
    combined_ic_df.plot(kind='bar')
    plt.title('不同组合方法的IC值比较')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('combined_factor_ic_comparison.png')
    plt.show()

    # 使用PCA组合因子
    print("\n使用PCA组合因子...")
    pca_result = combiner.perform_pca_combination(factors_df)

    print("\nPCA解释方差比例:")
    print(pca_result['explained_variance_ratio'])

    print("\n累积解释方差比例:")
    print(pca_result['cumulative_variance_ratio'])

    # 可视化PCA结果
    print("\n绘制PCA结果...")
    combiner.plot_pca_results(pca_result)

    # 评估PCA组合因子
    print("\n评估PCA组合因子...")
    pca_combined = pca_result['pca_factors']['PC1']  # 使用第一主成分
    pca_eval = combiner.evaluate_combined_factor(pca_combined, future_returns, evaluator)

    print(f"\nPCA组合因子IC: {pca_eval.get('IC', 0):.4f}")

    # 构建多因子模型
    print("\n构建多因子模型...")

    # 1. 回归模型
    print("\n1. 使用回归模型构建多因子模型...")
    regression_model = combiner.build_multi_factor_model(
        factors_df, future_returns, method='regression', model_type='ridge', alpha=0.1
    )

    print(f"\n回归模型R²: {regression_model.get('r2', 0):.4f}")
    print("\n回归系数:")
    print(regression_model.get('coefficients', pd.Series()))

    # 2. PCA模型
    print("\n2. 使用PCA构建多因子模型...")
    pca_model = combiner.build_multi_factor_model(
        factors_df, future_returns, method='pca', n_components=3
    )

    print(f"\nPCA模型解释方差比例: {sum(pca_model.get('explained_variance_ratio', [])):.4f}")

    # 3. 聚类模型
    print("\n3. 使用聚类构建多因子模型...")
    cluster_model = combiner.build_multi_factor_model(
        factors_df, future_returns, method='cluster', 
        n_clusters=4, weight_method='ic', ic_series=ic_series
    )

    print("\n聚类选择的因子:")
    print(cluster_model.get('selected_factors', pd.DataFrame()).columns.tolist())

    print("\n聚类模型权重:")
    print(cluster_model.get('weights', pd.Series()))

    # 比较不同模型的组合因子IC
    model_ic = {
        '回归模型': evaluator.calculate_ic(regression_model.get('combined_factor', pd.Series()), future_returns),
        'PCA模型': evaluator.calculate_ic(pca_model.get('combined_factor', pd.Series()), future_returns),
        '聚类模型': evaluator.calculate_ic(cluster_model.get('combined_factor', pd.Series()), future_returns)
    }

    model_ic_df = pd.Series(model_ic).sort_values(ascending=False)

    print("\n不同模型的组合因子IC值:")
    print(model_ic_df)

    # 可视化不同模型的组合因子IC
    plt.figure(figsize=(10, 6))
    model_ic_df.plot(kind='bar')
    plt.title('不同模型的组合因子IC值比较')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_factor_ic_comparison.png')
    plt.show()

    print("\n示例完成。")

if __name__ == "__main__":
    main()