# -*- coding: utf-8 -*-
"""
机器学习特征工程模块

从基础特征中构建高级特征
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from src.features.basic_features import BasicFeatureExtractor
from src.features.microstructure_factors import MicrostructureFactorExtractor
from src.features.time_series_factors import TimeSeriesFactorExtractor


class MLFeatureEngineer:
    """
    机器学习特征工程

    从基础特征中构建高级特征
    """

    def __init__(self):
        """
        初始化特征工程器
        """
        self.basic_extractor = BasicFeatureExtractor()
        self.microstructure_extractor = MicrostructureFactorExtractor()
        self.time_series_extractor = TimeSeriesFactorExtractor()
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.pca = PCA(n_components=0.95)  # 保留95%的方差
        self.selector = SelectKBest(f_regression, k=20)  # 默认选择20个最佳特征

    def extract_base_features(self, orderbook: pd.DataFrame, trades: pd.DataFrame = None) -> pd.DataFrame:
        """
        提取所有基础特征

        参数:
        orderbook (pd.DataFrame): 订单簿数据
        trades (pd.DataFrame): 成交数据，可选

        返回:
        pd.DataFrame: 包含所有基础特征的DataFrame
        """
        # 提取基本特征
        basic_features = self.basic_extractor.extract_all_basic_features(orderbook)

        # 提取微观结构因子
        micro_factors = self.microstructure_extractor.extract_all_microstructure_factors(orderbook, trades)

        # 提取时序特征因子
        ts_factors = self.time_series_extractor.extract_all_time_series_factors(orderbook)

        # 合并所有Series类型的特征
        series_features = {}

        for feature_dict in [basic_features, micro_factors, ts_factors]:
            for name, feature in feature_dict.items():
                if isinstance(feature, pd.Series):
                    series_features[name] = feature

        # 创建特征DataFrame
        features_df = pd.DataFrame(series_features)

        return features_df

    def create_feature_interactions(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        创建特征交互项

        参数:
        features_df (pd.DataFrame): 基础特征DataFrame

        返回:
        pd.DataFrame: 包含特征交互项的DataFrame
        """
        # 移除NaN值
        features_clean = features_df.dropna()

        if len(features_clean) == 0:
            return pd.DataFrame()

        # 应用多项式特征变换
        poly_features = self.poly.fit_transform(features_clean)

        # 创建新的特征名称
        feature_names = features_clean.columns
        poly_feature_names = self.poly.get_feature_names_out(feature_names)

        # 创建多项式特征DataFrame
        poly_df = pd.DataFrame(poly_features, index=features_clean.index, columns=poly_feature_names)

        # 只保留交互项（排除原始特征）
        interaction_df = poly_df.iloc[:, len(feature_names):]

        return interaction_df

    def create_nonlinear_transformations(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        创建非线性变换特征

        参数:
        features_df (pd.DataFrame): 基础特征DataFrame

        返回:
        pd.DataFrame: 包含非线性变换特征的DataFrame
        """
        # 移除NaN值
        features_clean = features_df.dropna()

        if len(features_clean) == 0:
            return pd.DataFrame()

        # 创建非线性变换
        nonlinear_features = pd.DataFrame(index=features_clean.index)

        for col in features_clean.columns:
            # 对数变换 (对正值)
            positive_mask = features_clean[col] > 0
            if positive_mask.any():
                nonlinear_features[f'log_{col}'] = np.nan
                nonlinear_features.loc[positive_mask, f'log_{col}'] = np.log(features_clean.loc[positive_mask, col])

            # 平方根变换 (对正值)
            if positive_mask.any():
                nonlinear_features[f'sqrt_{col}'] = np.nan
                nonlinear_features.loc[positive_mask, f'sqrt_{col}'] = np.sqrt(features_clean.loc[positive_mask, col])

            # 平方变换
            nonlinear_features[f'square_{col}'] = features_clean[col] ** 2

            # 立方变换
            nonlinear_features[f'cube_{col}'] = features_clean[col] ** 3

            # 倒数变换 (避免除以零)
            nonzero_mask = features_clean[col] != 0
            if nonzero_mask.any():
                nonlinear_features[f'reciprocal_{col}'] = np.nan
                nonlinear_features.loc[nonzero_mask, f'reciprocal_{col}'] = 1 / features_clean.loc[nonzero_mask, col]

            # 指数变换 (对小值)
            small_mask = features_clean[col] < 10  # 避免数值溢出
            if small_mask.any():
                nonlinear_features[f'exp_{col}'] = np.nan
                nonlinear_features.loc[small_mask, f'exp_{col}'] = np.exp(features_clean.loc[small_mask, col])

            # 符号变换
            nonlinear_features[f'sign_{col}'] = np.sign(features_clean[col])

        return nonlinear_features

    def create_time_lagged_features(self, features_df: pd.DataFrame, lags: List[int] = [1, 5, 10, 30]) -> pd.DataFrame:
        """
        创建时间滞后特征

        参数:
        features_df (pd.DataFrame): 基础特征DataFrame
        lags (List[int]): 滞后周期列表

        返回:
        pd.DataFrame: 包含时间滞后特征的DataFrame
        """
        lagged_features = pd.DataFrame(index=features_df.index)

        for col in features_df.columns:
            for lag in lags:
                lagged_features[f'{col}_lag_{lag}'] = features_df[col].shift(lag)

        return lagged_features

    def create_rolling_window_features(self, features_df: pd.DataFrame, windows: List[int] = [5, 10, 30, 60]) -> pd.DataFrame:
        """
        创建滚动窗口特征

        参数:
        features_df (pd.DataFrame): 基础特征DataFrame
        windows (List[int]): 窗口大小列表

        返回:
        pd.DataFrame: 包含滚动窗口特征的DataFrame
        """
        rolling_features = pd.DataFrame(index=features_df.index)

        for col in features_df.columns:
            for window in windows:
                # 滚动平均
                rolling_features[f'{col}_mean_{window}'] = features_df[col].rolling(window=window, min_periods=1).mean()

                # 滚动标准差
                rolling_features[f'{col}_std_{window}'] = features_df[col].rolling(window=window, min_periods=2).std()

                # 滚动最大值
                rolling_features[f'{col}_max_{window}'] = features_df[col].rolling(window=window, min_periods=1).max()

                # 滚动最小值
                rolling_features[f'{col}_min_{window}'] = features_df[col].rolling(window=window, min_periods=1).min()

                # 滚动中位数
                rolling_features[f'{col}_median_{window}'] = features_df[col].rolling(window=window, min_periods=1).median()

                # 滚动偏度
                rolling_features[f'{col}_skew_{window}'] = features_df[col].rolling(window=window, min_periods=3).skew()

                # 滚动峰度
                rolling_features[f'{col}_kurt_{window}'] = features_df[col].rolling(window=window, min_periods=3).kurt()

        return rolling_features

    def reduce_dimensions(self, features_df: pd.DataFrame, n_components: float = 0.95) -> pd.DataFrame:
        """
        使用PCA降维

        参数:
        features_df (pd.DataFrame): 特征DataFrame
        n_components (float): 保留的方差比例

        返回:
        pd.DataFrame: 降维后的特征DataFrame
        """
        # 移除NaN值
        features_clean = features_df.dropna()

        if len(features_clean) == 0:
            return pd.DataFrame()

        # 标准化特征
        scaled_features = self.scaler.fit_transform(features_clean)

        # 设置PCA组件数
        self.pca.n_components = n_components

        # 应用PCA
        pca_features = self.pca.fit_transform(scaled_features)

        # 创建PCA特征DataFrame
        pca_df = pd.DataFrame(
            pca_features, 
            index=features_clean.index,
            columns=[f'pca_{i+1}' for i in range(pca_features.shape[1])]
        )

        return pca_df

    def select_best_features(self, features_df: pd.DataFrame, target: pd.Series, k: int = 20) -> pd.DataFrame:
        """
        选择最佳特征

        参数:
        features_df (pd.DataFrame): 特征DataFrame
        target (pd.Series): 目标变量
        k (int): 选择的特征数量

        返回:
        pd.DataFrame: 选择的最佳特征DataFrame
        """
        # 确保特征和目标具有相同的索引
        common_index = features_df.index.intersection(target.index)
        features_aligned = features_df.loc[common_index]
        target_aligned = target.loc[common_index]

        # 移除NaN值
        mask = ~(features_aligned.isna().any(axis=1) | target_aligned.isna())
        features_clean = features_aligned[mask]
        target_clean = target_aligned[mask]

        if len(features_clean) == 0:
            return pd.DataFrame()

        # 设置选择器的k值
        self.selector.k = min(k, features_clean.shape[1])

        # 应用特征选择
        selected_features = self.selector.fit_transform(features_clean, target_clean)

        # 获取选择的特征索引
        selected_indices = self.selector.get_support(indices=True)
        selected_columns = features_clean.columns[selected_indices]

        # 创建选择的特征DataFrame
        selected_df = pd.DataFrame(
            selected_features,
            index=features_clean.index,
            columns=selected_columns
        )

        return selected_df

    def engineer_all_features(self, orderbook: pd.DataFrame, trades: pd.DataFrame = None, target: pd.Series = None) -> pd.DataFrame:
        """
        执行所有特征工程步骤

        参数:
        orderbook (pd.DataFrame): 订单簿数据
        trades (pd.DataFrame): 成交数据，可选
        target (pd.Series): 目标变量，可选

        返回:
        pd.DataFrame: 工程化的特征DataFrame
        """
        print("提取基础特征...")
        base_features = self.extract_base_features(orderbook, trades)

        print(f"基础特征数量: {base_features.shape[1]}")

        print("创建特征交互项...")
        interaction_features = self.create_feature_interactions(base_features)

        print(f"交互特征数量: {interaction_features.shape[1]}")

        print("创建非线性变换特征...")
        nonlinear_features = self.create_nonlinear_transformations(base_features)

        print(f"非线性特征数量: {nonlinear_features.shape[1]}")

        print("创建时间滞后特征...")
        lagged_features = self.create_time_lagged_features(base_features)

        print(f"滞后特征数量: {lagged_features.shape[1]}")

        print("创建滚动窗口特征...")
        rolling_features = self.create_rolling_window_features(base_features)

        print(f"滚动窗口特征数量: {rolling_features.shape[1]}")

        # 合并所有特征
        all_features = pd.concat(
            [base_features, interaction_features, nonlinear_features, lagged_features, rolling_features],
            axis=1
        )

        print(f"合并后的特征总数: {all_features.shape[1]}")

        # 如果提供了目标变量，执行特征选择
        if target is not None:
            print("执行特征选择...")
            selected_features = self.select_best_features(all_features, target)
            print(f"选择的特征数量: {selected_features.shape[1]}")
            return selected_features
        else:
            # 否则执行降维
            print("执行降维...")
            reduced_features = self.reduce_dimensions(all_features)
            print(f"降维后的特征数量: {reduced_features.shape[1]}")
            return reduced_features