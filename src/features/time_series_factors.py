# -*- coding: utf-8 -*-
"""
时序特征因子模块

从orderbook数据中提取时间序列特征因子
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union
from src.features.basic_features import BasicFeatureExtractor


class TimeSeriesFactorExtractor:
    """
    时序特征因子提取器

    从orderbook数据中提取时间序列特征因子
    """

    def __init__(self):
        """
        初始化因子提取器
        """
        self.basic_extractor = BasicFeatureExtractor()

    def extract_momentum_factors(self, orderbook: pd.DataFrame, windows: List[int] = [5, 10, 20, 60, 120]) -> Dict[str, pd.Series]:
        """
        提取价格动量因子

        参数:
        orderbook (pd.DataFrame): 订单簿数据
        windows (List[int]): 滚动窗口大小列表（秒）

        返回:
        Dict[str, pd.Series]: 包含不同时间窗口动量因子的字典
        """
        # 计算中间价
        mid_price = self.basic_extractor.extract_mid_price(orderbook)

        # 计算不同窗口的动量因子
        momentum_factors = {}

        for window in windows:
            # 计算过去window秒的收益率
            momentum = mid_price.pct_change(window)
            momentum_factors[f'momentum_{window}s'] = momentum

        return momentum_factors

    def extract_volatility_factors(self, orderbook: pd.DataFrame, windows: List[int] = [30, 60, 120, 300]) -> Dict[str, pd.Series]:
        """
        提取波动率因子

        参数:
        orderbook (pd.DataFrame): 订单簿数据
        windows (List[int]): 滚动窗口大小列表（秒）

        返回:
        Dict[str, pd.Series]: 包含不同时间窗口波动率因子的字典
        """
        # 计算中间价
        mid_price = self.basic_extractor.extract_mid_price(orderbook)

        # 计算收益率
        returns = mid_price.pct_change()

        # 计算不同窗口的波动率因子
        volatility_factors = {}

        for window in windows:
            # 计算过去window秒的波动率（标准差）
            volatility = returns.rolling(window=window, min_periods=max(5, window//10)).std()
            volatility_factors[f'volatility_{window}s'] = volatility

            # 计算过去window秒的波动率（基于极差）
            high_low_volatility = mid_price.rolling(window=window, min_periods=max(5, window//10)).apply(
                lambda x: (x.max() - x.min()) / x.mean() if len(x) > 0 else np.nan
            )
            volatility_factors[f'high_low_volatility_{window}s'] = high_low_volatility

        return volatility_factors

    def extract_mean_reversion_factors(self, orderbook: pd.DataFrame, windows: List[int] = [30, 60, 120, 300]) -> Dict[str, pd.Series]:
        """
        提取均值回归因子

        参数:
        orderbook (pd.DataFrame): 订单簿数据
        windows (List[int]): 滚动窗口大小列表（秒）

        返回:
        Dict[str, pd.Series]: 包含不同时间窗口均值回归因子的字典
        """
        # 计算中间价
        mid_price = self.basic_extractor.extract_mid_price(orderbook)

        # 计算不同窗口的均值回归因子
        mean_reversion_factors = {}

        for window in windows:
            # 计算过去window秒的移动平均
            ma = mid_price.rolling(window=window, min_periods=max(5, window//10)).mean()

            # 计算当前价格与移动平均的偏离度
            deviation = (mid_price - ma) / ma
            mean_reversion_factors[f'price_deviation_{window}s'] = deviation

            # 计算z-score（标准化偏离度）
            rolling_std = mid_price.rolling(window=window, min_periods=max(5, window//10)).std()
            z_score = (mid_price - ma) / rolling_std.replace(0, np.nan)
            mean_reversion_factors[f'z_score_{window}s'] = z_score

        return mean_reversion_factors

    def extract_price_jump_factors(self, orderbook: pd.DataFrame, windows: List[int] = [10, 30, 60], threshold: float = 3.0) -> Dict[str, pd.Series]:
        """
        提取价格跳跃因子

        参数:
        orderbook (pd.DataFrame): 订单簿数据
        windows (List[int]): 滚动窗口大小列表（秒）
        threshold (float): 跳跃识别阈值（标准差的倍数）

        返回:
        Dict[str, pd.Series]: 包含价格跳跃因子的字典
        """
        # 计算中间价
        mid_price = self.basic_extractor.extract_mid_price(orderbook)

        # 计算收益率
        returns = mid_price.pct_change()

        # 计算不同窗口的价格跳跃因子
        jump_factors = {}

        for window in windows:
            # 计算滚动标准差
            rolling_std = returns.rolling(window=window, min_periods=max(5, window//10)).std()

            # 识别跳跃（收益率超过threshold倍标准差）
            jumps = (returns.abs() > threshold * rolling_std).astype(int)
            jump_factors[f'jump_indicator_{window}s'] = jumps

            # 计算跳跃强度（收益率与标准差的比值）
            jump_intensity = returns.abs() / rolling_std.replace(0, np.nan)
            jump_factors[f'jump_intensity_{window}s'] = jump_intensity

            # 计算跳跃累积次数（滚动窗口内）
            jump_count = jumps.rolling(window=window, min_periods=1).sum()
            jump_factors[f'jump_count_{window}s'] = jump_count

        return jump_factors

    def extract_trend_factors(self, orderbook: pd.DataFrame, windows: List[int] = [60, 120, 300]) -> Dict[str, pd.Series]:
        """
        提取趋势因子

        参数:
        orderbook (pd.DataFrame): 订单簿数据
        windows (List[int]): 滚动窗口大小列表（秒）

        返回:
        Dict[str, pd.Series]: 包含趋势因子的字典
        """
        # 计算中间价
        mid_price = self.basic_extractor.extract_mid_price(orderbook)

        # 计算不同窗口的趋势因子
        trend_factors = {}

        for window in windows:
            # 计算短期和长期移动平均
            short_ma = mid_price.rolling(window=window//4, min_periods=max(5, window//20)).mean()
            long_ma = mid_price.rolling(window=window, min_periods=max(5, window//10)).mean()

            # 计算移动平均交叉指标
            ma_cross = short_ma / long_ma - 1
            trend_factors[f'ma_cross_{window}s'] = ma_cross

            # 计算价格方向一致性（过去window秒内价格变动方向的一致性）
            # 使用收益率符号的滚动平均
            returns = mid_price.pct_change()
            direction = returns.apply(np.sign)  # 1表示上涨，-1表示下跌，0表示不变
            direction_consistency = direction.rolling(window=window, min_periods=max(5, window//10)).mean().abs()
            trend_factors[f'direction_consistency_{window}s'] = direction_consistency

        return trend_factors

    def extract_all_time_series_factors(self, orderbook: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        提取所有时序特征因子

        参数:
        orderbook (pd.DataFrame): 订单簿数据

        返回:
        Dict[str, pd.Series]: 包含所有时序特征因子的字典
        """
        factors = {}

        # 提取动量因子
        factors.update(self.extract_momentum_factors(orderbook))

        # 提取波动率因子
        factors.update(self.extract_volatility_factors(orderbook))

        # 提取均值回归因子
        factors.update(self.extract_mean_reversion_factors(orderbook))

        # 提取价格跳跃因子
        factors.update(self.extract_price_jump_factors(orderbook))

        # 提取趋势因子
        factors.update(self.extract_trend_factors(orderbook))

        return factors