# -*- coding: utf-8 -*-
"""
因子评估模块

评估因子的预测能力和稳定性
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class FactorEvaluator:
    """
    因子评估器

    评估因子的预测能力和稳定性
    """

    def __init__(self):
        """
        初始化因子评估器
        """
        pass

    def calculate_ic(self, factor: pd.Series, returns: pd.Series, method: str = 'spearman') -> float:
        """
        计算信息系数(IC)

        参数:
        factor (pd.Series): 因子值
        returns (pd.Series): 收益率
        method (str): 相关系数计算方法，'spearman'或'pearson'

        返回:
        float: 信息系数
        """
        # 确保因子和收益率具有相同的索引
        common_index = factor.index.intersection(returns.index)
        factor_aligned = factor.loc[common_index]
        returns_aligned = returns.loc[common_index]

        # 移除NaN值
        mask = ~(factor_aligned.isna() | returns_aligned.isna())
        factor_clean = factor_aligned[mask]
        returns_clean = returns_aligned[mask]

        if len(factor_clean) == 0:
            return np.nan

        # 计算相关系数
        if method == 'spearman':
            ic, _ = spearmanr(factor_clean, returns_clean)
        elif method == 'pearson':
            ic, _ = pearsonr(factor_clean, returns_clean)
        else:
            raise ValueError("method必须是'spearman'或'pearson'")

        return ic

    def calculate_ic_series(self, factor: pd.Series, returns: pd.Series, 
                           method: str = 'spearman', window: int = None) -> pd.Series:
        """
        计算滚动信息系数序列

        参数:
        factor (pd.Series): 因子值
        returns (pd.Series): 收益率
        method (str): 相关系数计算方法，'spearman'或'pearson'
        window (int): 滚动窗口大小，如果为None则计算全样本IC

        返回:
        pd.Series: 信息系数时间序列
        """
        if window is None:
            # 计算全样本IC
            ic = self.calculate_ic(factor, returns, method)
            return pd.Series([ic], index=[returns.index[-1]])

        # 计算滚动IC
        ic_values = []
        ic_dates = []

        # 确保因子和收益率具有相同的索引
        common_index = factor.index.intersection(returns.index)
        factor_aligned = factor.loc[common_index]
        returns_aligned = returns.loc[common_index]

        # 移除NaN值
        mask = ~(factor_aligned.isna() | returns_aligned.isna())
        factor_clean = factor_aligned[mask]
        returns_clean = returns_aligned[mask]

        # 计算滚动IC
        for i in range(window, len(factor_clean) + 1):
            factor_window = factor_clean.iloc[i-window:i]
            returns_window = returns_clean.iloc[i-window:i]

            if len(factor_window) > 5:  # 确保有足够的数据点
                if method == 'spearman':
                    ic, _ = spearmanr(factor_window, returns_window)
                elif method == 'pearson':
                    ic, _ = pearsonr(factor_window, returns_window)
                else:
                    raise ValueError("method必须是'spearman'或'pearson'")

                ic_values.append(ic)
                ic_dates.append(factor_clean.index[i-1])

        return pd.Series(ic_values, index=ic_dates)

    def calculate_factor_returns(self, factor: pd.Series, returns: pd.Series, 
                                n_groups: int = 5) -> pd.DataFrame:
        """
        计算分组因子收益率

        参数:
        factor (pd.Series): 因子值
        returns (pd.Series): 收益率
        n_groups (int): 分组数量

        返回:
        pd.DataFrame: 分组收益率
        """
        # 确保因子和收益率具有相同的索引
        common_index = factor.index.intersection(returns.index)
        factor_aligned = factor.loc[common_index]
        returns_aligned = returns.loc[common_index]

        # 移除NaN值
        mask = ~(factor_aligned.isna() | returns_aligned.isna())
        factor_clean = factor_aligned[mask]
        returns_clean = returns_aligned[mask]

        if len(factor_clean) == 0:
            return pd.DataFrame()

        # 按因子值分组
        factor_quantiles = pd.qcut(factor_clean, n_groups, labels=False)

        # 计算每组的平均收益率
        group_returns = pd.DataFrame(index=range(n_groups))
        group_returns['mean_return'] = returns_clean.groupby(factor_quantiles).mean()

        # 计算多空组合收益率
        long_short_return = group_returns['mean_return'].iloc[-1] - group_returns['mean_return'].iloc[0]

        # 添加多空组合收益率
        group_returns.loc[n_groups] = [long_short_return]
        group_returns.index = list(range(1, n_groups + 1)) + ['long_short']

        return group_returns

    def calculate_factor_turnover(self, factor: pd.Series, window: int = 20) -> pd.Series:
        """
        计算因子换手率

        参数:
        factor (pd.Series): 因子值
        window (int): 滚动窗口大小

        返回:
        pd.Series: 因子换手率时间序列
        """
        # 移除NaN值
        factor_clean = factor.dropna()

        if len(factor_clean) <= window:
            return pd.Series()

        # 计算因子排名
        factor_rank = factor_clean.rank(pct=True)

        # 计算滚动换手率
        turnover = []
        dates = []

        for i in range(window, len(factor_rank)):
            # 计算相邻时间点的排名变化
            rank_change = abs(factor_rank.iloc[i] - factor_rank.iloc[i-1])

            # 换手率定义为排名变化的平均值
            turnover.append(rank_change)
            dates.append(factor_rank.index[i])

        return pd.Series(turnover, index=dates)

    def calculate_factor_stability(self, factor: pd.Series, windows: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
        """
        计算因子稳定性

        参数:
        factor (pd.Series): 因子值
        windows (List[int]): 滚动窗口大小列表

        返回:
        pd.DataFrame: 不同时间窗口的因子自相关系数
        """
        # 移除NaN值
        factor_clean = factor.dropna()

        if len(factor_clean) == 0:
            return pd.DataFrame()

        # 计算不同滞后期的自相关系数
        autocorr = pd.DataFrame(index=windows)

        for window in windows:
            if len(factor_clean) > window:
                # 计算滞后window期的自相关系数
                lagged_factor = factor_clean.shift(window)
                common_index = factor_clean.index.intersection(lagged_factor.index)

                if len(common_index) > 0:
                    factor_aligned = factor_clean.loc[common_index]
                    lagged_aligned = lagged_factor.loc[common_index]

                    # 移除NaN值
                    mask = ~(factor_aligned.isna() | lagged_aligned.isna())

                    if mask.sum() > 5:  # 确保有足够的数据点
                        corr, _ = pearsonr(factor_aligned[mask], lagged_aligned[mask])
                        autocorr.loc[window, 'autocorrelation'] = corr
                    else:
                        autocorr.loc[window, 'autocorrelation'] = np.nan
                else:
                    autocorr.loc[window, 'autocorrelation'] = np.nan
            else:
                autocorr.loc[window, 'autocorrelation'] = np.nan

        return autocorr

    def calculate_factor_decay(self, factor: pd.Series, returns: pd.Series, 
                              horizons: List[int] = [1, 5, 10, 20, 60]) -> pd.DataFrame:
        """
        计算因子衰减特性

        参数:
        factor (pd.Series): 因子值
        returns (pd.Series): 收益率
        horizons (List[int]): 预测周期列表

        返回:
        pd.DataFrame: 不同预测周期的信息系数
        """
        # 确保因子和收益率具有相同的索引
        common_index = factor.