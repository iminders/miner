# -*- coding: utf-8 -*-
"""
交易时机优化模块

用于优化交易时机
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.backtest.backtest_engine import BacktestEngine
from src.strategy.strategy import Strategy


class TimingOptimizer:
    """
    交易时机优化器

    用于优化交易时机
    """

    def __init__(self, 
                 strategy: Strategy,
                 price_data: Dict[str, pd.DataFrame],
                 factor_data: Dict[str, pd.DataFrame],
                 start_date: Union[str, pd.Timestamp],
                 end_date: Union[str, pd.Timestamp],
                 commission: float = 0.0003,
                 slippage: float = 0.0001,
                 initial_capital: float = 1000000.0):
        """
        初始化交易时机优化器

        参数:
        strategy (Strategy): 策略实例
        price_data (Dict[str, pd.DataFrame]): 价格数据
        factor_data (Dict[str, pd.DataFrame]): 因子数据
        start_date (Union[str, pd.Timestamp]): 回测开始日期
        end_date (Union[str, pd.Timestamp]): 回测结束日期
        commission (float): 手续费率
        slippage (float): 滑点率
        initial_capital (float): 初始资金
        """
        self.strategy = strategy
        self.price_data = price_data
        self.factor_data = factor_data
        self.start_date = start_date
        self.end_date = end_date
        self.commission = commission
        self.slippage = slippage
        self.initial_capital = initial_capital

        # 优化结果
        self.optimization_results = None
        self.best_timing_rules = None

    def analyze_market_timing(self, time_periods: List[str] = None) -> pd.DataFrame:
        """
        分析不同时间段的市场表现

        参数:
        time_periods (List[str]): 时间段列表，如['09:30-10:00', '10:00-11:30', '13:00-14:30', '14:30-15:00']

        返回:
        pd.DataFrame: 时间段分析结果
        """
        if time_periods is None:
            # 默认A股交易时间段
            time_periods = ['09:30-10:00', '10:00-11:30', '13:00-14:30', '14:30-15:00']

        # 创建回测引擎
        backtest_engine = BacktestEngine(
            strategy=self.strategy,
            start_date=self.start_date,
            end_date=self.end_date,
            commission=self.commission,
            slippage=self.slippage,
            initial_capital=self.initial_capital
        )

        # 运行回测
        backtest_result = backtest_engine.run_backtest(
            price_data=self.price_data,
            factor_data=self.factor_data
        )

        # 获取交易记录
        trades = backtest_result['trades']

        # 转换为DataFrame
        trades_df = pd.DataFrame(trades)

        # 确保时间戳列存在
        if 'timestamp' not in trades_df.columns:
            print("交易记录中缺少时间戳")
            return pd.DataFrame()

        # 设置时间索引
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])

        # 提取时间
        trades_df['time'] = trades_df['timestamp'].dt.strftime('%H:%M')

        # 分析不同时间段的交易
        time_period_results = []

        for period in time_periods:
            start_time, end_time = period.split('-')

            # 筛选时间段内的交易
            period_trades = trades_df[(trades_df['time'] >= start_time) & (trades_df['time'] < end_time)]

            # 计算时间段内的交易统计
            if len(period_trades) > 0:
                profitable_trades = period_trades[period_trades['profit'] > 0]
                win_rate = len(profitable_trades) / len(period_trades) if len(period_trades) > 0 else 0

                result = {
                    '时间段': period,
                    '交易次数': len(period_trades),
                    '盈利交易': len(profitable_trades),
                    '亏损交易': len(period_trades) - len(profitable_trades),
                    '胜率': win_rate,
                    '平均收益': period_trades['profit'].mean() if 'profit' in period_trades.columns else 0,
                    '总收益': period_trades['profit'].sum() if 'profit' in period_trades.columns else 0
                }
                time_period_results.append(result)

        # 转换为DataFrame
        results_df = pd.DataFrame(time_period_results)

        return results_df

    def optimize_time_windows(self, time_periods: List[str] = None) -> Dict:
        """
        优化交易时间窗口

        参数:
        time_periods (List[str]): 时间段列表，如['09:30-10:00', '10:00-11:30', '13:00-14:30', '14:30-15:00']

        返回:
        Dict: 优化结果
        """
        # 分析不同时间段的市场表现
        time_analysis = self.analyze_market_timing(time_periods)

        if time_analysis.empty:
            print("无法分析时间段")
            return {}

        # 按胜率排序
        time_analysis = time_analysis.sort_values(by='胜率', ascending=False)

        # 选择胜率最高的时间段
        best_periods = time_analysis[time_analysis['胜率'] > 0.5]['时间段'].tolist()

        # 创建时间窗口规则
        timing_rules = {
            'active_periods': best_periods,
            'analysis': time_analysis
        }

        self.best_timing_rules = timing_rules

        return timing_rules

    def optimize_day_of_week(self) -> Dict:
        """
        优化交易日（星期几）

        返回:
        Dict: 优化结果
        """
        # 创建回测引擎
        backtest_engine = BacktestEngine(
            strategy=self.strategy,
            start_date=self.start_date,
            end_date=self.end_date,
            commission=self.commission,
            slippage=self.slippage,
            initial_capital=self.initial_capital
        )

        # 运行回测
        backtest_result = backtest_engine.run_backtest(
            price_data=self.price_data,
            factor_data=self.factor_data
        )

        # 获取交易记录
        trades = backtest_result['trades']

        # 转换为DataFrame
        trades_df = pd.DataFrame(trades)

        # 确保时间戳列存在
        if 'timestamp' not in trades_df.columns:
            print("交易记录中缺少时间戳")
            return {}

        # 设置时间索引
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])

        # 提取星期几
        trades_df['day_of_week'] = trades_df['timestamp'].dt.day_name()

        # 分析不同星期几的交易
        day_results = []

        for day, group in trades_df.groupby('day_of_week'):
            profitable_trades = group[group['profit'] > 0]
            win_rate = len(profitable_trades) / len(group) if len(group) > 0 else 0

            result = {
                '星期': day,
                '交易次数': len(group),
                '盈利交易': len(profitable_trades),
                '亏损交易': len(group) - len(profitable_trades),
                '胜率': win_rate,
                '平均收益': group['profit'].mean() if 'profit' in group.columns else 0,
                '总收益': group['profit'].sum() if 'profit' in group.columns else 0
            }
            day_results.append(result)

        # 转换为DataFrame
        results_df = pd.DataFrame(day_results)

        # 按胜率排序
        results_df = results_df.sort_values(by='胜率', ascending=False)

        # 选择胜率最高的星期几
        best_days = results_df[results_df['胜率'] > 0.5]['星期'].tolist()

        # 创建星期几规则
        day_rules = {
            'active_days': best_days,
            'analysis': results_df
        }

        return day_rules

    def optimize_volatility_conditions(self, volatility_percentiles: List[float] = None) -> Dict:
        """
        优化波动率条件

        参数:
        volatility_percentiles (List[float]): 波动率百分位列表，如[0.25, 0.5, 0.75]

        返回:
        Dict: 优化结果
        """
        if volatility_percentiles is None:
            volatility_percentiles = [0.25, 0.5, 0.75]

        # 计算各股票的波动率
        volatility_data = {}

        for symbol, price_df in self.price_data.items():
            # 确保价格列存在
            price_column = None
            for col in ['price', 'close', 'mid_price', 'last_price']:
                if col in price_df.columns:
                    price_column = col
                    break

            if price_column is None:
                continue

            # 计算收益率
            returns = price_df[price_column].pct_change().dropna()

            # 计算波动率（20周期滚动标准差）
            volatility = returns.rolling(window=20).std()

            volatility_data[symbol] = volatility

        # 创建回测引擎
        backtest_engine = BacktestEngine(
            strategy=self.strategy,
            start_date=self.start_date,
            end_date=self.end_date,
            commission=self.commission,
            slippage=self.slippage,
            initial_capital=self.initial_capital
        )

        # 运行回测
        backtest_result = backtest_engine.run_backtest(
            price_data=self.price_data,
            factor_data=self.factor_data
        )

        # 获取交易记录
        trades = backtest_result['trades']

        # 转换为DataFrame
        trades_df = pd.DataFrame(trades)

        # 确保必要列存在
        if 'timestamp' not in trades_df.columns or 'symbol' not in trades_df.columns:
            print("交易记录中缺少必要列")
            return {}

        # 设置时间索引
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])

        # 添加波动率信息
        trades_df['volatility'] = np.nan

        for i, row in trades_df.iterrows():
            symbol = row['symbol']
            timestamp = row['timestamp']

            if symbol in volatility_data:
                vol_series = volatility_data[symbol]

                # 找到最接近的时间点
                if timestamp in vol_series.index:
                    trades_df.at[i, 'volatility'] = vol_series[timestamp]
                else:
                    # 找到小于等于交易时间的最大时间点
                    prev_times = vol_series.index[vol_series.index <= timestamp]
                    if len(prev_times) > 0:
                        trades_df.at[i, 'volatility'] = vol_series[prev_times[-1]]

        # 移除缺失波动率的交易
        trades_df = trades_df.dropna(subset=['volatility'])

        if trades_df.empty:
            print("没有有效的波动率数据")
            return {}

        # 计算波动率分位数
        volatility_quantiles = trades_df['volatility'].quantile(volatility_percentiles).to_dict()

        # 分析不同波动率区间的交易
        volatility_results = []

        # 添加最低区间
        low_vol_trades = trades_df[trades_df['volatility'] <= volatility_quantiles[volatility_percentiles[0]]]
        if len(low_vol_trades) > 0:
            profitable_trades = low_vol_trades[low_vol_trades['profit'] > 0]
            win_rate = len(profitable_trades) / len(low_vol_trades) if len(low_vol_trades) > 0 else 0

            result = {
                '波动率区间': f'<= {volatility_percentiles[0]*100}%分位',
                '交易次数': len(low_vol_trades),
                '盈利交易': len(profitable_trades),
                '亏损交易': len(low_vol_trades) - len(profitable_trades),
                '胜率': win_rate,
                '平均收益': low_vol_trades['profit'].mean() if 'profit' in low_vol_trades.columns else 0,
                '总收益': low_vol_trades['profit'].sum() if 'profit' in low_vol_trades.columns else 0
            }
            volatility_results.append(result)

        # 添加中间区间
        for i in range(len(volatility_percentiles)-1):
            lower = volatility_quantiles[volatility_percentiles[i]]
            upper = volatility_quantiles[volatility_percentiles[i+1]]

            mid_vol_trades = trades_df[(trades_df['volatility'] > lower) & (trades_df['volatility'] <= upper)]

            if len(mid_vol_trades) > 0:
                profitable_trades = mid_vol_trades[mid_vol_trades['profit'] > 0]
                win_rate = len(profitable_trades) / len(mid_vol_trades) if len(mid_vol_trades) > 0 else 0

                result = {
                    '波动率区间': f'{volatility_percentiles[i]*100}%-{volatility_percentiles[i+1]*100}%分位',
                    '交易次数': len(mid_vol_trades),
                    '盈利交易': len(profitable_trades),
                    '亏损交易': len(mid_vol_trades) - len(profitable_trades),
                    '胜率': win_rate,
                    '平均收益': mid_vol_trades['profit'].mean() if 'profit' in mid_vol_trades.columns else 0,
                    '总收益': mid_vol_trades['profit'].sum() if 'profit' in mid_vol_trades.columns else 0
                }
                volatility_results.append(result)

        # 添加最高区间
        high_vol_trades = trades_df[trades_df['volatility'] > volatility_quantiles[volatility_percentiles[-1]]]
        if len(high_vol_trades) > 0:
            profitable_trades = high_vol_trades[high_vol_trades['profit'] > 0]
            win_rate = len(profitable_trades) / len(high_vol_trades) if len(high_vol_trades) > 0 else 0

            result = {
                '波动率区间': f'> {volatility_percentiles[-1]*100}%分位',
                '交易次数': len(high_vol_trades),
                '盈利交易': len(profitable_trades),
                '亏损交易': len(high_vol_trades) - len(profitable_trades),
                '胜率': win_rate,
                '平均收益': high_vol_trades['profit'].mean() if 'profit' in high_vol_trades.columns else 0,
                '总收益': high_vol_trades['profit'].sum() if 'profit' in high_vol_trades.columns else 0
            }
            volatility_results.append(result)

        # 转换为DataFrame
        results_df = pd.DataFrame(volatility_results)

        # 按胜率排序
        results_df = results_df.sort_values(by='胜率', ascending=False)

        # 选择胜率最高的波动率区间
        best_volatility_ranges = results_df[results_df['胜率'] > 0.5]['波动率区间'].tolist()

        # 创建波动率规则
        volatility_rules = {
            'best_volatility_ranges': best_volatility_ranges,
            'volatility_quantiles': volatility_quantiles,
            'analysis': results_df
        }

        return volatility_rules

    def plot_timing_analysis(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        绘制时间分析结果

        参数:
        figsize (Tuple[int, int]): 图形大小
        """
        if self.best_timing_rules is None:
            print("请先运行时间窗口优化")
            return

        # 获取时间段分析结果
        time_analysis = self.best_timing_rules['analysis']

        if time_analysis.empty:
            print("没有时间段分析结果")
            return

        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 绘制交易次数
        sns.barplot(x='时间段', y='交易次数', data=time_analysis, ax=axes[0, 0])
        axes[0, 0].set_title('各时间段交易次数')
        axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45)

        # 绘制胜率
        sns.barplot(x='时间段', y='胜率', data=time_analysis, ax=axes[0, 1])
        axes[0, 1].set_title('各时间段胜率')
        axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)

        # 绘制平均收益
        sns.barplot(x='时间段', y='平均收益', data=time_analysis, ax=axes[1, 0])
        axes[1, 0].set_title('各时间段平均收益')
        axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45)

        # 绘制总收益
        sns.barplot(x='时间段', y='总收益', data=time_analysis, ax=axes[1, 1])
        axes[1, 1].set_title('各时间段总收益')
        axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45)

        plt.tight_layout()
        plt.show()

    def apply_timing_rules(self, strategy: Strategy) -> Strategy:
        """
        应用时机优化规则到策略

        参数:
        strategy (Strategy): 策略实例

        返回:
        Strategy: 应用了时机规则的策略
        """
        if self.best_timing_rules is None:
            print("请先运行时间窗口优化")
            return strategy

        # 获取最佳时间段
        best_periods = self.best_timing_rules.get('active_periods', [])

        # 设置策略的交易时间窗口
        if hasattr(strategy, 'trading_rules'):
            if not hasattr(strategy.trading_rules, 'trading_time_windows'):
                strategy.trading_rules.trading_time_windows = []

            strategy.trading_rules.trading_time_windows = best_periods

        return strategy