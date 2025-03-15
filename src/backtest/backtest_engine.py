# -*- coding: utf-8 -*-
"""
高频回测引擎模块

实现高频回测框架
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.strategy.strategy import Strategy


class BacktestEngine:
    """
    高频回测引擎

    实现高频回测框架
    """

    def __init__(self, 
                 strategy: Strategy,
                 start_date: Union[str, datetime],
                 end_date: Union[str, datetime],
                 data_frequency: str = '1s',
                 commission: float = 0.0003,
                 slippage: float = 0.0001,
                 initial_capital: float = 1000000.0):
        """
        初始化回测引擎

        参数:
        strategy (Strategy): 交易策略
        start_date (Union[str, datetime]): 回测开始日期
        end_date (Union[str, datetime]): 回测结束日期
        data_frequency (str): 数据频率，如'1s', '1m', '1h'
        commission (float): 手续费率
        slippage (float): 滑点率
        initial_capital (float): 初始资金
        """
        self.strategy = strategy

        # 设置回测日期
        if isinstance(start_date, str):
            self.start_date = pd.to_datetime(start_date)
        else:
            self.start_date = start_date

        if isinstance(end_date, str):
            self.end_date = pd.to_datetime(end_date)
        else:
            self.end_date = end_date

        self.data_frequency = data_frequency
        self.commission = commission
        self.slippage = slippage
        self.initial_capital = initial_capital

        # 回测结果
        self.backtest_result = None

        # 设置交易规则中的交易成本
        self.strategy.trading_rules.trading_fee = commission
        self.strategy.trading_rules.slippage = slippage

        # 设置策略初始资金
        self.strategy.initial_capital = initial_capital

    def load_price_data(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[pd.Timestamp, float]]:
        """
        加载价格数据

        参数:
        price_data (Dict[str, pd.DataFrame]): 价格数据，键为标的，值为DataFrame

        返回:
        Dict[str, Dict[pd.Timestamp, float]]: 处理后的价格数据
        """
        processed_data = {}

        for symbol, df in price_data.items():
            # 确保时间索引
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df.set_index('timestamp', inplace=True)
                else:
                    raise ValueError(f"DataFrame for {symbol} must have a timestamp column or DatetimeIndex")

            # 过滤日期范围
            df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]

            # 确保价格列存在
            price_column = None
            for col in ['price', 'close', 'mid_price', 'last_price']:
                if col in df.columns:
                    price_column = col
                    break

            if price_column is None:
                raise ValueError(f"DataFrame for {symbol} must have a price column (price, close, mid_price, or last_price)")

            # 转换为字典格式
            price_dict = df[price_column].to_dict()
            processed_data[symbol] = price_dict

        return processed_data

    def load_factor_data(self, factor_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[pd.Timestamp, Dict[str, float]]]:
        """
        加载因子数据

        参数:
        factor_data (Dict[str, pd.DataFrame]): 因子数据，键为标的，值为DataFrame

        返回:
        Dict[str, Dict[pd.Timestamp, Dict[str, float]]]: 处理后的因子数据
        """
        processed_data = {}

        for symbol, df in factor_data.items():
            # 确保时间索引
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df.set_index('timestamp', inplace=True)
                else:
                    raise ValueError(f"DataFrame for {symbol} must have a timestamp column or DatetimeIndex")

            # 过滤日期范围
            df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]

            # 转换为嵌套字典格式
            symbol_data = {}
            for timestamp in df.index:
                factor_values = {}
                for col in df.columns:
                    factor_values[col] = df.loc[timestamp, col]
                symbol_data[timestamp] = factor_values

            processed_data[symbol] = symbol_data

        return processed_data

    def generate_timestamps(self, price_data: Dict[str, Dict[pd.Timestamp, float]]) -> List[pd.Timestamp]:
        """
        生成时间戳列表

        参数:
        price_data (Dict[str, Dict[pd.Timestamp, float]]): 价格数据

        返回:
        List[pd.Timestamp]: 时间戳列表
        """
        # 收集所有时间戳
        all_timestamps = set()
        for symbol, price_dict in price_data.items():
            all_timestamps.update(price_dict.keys())

        # 排序时间戳
        timestamps = sorted(list(all_timestamps))

        return timestamps

    def run_backtest(self, price_data: Dict[str, pd.DataFrame], 
                    factor_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        运行回测

        参数:
        price_data (Dict[str, pd.DataFrame]): 价格数据
        factor_data (Dict[str, pd.DataFrame]): 因子数据

        返回:
        Dict: 回测结果
        """
        # 加载数据
        processed_price_data = self.load_price_data(price_data)
        processed_factor_data = self.load_factor_data(factor_data)

        # 生成时间戳列表
        timestamps = self.generate_timestamps(processed_price_data)

        # 运行策略回测
        print(f"开始回测: {self.start_date} 到 {self.end_date}")
        self.backtest_result = self.strategy.run_backtest(timestamps, processed_price_data, processed_factor_data)
        print(f"回测完成: 最终资金 {self.backtest_result['final_portfolio_value']:,.2f}")

        return self.backtest_result

    def plot_results(self) -> None:
        """
        绘制回测结果
        """
        if self.backtest_result is None:
            print("请先运行回测")
            return

        # 绘制权益曲线
        self.strategy.plot_equity_curve()

        # 绘制回撤曲线
        self.strategy.plot_drawdown()

        # 绘制交易分析
        self.strategy.plot_trade_analysis()

    def print_performance_summary(self) -> None:
        """
        打印性能摘要
        """
        if self.backtest_result is None:
            print("请先运行回测")
            return

        self.strategy.print_performance_summary()

    def analyze_trades(self) -> pd.DataFrame:
        """
        分析交易记录

        返回:
        pd.DataFrame: 交易分析结果
        """
        if self.backtest_result is None or not self.backtest_result['trades']:
            print("没有交易记录")
            return pd.DataFrame()

        # 转换交易记录为DataFrame
        trades_df = pd.DataFrame(self.backtest_result['trades'])

        # 设置时间索引
        trades_df.set_index('timestamp', inplace=True)

        # 计算交易统计
        trade_stats = {
            '总交易次数': len(trades_df),
            '买入交易': len(trades_df[trades_df['direction'] == 'BUY']),
            '卖出交易': len(trades_df[trades_df['direction'] == 'SELL']),
            '开仓交易': len(trades_df[trades_df['type'] == 'OPEN']),
            '平仓交易': len(trades_df[trades_df['type'] == 'CLOSE']),
            '调整仓位': len(trades_df[trades_df['type'] == 'ADJUST']),
            '平均交易金额': trades_df['value'].mean(),
            '最大交易金额': trades_df['value'].max(),
            '最小交易金额': trades_df['value'].min(),
        }

        # 计算盈亏统计
        if 'profit' in trades_df.columns:
            profit_trades = trades_df[trades_df['profit'] > 0]
            loss_trades = trades_df[trades_df['profit'] <= 0]

            trade_stats.update({
                '盈利交易': len(profit_trades),
                '亏损交易': len(loss_trades),
                '胜率': len(profit_trades) / len(trades_df) if len(trades_df) > 0 else 0,
                '平均盈利': profit_trades['profit'].mean() if len(profit_trades) > 0 else 0,
                '平均亏损': loss_trades['profit'].mean() if len(loss_trades) > 0 else 0,
                '最大盈利': profit_trades['profit'].max() if len(profit_trades) > 0 else 0,
                '最大亏损': loss_trades['profit'].min() if len(loss_trades) > 0 else 0,
                '总盈利': profit_trades['profit'].sum() if len(profit_trades) > 0 else 0,
                '总亏损': loss_trades['profit'].sum() if len(loss_trades) > 0 else 0,
                '净盈亏': trades_df['profit'].sum(),
            })

        # 转换为DataFrame
        stats_df = pd.DataFrame(list(trade_stats.items()), columns=['指标', '值'])

        return stats_df

    def analyze_by_symbol(self) -> pd.DataFrame:
        """
        按标的分析交易

        返回:
        pd.DataFrame: 按标的分析结果
        """
        if self.backtest_result is None or not self.backtest_result['trades']:
            print("没有交易记录")
            return pd.DataFrame()

        # 转换交易记录为DataFrame
        trades_df = pd.DataFrame(self.backtest_result['trades'])

        # 按标的分组
        symbol_groups = trades_df.groupby('symbol')

        # 初始化结果
        results = []

        for symbol, group in symbol_groups:
            # 计算基本统计
            stats = {
                '标的': symbol,
                '交易次数': len(group),
                '买入次数': len(group[group['direction'] == 'BUY']),
                '卖出次数': len(group[group['direction'] == 'SELL']),
            }

            # 计算盈亏统计
            if 'profit' in group.columns:
                profit_trades = group[group['profit'] > 0]
                loss_trades = group[group['profit'] <= 0]

                stats.update({
                    '盈利交易': len(profit_trades),
                    '亏损交易': len(loss_trades),
                    '胜率': len(profit_trades) / len(group) if len(group) > 0 else 0,
                    '平均盈利': profit_trades['profit'].mean() if len(profit_trades) > 0 else 0,
                    '平均亏损': loss_trades['profit'].mean() if len(loss_trades) > 0 else 0,
                    '净盈亏': group['profit'].sum(),
                })

            results.append(stats)

        # 转换为DataFrame
        results_df = pd.DataFrame(results)

        return results_df

    def analyze_by_time(self, time_unit: str = 'D') -> pd.DataFrame:
        """
        按时间分析交易

        参数:
        time_unit (str): 时间单位，如'D'表示按天，'H'表示按小时

        返回:
        pd.DataFrame: 按时间分析结果
        """
        if self.backtest_result is None or not self.backtest_result['trades']:
            print("没有交易记录")
            return pd.DataFrame()

        # 转换交易记录为DataFrame
        trades_df = pd.DataFrame(self.backtest_result['trades'])

        # 设置时间索引
        trades_df.set_index('timestamp', inplace=True)

        # 按时间单位重采样
        time_groups = trades_df.resample(time_unit)

        # 初始化结果
        results = []

        for time_period, group in time_groups:
            if len(group) == 0:
                continue

            # 计算基本统计
            stats = {
                '时间': time_period,
                '交易次数': len(group),
                '买入次数': len(group[group['direction'] == 'BUY']),
                '卖出次数': len(group[group['direction'] == 'SELL']),
            }

            # 计算盈亏统计
            if 'profit' in group.columns:
                profit_trades = group[group['profit'] > 0]
                loss_trades = group[group['profit'] <= 0]

                stats.update({
                    '盈利交易': len(profit_trades),
                    '亏损交易': len(loss_trades),
                    '胜率': len(profit_trades) / len(group) if len(group) > 0 else 0,
                    '净盈亏': group['profit'].sum(),
                })

            results.append(stats)

        # 转换为DataFrame
        results_df = pd.DataFrame(results)

        return results_df

    def save_results(self, output_dir: str) -> None:
        """
        保存回测结果

        参数:
        output_dir (str): 输出目录
        """
        import os
        import json
        from datetime import datetime

        if self.backtest_result is None:
            print("请先运行回测")
            return

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存交易记录
        trades_df = pd.DataFrame(self.backtest_result['trades'])
        trades_df.to_csv(os.path.join(output_dir, f'trades_{timestamp}.csv'), index=False)

        # 保存权益曲线
        equity_curve = pd.DataFrame(self.backtest_result['equity_curve'], columns=['timestamp', 'equity'])
        equity_curve.to_csv(os.path.join(output_dir, f'equity_curve_{timestamp}.csv'), index=False)

        # 保存性能指标
        metrics = self.backtest_result['performance_metrics']
        # 移除不可序列化的对象
        if 'equity_curve' in metrics:
            del metrics['equity_curve']

        with open(os.path.join(output_dir, f'metrics_{timestamp}.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

        print(f"回测结果已保存到 {output_dir}")