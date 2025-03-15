# -*- coding: utf-8 -*-
"""
止盈止损优化模块

用于优化止盈止损策略
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.backtest.backtest_engine import BacktestEngine
from src.strategy.strategy import Strategy
from src.strategy.risk_control import RiskControl


class StopOptimizer:
    """
    止盈止损优化器

    用于优化止盈止损策略
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
        初始化止盈止损优化器

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
        self.best_stop_params = None

    def optimize_stop_loss(self, stop_loss_levels: List[float]) -> pd.DataFrame:
        """
        优化止损水平

        参数:
        stop_loss_levels (List[float]): 止损水平列表，如[0.01, 0.02, 0.03, 0.05, 0.08, 0.10]

        返回:
        pd.DataFrame: 优化结果
        """
        # 初始化结果列表
        results = []

        # 遍历止损水平
        for stop_loss in tqdm(stop_loss_levels, desc="优化止损水平"):
            # 创建策略副本
            strategy_copy = self.strategy

            # 设置止损水平
            if hasattr(strategy_copy, 'risk_control'):
                strategy_copy.risk_control.stop_loss = stop_loss
            else:
                strategy_copy.risk_control = RiskControl(stop_loss=stop_loss)

            # 创建回测引擎
            backtest_engine = BacktestEngine(
                strategy=strategy_copy,
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

            # 获取性能指标
            metrics = backtest_result['performance_metrics']

            # 记录结果
            result = {
                '止损水平': stop_loss,
                '总收益率': metrics.get('total_return', 0),
                '年化收益率': metrics.get('annualized_return', 0),
                '最大回撤': metrics.get('max_drawdown', 0),
                '夏普比率': metrics.get('sharpe_ratio', 0),
                '索提诺比率': metrics.get('sortino_ratio', 0),
                '卡玛比率': metrics.get('calmar_ratio', 0),
                '胜率': metrics.get('win_rate', 0),
                '盈亏比': metrics.get('profit_loss_ratio', 0),
                '总交易次数': metrics.get('total_trades', 0)
            }

            results.append(result)

        # 转换为DataFrame
        results_df = pd.DataFrame(results)

        return results_df

    def optimize_take_profit(self, take_profit_levels: List[float]) -> pd.DataFrame:
        """
        优化止盈水平

        参数:
        take_profit_levels (List[float]): 止盈水平列表，如[0.02, 0.03, 0.05, 0.08, 0.10, 0.15]

        返回:
        pd.DataFrame: 优化结果
        """
        # 初始化结果列表
        results = []

        # 遍历止盈水平
        for take_profit in tqdm(take_profit_levels, desc="优化止盈水平"):
            # 创建策略副本
            strategy_copy = self.strategy

            # 设置止盈水平
            if hasattr(strategy_copy, 'risk_control'):
                strategy_copy.risk_control.take_profit = take_profit
            else:
                strategy_copy.risk_control = RiskControl(take_profit=take_profit)

            # 创建回测引擎
            backtest_engine = BacktestEngine(
                strategy=strategy_copy,
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

            # 获取性能指标
            metrics = backtest_result['performance_metrics']

            # 记录结果
            result = {
                '止盈水平': take_profit,
                '总收益率': metrics.get('total_return', 0),
                '年化收益率': metrics.get('annualized_return', 0),
                '最大回撤': metrics.get('max_drawdown', 0),
                '夏普比率': metrics.get('sharpe_ratio', 0),
                '索提诺比率': metrics.get('sortino_ratio', 0),
                '卡玛比率': metrics.get('calmar_ratio', 0),
                '胜率': metrics.get('win_rate', 0),
                '盈亏比': metrics.get('profit_loss_ratio', 0),
                '总交易次数': metrics.get('total_trades', 0)
            }

            results.append(result)

        # 转换为DataFrame
        results_df = pd.DataFrame(results)

        return results_df

    def optimize_trailing_stop(self, trailing_stop_levels: List[float]) -> pd.DataFrame:
        """
        优化追踪止损水平

        参数:
        trailing_stop_levels (List[float]): 追踪止损水平列表，如[0.01, 0.02, 0.03, 0.05, 0.08]

        返回:
        pd.DataFrame: 优化结果
        """
        # 初始化结果列表
        results = []

        # 遍历追踪止损水平
        for trailing_stop in tqdm(trailing_stop_levels, desc="优化追踪止损水平"):
            # 创建策略副本
            strategy_copy = self.strategy

            # 设置追踪止损水平
            if hasattr(strategy_copy, 'risk_control'):
                strategy_copy.risk_control.trailing_stop = trailing_stop
            else:
                strategy_copy.risk_control = RiskControl(trailing_stop=trailing_stop)

            # 创建回测引擎
            backtest_engine = BacktestEngine(
                strategy=strategy_copy,
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

            # 获取性能指标
            metrics = backtest_result['performance_metrics']

            # 记录结果
            result = {
                '追踪止损水平': trailing_stop,
                '总收益率': metrics.get('total_return', 0),
                '年化收益率': metrics.get('annualized_return', 0),
                '最大回撤': metrics.get('max_drawdown', 0),
                '夏普比率': metrics.get('sharpe_ratio', 0),
                '索提诺比率': metrics.get('sortino_ratio', 0),
                '卡玛比率': metrics.get('calmar_ratio', 0),
                '胜率': metrics.get('win_rate', 0),
                '盈亏比': metrics.get('profit_loss_ratio', 0),
                '总交易次数': metrics.get('total_trades', 0)
            }

            results.append(result)

        # 转换为DataFrame
        results_df = pd.DataFrame(results)

        return results_df

    def optimize_stop_combination(self, 
                                 stop_loss_levels: List[float],
                                 take_profit_levels: List[float]) -> pd.DataFrame:
        """
        优化止损和止盈组合

        参数:
        stop_loss_levels (List[float]): 止损水平列表
        take_profit_levels (List[float]): 止盈水平列表

        返回:
        pd.DataFrame: 优化结果
        """
        # 初始化结果列表
        results = []

        # 遍历止损和止盈组合
        total_combinations = len(stop_loss_levels) * len(take_profit_levels)
        with tqdm(total=total_combinations, desc="优化止损止盈组合") as pbar:
            for stop_loss in stop_loss_levels:
                for take_profit in take_profit_levels:
                    # 创建策略副本
                    strategy_copy = self.strategy

                    # 设置止损和止盈水平
                    if hasattr(strategy_copy, 'risk_control'):
                        strategy_copy.risk_control.stop_loss = stop_loss
                        strategy_copy.risk_control.take_profit = take_profit
                    else:
                        strategy_copy.risk_control = RiskControl(
                            stop_loss=stop_loss,
                            take_profit=take_profit
                        )

                    # 创建回测引擎
                    backtest_engine = BacktestEngine(
                        strategy=strategy_copy,
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

                    # 获取性能指标
                    metrics = backtest_result['performance_metrics']

                    # 记录结果
                    result = {
                        '止损水平': stop_loss,
                        '止盈水平': take_profit,
                        '总收益率': metrics.get('total_return', 0),
                        '年化收益率': metrics.get('annualized_return', 0),
                        '最大回撤': metrics.get('max_drawdown', 0),
                        '夏普比率': metrics.get('sharpe_ratio', 0),
                        '索提诺比率': metrics.get('sortino_ratio', 0),
                        '卡玛比率': metrics.get('calmar_ratio', 0),
                        '胜率': metrics.get('win_rate', 0),
                        '盈亏比': metrics.get('profit_loss_ratio', 0),
                        '总交易次数': metrics.get('total_trades', 0)
                    }

                    results.append(result)
                    pbar.update(1)

        # 转换为DataFrame
        results_df = pd.DataFrame(results)

        return results_df

    def run_optimization(self, 
                        stop_loss_levels: List[float] = None,
                        take_profit_levels: List[float] = None,
                        trailing_stop_levels: List[float] = None,
                        optimization_target: str = 'sharpe_ratio') -> Dict:
        """
        运行完整的止盈止损优化

        参数:
        stop_loss_levels (List[float]): 止损水平列表
        take_profit_levels (List[float]): 止盈水平列表
        trailing_stop_levels (List[float]): 追踪止损水平列表
        optimization_target (str): 优化目标，如'sharpe_ratio', 'total_return', 'calmar_ratio'等

        返回:
        Dict: 优化结果
        """
        # 设置默认参数
        if stop_loss_levels is None:
            stop_loss_levels = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]

        if take_profit_levels is None:
            take_profit_levels = [0.02, 0.03, 0.05, 0.08, 0.10, 0.15]

        if trailing_stop_levels is None:
            trailing_stop_levels = [0.01, 0.02, 0.03, 0.05, 0.08]

        # 运行止损优化
        print("优化止损水平...")
        stop_loss_results = self.optimize_stop_loss(stop_loss_levels)

        # 运行止盈优化
        print("优化止盈水平...")
        take_profit_results = self.optimize_take_profit(take_profit_levels)

        # 运行追踪止损优化
        print("优化追踪止损水平...")
        trailing_stop_results = self.optimize_trailing_stop(trailing_stop_levels)

        # 选择最佳止损和止盈水平
        best_stop_loss = stop_loss_results.sort_values(by=optimization_target, ascending=False).iloc[0]['止损水平']
        best_take_profit = take_profit_results.sort_values(by=optimization_target, ascending=False).iloc[0]['止盈水平']
        best_trailing_stop = trailing_stop_results.sort_values(by=optimization_target, ascending=False).iloc[0]['追踪止损水平']

        # 运行止损和止盈组合优化
        print("优化止损止盈组合...")
        # 选择最佳止损和止盈附近的值进行组合优化
        refined_stop_loss_levels = [
            max(0.005, best_stop_loss - 0.02),
            max(0.005, best_stop_loss - 0.01),
            best_stop_loss,
            best_stop_loss + 0.01,
            best_stop_loss + 0.02
        ]

        refined_take_profit_levels = [
            max(0.01, best_take_profit - 0.03),
            max(0.01, best_take_profit - 0.015),
            best_take_profit,
            best_take_profit + 0.015,
            best_take_profit + 0.03
        ]

        combination_results = self.optimize_stop_combination(
            refined_stop_loss_levels,
            refined_take_profit_levels
        )

        # 选择最佳组合
        best_combination = combination_results.sort_values(by=optimization_target, ascending=False).iloc[0]

        # 保存最佳参数
        self.best_stop_params = {
            'stop_loss': best_combination['止损水平'],
            'take_profit': best_combination['止盈水平'],
            'trailing_stop': best_trailing_stop,
            'performance': {
                'total_return': best_combination['总收益率'],
                'annualized_return': best_combination['年化收益率'],
                'max_drawdown': best_combination['最大回撤'],
                'sharpe_ratio': best_combination['夏普比率'],
                'sortino_ratio': best_combination['索提诺比率'],
                'calmar_ratio': best_combination['卡玛比率'],
                'win_rate': best_combination['胜率'],
                'profit_loss_ratio': best_combination['盈亏比'],
                'total_trades': best_combination['总交易次数']
            }
        }

        # 保存优化结果
        self.optimization_results = {
            'stop_loss_results': stop_loss_results,
            'take_profit_results': take_profit_results,
            'trailing_stop_results': trailing_stop_results,
            'combination_results': combination_results,
            'best_params': self.best_stop_params
        }

        return self.best_stop_params

    def plot_optimization_results(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        绘制优化结果

        参数:
        figsize (Tuple[int, int]): 图形大小
        """
        if self.optimization_results is None:
            print("请先运行优化")
            return

        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 绘制止损优化结果
        stop_loss_results = self.optimization_results['stop_loss_results']
        sns.lineplot(x='止损水平', y='夏普比率', data=stop_loss_results, ax=axes[0, 0], marker='o')
        axes[0, 0].set_title('止损水平与夏普比率关系')

        # 绘制约止盈优化结果
        take_profit_results = self.optimization_results['take_profit_results']
        sns.lineplot(x='止盈水平', y='夏普比率', data=take_profit_results, ax=axes[0, 1], marker='o')
        axes[0, 1].set_title('止盈水平与夏普比率关系')

        # 绘制追踪止损优化结果
        trailing_stop_results = self.optimization_results['trailing_stop_results']
        sns.lineplot(x='追踪止损水平', y='夏普比率', data=trailing_stop_results, ax=axes[1, 0], marker='o')
        axes[1, 0].set_title('追踪止损水平与夏普比率关系')

        # 绘制组合优化结果热图
        combination_results = self.optimization_results['combination_results']
        pivot_table = combination_results.pivot_table(
            values='夏普比率', 
            index='止损水平', 
            columns='止盈水平'
        )
        sns.heatmap(pivot_table, annot=True, cmap='viridis', ax=axes[1, 1])
        axes[1, 1].set_title('止损止盈组合与夏普比率关系')

        plt.tight_layout()
        plt.show()

    def apply_best_params(self, strategy: Strategy) -> Strategy:
        """
        应用最佳止盈止损参数到策略

        参数:
        strategy (Strategy): 策略实例

        返回:
        Strategy: 应用了最佳参数的策略
        """
        if self.best_stop_params is None:
            print("请先运行优化")
            return strategy

        # 设置最佳止盈止损参数
        if hasattr(strategy, 'risk_control'):
            strategy.risk_control.stop_loss = self.best_stop_params['stop_loss']
            strategy.risk_control.take_profit = self.best_stop_params['take_profit']
            strategy.risk_control.trailing_stop = self.best_stop_params['trailing_stop']
        else:
            strategy.risk_control = RiskControl(
                stop_loss=self.best_stop_params['stop_loss'],
                take_profit=self.best_stop_params['take_profit'],
                trailing_stop=self.best_stop_params['trailing_stop']
            )

        return strategy