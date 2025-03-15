# -*- coding: utf-8 -*-
"""
参数优化模块

用于优化策略参数
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Callable
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.backtest.backtest_engine import BacktestEngine
from src.strategy.strategy import Strategy


class ParameterOptimizer:
    """
    参数优化器

    用于优化策略参数
    """

    def __init__(self, 
                 strategy_class: type,
                 param_grid: Dict[str, List],
                 price_data: Dict[str, pd.DataFrame],
                 factor_data: Dict[str, pd.DataFrame],
                 start_date: Union[str, pd.Timestamp],
                 end_date: Union[str, pd.Timestamp],
                 commission: float = 0.0003,
                 slippage: float = 0.0001,
                 initial_capital: float = 1000000.0,
                 optimization_target: str = 'sharpe_ratio',
                 n_jobs: int = 1):
        """
        初始化参数优化器

        参数:
        strategy_class (type): 策略类
        param_grid (Dict[str, List]): 参数网格，格式为{参数名: [参数值列表]}
        price_data (Dict[str, pd.DataFrame]): 价格数据
        factor_data (Dict[str, pd.DataFrame]): 因子数据
        start_date (Union[str, pd.Timestamp]): 回测开始日期
        end_date (Union[str, pd.Timestamp]): 回测结束日期
        commission (float): 手续费率
        slippage (float): 滑点率
        initial_capital (float): 初始资金
        optimization_target (str): 优化目标，如'sharpe_ratio', 'total_return', 'calmar_ratio'等
        n_jobs (int): 并行任务数
        """
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.price_data = price_data
        self.factor_data = factor_data
        self.start_date = start_date
        self.end_date = end_date
        self.commission = commission
        self.slippage = slippage
        self.initial_capital = initial_capital
        self.optimization_target = optimization_target
        self.n_jobs = n_jobs

        # 优化结果
        self.optimization_results = None
        self.best_params = None
        self.best_performance = None

    def _run_backtest(self, params: Dict) -> Tuple[Dict, Dict]:
        """
        运行单次回测

        参数:
        params (Dict): 策略参数

        返回:
        Tuple[Dict, Dict]: (参数, 性能指标)
        """
        # 创建策略实例
        strategy = self.strategy_class(**params)

        # 创建回测引擎
        backtest_engine = BacktestEngine(
            strategy=strategy,
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
        performance_metrics = backtest_result['performance_metrics']

        return params, performance_metrics

    def grid_search(self) -> pd.DataFrame:
        """
        网格搜索优化

        返回:
        pd.DataFrame: 优化结果
        """
        # 生成参数组合
        param_combinations = list(itertools.product(*self.param_grid.values()))
        param_names = list(self.param_grid.keys())

        # 创建参数字典列表
        param_dicts = []
        for combo in param_combinations:
            param_dict = {name: value for name, value in zip(param_names, combo)}
            param_dicts.append(param_dict)

        # 运行回测
        results = []

        if self.n_jobs > 1:
            # 并行运行
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [executor.submit(self._run_backtest, params) for params in param_dicts]

                for future in tqdm(as_completed(futures), total=len(futures), desc="参数优化进度"):
                    params, metrics = future.result()
                    results.append((params, metrics))
        else:
            # 串行运行
            for params in tqdm(param_dicts, desc="参数优化进度"):
                params, metrics = self._run_backtest(params)
                results.append((params, metrics))

        # 整理结果
        optimization_results = []

        for params, metrics in results:
            result = params.copy()
            result.update({
                'total_return': metrics.get('total_return', 0),
                'annualized_return': metrics.get('annualized_return', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'sortino_ratio': metrics.get('sortino_ratio', 0),
                'calmar_ratio': metrics.get('calmar_ratio', 0),
                'win_rate': metrics.get('win_rate', 0),
                'profit_loss_ratio': metrics.get('profit_loss_ratio', 0),
                'total_trades': metrics.get('total_trades', 0)
            })
            optimization_results.append(result)

        # 转换为DataFrame
        results_df = pd.DataFrame(optimization_results)

        # 按优化目标排序
        results_df = results_df.sort_values(by=self.optimization_target, ascending=False)

        # 保存结果
        self.optimization_results = results_df
        self.best_params = results_df.iloc[0][param_names].to_dict()
        self.best_performance = results_df.iloc[0]

        return results_df

    def random_search(self, n_iter: int = 100) -> pd.DataFrame:
        """
        随机搜索优化

        参数:
        n_iter (int): 迭代次数

        返回:
        pd.DataFrame: 优化结果
        """
        # 生成随机参数组合
        param_dicts = []
        param_names = list(self.param_grid.keys())

        for _ in range(n_iter):
            param_dict = {}
            for name in param_names:
                values = self.param_grid[name]
                param_dict[name] = np.random.choice(values)
            param_dicts.append(param_dict)

        # 运行回测
        results = []

        if self.n_jobs > 1:
            # 并行运行
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [executor.submit(self._run_backtest, params) for params in param_dicts]

                for future in tqdm(as_completed(futures), total=len(futures), desc="参数优化进度"):
                    params, metrics = future.result()
                    results.append((params, metrics))
        else:
            # 串行运行
            for params in tqdm(param_dicts, desc="参数优化进度"):
                params, metrics = self._run_backtest(params)
                results.append((params, metrics))

        # 整理结果
        optimization_results = []

        for params, metrics in results:
            result = params.copy()
            result.update({
                'total_return': metrics.get('total_return', 0),
                'annualized_return': metrics.get('annualized_return', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'sortino_ratio': metrics.get('sortino_ratio', 0),
                'calmar_ratio': metrics.get('calmar_ratio', 0),
                'win_rate': metrics.get('win_rate', 0),
                'profit_loss_ratio': metrics.get('profit_loss_ratio', 0),
                'total_trades': metrics.get('total_trades', 0)
            })
            optimization_results.append(result)

        # 转换为DataFrame
        results_df = pd.DataFrame(optimization_results)

        # 按优化目标排序
        results_df = results_df.sort_values(by=self.optimization_target, ascending=False)

        # 保存结果
        self.optimization_results = results_df
        self.best_params = results_df.iloc[0][param_names].to_dict()
        self.best_performance = results_df.iloc[0]

        return results_df

    def plot_optimization_results(self, top_n: int = 20, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        绘制优化结果

        参数:
        top_n (int): 显示前N个结果
        figsize (Tuple[int, int]): 图形大小
        """
        if self.optimization_results is None:
            print("请先运行优化")
            return

        # 获取前N个结果
        top_results = self.optimization_results.head(top_n)

        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 绘制总收益率
        sns.barplot(x=top_results.index, y='total_return', data=top_results, ax=axes[0, 0])
        axes[0, 0].set_title('总收益率')
        axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45)

        # 绘制夏普比率
        sns.barplot(x=top_results.index, y='sharpe_ratio', data=top_results, ax=axes[0, 1])
        axes[0, 1].set_title('夏普比率')
        axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)

        # 绘制最大回撤
        sns.barplot(x=top_results.index, y='max_drawdown', data=top_results, ax=axes[1, 0])
        axes[1, 0].set_title('最大回撤')
        axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45)

        # 绘制胜率
        sns.barplot(x=top_results.index, y='win_rate', data=top_results, ax=axes[1, 1])
        axes[1, 1].set_title('胜率')
        axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45)

        plt.tight_layout()
        plt.show()

    def plot_parameter_impact(self, param_name: str, metric: str = None, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        绘制参数影响

        参数:
        param_name (str): 参数名
        metric (str): 性能指标，默认为优化目标
        figsize (Tuple[int, int]): 图形大小
        """
        if self.optimization_results is None:
            print("请先运行优化")
            return

        if metric is None:
            metric = self.optimization_target

        # 按参数分组
        grouped = self.optimization_results.groupby(param_name)[metric].mean().reset_index()

        # 绘制参数影响
        plt.figure(figsize=figsize)
        sns.barplot(x=param_name, y=metric, data=grouped)
        plt.title(f'参数 {param_name} 对 {metric} 的影响')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def get_best_strategy(self) -> Strategy:
        """
        获取最优策略

        返回:
        Strategy: 最优策略实例
        """
        if self.best_params is None:
            print("请先运行优化")
            return None

        # 创建最优策略实例
        best_strategy = self.strategy_class(**self.best_params)

        return best_strategy