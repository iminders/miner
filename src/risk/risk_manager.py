# -*- coding: utf-8 -*-
"""
风险管理主模块

该模块提供了风险管理的核心功能，包括风险指标计算、风险分析、压力测试等。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Optional, Tuple

from src.risk.risk_metrics import (
    calculate_var, calculate_cvar, calculate_drawdown,
    calculate_volatility, calculate_sharpe_ratio
)
from src.risk.risk_analysis import (
    analyze_returns_distribution, analyze_correlation,
    analyze_factor_exposure, analyze_regime
)
from src.risk.stress_testing import (
    historical_stress_test, monte_carlo_simulation,
    custom_stress_test
)
from src.risk.portfolio_optimization import (
    optimize_portfolio, calculate_efficient_frontier
)
from src.risk.visualization import (
    plot_risk_metrics, plot_drawdown, plot_rolling_metrics,
    plot_correlation_matrix, plot_stress_test_results
)


class RiskManager:
    """
    风险管理器类

    提供全面的风险管理功能，包括风险指标计算、风险分析、压力测试、
    投资组合优化等。
    """

    def __init__(
        self,
        returns: Union[pd.Series, pd.DataFrame],
        weights: Optional[np.ndarray] = None,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.0,
        frequency: str = 'D'
    ):
        """
        初始化风险管理器

        参数:
        returns (pd.Series or pd.DataFrame): 资产收益率序列或数据框
        weights (np.ndarray, optional): 资产权重数组
        benchmark_returns (pd.Series, optional): 基准收益率序列
        risk_free_rate (float): 无风险利率，默认为0
        frequency (str): 数据频率，'D'表示日度，'W'表示周度，'M'表示月度
        """
        self.returns = returns
        self.weights = weights
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
        self.frequency = frequency

        # 计算投资组合收益率
        if isinstance(returns, pd.DataFrame) and weights is not None:
            self.portfolio_returns = returns.dot(weights)
        else:
            self.portfolio_returns = returns

    def calculate_basic_metrics(self) -> Dict[str, float]:
        """
        计算基本风险指标

        返回:
        Dict[str, float]: 包含各种风险指标的字典
        """
        # 计算年化系数
        if self.frequency == 'D':
            annualization_factor = 252
        elif self.frequency == 'W':
            annualization_factor = 52
        elif self.frequency == 'M':
            annualization_factor = 12
        else:
            annualization_factor = 252  # 默认使用日度数据的年化系数

        # 计算基本指标
        total_return = (1 + self.portfolio_returns).prod() - 1
        annualized_return = (1 + total_return) ** (annualization_factor / len(self.portfolio_returns)) - 1
        volatility = calculate_volatility(self.portfolio_returns, annualization_factor)
        sharpe = calculate_sharpe_ratio(self.portfolio_returns, self.risk_free_rate, annualization_factor)

        # 计算最大回撤
        max_drawdown = calculate_drawdown(self.portfolio_returns)['max_drawdown']

        # 计算VaR和CVaR
        var_95 = calculate_var(self.portfolio_returns, alpha=0.05)
        cvar_95 = calculate_cvar(self.portfolio_returns, alpha=0.05)

        # 计算偏度和峰度
        skewness = self.portfolio_returns.skew()
        kurtosis = self.portfolio_returns.kurtosis()

        # 计算下行风险
        downside_returns = self.portfolio_returns[self.portfolio_returns < 0]
        downside_risk = np.sqrt((downside_returns**2).mean()) * np.sqrt(annualization_factor)

        # 计算Sortino比率
        sortino_ratio = (annualized_return - self.risk_free_rate) / downside_risk if downside_risk != 0 else 0

        # 计算Calmar比率
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # 如果有基准收益率，计算相对指标
        if self.benchmark_returns is not None:
            # 计算超额收益
            excess_returns = self.portfolio_returns - self.benchmark_returns

            # 计算年化超额收益
            total_excess_return = (1 + excess_returns).prod() - 1
            annualized_excess_return = (1 + total_excess_return) ** (annualization_factor / len(excess_returns)) - 1

            # 计算跟踪误差
            tracking_error = excess_returns.std() * np.sqrt(annualization_factor)

            # 计算信息比率
            information_ratio = annualized_excess_return / tracking_error if tracking_error != 0 else 0

            # 计算Beta和Alpha
            cov_matrix = np.cov(self.portfolio_returns, self.benchmark_returns)
            beta = cov_matrix[0, 1] / cov_matrix[1, 1]

            benchmark_annualized_return = (1 + self.benchmark_returns.sum()) ** (annualization_factor / len(self.benchmark_returns)) - 1
            alpha = annualized_return - (self.risk_free_rate + beta * (benchmark_annualized_return - self.risk_free_rate))

            # 添加相对指标
            metrics = {
                '总收益率': total_return,
                '年化收益率': annualized_return,
                '波动率': volatility,
                '夏普比率': sharpe,
                '最大回撤': max_drawdown,
                'VaR (95%)': var_95,
                'CVaR (95%)': cvar_95,
                '偏度': skewness,
                '峰度': kurtosis,
                '下行风险': downside_risk,
                'Sortino比率': sortino_ratio,
                'Calmar比率': calmar_ratio,
                '年化超额收益': annualized_excess_return,
                '跟踪误差': tracking_error,
                '信息比率': information_ratio,
                'Beta': beta,
                'Alpha': alpha
            }
        else:
            # 只包含绝对指标
            metrics = {
                '总收益率': total_return,
                '年化收益率': annualized_return,
                '波动率': volatility,
                '夏普比率': sharpe,
                '最大回撤': max_drawdown,
                'VaR (95%)': var_95,
                'CVaR (95%)': cvar_95,
                '偏度': skewness,
                '峰度': kurtosis,
                '下行风险': downside_risk,
                'Sortino比率': sortino_ratio,
                'Calmar比率': calmar_ratio
            }

        return metrics

    def calculate_rolling_metrics(self, window: int = 63) -> pd.DataFrame:
        """
        计算滚动风险指标

        参数:
        window (int): 滚动窗口大小，默认为63（约3个月的交易日）

        返回:
        pd.DataFrame: 包含滚动风险指标的数据框
        """
        # 计算年化系数
        if self.frequency == 'D':
            annualization_factor = 252
        elif self.frequency == 'W':
            annualization_factor = 52
        elif self.frequency == 'M':
            annualization_factor = 12
        else:
            annualization_factor = 252

        # 初始化结果数据框
        rolling_metrics = pd.DataFrame(index=self.portfolio_returns.index[window-1:])

        # 计算滚动收益率
        rolling_returns = self.portfolio_returns.rolling(window=window)

        # 计算滚动波动率
        rolling_metrics['波动率'] = rolling_returns.std() * np.sqrt(annualization_factor)

        # 计算滚动夏普比率
        rolling_metrics['夏普比率'] = (rolling_returns.mean() * annualization_factor - self.risk_free_rate) / (rolling_returns.std() * np.sqrt(annualization_factor))

        # 计算滚动最大回撤
        rolling_max_drawdown = []

        for i in range(window-1, len(self.portfolio_returns)):
            window_returns = self.portfolio_returns.iloc[i-window+1:i+1]
            drawdown_info = calculate_drawdown(window_returns)
            rolling_max_drawdown.append(drawdown_info['max_drawdown'])

        rolling_metrics['最大回撤'] = rolling_max_drawdown

        # 计算滚动VaR和CVaR
        rolling_var = []
        rolling_cvar = []

        for i in range(window-1, len(self.portfolio_returns)):
            window_returns = self.portfolio_returns.iloc[i-window+1:i+1]
            var_95 = calculate_var(window_returns, alpha=0.05)
            cvar_95 = calculate_cvar(window_returns, alpha=0.05)
            rolling_var.append(var_95)
            rolling_cvar.append(cvar_95)

        rolling_metrics['VaR (95%)'] = rolling_var
        rolling_metrics['CVaR (95%)'] = rolling_cvar

        # 如果有基准收益率，计算相对指标
        if self.benchmark_returns is not None:
            # 计算滚动超额收益
            excess_returns = self.portfolio_returns - self.benchmark_returns
            rolling_excess_returns = excess_returns.rolling(window=window)

            # 计算滚动跟踪误差
            rolling_metrics['跟踪误差'] = rolling_excess_returns.std() * np.sqrt(annualization_factor)

            # 计算滚动信息比率
            rolling_metrics['信息比率'] = (rolling_excess_returns.mean() * annualization_factor) / (rolling_excess_returns.std() * np.sqrt(annualization_factor))

            # 计算滚动Beta
            rolling_beta = []

            for i in range(window-1, len(self.portfolio_returns)):
                window_portfolio = self.portfolio_returns.iloc[i-window+1:i+1]
                window_benchmark = self.benchmark_returns.iloc[i-window+1:i+1]

                cov_matrix = np.cov(window_portfolio, window_benchmark)
                beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 0

                rolling_beta.append(beta)

            rolling_metrics['Beta'] = rolling_beta

        return rolling_metrics

    def plot_risk_dashboard(self, figsize: Tuple[int, int] = (15, 10)):
        """
        绘制风险仪表盘

        参数:
        figsize (tuple): 图表大小
        """
        # 计算基本风险指标
        metrics = self.calculate_basic_metrics()

        # 计算累积收益率
        cumulative_returns = (1 + self.portfolio_returns).cumprod()

        # 计算回撤
        drawdown_info = calculate_drawdown(self.portfolio_returns)
        drawdowns = drawdown_info['drawdowns']

        # 创建图表
        fig = plt.figure(figsize=figsize)

        # 绘制累积收益率
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
        ax1.plot(cumulative_returns.index, cumulative_returns, 'b-', label='投资组合')

        if self.benchmark_returns is not None:
            benchmark_cumulative_returns = (1 + self.benchmark_returns).cumprod()
            ax1.plot(benchmark_cumulative_returns.index, benchmark_cumulative_returns, 'r--', label='基准')
            ax1.legend()

        ax1.set_title('累积收益率', fontsize=12)
        ax1.set_xlabel('')
        ax1.set_ylabel('累积收益率', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 绘制回撤
        ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
        ax2.fill_between(drawdowns.index, drawdowns, 0, color='r', alpha=0.3)
        ax2.set_title('回撤', fontsize=12)
        ax2.set_xlabel('')
        ax2.set_ylabel('回撤', fontsize=10)
        ax2.grid(True, alpha=0.3)

        # 绘制收益率分布
        ax3 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
        ax3.hist(self.portfolio_returns, bins=30, alpha=0.5, color='b')

        # 添加VaR和CVaR线
        ax3.axvline(x=metrics['VaR (95%)'], color='r', linestyle='--', label=f'VaR (95%): {metrics["VaR (95%)"]:.2%}')
        ax3.axvline(x=metrics['CVaR (95%)'], color='purple', linestyle='--', label=f'CVaR (95%): {metrics["CVaR (95%)"]:.2%}')

        ax3.set_title('收益率分布', fontsize=12)
        ax3.set_xlabel('收益率', fontsize=10)
        ax3.set_ylabel('频率', fontsize=10)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # 绘制风险指标表格
        ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
        ax4.axis('off')

        # 选择要显示的指标
        display_metrics = [
            '年化收益率', '波动率', '夏普比率', '最大回撤', 
            'Sortino比率', 'Calmar比率', 'VaR (95%)', 'CVaR (95%)'
        ]

        if self.benchmark_returns is not None:
            display_metrics.extend(['Beta', 'Alpha', '跟踪误差', '信息比率'])

        # 创建表格数据
        table_data = []
        for metric in display_metrics:
            if metric in metrics:
                # 格式化显示
                if metric in ['年化收益率', '波动率', '最大回撤', 'VaR (95%)', 'CVaR (95%)', '下行风险', 'Alpha', '跟踪误差']:
                    value = f"{metrics[metric]:.2%}"
                elif metric in ['夏普比率', 'Sortino比率', 'Calmar比率', 'Beta', '信息比率']:
                    value = f"{metrics[metric]:.2f}"
                else:
                    value = f"{metrics[metric]}"

                table_data.append([metric, value])

        # 创建表格
        table = ax4.table(
            cellText=table_data,
            colLabels=['风险指标', '数值'],
            loc='center',
            cellLoc='center',
            colWidths=[0.4, 0.4]
        )

        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        plt.tight_layout()
        plt.show()

    def analyze_returns_distribution(self, figsize: Tuple[int, int] = (12, 8)):
        """
        分析收益率分布

        参数:
        figsize (tuple): 图表大小
        """
        # 导入统计模块
        from scipy import stats

        # 计算统计量
        mean = self.portfolio_returns.mean()
        std = self.portfolio_returns.std()
        skewness = self.portfolio_returns.skew()
        kurtosis = self.portfolio_returns.kurtosis()

        # 进行正态性检验
        k2, p_value = stats.normaltest(self.portfolio_returns)

        # 创建图表
        plt.figure(figsize=figsize)

        # 绘制直方图和核密度估计
        plt.subplot(2, 2, 1)
        sns.histplot(self.portfolio_returns, kde=True, color='blue', stat='density', alpha=0.6)

        # 添加正态分布曲线
        x = np.linspace(self.portfolio_returns.min(), self.portfolio_returns.max(), 100)
        plt.plot(x, stats.norm.pdf(x, mean, std), 'r-', label='正态分布')

        plt.title('收益率分布', fontsize=12)
        plt.xlabel('收益率', fontsize=10)
        plt.ylabel('密度', fontsize=10)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 绘制QQ图
        plt.subplot(2, 2, 2)
        stats.probplot(self.portfolio_returns, dist="norm", plot=plt)
        plt.title('QQ图 (正态性检验)', fontsize=12)
        plt.grid(True, alpha=0.3)

        # 绘制自相关图
        plt.subplot(2, 2, 3)
        pd.plotting.autocorrelation_plot(self.portfolio_returns)
        plt.title('收益率自相关图', fontsize=12)
        plt.xlabel('滞后期数', fontsize=10)
        plt.ylabel('自相关系数', fontsize=10)
        plt.grid(True, alpha=0.3)

        # 绘制统计量表格
        plt.subplot(2, 2, 4)
        plt.axis('off')

        # 创建表格数据
        table_data = [
            ['均值', f"{mean:.6f}"],
            ['标准差', f"{std:.6f}"],
            ['偏度', f"{skewness:.4f}"],
            ['峰度', f"{kurtosis:.4f}"],
            ['正态性检验统计量', f"{k2:.4f}"],
            ['p值', f"{p_value:.6f}"],
            ['正态分布假设', "拒绝" if p_value < 0.05 else "接受"]
        ]

        # 创建表格
        table = plt.table(
            cellText=table_data,
            colLabels=['统计量', '数值'],
            loc='center',
            cellLoc='center',
            colWidths=[0.4, 0.4]
        )

        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        plt.tight_layout()
        plt.show()

        # 打印分析结果
        print("收益率分布分析结果:")
        print(f"均值: {mean:.6f}")
        print(f"标准差: {std:.6f}")
        print(f"偏度: {skewness:.4f} (正态分布为0，负值表示左偏，正值表示右偏)")
        print(f"峰度: {kurtosis:.4f} (正态分布为0，正值表示尾部更厚)")
        print(f"正态性检验 (D'Agostino-Pearson):")
        print(f"  统计量: {k2:.4f}")
        print(f"  p值: {p_value:.6f}")

        if p_value < 0.05:
            print("  结论: 拒绝正态分布假设 (p < 0.05)")
        else:
            print("  结论: 无法拒绝正态分布假设 (p >= 0.05)")

    def analyze_correlation(self, method: str = 'pearson', figsize: Tuple[int, int] = (10, 8)):
        """
        分析资产相关性

        参数:
        method (str): 相关系数计算方法，'pearson', 'spearman', 或 'kendall'
        figsize (tuple): 图表大小
        """
        if not isinstance(self.returns, pd.DataFrame):
            raise ValueError("相关性分析需要资产收益率DataFrame")

        # 计算相关系数矩阵
        correlation_matrix = self.returns.corr(method=method)

        # 创建图表
        plt.figure(figsize=figsize)

        # 绘制热图
        sns.heatmap(
            correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            vmin=-1, 
            vmax=1, 
            center=0,
            linewidths=0.5,
            fmt='.2f'
        )

        plt.title(f'资产相关性矩阵 ({method.capitalize()})', fontsize=14)
        plt.tight_layout()
        plt.show()

        # 计算平均相关系数
        n = correlation_matrix.shape[0]
        total_corr = correlation_matrix.sum().sum()
        avg_corr = (total_corr - n) / (n * (n - 1))  # 减去对角线的1

        # 打印分析结果
        print("相关性分析结果:")
        print(f"相关系数计算方法: {method.capitalize()}")
        print(f"平均相关系数: {avg_corr:.4f}")

        # 找出相关性最高和最低的资产对
        # 创建上三角矩阵（排除对角线）
        mask = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        upper_tri = correlation_matrix.where(mask)

        # 找出最高相关性
        max_corr_idx = upper_tri.stack().idxmax()
        max_corr_value = upper_tri.stack().max()

        # 找出最低相关性
        min_corr_idx = upper_tri.stack().idxmin()
        min_corr_value = upper_tri.stack().min()

        print(f"相关性最高的资产对: {max_corr_idx[0]} 和 {max_corr_idx[1]} (相关系数: {max_corr_value:.4f})")
        print(f"相关性最低的资产对: {min_corr_idx[0]} 和 {min_corr_idx[1]} (相关系数: {min_corr_value:.4f})")

    def analyze_drawdowns(self, top_n: int = 5, figsize: Tuple[int, int] = (12, 8)):
        """
        分析最大回撤

        参数:
        top_n (int): 显示前N个最大回撤
        figsize (tuple): 图表大小
        """
        # 计算累积收益率
        cumulative_returns = (1 + self.portfolio_returns).cumprod()

        # 计算回撤
        drawdown_info = calculate_drawdown(self.portfolio_returns, top_n=top_n)
        drawdowns = drawdown_info['drawdowns']
        drawdown_details = drawdown_info['drawdown_details']

        # 创建图表
        plt.figure(figsize=figsize)

        # 绘制累积收益率和回撤
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(cumulative_returns.index, cumulative_returns, 'b-', label='累积收益率')

        # 标记最大回撤区间
        for i, drawdown in enumerate(drawdown_details[:top_n]):
            start_date = drawdown['peak_date']
            end_date = drawdown['recovery_date'] if drawdown['recovery_date'] else drawdown['valley_date']

            # 如果没有恢复日期，使用最后一个日期
            if end_date == drawdown['valley_date'] and i == 0:
                end_date = cumulative_returns.index[-1]

            ax1.axvspan(start_date, end_date, color=f'C{i+1}', alpha=0.2, label=f'回撤 #{i+1}: {drawdown["drawdown"]:.2%}')

        ax1.set_title('累积收益率和最大回撤', fontsize=12)
        ax1.set_xlabel('')
        ax1.set_ylabel('累积收益率', fontsize=10)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 绘制回撤
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        ax2.fill_between(drawdowns.index, drawdowns, 0, color='r', alpha=0.3)

        # 标记最大回撤点
        for i, drawdown in enumerate(drawdown_details[:top_n]):
            valley_date = drawdown['valley_date']
            valley_value = drawdowns.loc[valley_date]

            ax2.scatter(valley_date, valley_value, color=f'C{i+1}', s=50, zorder=5)
            ax2.annotate(
                f'#{i+1}: {drawdown["drawdown"]:.2%}',
                (valley_date, valley_value),
                xytext=(10, -20),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color=f'C{i+1}')
            )

        ax2.set_title('回撤', fontsize=12)
        ax2.set_xlabel('')
        ax2.set_ylabel('回撤', fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 打印回撤分析结果
        print("回撤分析结果:")
        print(f"最大回撤: {drawdown_details[0]['drawdown']:.2%}")

        print("\n前", top_n, "个最大回撤:")
        for i, drawdown in enumerate(drawdown_details[:top_n]):
            recovery_time = "尚未恢复" if drawdown['recovery_date'] is None else f"{(drawdown['recovery_date'] - drawdown['valley_date']).days}天"
            drawdown_time = f"{(drawdown['valley_date'] - drawdown['peak_date']).days}天"

            print(f"#{i+1}: {drawdown['drawdown']:.2%}")
            print(f"  峰值日期: {drawdown['peak_date'].strftime('%Y-%m-%d')}")
            print(f"  谷值日期: {drawdown['valley_date'].strftime('%Y-%m-%d')}")
            print(f"  恢复日期: {drawdown['recovery_date'].strftime('%Y-%m-%d') if drawdown['recovery_date'] else '尚未恢复'}")
            print(f"  下跌时间: {drawdown_time}")
            print(f"  恢复时间: {recovery_time}")
            print(f"  总持续时间: {(drawdown['recovery_date'] - drawdown['peak_date']).days if drawdown['recovery_date'] else '尚未结束'}天")
            print("")

    def analyze_rolling_performance(self, window: int = 63, figsize: Tuple[int, int] = (12, 10)):
        """
        分析滚动业绩表现

        参数:
        window (int): 滚动窗口大小，默认为63（约3个月的交易日）
        figsize (tuple): 图表大小
        """
        # 计算滚动指标
        rolling_metrics = self.calculate_rolling_metrics(window)

        # 计算年化系数
        if self.frequency == 'D':
            annualization_factor = 252
        elif self.frequency == 'W':
            annualization_factor = 52
        elif self.frequency == 'M':
            annualization_factor = 12
        else:
            annualization_factor = 252

        # 计算滚动收益率
        rolling_returns = self.portfolio_returns.rolling(window=window).mean() * annualization_factor

        # 创建图表
        plt.figure(figsize=figsize)

        # 绘制滚动收益率
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(rolling_returns.index, rolling_returns, 'b-', label='滚动年化收益率')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)

        if self.benchmark_returns is not None:
            rolling_benchmark_returns = self.benchmark_returns.rolling(window=window).mean() * annualization_factor
            ax1.plot(rolling_benchmark_returns.index, rolling_benchmark_returns, 'r--', label='基准滚动年化收益率')

            # 计算滚动超额收益率
            rolling_excess_returns = rolling_returns - rolling_benchmark_returns
            ax1.plot(rolling_excess_returns.index, rolling_excess_returns, 'g-', label='滚动超额收益率')

        ax1.set_title('滚动年化收益率', fontsize=12)
        ax1.set_xlabel('')
        ax1.set_ylabel('收益率', fontsize=10)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 绘制滚动波动率和最大回撤
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.plot(rolling_metrics.index, rolling_metrics['波动率'], 'r-', label='滚动波动率')
        ax2.set_ylabel('波动率', fontsize=10)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        ax2_twin = ax2.twinx()
        ax2_twin.plot(rolling_metrics.index, rolling_metrics['最大回撤'], 'g-', label='滚动最大回撤')
        ax2_twin.set_ylabel('最大回撤', fontsize=10)
        ax2_twin.legend(loc='upper right')

        ax2.set_title('滚动波动率和最大回撤', fontsize=12)

        # 绘制滚动风险调整收益指标
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        ax3.plot(rolling_metrics.index, rolling_metrics['夏普比率'], 'b-', label='滚动夏普比率')
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)

        if '信息比率' in rolling_metrics.columns:
            ax3.plot(rolling_metrics.index, rolling_metrics['信息比率'], 'g-', label='滚动信息比率')

        ax3.set_title('滚动风险调整收益指标', fontsize=12)
        ax3.set_xlabel('')
        ax3.set_ylabel('比率', fontsize=10)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 打印滚动业绩分析结果
        print("滚动业绩分析结果:")
        print(f"滚动窗口: {window}天")

        # 计算各指标的均值、最大值、最小值和当前值
        metrics_summary = pd.DataFrame(index=rolling_metrics.columns)
        metrics_summary['均值'] = rolling_metrics.mean()
        metrics_summary['最大值'] = rolling_metrics.max()
        metrics_summary['最小值'] = rolling_metrics.min()
        metrics_summary['当前值'] = rolling_metrics.iloc[-1]

        # 格式化显示
        pd.set_option('display.float_format', '{:.4f}'.format)
        print("\n滚动指标统计:")
        print(metrics_summary)

        # 计算正收益和负收益的比例
        positive_returns_ratio = (rolling_returns > 0).mean()
        print(f"\n正收益比例: {positive_returns_ratio:.2%}")

        if self.benchmark_returns is not None:
            outperform_ratio = (rolling_returns > rolling_benchmark_returns).mean()
            print(f"跑赢基准比例: {outperform_ratio:.2%}")

    def run_historical_stress_test(self, stress_periods: Dict[str, Tuple[str, str]], figsize: Tuple[int, int] = (12, 8)):
        """
        运行历史压力测试

        参数:
        stress_periods (dict): 压力期间字典，格式为 {事件名称: (开始日期, 结束日期)}
        figsize (tuple): 图表大小
        """
        # 导入压力测试模块
        from src.risk.stress_testing import historical_stress_test

        # 运行历史压力测试
        stress_test_results = historical_stress_test(
            self.portfolio_returns, 
            stress_periods, 
            self.benchmark_returns
        )

        # 创建图表
        plt.figure(figsize=figsize)

        # 绘制压力测试结果
        ax = plt.subplot(1, 1, 1)

        # 准备数据
        events = list(stress_test_results.keys())
        portfolio_returns = [stress_test_results[event]['portfolio_return'] * 100 for event in events]

        # 设置条形图位置
        x = np.arange(len(events))
        width = 0.35

        # 绘制投资组合收益率
        bars1 = ax.bar(x, portfolio_returns, width, label='投资组合', color='b', alpha=0.7)

        # 如果有基准收益率，也绘制基准收益率
        if self.benchmark_returns is not None:
            benchmark_returns = [stress_test_results[event]['benchmark_return'] * 100 for event in events]
            bars2 = ax.bar(x + width, benchmark_returns, width, label='基准', color='r', alpha=0.7)

        # 添加标签和图例
        ax.set_xlabel('历史压力事件', fontsize=12)
        ax.set_ylabel('累积收益率 (%)', fontsize=12)
        ax.set_title('历史压力测试结果', fontsize=14)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(events, rotation=45, ha='right')
        ax.legend()

        # 添加数据标签
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3点垂直偏移
                            textcoords="offset points",
                            ha='center', va='bottom')

        add_labels(bars1)
        if self.benchmark_returns is not None:
            add_labels(bars2)

        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        plt.show()

        # 打印压力测试结果
        print("历史压力测试结果:")
        for event, result in stress_test_results.items():
            print(f"\n事件: {event}")
            print(f"  时间段: {result['start_date']} 至 {result['end_date']} ({result['duration']}天)")
            print(f"  投资组合收益率: {result['portfolio_return']:.2%}")

            if 'benchmark_return' in result:
                print(f"  基准收益率: {result['benchmark_return']:.2%}")
                print(f"  超额收益率: {result['excess_return']:.2%}")

            print(f"  最大回撤: {result['max_drawdown']:.2%}")
            print(f"  波动率: {result['volatility']:.2%}")
            print(f"  夏普比率: {result['sharpe']:.2f}")

    def run_monte_carlo_simulation(
        self, 
        n_simulations: int = 1000, 
        n_days: int = 252, 
        method: str = 'bootstrap', 
        confidence_level: float = 0.95,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        运行蒙特卡洛模拟

        参数:
        n_simulations (int): 模拟次数
        n_days (int): 模拟天数
        method (str): 模拟方法，'bootstrap'（历史抽样）或'normal'（正态分布）
        confidence_level (float): 置信水平
        figsize (tuple): 图表大小
        """
        # 导入蒙特卡洛模拟模块
        from src.risk.stress_testing import monte_carlo_simulation

        # 运行蒙特卡洛模拟
        simulation_results = monte_carlo_simulation(
            self.portfolio_returns, 
            n_simulations, 
            n_days, 
            method
        )

        # 计算统计量
        final_returns = simulation_results['final_returns']
        mean_return = np.mean(final_returns)
        median_return = np.median(final_returns)
        std_return = np.std(final_returns)

        # 计算置信区间
        lower_percentile = (1 - confidence_level) / 2 * 100
        upper_percentile = (1 + confidence_level) / 2 * 100
        lower_bound = np.percentile(final_returns, lower_percentile)
        upper_bound = np.percentile(final_returns, upper_percentile)

        # 计算VaR和CVaR
        var_95 = np.percentile(final_returns, 5)
        cvar_95 = np.mean(final_returns[final_returns <= var_95])

        # 创建图表
        plt.figure(figsize=figsize)

        # 绘制模拟路径
        ax1 = plt.subplot(2, 2, 1)

        # 绘制部分模拟路径
        paths = simulation_results['paths']
        for i in range(min(100, n_simulations)):  # 只绘制部分路径以避免图表过于拥挤
            ax1.plot(paths[:, i], 'b-', alpha=0.1)

        # 绘制均值路径
        mean_path = np.mean(paths, axis=1)
        ax1.plot(mean_path, 'r-', linewidth=2, label='均值路径')

        # 绘制分位数路径
        percentile_5 = np.percentile(paths, 5, axis=1)
        percentile_95 = np.percentile(paths, 95, axis=1)

        ax1.plot(percentile_5, 'g--', linewidth=1.5, label='5%分位数')
        ax1.plot(percentile_95, 'g--', linewidth=1.5, label='95%分位数')

        ax1.set_title('蒙特卡洛模拟路径', fontsize=12)
        ax1.set_xlabel('天数', fontsize=10)
        ax1.set_ylabel('累积收益率', fontsize=10)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 绘制最终收益率分布
        ax2 = plt.subplot(2, 2, 2)

        ax2.hist(final_returns, bins=30, density=True, alpha=0.7, color='b')

        # 绘制核密度估计
        from scipy import stats
        kde = stats.gaussian_kde(final_returns)
        x = np.linspace(final_returns.min(), final_returns.max(), 1000)
        ax2.plot(x, kde(x), 'r-', label='核密度估计')

        # 绘制置信区间
        ax2.axvline(x=lower_bound, color='g', linestyle='--', label=f'{lower_percentile:.1f}%分位数: {lower_bound:.2%}')
        ax2.axvline(x=upper_bound, color='g', linestyle='--', label=f'{upper_percentile:.1f}%分位数: {upper_bound:.2%}')

        # 绘制VaR和CVaR
        ax2.axvline(x=var_95, color='orange', linestyle='--', label=f'VaR (95%): {var_95:.2%}')
        ax2.axvline(x=cvar_95, color='purple', linestyle='--', label=f'CVaR (95%): {cvar_95:.2%}')

        ax2.set_title('最终收益率分布', fontsize=12)
        ax2.set_xlabel('累积收益率', fontsize=10)
        ax2.set_ylabel('密度', fontsize=10)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 绘制统计量
        ax3 = plt.subplot(2, 2, 3)

        metrics = [
            ('均值', mean_return * 100),
            ('中位数', median_return * 100),
            ('标准差', std_return * 100),
            ('VaR (95%)', -var_95 * 100),
            ('CVaR (95%)', -cvar_95 * 100)
        ]

        x = np.arange(len(metrics))
        values = [m[1] for m in metrics]

        bars = ax3.bar(x, values, color=['blue', 'green', 'red', 'orange', 'purple'])

        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3点垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom')

        ax3.set_title('蒙特卡洛模拟统计量', fontsize=12)
        ax3.set_ylabel('百分比 (%)', fontsize=10)
        ax3.set_xticks(x)
        ax3.set_xticklabels([m[0] for m in metrics])
        ax3.grid(True, alpha=0.3)

        # 绘制概率表格
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')

        # 计算不同收益率的概率
        prob_positive = (final_returns > 0).mean() * 100
        prob_negative = (final_returns <= 0).mean() * 100
        prob_above_10 = (final_returns > 0.1).mean() * 100
        prob_below_minus_10 = (final_returns < -0.1).mean() * 100

        # 创建表格数据
        table_data = [
            ['正收益概率', f"{prob_positive:.1f}%"],
            ['负收益概率', f"{prob_negative:.1f}%"],
            ['收益率>10%概率', f"{prob_above_10:.1f}%"],
            ['收益率<-10%概率', f"{prob_below_minus_10:.1f}%"],
            ['95%置信区间', f"{lower_bound:.2%} 至 {upper_bound:.2%}"],
            ['模拟方法', method.capitalize()]
        ]

        # 创建表格
        table = plt.table(
            cellText=table_data,
            colLabels=['指标', '数值'],
            loc='center',
            cellLoc='center',
            colWidths=[0.4, 0.4]
        )

        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        plt.tight_layout()
        plt.show()

        # 打印模拟结果
        print("蒙特卡洛模拟结果:")
        print(f"模拟方法: {method.capitalize()}")
        print(f"模拟次数: {n_simulations}")
        print(f"模拟天数: {n_days}")
        print(f"均值收益率: {mean_return:.2%}")
        print(f"中位数收益率: {median_return:.2%}")
        print(f"标准差: {std_return:.2%}")
        print(f"VaR (95%): {-var_95:.2%}")
        print(f"CVaR (95%): {-cvar_95:.2%}")
        print(f"95%置信区间: {lower_bound:.2%} 至 {upper_bound:.2%}")
        print(f"正收益概率: {prob_positive:.1f}%")
        print(f"负收益概率: {prob_negative:.1f}%")
        print(f"收益率>10%概率: {prob_above_10:.1f}%")
        print(f"收益率<-10%概率: {prob_below_minus_10:.1f}%")

    def optimize_portfolio(
        self, 
        optimization_method: str = 'mean_variance', 
        risk_free_rate: float = None,
        target_return: float = None,
        target_risk: float = None,
        constraints: Dict = None,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        优化投资组合

        参数:
        optimization_method (str): 优化方法，'mean_variance'（均值方差）, 'min_variance'（最小方差）, 
                                  'max_sharpe'（最大夏普比率）, 'risk_parity'（风险平价）
        risk_free_rate (float): 无风险利率，默认使用初始化时设置的值
        target_return (float): 目标收益率，用于均值方差优化
        target_risk (float): 目标风险，用于均值方差优化
        constraints (dict): 优化约束条件
        figsize (tuple): 图表大小

        返回:
        Dict: 包含优化结果的字典
        """
        # 导入投资组合优化模块
        from src.risk.portfolio_optimization import optimize_portfolio

        # 如果未提供无风险利率，使用初始化时设置的值
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        # 检查输入数据
        if not isinstance(self.returns, pd.DataFrame):
            raise ValueError("投资组合优化需要资产收益率DataFrame")

        # 设置默认约束条件
        if constraints is None:
            constraints = {
                'weight_sum': 1.0,  # 权重和为1
                'weight_bounds': (0.0, 1.0)  # 权重范围在0到1之间（不允许卖空）
            }

        # 运行投资组合优化
        optimization_results = optimize_portfolio(
            self.returns,
            method=optimization_method,
            risk_free_rate=risk_free_rate,
            target_return=target_return,
            target_risk=target_risk,
            constraints=constraints
        )

        # 提取优化结果
        optimal_weights = optimization_results['weights']
        expected_return = optimization_results['expected_return']
        expected_risk = optimization_results['expected_risk']
        sharpe_ratio = optimization_results['sharpe_ratio']

        # 创建图表
        plt.figure(figsize=figsize)

        # 绘制优化权重
        ax1 = plt.subplot(2, 2, 1)

        # 准备数据
        assets = self.returns.columns
        x = np.arange(len(assets))

        # 绘制条形图
        bars = ax1.bar(x, optimal_weights, color='blue', alpha=0.7)

        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.2%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3点垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom')

        ax1.set_title('优化投资组合权重', fontsize=12)
        ax1.set_ylabel('权重', fontsize=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(assets, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)

        # 如果是均值方差优化，绘制有效前沿
        if optimization_method in ['mean_variance', 'min_variance', 'max_sharpe']:
            # 导入有效前沿计算模块
            from src.risk.portfolio_optimization import calculate_efficient_frontier

            # 计算有效前沿
            frontier_results = calculate_efficient_frontier(
                self.returns,
                risk_free_rate=risk_free_rate,
                n_points=50,
                constraints=constraints
            )

            # 绘制有效前沿
            ax2 = plt.subplot(2, 2, 2)

            # 绘制有效前沿曲线
            ax2.plot(
                frontier_results['risks'] * 100, 
                frontier_results['returns'] * 100, 
                'b-', 
                linewidth=2,
                label='有效前沿'
            )

            # 绘制最小方差点
            min_var_idx = np.argmin(frontier_results['risks'])
            min_var_risk = frontier_results['risks'][min_var_idx] * 100
            min_var_return = frontier_results['returns'][min_var_idx] * 100

            ax2.scatter(
                min_var_risk, 
                min_var_return, 
                color='g', 
                marker='*', 
                s=100,
                label='最小方差'
            )

            # 绘制最大夏普比率点
            max_sharpe_idx = np.argmax(frontier_results['sharpe_ratios'])
            max_sharpe_risk = frontier_results['risks'][max_sharpe_idx] * 100
            max_sharpe_return = frontier_results['returns'][max_sharpe_idx] * 100

            ax2.scatter(
                max_sharpe_risk, 
                max_sharpe_return, 
                color='r', 
                marker='*', 
                s=100,
                label='最大夏普比率'
            )

            # 绘制当前优化点
            ax2.scatter(
                expected_risk * 100, 
                expected_return * 100, 
                color='purple', 
                marker='o', 
                s=100,
                label='当前优化点'
            )

            # 绘制资本市场线（如果有无风险利率）
            if risk_free_rate is not None:
                # 计算资本市场线
                max_sharpe_slope = (max_sharpe_return - risk_free_rate * 100) / max_sharpe_risk
                x_cml = np.linspace(0, max(frontier_results['risks']) * 100 * 1.2, 100)
                y_cml = risk_free_rate * 100 + max_sharpe_slope * x_cml

                ax2.plot(x_cml, y_cml, 'g--', label='资本市场线')

                # 标记无风险利率点
                ax2.scatter(0, risk_free_rate * 100, color='k', marker='o', label='无风险利率')

            # 绘制单个资产点
            for i, asset in enumerate(assets):
                asset_return = self.returns[asset].mean() * 252 * 100  # 年化
                asset_risk = self.returns[asset].std() * np.sqrt(252) * 100  # 年化

                ax2.scatter(
                    asset_risk, 
                    asset_return, 
                    color=f'C{i}', 
                    marker='o',
                    label=asset
                )

            ax2.set_title('有效前沿', fontsize=12)
            ax2.set_xlabel('年化波动率 (%)', fontsize=10)
            ax2.set_ylabel('年化收益率 (%)', fontsize=10)
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)

        # 绘制风险贡献
        ax3 = plt.subplot(2, 2, 3)

        # 计算协方差矩阵
        cov_matrix = self.returns.cov() * 252  # 年化

        # 计算投资组合方差
        portfolio_variance = optimal_weights.T @ cov_matrix @ optimal_weights

        # 计算边际风险贡献
        marginal_contribution = cov_matrix @ optimal_weights

        # 计算风险贡献
        risk_contribution = optimal_weights * marginal_contribution / np.sqrt(portfolio_variance)

        # 计算风险贡献百分比
        risk_contribution_pct = risk_contribution / risk_contribution.sum()

        # 绘制饼图
        ax3.pie(
            risk_contribution_pct, 
            labels=assets, 
            autopct='%1.1f%%',
            startangle=90, 
            shadow=False
        )

        ax3.set_title('风险贡献', fontsize=12)

        # 绘制优化结果表格
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')

        # 创建表格数据
        table_data = [
            ['优化方法', optimization_method.replace('_', ' ').title()],
            ['预期年化收益率', f"{expected_return:.2%}"],
            ['预期年化波动率', f"{expected_risk:.2%}"],
            ['夏普比率', f"{sharpe_ratio:.2f}"],
            ['无风险利率', f"{risk_free_rate:.2%}"]
        ]

        # 添加约束条件信息
        for key, value in constraints.items():
            if key == 'weight_bounds':
                table_data.append(['权重范围', f"{value[0]:.2f} 至 {value[1]:.2f}"])
            elif key == 'weight_sum':
                table_data.append(['权重和', f"{value:.2f}"])
            else:
                table_data.append([key, str(value)])

        # 创建表格
        table = plt.table(
            cellText=table_data,
            colLabels=['参数', '数值'],
            loc='center',
            cellLoc='center',
            colWidths=[0.4, 0.4]
        )

        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        plt.tight_layout()
        plt.show()

        # 打印优化结果
        print("投资组合优化结果:")
        print(f"优化方法: {optimization_method.replace('_', ' ').title()}")
        print(f"预期年化收益率: {expected_return:.2%}")
        print(f"预期年化波动率: {expected_risk:.2%}")
        print(f"夏普比率: {sharpe_ratio:.2f}")

        print("\n优化权重:")
        for i, asset in enumerate(assets):
            print(f"  {asset}: {optimal_weights[i]:.2%}")

        print("\n风险贡献:")
        for i, asset in enumerate(assets):
            print(f"  {asset}: {risk_contribution_pct[i]:.2%}")

        return optimization_results

    def analyze_factor_exposure(self, factors: pd.DataFrame, figsize: Tuple[int, int] = (12, 8)):
        """
        分析投资组合的因子暴露

        参数:
        factors (pd.DataFrame): 因子收益率数据框，索引为日期，列为因子名称
        figsize (tuple): 图表大小

        返回:
        Dict: 包含因子分析结果的字典
        """
        # 检查输入数据
        if not isinstance(factors, pd.DataFrame):
            raise ValueError("因子分析需要因子收益率DataFrame")

        # 确保因子数据和投资组合收益率有相同的索引
        common_index = factors.index.intersection(self.portfolio_returns.index)
        if len(common_index) == 0:
            raise ValueError("因子数据和投资组合收益率没有共同的日期")

        # 对齐数据
        aligned_factors = factors.loc[common_index]
        aligned_returns = self.portfolio_returns.loc[common_index]

        # 添加常数项（Alpha）
        X = sm.add_constant(aligned_factors)

        # 运行因子回归
        model = sm.OLS(aligned_returns, X)
        results = model.fit()

        # 提取因子暴露（系数）
        factor_exposures = results.params
        factor_tvalues = results.tvalues
        factor_pvalues = results.pvalues
        r_squared = results.rsquared
        adj_r_squared = results.rsquared_adj

        # 计算因子贡献
        factor_contributions = pd.DataFrame(index=aligned_factors.index)

        # 计算每个因子的贡献
        for factor in aligned_factors.columns:
            factor_contributions[factor] = aligned_factors[factor] * factor_exposures[factor]

        # 添加Alpha贡献
        factor_contributions['Alpha'] = factor_exposures['const']

        # 计算累积因子贡献
        cumulative_contributions = factor_contributions.cumsum()

        # 创建图表
        plt.figure(figsize=figsize)

        # 绘制因子暴露
        ax1 = plt.subplot(2, 2, 1)

        # 准备数据
        factors_list = aligned_factors.columns
        x = np.arange(len(factors_list) + 1)  # +1 for Alpha
        exposures = [factor_exposures['const']] + [factor_exposures[f] for f in factors_list]

        # 设置颜色
        colors = ['red' if e < 0 else 'green' for e in exposures]

        # 绘制条形图
        bars = ax1.bar(x, exposures, color=colors, alpha=0.7)

        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -15),  # 根据高度调整垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top')

        ax1.set_title('因子暴露（系数）', fontsize=12)
        ax1.set_ylabel('暴露', fontsize=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Alpha'] + list(factors_list), rotation=45, ha='right')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.grid(True, alpha=0.3)

        # 绘制因子贡献
        ax2 = plt.subplot(2, 2, 2)

        # 计算每个因子的平均贡献
        mean_contributions = factor_contributions.mean()

        # 按贡献绝对值排序
        sorted_contributions = mean_contributions.abs().sort_values(ascending=False)
        sorted_factors = sorted_contributions.index

        # 设置颜色
        colors = ['red' if mean_contributions[f] < 0 else 'green' for f in sorted_factors]

        # 绘制条形图
        bars = ax2.bar(np.arange(len(sorted_factors)), [mean_contributions[f] for f in sorted_factors], color=colors, alpha=0.7)

        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.6f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -15),  # 根据高度调整垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top')

        ax2.set_title('平均因子贡献', fontsize=12)
        ax2.set_ylabel('贡献', fontsize=10)
        ax2.set_xticks(np.arange(len(sorted_factors)))
        ax2.set_xticklabels(sorted_factors, rotation=45, ha='right')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.grid(True, alpha=0.3)

        # 绘制累积因子贡献
        ax3 = plt.subplot(2, 2, 3)

        # 绘制堆叠面积图
        ax3.stackplot(
            cumulative_contributions.index,
            [cumulative_contributions[factor] for factor in factor_contributions.columns],
            labels=factor_contributions.columns,
            alpha=0.7
        )

        # 绘制总的累积收益
        total_cumulative = cumulative_contributions.sum(axis=1)
        ax3.plot(total_cumulative.index, total_cumulative, 'k-', linewidth=2, label='总累积贡献')

        ax3.set_title('累积因子贡献', fontsize=12)
        ax3.set_xlabel('日期', fontsize=10)
        ax3.set_ylabel('累积贡献', fontsize=10)
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)

        # 绘制回归统计量
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')

        # 创建表格数据
        table_data = [
            ['R²', f"{r_squared:.4f}"],
            ['调整R²', f"{adj_r_squared:.4f}"],
            ['Alpha', f"{factor_exposures['const']:.6f}"],
            ['Alpha t值', f"{factor_tvalues['const']:.4f}"],
            ['Alpha p值', f"{factor_pvalues['const']:.4f}"],
            ['显著因子数', f"{sum(factor_pvalues[1:] < 0.05)}/{len(factors_list)}"]
        ]

        # 创建表格
        table = plt.table(
            cellText=table_data,
            colLabels=['统计量', '数值'],
            loc='center',
            cellLoc='center',
            colWidths=[0.4, 0.4]
        )

        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        plt.tight_layout()
        plt.show()

        # 打印因子分析结果
        print("因子分析结果:")
        print(f"R²: {r_squared:.4f}")
        print(f"调整R²: {adj_r_squared:.4f}")
        print(f"Alpha: {factor_exposures['const']:.6f} (t值: {factor_tvalues['const']:.4f}, p值: {factor_pvalues['const']:.4f})")

        print("\n因子暴露:")
        for factor in factors_list:
            significance = "***" if factor_pvalues[factor] < 0.01 else "**" if factor_pvalues[factor] < 0.05 else "*" if factor_pvalues[factor] < 0.1 else ""
            print(f"  {factor}: {factor_exposures[factor]:.6f} (t值: {factor_tvalues[factor]:.4f}, p值: {factor_pvalues[factor]:.4f}) {significance}")

        print("\n平均因子贡献:")
        for factor in sorted_factors:
            if factor != 'Alpha':  # Alpha已经单独显示
                print(f"  {factor}: {mean_contributions[factor]:.6f}")

        # 返回分析结果
        return {
            'exposures': factor_exposures,
            'tvalues': factor_tvalues,
            'pvalues': factor_pvalues,
            'r_squared': r_squared,
            'adj_r_squared': adj_r_squared,
            'contributions': factor_contributions,
            'mean_contributions': mean_contributions,
            'cumulative_contributions': cumulative_contributions
        }

    def analyze_risk_concentration(self, figsize: Tuple[int, int] = (12, 8)):
        """
        分析投资组合的风险集中度

        参数:
        figsize (tuple): 图表大小

        返回:
        Dict: 包含风险集中度分析结果的字典
        """
        # 检查输入数据
        if not isinstance(self.returns, pd.DataFrame):
            raise ValueError("风险集中度分析需要资产收益率DataFrame")

        # 计算协方差矩阵
        cov_matrix = self.returns.cov() * 252  # 年化

        # 计算相关系数矩阵
        corr_matrix = self.returns.corr()

        # 计算资产波动率
        asset_volatilities = self.returns.std() * np.sqrt(252)  # 年化

        # 如果有投资组合权重，计算风险贡献
        if hasattr(self, 'weights') and self.weights is not None:
            weights = self.weights

            # 确保权重和资产收益率有相同的列
            if not all(asset in self.returns.columns for asset in weights.index):
                raise ValueError("权重和资产收益率的资产不匹配")

            # 对齐权重和资产收益率
            aligned_weights = pd.Series(0, index=self.returns.columns)
            for asset in weights.index:
                if asset in self.returns.columns:
                    aligned_weights[asset] = weights[asset]

            # 计算投资组合方差
            portfolio_variance = aligned_weights.T @ cov_matrix @ aligned_weights

            # 计算边际风险贡献
            marginal_contribution = cov_matrix @ aligned_weights

            # 计算风险贡献
            risk_contribution = aligned_weights * marginal_contribution / np.sqrt(portfolio_variance)

            # 计算风险贡献百分比
            risk_contribution_pct = risk_contribution / risk_contribution.sum()

            # 计算风险集中度指标
            hhi_weights = (aligned_weights ** 2).sum()  # 权重的赫芬达尔-赫希曼指数
            hhi_risk = (risk_contribution_pct ** 2).sum()  # 风险贡献的赫芬达尔-赫希曼指数

            # 计算有效资产数量
            effective_assets_weights = 1 / hhi_weights
            effective_assets_risk = 1 / hhi_risk
        else:
            # 如果没有权重，只分析资产间的相关性和波动率
            aligned_weights = None
            risk_contribution = None
            risk_contribution_pct = None
            hhi_weights = None
            hhi_risk = None
            effective_assets_weights = None
            effective_assets_risk = None

        # 创建图表
        plt.figure(figsize=figsize)

        # 绘制相关系数热图
        ax1 = plt.subplot(2, 2, 1)
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            vmin=-1, 
            vmax=1, 
            center=0,
            linewidths=0.5,
            fmt='.2f',
            ax=ax1
        )
        ax1.set_title('资产相关系数矩阵', fontsize=12)

        # 绘制资产波动率
        ax2 = plt.subplot(2, 2, 2)

        # 按波动率排序
        sorted_volatilities = asset_volatilities.sort_values(ascending=False)

        # 绘制条形图
        bars = ax2.bar(np.arange(len(sorted_volatilities)), sorted_volatilities * 100, color='blue', alpha=0.7)

        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3点垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom')

        ax2.set_title('资产年化波动率', fontsize=12)
        ax2.set_ylabel('波动率 (%)', fontsize=10)
        ax2.set_xticks(np.arange(len(sorted_volatilities)))
        ax2.set_xticklabels(sorted_volatilities.index, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)

        # 如果有权重，绘制风险贡献
        if aligned_weights is not None:
            # 绘制风险贡献
            ax3 = plt.subplot(2, 2, 3)

            # 按风险贡献排序
            sorted_risk_contribution = risk_contribution_pct.sort_values(ascending=False)

            # 绘制条形图
            bars = ax3.bar(np.arange(len(sorted_risk_contribution)), sorted_risk_contribution, color='blue', alpha=0.7)

            # 添加数据标签
            for bar in bars:
                height = bar.get_height()
                ax3.annotate(f'{height:.2%}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3点垂直偏移
                            textcoords="offset points",
                            ha='center', va='bottom')

            ax3.set_title('风险贡献', fontsize=12)
            ax3.set_ylabel('风险贡献', fontsize=10)
            ax3.set_xticks(np.arange(len(sorted_risk_contribution)))
            ax3.set_xticklabels(sorted_risk_contribution.index, rotation=45, ha='right')
            ax3.grid(True, alpha=0.3)

            # 绘制风险集中度指标
            ax4 = plt.subplot(2, 2, 4)
            ax4.axis('off')

            # 创建表格数据
            table_data = [
                ['权重赫芬达尔指数', f"{hhi_weights:.4f}"],
                ['风险贡献赫芬达尔指数', f"{hhi_risk:.4f}"],
                ['有效资产数量 (权重)', f"{effective_assets_weights:.2f}"],
                ['有效资产数量 (风险)', f"{effective_assets_risk:.2f}"],
                ['最大权重', f"{aligned_weights.max():.2%}"],
                ['最小权重', f"{aligned_weights.min():.2%}"],
                ['最大风险贡献', f"{risk_contribution_pct.max():.2%}"],
                ['最小风险贡献', f"{risk_contribution_pct.min():.2%}"]
            ]
        else:
            # 如果没有权重，绘制相关性网络图
            ax3 = plt.subplot(2, 2, 3)

            # 计算相关性网络
            # 只保留绝对值大于0.3的相关系数
            threshold = 0.3
            filtered_corr = corr_matrix.copy()
            filtered_corr[np.abs(filtered_corr) < threshold] = 0

            # 创建网络图
            G = nx.from_pandas_adjacency(filtered_corr)

            # 设置节点大小（基于波动率）
            node_sizes = asset_volatilities * 5000

            # 设置边的宽度（基于相关系数的绝对值）
            edge_weights = [abs(corr_matrix.loc[u, v]) * 3 for u, v in G.edges()]

            # 设置边的颜色（正相关为绿色，负相关为红色）
            edge_colors = ['green' if corr_matrix.loc[u, v] > 0 else 'red' for u, v in G.edges()]

            # 绘制网络图
            pos = nx.spring_layout(G, seed=42)  # 使用弹簧布局
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)
            nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color=edge_colors, alpha=0.6)
            nx.draw_networkx_labels(G, pos, font_size=8)

            ax3.set_title('资产相关性网络', fontsize=12)

            # 绘制统计量表格
            ax4 = plt.subplot(2, 2, 4)
            ax4.axis('off')

            # 计算平均相关系数
            n = corr_matrix.shape[0]
            total_corr = corr_matrix.sum().sum()
            avg_corr = (total_corr - n) / (n * (n - 1))  # 减去对角线的1

            # 计算相关系数的标准差
            corr_std = np.sqrt(((corr_matrix - np.eye(n)).values.flatten() ** 2).sum() / (n * (n - 1)))

            # 创建表格数据
            table_data = [
                ['资产数量', f"{n}"],
                ['平均相关系数', f"{avg_corr:.4f}"],
                ['相关系数标准差', f"{corr_std:.4f}"],
                ['最大相关系数', f"{corr_matrix.values[~np.eye(n, dtype=bool)].max():.4f}"],
                ['最小相关系数', f"{corr_matrix.values[~np.eye(n, dtype=bool)].min():.4f}"],
                ['正相关比例', f"{(corr_matrix.values[~np.eye(n, dtype=bool)] > 0).mean():.2%}"],
                ['负相关比例', f"{(corr_matrix.values[~np.eye(n, dtype=bool)] < 0).mean():.2%}"],
                ['平均波动率', f"{asset_volatilities.mean():.2%}"]
            ]

        # 创建表格
        table = plt.table(
            cellText=table_data,
            colLabels=['指标', '数值'],
            loc='center',
            cellLoc='center',
            colWidths=[0.4, 0.4]
        )

        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        plt.tight_layout()
        plt.show()

        # 打印风险集中度分析结果
        print("风险集中度分析结果:")

        if aligned_weights is not None:
            print("\n权重分析:")
            print(f"权重赫芬达尔指数: {hhi_weights:.4f}")
            print(f"有效资产数量 (权重): {effective_assets_weights:.2f}")
            print(f"最大权重: {aligned_weights.max():.2%} ({aligned_weights.idxmax()})")
            print(f"最小权重: {aligned_weights.min():.2%} ({aligned_weights.idxmin()})")

            print("\n风险贡献分析:")
            print(f"风险贡献赫芬达尔指数: {hhi_risk:.4f}")
            print(f"有效资产数量 (风险): {effective_assets_risk:.2f}")
            print(f"最大风险贡献: {risk_contribution_pct.max():.2%} ({risk_contribution_pct.idxmax()})")
            print(f"最小风险贡献: {risk_contribution_pct.min():.2%} ({risk_contribution_pct.idxmin()})")

            # 计算风险集中度比率
            risk_concentration_ratio = hhi_risk / hhi_weights
            print(f"\n风险集中度比率: {risk_concentration_ratio:.4f}")
            if risk_concentration_ratio > 1.2:
                print("风险集中度高于权重集中度，表明投资组合风险过度集中于少数资产")
            elif risk_concentration_ratio < 0.8:
                print("风险集中度低于权重集中度，表明投资组合风险分散良好")
            else:
                print("风险集中度与权重集中度基本匹配")

        print("\n相关性分析:")
        print(f"资产数量: {n}")
        print(f"平均相关系数: {avg_corr:.4f}")
        print(f"相关系数标准差: {corr_std:.4f}")
        print(f"最大相关系数: {corr_matrix.values[~np.eye(n, dtype=bool)].max():.4f}")
        print(f"最小相关系数: {corr_matrix.values[~np.eye(n, dtype=bool)].min():.4f}")
        print(f"正相关比例: {(corr_matrix.values[~np.eye(n, dtype=bool)] > 0).mean():.2%}")
        print(f"负相关比例: {(corr_matrix.values[~np.eye(n, dtype=bool)] < 0).mean():.2%}")

        # 返回分析结果
        result = {
            'correlation_matrix': corr_matrix,
            'volatilities': asset_volatilities,
            'avg_correlation': avg_corr,
            'correlation_std': corr_std
        }

        if aligned_weights is not None:
            result.update({
                'weights': aligned_weights,
                'risk_contribution': risk_contribution,
                'risk_contribution_pct': risk_contribution_pct,
                'hhi_weights': hhi_weights,
                'hhi_risk': hhi_risk,
                'effective_assets_weights': effective_assets_weights,
                'effective_assets_risk': effective_assets_risk,
                'risk_concentration_ratio': risk_concentration_ratio
            })

        return result

    def analyze_factor_exposure(self, factors: pd.DataFrame, figsize: Tuple[int, int] = (12, 8)):
        """
        分析投资组合的因子暴露

        参数:
        factors (pd.DataFrame): 因子收益率数据框，索引为日期，列为因子名称
        figsize (tuple): 图表大小

        返回:
        Dict: 包含因子分析结果的字典
        """
        # 检查输入数据
        if not isinstance(factors, pd.DataFrame):
            raise ValueError("因子分析需要因子收益率DataFrame")

        # 确保因子数据和投资组合收益率有相同的索引
        common_index = factors.index.intersection(self.portfolio_returns.index)
        if len(common_index) == 0:
            raise ValueError("因子数据和投资组合收益率没有共同的日期")

        # 对齐数据
        aligned_factors = factors.loc[common_index]
        aligned_returns = self.portfolio_returns.loc[common_index]

        # 添加常数项（Alpha）
        X = sm.add_constant(aligned_factors)

        # 运行因子回归
        model = sm.OLS(aligned_returns, X)
        results = model.fit()

        # 提取因子暴露（系数）
        factor_exposures = results.params
        factor_tvalues = results.tvalues
        factor_pvalues = results.pvalues
        r_squared = results.rsquared
        adj_r_squared = results.rsquared_adj

        # 计算因子贡献
        factor_contributions = pd.DataFrame(index=aligned_factors.index)

        # 计算每个因子的贡献
        for factor in aligned_factors.columns:
            factor_contributions[factor] = aligned_factors[factor] * factor_exposures[factor]

        # 添加Alpha贡献
        factor_contributions['Alpha'] = factor_exposures['const']

        # 计算累积因子贡献
        cumulative_contributions = factor_contributions.cumsum()

        # 创建图表
        plt.figure(figsize=figsize)

        # 绘制因子暴露
        ax1 = plt.subplot(2, 2, 1)

        # 准备数据
        factors_list = aligned_factors.columns
        x = np.arange(len(factors_list) + 1)  # +1 for Alpha
        exposures = [factor_exposures['const']] + [factor_exposures[f] for f in factors_list]

        # 设置颜色
        colors = ['red' if e < 0 else 'green' for e in exposures]

        # 绘制条形图
        bars = ax1.bar(x, exposures, color=colors, alpha=0.7)

        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -15),  # 根据高度调整垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top')

        ax1.set_title('因子暴露（系数）', fontsize=12)
        ax1.set_ylabel('暴露', fontsize=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Alpha'] + list(factors_list), rotation=45, ha='right')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.grid(True, alpha=0.3)

        # 绘制因子贡献
        ax2 = plt.subplot(2, 2, 2)

        # 计算每个因子的平均贡献
        mean_contributions = factor_contributions.mean()

        # 按贡献绝对值排序
        sorted_contributions = mean_contributions.abs().sort_values(ascending=False)
        sorted_factors = sorted_contributions.index

        # 设置颜色
        colors = ['red' if mean_contributions[f] < 0 else 'green' for f in sorted_factors]

        # 绘制条形图
        bars = ax2.bar(np.arange(len(sorted_factors)), [mean_contributions[f] for f in sorted_factors], color=colors, alpha=0.7)

        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.6f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -15),  # 根据高度调整垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top')

        ax2.set_title('平均因子贡献', fontsize=12)
        ax2.set_ylabel('贡献', fontsize=10)
        ax2.set_xticks(np.arange(len(sorted_factors)))
        ax2.set_xticklabels(sorted_factors, rotation=45, ha='right')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.grid(True, alpha=0.3)

        # 绘制累积因子贡献
        ax3 = plt.subplot(2, 2, 3)

        # 绘制堆叠面积图
        ax3.stackplot(
            cumulative_contributions.index,
            [cumulative_contributions[factor] for factor in factor_contributions.columns],
            labels=factor_contributions.columns,
            alpha=0.7
        )

        # 绘制总的累积收益
        total_cumulative = cumulative_contributions.sum(axis=1)
        ax3.plot(total_cumulative.index, total_cumulative, 'k-', linewidth=2, label='总累积贡献')

        ax3.set_title('累积因子贡献', fontsize=12)
        ax3.set_xlabel('日期', fontsize=10)
        ax3.set_ylabel('累积贡献', fontsize=10)
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)

        # 绘制回归统计量
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')

        # 创建表格数据
        table_data = [
            ['R²', f"{r_squared:.4f}"],
            ['调整R²', f"{adj_r_squared:.4f}"],
            ['Alpha', f"{factor_exposures['const']:.6f}"],
            ['Alpha t值', f"{factor_tvalues['const']:.4f}"],
            ['Alpha p值', f"{factor_pvalues['const']:.4f}"],
            ['显著因子数', f"{sum(factor_pvalues[1:] < 0.05)}/{len(factors_list)}"]
        ]

        # 创建表格
        table = plt.table(
            cellText=table_data,
            colLabels=['统计量', '数值'],
            loc='center',
            cellLoc='center',
            colWidths=[0.4, 0.4]
        )

        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        plt.tight_layout()
        plt.show()

        # 打印因子分析结果
        print("因子分析结果:")
        print(f"R²: {r_squared:.4f}")
        print(f"调整R²: {adj_r_squared:.4f}")
        print(f"Alpha: {factor_exposures['const']:.6f} (t值: {factor_tvalues['const']:.4f}, p值: {factor_pvalues['const']:.4f})")

        print("\n因子暴露:")
        for factor in factors_list:
            significance = "***" if factor_pvalues[factor] < 0.01 else "**" if factor_pvalues[factor] < 0.05 else "*" if factor_pvalues[factor] < 0.1 else ""
            print(f"  {factor}: {factor_exposures[factor]:.6f} (t值: {factor_tvalues[factor]:.4f}, p值: {factor_pvalues[factor]:.4f}) {significance}")

        print("\n平均因子贡献:")
        for factor in sorted_factors:
            if factor != 'Alpha':  # Alpha已经单独显示
                print(f"  {factor}: {mean_contributions[factor]:.6f}")

        # 返回分析结果
        return {
            'exposures': factor_exposures,
            'tvalues': factor_tvalues,
            'pvalues': factor_pvalues,
            'r_squared': r_squared,
            'adj_r_squared': adj_r_squared,
            'contributions': factor_contributions,
            'mean_contributions': mean_contributions,
            'cumulative_contributions': cumulative_contributions
        }

    def analyze_tail_risk(self, confidence_levels: List[float] = [0.95, 0.99], 
                         window: int = None, 
                         method: str = 'historical',
                         figsize: Tuple[int, int] = (12, 8)):
        """
        分析投资组合的尾部风险

        参数:
        confidence_levels (list): VaR和CVaR的置信水平列表
        window (int): 滚动窗口大小，如果为None则使用全部历史数据
        method (str): 计算方法，'historical'（历史模拟法）, 'parametric'（参数法）, 'cornish_fisher'（Cornish-Fisher展开）
        figsize (tuple): 图表大小

        返回:
        Dict: 包含尾部风险分析结果的字典
        """
        # 检查输入数据
        if not isinstance(self.portfolio_returns, pd.Series):
            raise ValueError("尾部风险分析需要投资组合收益率Series")

        # 计算静态VaR和CVaR
        static_var = {}
        static_cvar = {}

        for cl in confidence_levels:
            # 计算VaR
            if method == 'historical':
                # 历史模拟法
                var = -np.percentile(self.portfolio_returns, 100 * (1 - cl))
            elif method == 'parametric':
                # 参数法（假设正态分布）
                z_score = stats.norm.ppf(cl)
                var = -(self.portfolio_returns.mean() - z_score * self.portfolio_returns.std())
            elif method == 'cornish_fisher':
                # Cornish-Fisher展开（考虑偏度和峰度）
                z_score = stats.norm.ppf(cl)
                skew = stats.skew(self.portfolio_returns)
                kurt = stats.kurtosis(self.portfolio_returns)

                # Cornish-Fisher展开
                z_cf = z_score + (z_score**2 - 1) * skew / 6 + (z_score**3 - 3*z_score) * kurt / 24 - (2*z_score**3 - 5*z_score) * skew**2 / 36

                var = -(self.portfolio_returns.mean() - z_cf * self.portfolio_returns.std())
            else:
                raise ValueError(f"不支持的方法: {method}")

            # 计算CVaR
            cvar = -self.portfolio_returns[self.portfolio_returns <= -var].mean()

            static_var[cl] = var
            static_cvar[cl] = cvar

        # 计算滚动VaR和CVaR
        if window is not None and window < len(self.portfolio_returns):
            rolling_var = {}
            rolling_cvar = {}

            for cl in confidence_levels:
                var_series = pd.Series(index=self.portfolio_returns.index)
                cvar_series = pd.Series(index=self.portfolio_returns.index)

                for i in range(window, len(self.portfolio_returns)):
                    window_returns = self.portfolio_returns.iloc[i-window:i]

                    # 计算VaR
                    if method == 'historical':
                        var = -np.percentile(window_returns, 100 * (1 - cl))
                    elif method == 'parametric':
                        z_score = stats.norm.ppf(cl)
                        var = -(window_returns.mean() - z_score * window_returns.std())
                    elif method == 'cornish_fisher':
                        z_score = stats.norm.ppf(cl)
                        skew = stats.skew(window_returns)
                        kurt = stats.kurtosis(window_returns)

                        # Cornish-Fisher展开
                        z_cf = z_score + (z_score**2 - 1) * skew / 6 + (z_score**3 - 3*z_score) * kurt / 24 - (2*z_score**3 - 5*z_score) * skew**2 / 36

                        var = -(window_returns.mean() - z_cf * window_returns.std())

                    # 计算CVaR
                    cvar = -window_returns[window_returns <= -var].mean()

                    var_series.iloc[i] = var
                    cvar_series.iloc[i] = cvar

                rolling_var[cl] = var_series.dropna()
                rolling_cvar[cl] = cvar_series.dropna()
        else:
            rolling_var = None
            rolling_cvar = None

        # 创建图表
        plt.figure(figsize=figsize)

        # 绘制收益率分布和VaR/CVaR
        ax1 = plt.subplot(2, 2, 1)

        # 绘制收益率直方图
        ax1.hist(self.portfolio_returns, bins=30, density=True, alpha=0.7, color='blue')

        # 绘制核密度估计
        kde = stats.gaussian_kde(self.portfolio_returns)
        x = np.linspace(self.portfolio_returns.min(), self.portfolio_returns.max(), 1000)
        ax1.plot(x, kde(x), 'r-', label='核密度估计')

        # 绘制VaR和CVaR线
        colors = ['green', 'orange', 'purple', 'brown']
        for i, cl in enumerate(confidence_levels):
            var = static_var[cl]
            cvar = static_cvar[cl]

            ax1.axvline(x=-var, color=colors[i % len(colors)], linestyle='--', 
                       label=f'VaR ({cl*100:.0f}%): {var:.2%}')
            ax1.axvline(x=-cvar, color=colors[i % len(colors)], linestyle=':',
                       label=f'CVaR ({cl*100:.0f}%): {cvar:.2%}')

        ax1.set_title('收益率分布与尾部风险', fontsize=12)
        ax1.set_xlabel('收益率', fontsize=10)
        ax1.set_ylabel('密度', fontsize=10)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 绘制QQ图（检验正态性）
        ax2 = plt.subplot(2, 2, 2)

        # 计算理论分位数和样本分位数
        from scipy import stats
        (quantiles, values), (slope, intercept, r) = stats.probplot(self.portfolio_returns, dist='norm')

        # 绘制QQ图
        ax2.scatter(quantiles, values, color='blue', alpha=0.7)
        ax2.plot(quantiles, intercept + slope * quantiles, 'r-', linewidth=2)

        # 计算正态性检验
        k2, p_value = stats.normaltest(self.portfolio_returns)

        ax2.set_title(f'正态QQ图 (p值: {p_value:.4f})', fontsize=12)
        ax2.set_xlabel('理论分位数', fontsize=10)
        ax2.set_ylabel('样本分位数', fontsize=10)
        ax2.grid(True, alpha=0.3)

        # 如果有滚动VaR和CVaR，绘制时间序列
        if rolling_var is not None:
            ax3 = plt.subplot(2, 2, 3)

            # 绘制滚动VaR
            for i, cl in enumerate(confidence_levels):
                ax3.plot(rolling_var[cl].index, rolling_var[cl] * 100, 
                        color=colors[i % len(colors)], linestyle='-',
                        label=f'VaR ({cl*100:.0f}%)')

            ax3.set_title('滚动VaR', fontsize=12)
            ax3.set_xlabel('日期', fontsize=10)
            ax3.set_ylabel('VaR (%)', fontsize=10)
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3)

            ax4 = plt.subplot(2, 2, 4)

            # 绘制滚动CVaR
            for i, cl in enumerate(confidence_levels):
                ax4.plot(rolling_cvar[cl].index, rolling_cvar[cl] * 100, 
                        color=colors[i % len(colors)], linestyle='-',
                        label=f'CVaR ({cl*100:.0f}%)')

            ax4.set_title('滚动CVaR', fontsize=12)
            ax4.set_xlabel('日期', fontsize=10)
            ax4.set_ylabel('CVaR (%)', fontsize=10)
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3)
        else:
            # 绘制尾部风险统计量
            ax3 = plt.subplot(2, 1, 2)
            ax3.axis('off')

            # 计算额外的尾部风险统计量
            mean_return = self.portfolio_returns.mean()
            std_return = self.portfolio_returns.std()
            skewness = stats.skew(self.portfolio_returns)
            kurtosis = stats.kurtosis(self.portfolio_returns)

            # 计算下行风险
            downside_returns = self.portfolio_returns[self.portfolio_returns < 0]
            downside_risk = np.sqrt((downside_returns**2).mean())

            # 计算最大回撤
            cumulative_returns = (1 + self.portfolio_returns).cumprod()
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns / running_max - 1)
            max_drawdown = drawdown.min()

            # 创建表格数据
            table_data = [
                ['平均收益率', f"{mean_return:.4%}"],
                ['标准差', f"{std_return:.4%}"],
                ['偏度', f"{skewness:.4f}"],
                ['峰度', f"{kurtosis:.4f}"],
                ['下行风险', f"{downside_risk:.4%}"],
                ['最大回撤', f"{max_drawdown:.2%}"]
            ]

            # 添加VaR和CVaR
            for cl in confidence_levels:
                table_data.append([f"VaR ({cl*100:.0f}%)", f"{static_var[cl]:.4%}"])
                table_data.append([f"CVaR ({cl*100:.0f}%)", f"{static_cvar[cl]:.4%}"])

            # 添加计算方法
            table_data.append(['计算方法', method.replace('_', ' ').title()])

            # 创建表格
            table = plt.table(
                cellText=table_data,
                colLabels=['统计量', '数值'],
                loc='center',
                cellLoc='center',
                colWidths=[0.4, 0.4]
            )

            # 设置表格样式
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)

        plt.tight_layout()
        plt.show()

        # 打印尾部风险分析结果
        print("尾部风险分析结果:")
        print(f"计算方法: {method.replace('_', ' ').title()}")

        print("\n收益率统计:")
        print(f"平均收益率: {self.portfolio_returns.mean():.4%}")
        print(f"标准差: {self.portfolio_returns.std():.4%}")
        print(f"偏度: {stats.skew(self.portfolio_returns):.4f}")
        print(f"峰度: {stats.kurtosis(self.portfolio_returns):.4f}")
        print(f"下行风险: {downside_risk:.4%}")
        print(f"最大回撤: {max_drawdown:.2%}")

        print("\n尾部风险指标:")
        for cl in confidence_levels:
            print(f"VaR ({cl*100:.0f}%): {static_var[cl]:.4%}")
            print(f"CVaR ({cl*100:.0f}%): {static_cvar[cl]:.4%}")

        # 正态性检验结果
        print(f"\n正态性检验 (D'Agostino-Pearson):")
        print(f"统计量: {k2:.4f}")
        print(f"p值: {p_value:.4f}")
        if p_value < 0.05:
            print("结论: 收益率分布显著偏离正态分布")
        else:
            print("结论: 收益率分布接近正态分布")

        # 返回分析结果
        result = {
            'method': method,
            'static_var': static_var,
            'static_cvar': static_cvar,
            'mean_return': mean_return,
            'std_return': std_return,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'downside_risk': downside_risk,
            'max_drawdown': max_drawdown,
            'normality_test': {
                'statistic': k2,
                'p_value': p_value
            }
        }

        if rolling_var is not None:
            result.update({
                'rolling_var': rolling_var,
                'rolling_cvar': rolling_cvar
            })

        return result

    def stress_test(self, scenarios: Dict[str, Dict[str, float]] = None, 
                   historical_events: List[str] = None,
                   monte_carlo_sims: int = None,
                   figsize: Tuple[int, int] = (12, 8)):
        """
        对投资组合进行压力测试

        参数:
        scenarios (dict): 自定义情景，格式为 {'情景名称': {'资产1': 变化率1, '资产2': 变化率2, ...}}
        historical_events (list): 历史事件列表，用于模拟历史情景
        monte_carlo_sims (int): 蒙特卡洛模拟次数，如果为None则不进行模拟
        figsize (tuple): 图表大小

        返回:
        Dict: 包含压力测试结果的字典
        """
        # 检查输入数据
        if not hasattr(self, 'weights') or self.weights is None:
            raise ValueError("压力测试需要投资组合权重")

        # 确保权重和资产收益率有相同的列
        if not all(asset in self.returns.columns for asset in self.weights.index):
            raise ValueError("权重和资产收益率的资产不匹配")

        # 对齐权重和资产收益率
        aligned_weights = pd.Series(0, index=self.returns.columns)
        for asset in self.weights.index:
            if asset in self.returns.columns:
                aligned_weights[asset] = self.weights[asset]

        # 初始化结果
        stress_results = {}

        # 处理自定义情景
        if scenarios is not None:
            custom_results = {}

            for scenario_name, scenario_changes in scenarios.items():
                # 确保所有资产都在情景中
                for asset in aligned_weights.index:
                    if asset not in scenario_changes:
                        scenario_changes[asset] = 0.0

                # 计算情景下的投资组合收益率
                scenario_return = sum(aligned_weights[asset] * scenario_changes[asset] for asset in aligned_weights.index)

                custom_results[scenario_name] = scenario_return

            stress_results['custom_scenarios'] = custom_results

        # 处理历史事件
        if historical_events is not None:
            historical_results = {}

            # 定义历史事件的时间范围
            historical_periods = {
                '2008金融危机': ('2008-09-01', '2009-03-31'),
                '2020新冠疫情': ('2020-02-15', '2020-04-15'),
                '2000科技泡沫': ('2000-03-01', '2000-12-31'),
                '2015中国股灾': ('2015-06-01', '2015-08-31'),
                '2011欧债危机': ('2011-07-01', '2011-12-31'),
                '1997亚洲金融危机': ('1997-07-01', '1997-12-31'),
                '2013美联储缩减恐慌': ('2013-05-01', '2013-09-30'),
                '2018年末抛售': ('2018-10-01', '2018-12-31'),
                '2016英国脱欧': ('2016-06-23', '2016-07-15'),
                '2022俄乌冲突': ('2022-02-24', '2022-03-15')
            }

            for event in historical_events:
                if event in historical_periods:
                    start_date, end_date = historical_periods[event]

                    # 检查数据是否覆盖该时间段
                    if start_date in self.returns.index and end_date in self.returns.index:
                        # 提取该时间段的收益率
                        event_returns = self.returns.loc[start_date:end_date]

                        # 计算累积收益率
                        cumulative_returns = (1 + event_returns).prod() - 1

                        # 计算投资组合在该事件下的累积收益率
                        portfolio_event_return = sum(aligned_weights[asset] * cumulative_returns[asset] for asset in aligned_weights.index)

                        historical_results[event] = portfolio_event_return
                    else:
                        print(f"警告: 数据不覆盖 {event} 的时间段 ({start_date} 至 {end_date})")
                else:
                    print(f"警告: 未定义历史事件 '{event}'")

            stress_results['historical_events'] = historical_results

        # 处理蒙特卡洛模拟
        if monte_carlo_sims is not None and monte_carlo_sims > 0:
            # 计算收益率的均值和协方差
            mean_returns = self.returns.mean()
            cov_matrix = self.returns.cov()

            # 生成多元正态分布的随机收益率
            np.random.seed(42)  # 设置随机种子以确保可重复性
            simulated_returns = np.random.multivariate_normal(
                mean_returns, 
                cov_matrix, 
                monte_carlo_sims
            )

            # 计算模拟的投资组合收益率
            portfolio_sim_returns = np.dot(simulated_returns, aligned_weights)

            # 计算VaR和CVaR
            var_95 = -np.percentile(portfolio_sim_returns, 5)
            var_99 = -np.percentile(portfolio_sim_returns, 1)
            cvar_95 = -portfolio_sim_returns[portfolio_sim_returns <= -var_95].mean()
            cvar_99 = -portfolio_sim_returns[portfolio_sim_returns <= -var_99].mean()

            # 计算最大损失和最大收益
            max_loss = -portfolio_sim_returns.min()
            max_gain = portfolio_sim_returns.max()

            stress_results['monte_carlo'] = {
                'simulations': monte_carlo_sims,
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'max_loss': max_loss,
                'max_gain': max_gain,
                'returns': portfolio_sim_returns
            }

        # 创建图表
        plt.figure(figsize=figsize)

        # 绘制自定义情景
        if 'custom_scenarios' in stress_results:
            ax1 = plt.subplot(2, 2, 1)

            # 准备数据
            scenario_names = list(stress_results['custom_scenarios'].keys())
            scenario_returns = [stress_results['custom_scenarios'][name] for name in scenario_names]

            # 设置颜色
            colors = ['red' if r < 0 else 'green' for r in scenario_returns]

            # 绘制条形图
            bars = ax1.bar(np.arange(len(scenario_names)), [r * 100 for r in scenario_returns], color=colors, alpha=0.7)

            # 添加数据标签
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3 if height >= 0 else -15),  # 根据高度调整垂直偏移
                            textcoords="offset points",
                            ha='center', va='bottom' if height >= 0 else 'top')

            ax1.set_title('自定义情景测试', fontsize=12)
            ax1.set_ylabel('收益率 (%)', fontsize=10)
            ax1.set_xticks(np.arange(len(scenario_names)))
            ax1.set_xticklabels(scenario_names, rotation=45, ha='right')
            ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax1.grid(True, alpha=0.3)

        # 绘制历史事件
        if 'historical_events' in stress_results:
            ax2 = plt.subplot(2, 2, 2)

            # 准备数据
            event_names = list(stress_results['historical_events'].keys())
            event_returns = [stress_results['historical_events'][name] for name in event_names]

            # 设置颜色
            colors = ['red' if r < 0 else 'green' for r in event_returns]

            # 绘制条形图
            bars = ax2.bar(np.arange(len(event_names)), [r * 100 for r in event_returns], color=colors, alpha=0.7)

            # 添加数据标签
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3 if height >= 0 else -15),  # 根据高度调整垂直偏移
                            textcoords="offset points",
                            ha='center', va='bottom' if height >= 0 else 'top')

            ax2.set_title('历史事件测试', fontsize=12)
            ax2.set_ylabel('收益率 (%)', fontsize=10)
            ax2.set_xticks(np.arange(len(event_names)))
            ax2.set_xticklabels(event_names, rotation=45, ha='right')
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax2.grid(True, alpha=0.3)

        # 绘制蒙特卡洛模拟
        if 'monte_carlo' in stress_results:
            ax3 = plt.subplot(2, 2, 3)

            # 绘制直方图
            ax3.hist(stress_results['monte_carlo']['returns'] * 100, bins=50, alpha=0.7, color='blue', density=True)

            # 绘制核密度估计
            kde = stats.gaussian_kde(stress_results['monte_carlo']['returns'] * 100)
            x = np.linspace(
                stress_results['monte_carlo']['returns'].min() * 100, 
                stress_results['monte_carlo']['returns'].max() * 100, 
                1000
            )
            ax3.plot(x, kde(x), 'r-', linewidth=2)

            # 绘制VaR线
            ax3.axvline(x=-stress_results['monte_carlo']['var_95'] * 100, color='orange', linestyle='--',
                       label=f"VaR (95%): {stress_results['monte_carlo']['var_95']:.2%}")
            ax3.axvline(x=-stress_results['monte_carlo']['var_99'] * 100, color='red', linestyle='--',
                       label=f"VaR (99%): {stress_results['monte_carlo']['var_99']:.2%}")

            ax3.set_title('蒙特卡洛模拟', fontsize=12)
            ax3.set_xlabel('收益率 (%)', fontsize=10)
            ax3.set_ylabel('密度', fontsize=10)
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3)

            # 绘制统计量表格
            ax4 = plt.subplot(2, 2, 4)
            ax4.axis('off')

            # 创建表格数据
            table_data = [
                ['模拟次数', f"{stress_results['monte_carlo']['simulations']}"],
                ['VaR (95%)', f"{stress_results['monte_carlo']['var_95']:.4%}"],
                ['CVaR (95%)', f"{stress_results['monte_carlo']['cvar_95']:.4%}"],
                ['VaR (99%)', f"{stress_results['monte_carlo']['var_99']:.4%}"],
                ['CVaR (99%)', f"{stress_results['monte_carlo']['cvar_99']:.4%}"],
                ['最大损失', f"{stress_results['monte_carlo']['max_loss']:.4%}"],
                ['最大收益', f"{stress_results['monte_carlo']['max_gain']:.4%}"]
            ]

            # 创建表格
            table = plt.table(
                cellText=table_data,
                colLabels=['统计量', '数值'],
                loc='center',
                cellLoc='center',
                colWidths=[0.4, 0.4]
            )

            # 设置表格样式
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
        else:
            # 如果没有蒙特卡洛模拟，绘制汇总表格
            ax3 = plt.subplot(2, 1, 2)
            ax3.axis('off')

            # 创建表格数据
            table_data = []

            if 'custom_scenarios' in stress_results:
                # 添加自定义情景的最坏和最好情况
                scenario_returns = list(stress_results['custom_scenarios'].values())
                worst_scenario = min(stress_results['custom_scenarios'].items(), key=lambda x: x[1])
                best_scenario = max(stress_results['custom_scenarios'].items(), key=lambda x: x[1])

                table_data.extend([
                    ['自定义情景数量', f"{len(stress_results['custom_scenarios'])}"],
                    ['最坏情景', f"{worst_scenario[0]} ({worst_scenario[1]:.2%})"],
                    ['最好情景', f"{best_scenario[0]} ({best_scenario[1]:.2%})"]
                ])

            if 'historical_events' in stress_results:
                # 添加历史事件的最坏和最好情况
                event_returns = list(stress_results['historical_events'].values())
                worst_event = min(stress_results['historical_events'].items(), key=lambda x: x[1])
                best_event = max(stress_results['historical_events'].items(), key=lambda x: x[1])

                table_data.extend([
                    ['历史事件数量', f"{len(stress_results['historical_events'])}"],
                    ['最坏历史事件', f"{worst_event[0]} ({worst_event[1]:.2%})"],
                    ['最好历史事件', f"{best_event[0]} ({best_event[1]:.2%})"]
                ])

            # 创建表格
            table = plt.table(
                cellText=table_data,
                colLabels=['统计量', '数值'],
                loc='center',
                cellLoc='center',
                colWidths=[0.4, 0.4]
            )

            # 设置表格样式
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)

        plt.tight_layout()
        plt.show()

        # 打印压力测试结果
        print("压力测试结果:")

        if 'custom_scenarios' in stress_results:
            print("\n自定义情景测试:")
            for scenario, return_value in stress_results['custom_scenarios'].items():
                print(f"  {scenario}: {return_value:.4%}")

        if 'historical_events' in stress_results:
            print("\n历史事件测试:")
            for event, return_value in stress_results['historical_events'].items():
                print(f"  {event}: {return_value:.4%}")

        if 'monte_carlo' in stress_results:
            print("\n蒙特卡洛模拟:")
            print(f"  模拟次数: {stress_results['monte_carlo']['simulations']}")
            print(f"  VaR (95%): {stress_results['monte_carlo']['var_95']:.4%}")
            print(f"  CVaR (95%): {stress_results['monte_carlo']['cvar_95']:.4%}")
            print(f"  VaR (99%): {stress_results['monte_carlo']['var_99']:.4%}")
            print(f"  CVaR (99%): {stress_results['monte_carlo']['cvar_99']:.4%}")
            print(f"  最大损失: {stress_results['monte_carlo']['max_loss']:.4%}")
            print(f"  最大收益: {stress_results['monte_carlo']['max_gain']:.4%}")

        return stress_results

    def analyze_liquidity_risk(self, volume_data: pd.DataFrame = None, 
                              bid_ask_data: pd.DataFrame = None,
                              position_sizes: pd.Series = None,
                              figsize: Tuple[int, int] = (12, 8)):
        """
        分析投资组合的流动性风险

        参数:
        volume_data (pd.DataFrame): 交易量数据，索引为日期，列为资产
        bid_ask_data (pd.DataFrame): 买卖价差数据，索引为日期，列为资产
        position_sizes (pd.Series): 持仓规模，索引为资产，值为持仓金额
        figsize (tuple): 图表大小

        返回:
        Dict: 包含流动性风险分析结果的字典
        """
        # 检查输入数据
        if volume_data is None and bid_ask_data is None:
            raise ValueError("流动性风险分析需要交易量数据或买卖价差数据")

        # 如果没有提供持仓规模，使用投资组合权重
        if position_sizes is None:
            if not hasattr(self, 'weights') or self.weights is None:
                raise ValueError("流动性风险分析需要持仓规模或投资组合权重")

            # 假设总资产为1
            position_sizes = self.weights

        # 初始化结果
        liquidity_results = {}

        # 分析交易量
        if volume_data is not None:
            # 确保交易量数据和持仓有相同的资产
            common_assets = set(volume_data.columns).intersection(set(position_sizes.index))
            if len(common_assets) == 0:
                raise ValueError("交易量数据和持仓没有共同的资产")

            # 对齐数据
            aligned_volume = volume_data[list(common_assets)]
            aligned_positions = position_sizes[list(common_assets)]

            # 计算平均日交易量
            avg_daily_volume = aligned_volume.mean()

            # 计算持仓占平均日交易量的比例
            position_to_volume_ratio = aligned_positions / avg_daily_volume

            # 计算清仓天数（假设每天最多交易日均交易量的20%）
            liquidation_days = aligned_positions / (avg_daily_volume * 0.2)

            # 计算流动性成本（假设交易成本与交易量成反比）
            liquidity_cost = 0.01 * position_to_volume_ratio  # 简化模型

            liquidity_results['volume'] = {
                'avg_daily_volume': avg_daily_volume,
                'position_to_volume_ratio': position_to_volume_ratio,
                'liquidation_days': liquidation_days,
                'liquidity_cost': liquidity_cost
            }

        # 分析买卖价差
        if bid_ask_data is not None:
            # 确保买卖价差数据和持仓有相同的资产
            common_assets = set(bid_ask_data.columns).intersection(set(position_sizes.index))
            if len(common_assets) == 0:
                raise ValueError("买卖价差数据和持仓没有共同的资产")

            # 对齐数据
            aligned_bid_ask = bid_ask_data[list(common_assets)]
            aligned_positions = position_sizes[list(common_assets)]

            # 计算平均买卖价差
            avg_bid_ask_spread = aligned_bid_ask.mean()

            # 计算买卖价差的波动性
            bid_ask_volatility = aligned_bid_ask.std()

            # 计算基于买卖价差的流动性成本
            spread_cost = aligned_positions * avg_bid_ask_spread

            liquidity_results['bid_ask'] = {
                'avg_bid_ask_spread': avg_bid_ask_spread,
                'bid_ask_volatility': bid_ask_volatility,
                'spread_cost': spread_cost
            }

        # 创建图表
        plt.figure(figsize=figsize)

        # 绘制持仓占交易量比例
        if 'volume' in liquidity_results:
            ax1 = plt.subplot(2, 2, 1)

            # 按比例排序
            sorted_ratio = liquidity_results['volume']['position_to_volume_ratio'].sort_values(ascending=False)

            # 设置颜色（比例越高，颜色越深）
            colors = plt.cm.YlOrRd(np.linspace(0.2, 0.8, len(sorted_ratio)))

            # 绘制条形图
            bars = ax1.bar(np.arange(len(sorted_ratio)), sorted_ratio, color=colors, alpha=0.7)

            # 添加数据标签
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.2%}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3点垂直偏移
                            textcoords="offset points",
                            ha='center', va='bottom')

            ax1.set_title('持仓占日均交易量比例', fontsize=12)
            ax1.set_ylabel('比例', fontsize=10)
            ax1.set_xticks(np.arange(len(sorted_ratio)))
            ax1.set_xticklabels(sorted_ratio.index, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)

            # 绘制清仓天数
            ax2 = plt.subplot(2, 2, 2)

            # 按天数排序
            sorted_days = liquidity_results['volume']['liquidation_days'].sort_values(ascending=False)

            # 设置颜色（天数越多，颜色越深）
            colors = plt.cm.YlOrRd(np.linspace(0.2, 0.8, len(sorted_days)))

            # 绘制条形图
            bars = ax2.bar(np.arange(len(sorted_days)), sorted_days, color=colors, alpha=0.7)

            # 添加数据标签
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3点垂直偏移
                            textcoords="offset points",
                            ha='center', va='bottom')

            ax2.set_title('预估清仓天数', fontsize=12)
            ax2.set_ylabel('天数', fontsize=10)
            ax2.set_xticks(np.arange(len(sorted_days)))
            ax2.set_xticklabels(sorted_days.index, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)

        # 绘制买卖价差
        if 'bid_ask' in liquidity_results:
            ax3 = plt.subplot(2, 2, 3)

            # 按价差排序
            sorted_spread = liquidity_results['bid_ask']['avg_bid_ask_spread'].sort_values(ascending=False)

            # 设置颜色（价差越大，颜色越深）
            colors = plt.cm.YlOrRd(np.linspace(0.2, 0.8, len(sorted_spread)))

            # 绘制条形图
            bars = ax3.bar(np.arange(len(sorted_spread)), sorted_spread * 100, color=colors, alpha=0.7)

            # 添加数据标签
            for bar in bars:
                height = bar.get_height()
                ax3.annotate(f'{height:.2f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3点垂直偏移
                            textcoords="offset points",
                            ha='center', va='bottom')

            ax3.set_title('平均买卖价差', fontsize=12)
            ax3.set_ylabel('价差 (%)', fontsize=10)
            ax3.set_xticks(np.arange(len(sorted_spread)))
            ax3.set_xticklabels(sorted_spread.index, rotation=45, ha='right')
            ax3.grid(True, alpha=0.3)

            # 绘制买卖价差的波动性
            ax4 = plt.subplot(2, 2, 4)

            # 按波动性排序
            sorted_volatility = liquidity_results['bid_ask']['bid_ask_volatility'].sort_values(ascending=False)

            # 设置颜色（波动性越大，颜色越深）
            colors = plt.cm.YlOrRd(np.linspace(0.2, 0.8, len(sorted_volatility)))

            # 绘制条形图
            bars = ax4.bar(np.arange(len(sorted_volatility)), sorted_volatility * 100, color=colors, alpha=0.7)

            # 添加数据标签
            for bar in bars:
                height = bar.get_height()
                ax4.annotate(f'{height:.2f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3点垂直偏移
                            textcoords="offset points",
                            ha='center', va='bottom')

            ax4.set_title('买卖价差波动性', fontsize=12)
            ax4.set_ylabel('波动性 (%)', fontsize=10)
            ax4.set_xticks(np.arange(len(sorted_volatility)))
            ax4.set_xticklabels(sorted_volatility.index, rotation=45, ha='right')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 打印流动性风险分析结果
        print("流动性风险分析结果:")

        if 'volume' in liquidity_results:
            print("\n交易量分析:")

            # 找出流动性最差的资产（持仓占交易量比例最高的）
            worst_liquidity = liquidity_results['volume']['position_to_volume_ratio'].sort_values(ascending=False).head(3)
            print("流动性最差的资产（持仓/日均交易量）:")
            for asset, ratio in worst_liquidity.items():
                print(f"  {asset}: {ratio:.4%}")

            # 找出清仓天数最长的资产
            longest_liquidation = liquidity_results['volume']['liquidation_days'].sort_values(ascending=False).head(3)
            print("\n清仓天数最长的资产:")
            for asset, days in longest_liquidation.items():
                print(f"  {asset}: {days:.2f}天")

            # 计算投资组合整体的流动性指标
            portfolio_liquidation_days = (liquidity_results['volume']['liquidation_days'] * position_sizes).sum() / position_sizes.sum()
            print(f"\n投资组合整体预估清仓天数: {portfolio_liquidation_days:.2f}天")

            portfolio_liquidity_cost = (liquidity_results['volume']['liquidity_cost'] * position_sizes).sum() / position_sizes.sum()
            print(f"投资组合整体流动性成本: {portfolio_liquidity_cost:.4%}")

        if 'bid_ask' in liquidity_results:
            print("\n买卖价差分析:")

            # 找出价差最大的资产
            widest_spread = liquidity_results['bid_ask']['avg_bid_ask_spread'].sort_values(ascending=False).head(3)
            print("价差最大的资产:")
            for asset, spread in widest_spread.items():
                print(f"  {asset}: {spread:.4%}")

            # 找出价差波动性最大的资产
            most_volatile = liquidity_results['bid_ask']['bid_ask_volatility'].sort_values(ascending=False).head(3)
            print("\n价差波动性最大的资产:")
            for asset, volatility in most_volatile.items():
                print(f"  {asset}: {volatility:.4%}")

            # 计算投资组合整体的价差成本
            portfolio_spread_cost = liquidity_results['bid_ask']['spread_cost'].sum() / position_sizes.sum()
            print(f"\n投资组合整体价差成本: {portfolio_spread_cost:.4%}")

        # 返回分析结果
        return liquidity_results

    def analyze_risk_attribution(self, factor_data: pd.DataFrame = None,
                               benchmark_returns: pd.Series = None,
                               figsize: Tuple[int, int] = (12, 8)):
        """
        分析投资组合收益的风险归因

        参数:
        factor_data (pd.DataFrame): 因子收益率数据，索引为日期，列为因子
        benchmark_returns (pd.Series): 基准收益率，索引为日期
        figsize (tuple): 图表大小

        返回:
        Dict: 包含风险归因分析结果的字典
        """
        # 检查输入数据
        if not isinstance(self.portfolio_returns, pd.Series):
            raise ValueError("风险归因分析需要投资组合收益率Series")

        # 初始化结果
        attribution_results = {}

        # 计算基本归因指标
        mean_return = self.portfolio_returns.mean()
        total_risk = self.portfolio_returns.std()

        # 计算相对于基准的超额收益
        if benchmark_returns is not None:
            # 确保基准收益率和投资组合收益率有相同的索引
            common_index = self.portfolio_returns.index.intersection(benchmark_returns.index)
            if len(common_index) == 0:
                raise ValueError("投资组合收益率和基准收益率没有共同的日期")

            # 对齐数据
            aligned_portfolio = self.portfolio_returns.loc[common_index]
            aligned_benchmark = benchmark_returns.loc[common_index]

            # 计算超额收益
            excess_returns = aligned_portfolio - aligned_benchmark

            # 计算超额收益的统计量
            excess_mean = excess_returns.mean()
            excess_risk = excess_returns.std()

            # 计算信息比率
            information_ratio = excess_mean / excess_risk if excess_risk > 0 else 0

            # 计算跟踪误差
            tracking_error = excess_risk

            # 计算捕获率
            up_market = aligned_benchmark > 0
            down_market = aligned_benchmark < 0

            up_capture = (aligned_portfolio[up_market].mean() / aligned_benchmark[up_market].mean()) if up_market.any() else 0
            down_capture = (aligned_portfolio[down_market].mean() / aligned_benchmark[down_market].mean()) if down_market.any() else 0

            # 计算Beta和Alpha
            cov_matrix = pd.concat([aligned_portfolio, aligned_benchmark], axis=1).cov()
            beta = cov_matrix.iloc[0, 1] / cov_matrix.iloc[1, 1] if cov_matrix.iloc[1, 1] > 0 else 0
            alpha = aligned_portfolio.mean() - beta * aligned_benchmark.mean()

            # 存储基准相关的归因结果
            attribution_results['benchmark'] = {
                'excess_returns': excess_returns,
                'excess_mean': excess_mean,
                'tracking_error': tracking_error,
                'information_ratio': information_ratio,
                'up_capture': up_capture,
                'down_capture': down_capture,
                'beta': beta,
                'alpha': alpha
            }

        # 因子归因分析
        if factor_data is not None:
            # 确保因子数据和投资组合收益率有相同的索引
            common_index = self.portfolio_returns.index.intersection(factor_data.index)
            if len(common_index) == 0:
                raise ValueError("投资组合收益率和因子数据没有共同的日期")

            # 对齐数据
            aligned_portfolio = self.portfolio_returns.loc[common_index]
            aligned_factors = factor_data.loc[common_index]

            # 添加常数项（Alpha）
            X = sm.add_constant(aligned_factors)

            # 进行回归分析
            model = sm.OLS(aligned_portfolio, X)
            results = model.fit()

            # 提取因子暴露（系数）
            factor_exposures = results.params

            # 计算每个因子的贡献
            factor_contributions = pd.DataFrame(index=common_index, columns=X.columns)
            for factor in X.columns:
                if factor == 'const':
                    # Alpha贡献是常数
                    factor_contributions[factor] = factor_exposures[factor] / len(common_index)
                else:
                    # 因子贡献是因子暴露乘以因子收益率
                    factor_contributions[factor] = factor_exposures[factor] * aligned_factors[factor]

            # 计算每个因子对总风险的贡献
            # 使用协方差矩阵和因子暴露计算
            factor_cov = aligned_factors.cov()
            factor_exposures_vector = factor_exposures.drop('const')

            # 计算因子风险贡献
            factor_risk_contribution = {}

            # 计算总的因子风险
            factor_risk = np.sqrt(factor_exposures_vector.dot(factor_cov).dot(factor_exposures_vector))

            # 计算每个因子的边际风险贡献
            for factor in factor_exposures_vector.index:
                # 计算因子的边际贡献
                marginal_contribution = factor_exposures_vector.dot(factor_cov[factor])

                # 计算因子的风险贡献
                factor_risk_contribution[factor] = factor_exposures_vector[factor] * marginal_contribution / factor_risk if factor_risk > 0 else 0

            # 存储因子归因结果
            attribution_results['factor'] = {
                'exposures': factor_exposures,
                'contributions': factor_contributions,
                'risk_contribution': factor_risk_contribution,
                'total_factor_risk': factor_risk,
                'r_squared': results.rsquared,
                'adj_r_squared': results.rsquared_adj
            }

        # 创建图表
        plt.figure(figsize=figsize)

        # 绘制收益归因
        if 'benchmark' in attribution_results:
            ax1 = plt.subplot(2, 2, 1)

            # 准备数据
            benchmark_data = attribution_results['benchmark']

            # 计算累积收益
            cum_portfolio = (1 + self.portfolio_returns).cumprod()
            cum_benchmark = (1 + benchmark_returns).cumprod()
            cum_excess = (1 + benchmark_data['excess_returns']).cumprod()

            # 绘制累积收益曲线
            ax1.plot(cum_portfolio.index, cum_portfolio, 'b-', label='投资组合')
            ax1.plot(cum_benchmark.index, cum_benchmark, 'r-', label='基准')
            ax1.plot(cum_excess.index, cum_excess, 'g-', label='超额收益')

            ax1.set_title('累积收益对比', fontsize=12)
            ax1.set_xlabel('日期', fontsize=10)
            ax1.set_ylabel('累积收益', fontsize=10)
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)

            # 绘制收益归因表格
            ax2 = plt.subplot(2, 2, 2)
            ax2.axis('off')

            # 创建表格数据
            table_data = [
                ['年化收益率', f"{mean_return * 252:.4%}"],
                ['年化波动率', f"{total_risk * np.sqrt(252):.4%}"],
                ['超额收益', f"{benchmark_data['excess_mean'] * 252:.4%}"],
                ['跟踪误差', f"{benchmark_data['tracking_error'] * np.sqrt(252):.4%}"],
                ['信息比率', f"{benchmark_data['information_ratio'] * np.sqrt(252):.4f}"],
                ['上行捕获率', f"{benchmark_data['up_capture']:.4f}"],
                ['下行捕获率', f"{benchmark_data['down_capture']:.4f}"],
                ['Beta', f"{benchmark_data['beta']:.4f}"],
                ['Alpha', f"{benchmark_data['alpha'] * 252:.4%}"]
            ]

            # 创建表格
            table = plt.table(
                cellText=table_data,
                colLabels=['指标', '数值'],
                loc='center',
                cellLoc='center',
                colWidths=[0.4, 0.4]
            )

            # 设置表格样式
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)

        # 绘制因子归因
        if 'factor' in attribution_results:
            ax3 = plt.subplot(2, 2, 3)

            # 准备数据
            factor_data = attribution_results['factor']

            # 计算平均因子贡献
            mean_contributions = factor_data['contributions'].mean()

            # 按贡献绝对值排序
            sorted_contributions = mean_contributions.abs().sort_values(ascending=False)
            sorted_factors = sorted_contributions.index

            # 设置颜色
            colors = ['red' if mean_contributions[f] < 0 else 'green' for f in sorted_factors]

            # 绘制条形图
            bars = ax3.bar(np.arange(len(sorted_factors)), [mean_contributions[f] for f in sorted_factors], color=colors, alpha=0.7)

            # 添加数据标签
            for bar in bars:
                height = bar.get_height()
                ax3.annotate(f'{height:.6f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3 if height >= 0 else -15),  # 根据高度调整垂直偏移
                            textcoords="offset points",
                            ha='center', va='bottom' if height >= 0 else 'top')

            ax3.set_title('平均因子贡献', fontsize=12)
            ax3.set_ylabel('贡献', fontsize=10)
            ax3.set_xticks(np.arange(len(sorted_factors)))
            ax3.set_xticklabels(sorted_factors, rotation=45, ha='right')
            ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax3.grid(True, alpha=0.3)

            # 绘制因子风险贡献
            ax4 = plt.subplot(2, 2, 4)

            # 准备数据
            risk_contribution = factor_data['risk_contribution']

            # 按风险贡献绝对值排序
            sorted_risk = pd.Series(risk_contribution).abs().sort_values(ascending=False)
            sorted_risk_factors = sorted_risk.index

            # 计算风险贡献百分比
            total_explained_risk = sum(abs(v) for v in risk_contribution.values())
            risk_pct = {k: abs(v) / total_explained_risk * 100 if total_explained_risk > 0 else 0 
                       for k, v in risk_contribution.items()}

            # 设置颜色
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_risk_factors)))

            # 绘制饼图
            wedges, texts, autotexts = ax4.pie(
                [risk_pct[f] for f in sorted_risk_factors],
                labels=[f for f in sorted_risk_factors],
                autopct='%1.1f%%',
                startangle=90,
                colors=colors
            )

            # 设置饼图样式
            for text in texts:
                text.set_fontsize(8)
            for autotext in autotexts:
                autotext.set_fontsize(8)
                autotext.set_color('white')

            ax4.set_title('因子风险贡献', fontsize=12)
            ax4.axis('equal')  # 确保饼图是圆形的

        plt.tight_layout()
        plt.show()

        # 打印风险归因分析结果
        print("风险归因分析结果:")

        if 'benchmark' in attribution_results:
            print("\n基准归因:")
            benchmark_data = attribution_results['benchmark']
            print(f"年化超额收益: {benchmark_data['excess_mean'] * 252:.4%}")
            print(f"年化跟踪误差: {benchmark_data['tracking_error'] * np.sqrt(252):.4%}")
            print(f"信息比率: {benchmark_data['information_ratio'] * np.sqrt(252):.4f}")
            print(f"上行捕获率: {benchmark_data['up_capture']:.4f}")
            print(f"下行捕获率: {benchmark_data['down_capture']:.4f}")
            print(f"Beta: {benchmark_data['beta']:.4f}")
            print(f"Alpha: {benchmark_data['alpha'] * 252:.4%}")

        if 'factor' in attribution_results:
            print("\n因子归因:")
            factor_data = attribution_results['factor']
            print(f"R²: {factor_data['r_squared']:.4f}")
            print(f"调整R²: {factor_data['adj_r_squared']:.4f}")

            print("\n因子暴露:")
            for factor, exposure in factor_data['exposures'].items():
                print(f"  {factor}: {exposure:.6f}")

            print("\n平均因子贡献:")
            mean_contributions = factor_data['contributions'].mean()
            for factor in sorted(mean_contributions.index, key=lambda x: abs(mean_contributions[x]), reverse=True):
                print(f"  {factor}: {mean_contributions[factor]:.6f}")

            print("\n因子风险贡献:")
            risk_contribution = factor_data['risk_contribution']
            total_explained_risk = sum(abs(v) for v in risk_contribution.values())
            for factor in sorted(risk_contribution.keys(), key=lambda x: abs(risk_contribution[x]), reverse=True):
                risk_pct = abs(risk_contribution[factor]) / total_explained_risk * 100 if total_explained_risk > 0 else 0
                print(f"  {factor}: {risk_contribution[factor]:.6f} ({risk_pct:.2f}%)")

        return attribution_results

    def analyze_concentration_risk(self, position_sizes: pd.Series = None,
                                 sector_mapping: Dict[str, str] = None,
                                 figsize: Tuple[int, int] = (12, 8)):
        """
        分析投资组合的集中度风险

        参数:
        position_sizes (pd.Series): 持仓规模，索引为资产，值为持仓金额
        sector_mapping (Dict[str, str]): 资产到行业的映射，格式为 {'资产1': '行业1', '资产2': '行业2', ...}
        figsize (tuple): 图表大小

        返回:
        Dict: 包含集中度风险分析结果的字典
        """
        # 检查输入数据
        if position_sizes is None:
            if not hasattr(self, 'weights') or self.weights is None:
                raise ValueError("集中度风险分析需要持仓规模或投资组合权重")

            # 假设总资产为1
            position_sizes = self.weights

        # 初始化结果
        concentration_results = {}

        # 计算基本集中度指标
        total_position = position_sizes.sum()
        position_weights = position_sizes / total_position

        # 计算资产集中度
        # 按权重排序
        sorted_weights = position_weights.sort_values(ascending=False)

        # 计算前N大持仓的权重和
        top_holdings = {
            'top_1': sorted_weights.iloc[0] if len(sorted_weights) >= 1 else 0,
            'top_3': sorted_weights.iloc[:3].sum() if len(sorted_weights) >= 3 else sorted_weights.sum(),
            'top_5': sorted_weights.iloc[:5].sum() if len(sorted_weights) >= 5 else sorted_weights.sum(),
            'top_10': sorted_weights.iloc[:10].sum() if len(sorted_weights) >= 10 else sorted_weights.sum()
        }

        # 计算赫芬达尔-赫希曼指数 (HHI)
        # HHI是权重平方和，范围从1/N（完全分散）到1（完全集中）
        hhi = (position_weights ** 2).sum()

        # 计算有效N值（Effective N）
        # 有效N是HHI的倒数，表示等权重资产的等效数量
        effective_n = 1 / hhi if hhi > 0 else float('inf')

        # 计算基尼系数
        # 基尼系数衡量分布的不平等程度，范围从0（完全平等）到1（完全不平等）
        sorted_weights_values = sorted(position_weights.values)
        n = len(sorted_weights_values)
        if n > 0:
            # 计算洛伦兹曲线下的面积
            lorenz_area = sum((n - i) * w for i, w in enumerate(sorted_weights_values)) / (n * sum(sorted_weights_values))
            # 计算基尼系数
            gini = 1 - 2 * lorenz_area
        else:
            gini = 0

        # 存储资产集中度结果
        concentration_results['asset'] = {
            'top_holdings': top_holdings,
            'hhi': hhi,
            'effective_n': effective_n,
            'gini': gini,
            'weights': sorted_weights
        }

        # 分析行业集中度
        if sector_mapping is not None:
            # 计算行业权重
            sector_weights = {}

            for asset, weight in position_weights.items():
                if asset in sector_mapping:
                    sector = sector_mapping[asset]
                    if sector in sector_weights:
                        sector_weights[sector] += weight
                    else:
                        sector_weights[sector] = weight
                else:
                    # 对于未映射的资产，归类为"其他"
                    if '其他' in sector_weights:
                        sector_weights['其他'] += weight
                    else:
                        sector_weights['其他'] = weight

            # 转换为Series并排序
            sector_weights = pd.Series(sector_weights).sort_values(ascending=False)

            # 计算行业HHI
            sector_hhi = (sector_weights ** 2).sum()

            # 计算行业有效N值
            sector_effective_n = 1 / sector_hhi if sector_hhi > 0 else float('inf')

            # 计算行业基尼系数
            sorted_sector_weights = sorted(sector_weights.values)
            n = len(sorted_sector_weights)
            if n > 0:
                sector_lorenz_area = sum((n - i) * w for i, w in enumerate(sorted_sector_weights)) / (n * sum(sorted_sector_weights))
                sector_gini = 1 - 2 * sector_lorenz_area
            else:
                sector_gini = 0

            # 存储行业集中度结果
            concentration_results['sector'] = {
                'weights': sector_weights,
                'hhi': sector_hhi,
                'effective_n': sector_effective_n,
                'gini': sector_gini
            }

        # 创建图表
        plt.figure(figsize=figsize)

        # 绘制资产权重分布
        ax1 = plt.subplot(2, 2, 1)

        # 准备数据
        asset_weights = concentration_results['asset']['weights']

        # 只显示前15个资产
        if len(asset_weights) > 15:
            displayed_weights = asset_weights.iloc[:15]
            # 如果有更多资产，将剩余资产合并为"其他"
            other_weight = asset_weights.iloc[15:].sum()
            displayed_weights = pd.concat([displayed_weights, pd.Series({'其他': other_weight})])
        else:
            displayed_weights = asset_weights

        # 设置颜色
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(displayed_weights)))

        # 绘制饼图
        wedges, texts, autotexts = ax1.pie(
            displayed_weights.values,
            labels=displayed_weights.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors
        )

        # 设置饼图样式
        for text in texts:
            text.set_fontsize(8)
        for autotext in autotexts:
            autotext.set_fontsize(8)
            autotext.set_color('white')

        ax1.set_title('资产权重分布', fontsize=12)
        ax1.axis('equal')  # 确保饼图是圆形的

        # 绘制资产集中度指标
        ax2 = plt.subplot(2, 2, 2)
        ax2.axis('off')

        # 创建表格数据
        table_data = [
            ['总资产数量', f"{len(position_weights)}"],
            ['最大单一持仓', f"{top_holdings['top_1']:.2%}"],
            ['前3大持仓', f"{top_holdings['top_3']:.2%}"],
            ['前5大持仓', f"{top_holdings['top_5']:.2%}"],
            ['前10大持仓', f"{top_holdings['top_10']:.2%}"],
            ['HHI指数', f"{hhi:.4f}"],
            ['有效N值', f"{effective_n:.2f}"],
            ['基尼系数', f"{gini:.4f}"]
        ]

        # 创建表格
        table = plt.table(
            cellText=table_data,
            colLabels=['指标', '数值'],
            loc='center',
            cellLoc='center',
            colWidths=[0.4, 0.4]
        )

        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # 绘制行业分布
        if 'sector' in concentration_results:
            ax3 = plt.subplot(2, 2, 3)

            # 准备数据
            sector_weights = concentration_results['sector']['weights']

            # 设置颜色
            colors = plt.cm.tab20(np.linspace(0, 1, len(sector_weights)))

            # 绘制饼图
            wedges, texts, autotexts = ax3.pie(
                sector_weights.values,
                labels=sector_weights.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors
            )

            # 设置饼图样式
            for text in texts:
                text.set_fontsize(8)
            for autotext in autotexts:
                autotext.set_fontsize(8)
                autotext.set_color('white')

            ax3.set_title('行业分布', fontsize=12)
            ax3.axis('equal')  # 确保饼图是圆形的

            # 绘制行业集中度指标
            ax4 = plt.subplot(2, 2, 4)
            ax4.axis('off')

            # 创建表格数据
            sector_table_data = [
                ['行业数量', f"{len(sector_weights)}"],
                ['最大行业占比', f"{sector_weights.iloc[0]:.2%}"],
                ['前3大行业占比', f"{sector_weights.iloc[:3].sum():.2%}" if len(sector_weights) >= 3 else f"{sector_weights.sum():.2%}"],
                ['HHI指数', f"{sector_hhi:.4f}"],
                ['有效N值', f"{sector_effective_n:.2f}"],
                ['基尼系数', f"{sector_gini:.4f}"]
            ]

            # 创建表格
            sector_table = plt.table(
                cellText=sector_table_data,
                colLabels=['指标', '数值'],
                loc='center',
                cellLoc='center',
                colWidths=[0.4, 0.4]
            )

            # 设置表格样式
            sector_table.auto_set_font_size(False)
            sector_table.set_fontsize(10)
            sector_table.scale(1, 1.5)

        plt.tight_layout()
        plt.show()

        # 打印集中度风险分析结果
        print("集中度风险分析结果:")

        print("\n资产集中度:")
        print(f"总资产数量: {len(position_weights)}")
        print(f"最大单一持仓: {top_holdings['top_1']:.2%}")
        print(f"前3大持仓: {top_holdings['top_3']:.2%}")
        print(f"前5大持仓: {top_holdings['top_5']:.2%}")
        print(f"前10大持仓: {top_holdings['top_10']:.2%}")
        print(f"HHI指数: {hhi:.4f}")
        print(f"有效N值: {effective_n:.2f}")
        print(f"基尼系数: {gini:.4f}")

        if 'sector' in concentration_results:
            print("\n行业集中度:")
            print(f"行业数量: {len(sector_weights)}")
            print(f"最大行业占比: {sector_weights.iloc[0]:.2%}")
            print(f"前3大行业占比: {sector_weights.iloc[:3].sum():.2%}" if len(sector_weights) >= 3 else f"{sector_weights.sum():.2%}")
            print(f"HHI指数: {sector_hhi:.4f}")
            print(f"有效N值: {sector_effective_n:.2f}")
            print(f"基尼系数: {sector_gini:.4f}")

            print("\n行业分布:")
            for sector, weight in sector_weights.items():
                print(f"  {sector}: {weight:.2%}")

        return concentration_results
