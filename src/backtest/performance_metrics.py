# -*- coding: utf-8 -*-
"""
性能评估指标计算模块

计算策略性能评估指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple


def calculate_returns(equity_curve: pd.Series) -> pd.Series:
    """
    计算收益率序列

    参数:
    equity_curve (pd.Series): 权益曲线

    返回:
    pd.Series: 收益率序列
    """
    return equity_curve.pct_change().dropna()


def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    计算累积收益率

    参数:
    returns (pd.Series): 收益率序列

    返回:
    pd.Series: 累积收益率序列
    """
    return (1 + returns).cumprod() - 1


def calculate_annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    计算年化收益率

    参数:
    returns (pd.Series): 收益率序列
    periods_per_year (int): 每年的周期数，默认252个交易日

    返回:
    float: 年化收益率
    """
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    return (1 + total_return) ** (periods_per_year / n_periods) - 1 if n_periods > 0 else 0


def calculate_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    计算波动率

    参数:
    returns (pd.Series): 收益率序列
    periods_per_year (int): 每年的周期数，默认252个交易日

    返回:
    float: 年化波动率
    """
    return returns.std() * np.sqrt(periods_per_year) if len(returns) > 1 else 0


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    计算夏普比率

    参数:
    returns (pd.Series): 收益率序列
    risk_free_rate (float): 无风险利率，年化
    periods_per_year (int): 每年的周期数，默认252个交易日

    返回:
    float: 夏普比率
    """
    # 计算超额收益
    excess_returns = returns - risk_free_rate / periods_per_year

    # 计算年化超额收益均值
    annual_excess_return = excess_returns.mean() * periods_per_year

    # 计算年化波动率
    annual_volatility = calculate_volatility(returns, periods_per_year)

    # 计算夏普比率
    return annual_excess_return / annual_volatility if annual_volatility > 0 else 0


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    计算索提诺比率

    参数:
    returns (pd.Series): 收益率序列
    risk_free_rate (float): 无风险利率，年化
    periods_per_year (int): 每年的周期数，默认252个交易日

    返回:
    float: 索提诺比率
    """
    # 计算超额收益
    excess_returns = returns - risk_free_rate / periods_per_year

    # 计算年化超额收益均值
    annual_excess_return = excess_returns.mean() * periods_per_year

    # 计算下行波动率
    downside_returns = returns[returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(periods_per_year) if len(downside_returns) > 1 else 0

    # 计算索提诺比率
    return annual_excess_return / downside_volatility if downside_volatility > 0 else 0


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    计算最大回撤

    参数:
    equity_curve (pd.Series): 权益曲线

    返回:
    float: 最大回撤
    """
    # 计算累计最大值
    running_max = equity_curve.cummax()

    # 计算回撤
    drawdown = (equity_curve / running_max - 1)

    # 计算最大回撤
    return drawdown.min()


def calculate_calmar_ratio(returns: pd.Series, equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """
    计算卡玛比率

    参数:
    returns (pd.Series): 收益率序列
    equity_curve (pd.Series): 权益曲线
    periods_per_year (int): 每年的周期数，默认252个交易日

    返回:
    float: 卡玛比率
    """
    # 计算年化收益率
    annualized_return = calculate_annualized_return(returns, periods_per_year)

    # 计算最大回撤
    max_drawdown = calculate_max_drawdown(equity_curve)

    # 计算卡玛比率
    return annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0


def calculate_omega_ratio(returns: pd.Series, threshold: float = 0.0, periods_per_year: int = 252) -> float:
    """
    计算欧米伽比率

    参数:
    returns (pd.Series): 收益率序列
    threshold (float): 阈值收益率，年化
    periods_per_year (int): 每年的周期数，默认252个交易日

    返回:
    float: 欧米伽比率
    """
    # 将年化阈值转换为对应周期的阈值
    threshold_return = threshold / periods_per_year

    # 计算超过阈值的收益和低于阈值的损失
    excess_returns = returns - threshold_return
    positive_returns = excess_returns[excess_returns > 0]
    negative_returns = excess_returns[excess_returns <= 0]

    # 计算正收益和负收益的总和
    positive_sum = positive_returns.sum() if len(positive_returns) > 0 else 0
    negative_sum = abs(negative_returns.sum()) if len(negative_returns) > 0 else 0

    # 计算欧米伽比率
    return positive_sum / negative_sum if negative_sum > 0 else float('inf')


def calculate_win_rate(trades: List[Dict]) -> float:
    """
    计算胜率

    参数:
    trades (List[Dict]): 交易记录列表

    返回:
    float: 胜率
    """
    if not trades:
        return 0

    # 统计盈利交易数量
    profitable_trades = sum(1 for trade in trades if 'profit' in trade and trade['profit'] > 0)

    # 计算胜率
    return profitable_trades / len(trades)


def calculate_profit_loss_ratio(trades: List[Dict]) -> float:
    """
    计算盈亏比

    参数:
    trades (List[Dict]): 交易记录列表

    返回:
    float: 盈亏比
    """
    if not trades:
        return 0

    # 筛选盈利和亏损交易
    profitable_trades = [trade for trade in trades if 'profit' in trade and trade['profit'] > 0]
    losing_trades = [trade for trade in trades if 'profit' in trade and trade['profit'] <= 0]

    if not profitable_trades or not losing_trades:
        return 0

    # 计算平均盈利和平均亏损
    avg_profit = sum(trade['profit'] for trade in profitable_trades) / len(profitable_trades)
    avg_loss = abs(sum(trade['profit'] for trade in losing_trades) / len(losing_trades))

    # 计算盈亏比
    return avg_profit / avg_loss if avg_loss > 0 else float('inf')


def calculate_expectancy(trades: List[Dict]) -> float:
    """
    计算期望收益

    参数:
    trades (List[Dict]): 交易记录列表

    返回:
    float: 期望收益
    """
    if not trades:
        return 0

    # 计算胜率
    win_rate = calculate_win_rate(trades)

    # 计算盈亏比
    profit_loss_ratio = calculate_profit_loss_ratio(trades)

    # 计算期望收益
    return (win_rate * profit_loss_ratio) - (1 - win_rate)


def calculate_max_consecutive_wins_losses(trades: List[Dict]) -> Tuple[int, int]:
    """
    计算最大连续盈利和亏损次数

    参数:
    trades (List[Dict]): 交易记录列表

    返回:
    Tuple[int, int]: (最大连续盈利次数, 最大连续亏损次数)
    """
    if not trades:
        return 0, 0

    # 初始化计数器
    consecutive_wins = 0
    consecutive_losses = 0
    max_consecutive_wins = 0
    max_consecutive_losses = 0

    for trade in trades:
        if 'profit' not in trade:
            continue

        if trade['profit'] > 0:
            # 盈利交易
            consecutive_wins += 1
            consecutive_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
        else:
            # 亏损交易
            consecutive_losses += 1
            consecutive_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)

    return max_consecutive_wins, max_consecutive_losses


def calculate_average_trade_duration(trades: List[Dict]) -> float:
    """
    计算平均交易持续时间（以天为单位）

    参数:
    trades (List[Dict]): 交易记录列表

    返回:
    float: 平均交易持续时间（天）
    """
    if not trades:
        return 0

    # 筛选有开仓和平仓时间的交易
    valid_trades = [trade for trade in trades if 'open_time' in trade and 'close_time' in trade]

    if not valid_trades:
        return 0

    # 计算每笔交易的持续时间
    durations = [(trade['close_time'] - trade['open_time']).total_seconds() / (24 * 3600) for trade in valid_trades]

    # 计算平均持续时间
    return sum(durations) / len(durations)


def calculate_all_metrics(equity_curve: pd.Series, trades: List[Dict], 
                         risk_free_rate: float = 0.0, periods_per_year: int = 252) -> Dict:
    """
    计算所有性能指标

    参数:
    equity_curve (pd.Series): 权益曲线
    trades (List[Dict]): 交易记录列表
    risk_free_rate (float): 无风险利率，年化
    periods_per_year (int): 每年的周期数，默认252个交易日

    返回:
    Dict: 性能指标字典
    """
    # 计算收益率序列
    returns = calculate_returns(equity_curve)

    # 计算各项指标
    metrics = {
        'total_return': (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1 if len(equity_curve) > 1 else 0,
        'annualized_return': calculate_annualized_return(returns, periods_per_year),
        'volatility': calculate_volatility(returns, periods_per_year),
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year),
        'sortino_ratio': calculate_sortino_ratio(returns, risk_free_rate, periods_per_year),
        'max_drawdown': calculate_max_drawdown(equity_curve),
        'calmar_ratio': calculate_calmar_ratio(returns, equity_curve, periods_per_year),
        'omega_ratio': calculate_omega_ratio(returns, risk_free_rate, periods_per_year),
        'win_rate': calculate_win_rate(trades),
        'profit_loss_ratio': calculate_profit_loss_ratio(trades),
        'expectancy': calculate_expectancy(trades),
        'total_trades': len(trades),
        'profitable_trades': sum(1 for trade in trades if 'profit' in trade and trade['profit'] > 0),
        'losing_trades': sum(1 for trade in trades if 'profit' in trade and trade['profit'] <= 0),
    }

    # 计算最大连续盈亏次数
    max_consecutive_wins, max_consecutive_losses = calculate_max_consecutive_wins_losses(trades)
    metrics['max_consecutive_wins'] = max_consecutive_wins
    metrics['max_consecutive_losses'] = max_consecutive_losses

    return metrics