# -*- coding: utf-8 -*-
"""
策略模块

整合信号生成、交易规则和风险控制
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple
from datetime import datetime

from src.strategy.signal_generator import SignalGenerator
from src.strategy.trading_rules import TradingRules, OrderType, OrderDirection
from src.strategy.risk_control import RiskController


class Strategy:
    """
    交易策略

    整合信号生成、交易规则和风险控制
    """

    def __init__(self, 
                 signal_generator: SignalGenerator,
                 trading_rules: TradingRules,
                 risk_controller: RiskController,
                 initial_capital: float = 1000000.0):
        """
        初始化策略

        参数:
        signal_generator (SignalGenerator): 信号生成器
        trading_rules (TradingRules): 交易规则
        risk_controller (RiskController): 风险控制器
        initial_capital (float): 初始资金
        """
        self.signal_generator = signal_generator
        self.trading_rules = trading_rules
        self.risk_controller = risk_controller
        self.initial_capital = initial_capital

        # 策略状态
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.positions = {}  # 持仓字典，键为标的，值为持仓数量
        self.entry_prices = {}  # 入场价格字典
        self.holding_periods = {}  # 持仓周期字典
        self.returns_history = {}  # 收益率历史
        self.trades = []  # 交易记录
        self.equity_curve = []  # 权益曲线

        # 性能指标
        self.performance_metrics = {}

    def update_market_data(self, timestamp: pd.Timestamp, 
                          prices: Dict[str, float], 
                          factors: Dict[str, Dict[str, float]]) -> None:
        """
        更新市场数据

        参数:
        timestamp (pd.Timestamp): 时间戳
        prices (Dict[str, float]): 价格字典，键为标的，值为价格
        factors (Dict[str, Dict[str, float]]): 因子字典，键为标的，值为因子字典
        """
        # 更新持仓周期
        for symbol in self.positions:
            if self.positions[symbol] != 0:
                if symbol not in self.holding_periods:
                    self.holding_periods[symbol] = 0
                self.holding_periods[symbol] += 1

        # 更新投资组合价值
        self.update_portfolio_value(prices)

        # 更新收益率历史
        self.update_returns_history(timestamp, prices)

        # 更新权益曲线
        self.equity_curve.append((timestamp, self.portfolio_value))

    def update_portfolio_value(self, prices: Dict[str, float]) -> None:
        """
        更新投资组合价值

        参数:
        prices (Dict[str, float]): 价格字典
        """
        portfolio_value = self.cash

        for symbol, shares in self.positions.items():
            if symbol in prices:
                position_value = shares * prices[symbol]
                portfolio_value += position_value

        self.portfolio_value = portfolio_value

    def update_returns_history(self, timestamp: pd.Timestamp, 
                              prices: Dict[str, float]) -> None:
        """
        更新收益率历史

        参数:
        timestamp (pd.Timestamp): 时间戳
        prices (Dict[str, float]): 价格字典
        """
        for symbol, price in prices.items():
            if symbol not in self.returns_history:
                self.returns_history[symbol] = pd.Series(dtype=float)

            # 如果有前一个价格，计算收益率
            if len(self.returns_history[symbol]) > 0:
                prev_price = self.returns_history[symbol].iloc[-1]
                if prev_price > 0:
                    returns = (price / prev_price) - 1
                    self.returns_history[symbol][timestamp] = returns
            else:
                self.returns_history[symbol][timestamp] = 0.0

    def generate_signals(self, factors: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        生成交易信号

        参数:
        factors (Dict[str, Dict[str, float]]): 因子字典

        返回:
        Dict[str, float]: 信号字典，键为标的，值为信号强度
        """
        signals = {}

        for symbol, factor_dict in factors.items():
            # 将因子字典转换为Series
            factor_series = pd.Series(factor_dict)

            # 生成信号
            signal = self.signal_generator.generate_signals(factor_series)

            # 取最新的信号
            if len(signal) > 0:
                signals[symbol] = signal.iloc[-1]
            else:
                signals[symbol] = 0.0

        return signals

    def execute_trades(self, timestamp: pd.Timestamp, 
                      signals: Dict[str, float], 
                      prices: Dict[str, float]) -> List[Dict]:
        """
        执行交易

        参数:
        timestamp (pd.Timestamp): 时间戳
        signals (Dict[str, float]): 信号字典
        prices (Dict[str, float]): 价格字典

        返回:
        List[Dict]: 交易记录列表
        """
        executed_trades = []

        # 计算风险指标
        risk_metrics = self.risk_controller.calculate_portfolio_risk(
            self.portfolio_value, self.positions, prices, self.returns_history
        )

        # 检查风险限制
        risk_passed, warnings = self.risk_controller.check_risk_limits(risk_metrics)

        # 如果风险检查不通过，可以选择不交易或减少仓位
        if not risk_passed:
            print(f"风险检查不通过: {warnings}")
            # 这里可以选择减仓或其他风险控制措施

        # 检查追踪止损
        trailing_stops = self.risk_controller.update_trailing_stops(
            self.positions, self.entry_prices, prices
        )

        # 处理止损平仓
        for symbol, triggered in trailing_stops.items():
            if triggered and symbol in self.positions and self.positions[symbol] != 0:
                # 平仓
                trade = self.close_position(timestamp, symbol, prices[symbol], "追踪止损")
                if trade:
                    executed_trades.append(trade)

        # 处理信号交易
        for symbol, signal in signals.items():
            if symbol not in prices:
                continue

            current_price = prices[symbol]
            current_position = self.positions.get(symbol, 0)

            # 检查是否应该平仓
            if current_position != 0 and symbol in self.entry_prices:
                should_close = self.trading_rules.should_close_position(
                    symbol, current_position, self.holding_periods,
                    current_price, self.entry_prices[symbol],
                    self.risk_controller.stop_loss_ratio,
                    self.risk_controller.take_profit_ratio
                )

                if should_close:
                    trade = self.close_position(timestamp, symbol, current_price, "止盈止损")
                    if trade:
                        executed_trades.append(trade)
                    continue

            # 根据信号开仓或调整仓位
            if signal != 0 and abs(signal) > self.signal_generator.signal_threshold:
                # 检查是否可以下单
                if self.trading_rules.can_place_order(timestamp, symbol, current_position, self.holding_periods):
                    # 计算目标仓位
                    target_position = self.trading_rules.calculate_position_size(
                        signal, current_price, self.portfolio_value
                    )

                    # 根据风险指标调整仓位
                    adjusted_position = self.risk_controller.adjust_position_size(
                        target_position, symbol, current_price, self.portfolio_value, risk_metrics
                    )

                    # 计算需要交易的数量
                    trade_size = adjusted_position - current_position

                    if trade_size != 0:
                        # 执行交易
                        trade = self.execute_order(timestamp, symbol, trade_size, current_price)
                        if trade:
                            executed_trades.append(trade)

        return executed_trades

    def execute_order(self, timestamp: pd.Timestamp, 
                     symbol: str, shares: int, price: float) -> Dict:
        """
        执行订单

        参数:
        timestamp (pd.Timestamp): 时间戳
        symbol (str): 交易标的
        shares (int): 交易数量
        price (float): 交易价格

        返回:
        Dict: 交易记录
        """
        # 确定交易方向
        direction = OrderDirection.BUY if shares > 0 else OrderDirection.SELL

        # 应用交易成本
        execution_price = self.trading_rules.apply_trading_costs(price, direction)

        # 计算交易金额
        trade_value = abs(shares) * execution_price

        # 检查资金是否足够
        if direction == OrderDirection.BUY and trade_value > self.cash:
            # 资金不足，调整交易数量
            adjusted_shares = int(self.cash / execution_price)
            if adjusted_shares <= 0:
                return None
            shares = adjusted_shares
            trade_value = abs(shares) * execution_price

        # 更新现金
        if direction == OrderDirection.BUY:
            self.cash -= trade_value
        else:
            self.cash += trade_value

        # 更新持仓
        if symbol not in self.positions:
            self.positions[symbol] = 0
        self.positions[symbol] += shares

        # 更新入场价格（如果是新开仓）
        if (self.positions[symbol] > 0 and direction == OrderDirection.BUY) or \
           (self.positions[symbol] < 0 and direction == OrderDirection.SELL):
            self.entry_prices[symbol] = execution_price

        # 重置持仓周期
        self.holding_periods[symbol] = 0

        # 更新交易计数器
        date_str = timestamp.strftime('%Y-%m-%d')
        if date_str not in self.trading_rules.order_counter:
            self.trading_rules.order_counter[date_str] = 0
        self.trading_rules.order_counter[date_str] += 1

        # 创建交易记录
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'direction': direction.name,
            'shares': abs(shares),
            'price': execution_price,
            'value': trade_value,
            'type': 'OPEN' if self.positions[symbol] != 0 else 'ADJUST'
        }

        # 添加到交易记录
        self.trades.append(trade)

        return trade

    def close_position(self, timestamp: pd.Timestamp, 
                      symbol: str, price: float, reason: str) -> Dict:
        """
        平仓

        参数:
        timestamp (pd.Timestamp): 时间戳
        symbol (str): 交易标的
        price (float): 交易价格
        reason (str): 平仓原因

        返回:
        Dict: 交易记录
        """
        if symbol not in self.positions or self.positions[symbol] == 0:
            return None

        # 获取当前持仓
        current_position = self.positions[symbol]

        # 确定交易方向
        direction = OrderDirection.SELL if current_position > 0 else OrderDirection.BUY

        # 应用交易成本
        execution_price = self.trading_rules.apply_trading_costs(price, direction)

        # 计算交易金额
        trade_value = abs(current_position) * execution_price

        # 更新现金
        if direction == OrderDirection.SELL:
            self.cash += trade_value
        else:
            self.cash -= trade_value

        # 计算收益
        if symbol in self.entry_prices:
            entry_price = self.entry_prices[symbol]
            if current_position > 0:
                profit = (execution_price - entry_price) * current_position
            else:
                profit = (entry_price - execution_price) * abs(current_position)
        else:
            profit = 0

        # 更新持仓
        self.positions[symbol] = 0

        # 清除入场价格和持仓周期
        if symbol in self.entry_prices:
            del self.entry_prices[symbol]
        if symbol in self.holding_periods:
            del self.holding_periods[symbol]

        # 更新交易计数器
        date_str = timestamp.strftime('%Y-%m-%d')
        if date_str not in self.trading_rules.order_counter:
            self.trading_rules.order_counter[date_str] = 0
        self.trading_rules.order_counter[date_str] += 1

        # 创建交易记录
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'direction': direction.name,
            'shares': abs(current_position),
            'price': execution_price,
            'value': trade_value,
            'profit': profit,
            'type': 'CLOSE',
            'reason': reason
        }

        # 添加到交易记录
        self.trades.append(trade)

        return trade

    def calculate_performance_metrics(self) -> Dict:
        """
        计算策略性能指标

        返回:
        Dict: 性能指标字典
        """
        if not self.equity_curve:
            return {}

        # 提取权益曲线数据
        dates = [entry[0] for entry in self.equity_curve]
        equity = [entry[1] for entry in self.equity_curve]

        # 创建权益曲线Series
        equity_series = pd.Series(equity, index=dates)

        # 计算收益率
        returns = equity_series.pct_change().dropna()

        # 计算累积收益率
        cumulative_returns = (1 + returns).cumprod() - 1

        # 计算总收益率
        total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1 if len(equity_series) > 1 else 0

        # 计算年化收益率
        days = (dates[-1] - dates[0]).days
        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0

        # 计算最大回撤
        running_max = equity_series.cummax()
        drawdown = (equity_series / running_max - 1)
        max_drawdown = drawdown.min()

        # 计算夏普比率
        risk_free_rate = 0.02  # 假设无风险利率为2%
        excess_returns = returns - risk_free_rate / 252  # 假设252个交易日
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0

        # 计算索提诺比率
        downside_returns = returns[returns < 0]
        sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0

        # 计算卡玛比率
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # 计算胜率
        profitable_trades = [trade for trade in self.trades if 'profit' in trade and trade['profit'] > 0]
        win_rate = len(profitable_trades) / len(self.trades) if len(self.trades) > 0 else 0

        # 计算盈亏比
        if profitable_trades:
            avg_profit = sum(trade['profit'] for trade in profitable_trades) / len(profitable_trades)
            losing_trades = [trade for trade in self.trades if 'profit' in trade and trade['profit'] <= 0]
            avg_loss = abs(sum(trade['profit'] for trade in losing_trades) / len(losing_trades)) if losing_trades else 1
            profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0
        else:
            profit_loss_ratio = 0

        # 计算最大连续亏损次数
        consecutive_losses = 0
        max_consecutive_losses = 0
        for trade in self.trades:
            if 'profit' in trade:
                if trade['profit'] <= 0:
                    consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                else:
                    consecutive_losses = 0

        # 计算交易频率
        if len(dates) > 1:
            trading_days = (dates[-1] - dates[0]).days
            trades_per_day = len(self.trades) / trading_days if trading_days > 0 else 0
        else:
            trades_per_day = 0

        # 创建性能指标字典
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'max_consecutive_losses': max_consecutive_losses,
            'trades_per_day': trades_per_day,
            'total_trades': len(self.trades),
            'equity_curve': equity_series
        }

        self.performance_metrics = metrics
        return metrics

    def run_backtest(self, timestamps: List[pd.Timestamp], 
                    prices_data: Dict[str, Dict[pd.Timestamp, float]],
                    factors_data: Dict[str, Dict[pd.Timestamp, Dict[str, float]]]) -> Dict:
        """
        运行回测

        参数:
        timestamps (List[pd.Timestamp]): 时间戳列表
        prices_data (Dict[str, Dict[pd.Timestamp, float]]): 价格数据，格式为{symbol: {timestamp: price}}
        factors_data (Dict[str, Dict[pd.Timestamp, Dict[str, float]]]): 因子数据，格式为{symbol: {timestamp: {factor: value}}}

        返回:
        Dict: 回测结果
        """
        # 重置策略状态
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions = {}
        self.entry_prices = {}
        self.holding_periods = {}
        self.returns_history = {}
        self.trades = []
        self.equity_curve = []

        # 遍历每个时间点
        for timestamp in timestamps:
            # 获取当前价格
            current_prices = {}
            for symbol, price_dict in prices_data.items():
                if timestamp in price_dict:
                    current_prices[symbol] = price_dict[timestamp]

            # 获取当前因子
            current_factors = {}
            for symbol, factor_dict in factors_data.items():
                if timestamp in factor_dict:
                    current_factors[symbol] = factor_dict[timestamp]

            # 更新市场数据
            self.update_market_data(timestamp, current_prices, current_factors)

            # 生成信号
            signals = self.generate_signals(current_factors)

            # 执行交易
            executed_trades = self.execute_trades(timestamp, signals, current_prices)

        # 计算性能指标
        performance_metrics = self.calculate_performance_metrics()

        # 创建回测结果
        backtest_result = {
            'initial_capital': self.initial_capital,
            'final_portfolio_value': self.portfolio_value,
            'performance_metrics': performance_metrics,
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'positions': self.positions
        }

        return backtest_result

    def plot_equity_curve(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        绘制权益曲线

        参数:
        figsize (Tuple[int, int]): 图形大小
        """
        import matplotlib.pyplot as plt

        if not self.equity_curve:
            print("没有权益曲线数据")
            return

        # 提取权益曲线数据
        dates = [entry[0] for entry in self.equity_curve]
        equity = [entry[1] for entry in self.equity_curve]

        # 创建权益曲线Series
        equity_series = pd.Series(equity, index=dates)

        # 计算基准线（初始资金）
        benchmark = pd.Series(self.initial_capital, index=dates)

        # 绘制权益曲线
        plt.figure(figsize=figsize)
        plt.plot(equity_series, label='策略权益')
        plt.plot(benchmark, label='基准', linestyle='--')
        plt.title('策略权益曲线')
        plt.xlabel('日期')
        plt.ylabel('权益')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_drawdown(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        绘制回撤曲线

        参数:
        figsize (Tuple[int, int]): 图形大小
        """
        import matplotlib.pyplot as plt

        if not self.equity_curve:
            print("没有权益曲线数据")
            return

        # 提取权益曲线数据
        dates = [entry[0] for entry in self.equity_curve]
        equity = [entry[1] for entry in self.equity_curve]

        # 创建权益曲线Series
        equity_series = pd.Series(equity, index=dates)

        # 计算回撤
        running_max = equity_series.cummax()
        drawdown = (equity_series / running_max - 1) * 100  # 转换为百分比

        # 绘制回撤曲线
        plt.figure(figsize=figsize)
        plt.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        plt.plot(drawdown, color='red', label='回撤')
        plt.title('策略回撤曲线')
        plt.xlabel('日期')
        plt.ylabel('回撤 (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_trade_analysis(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        绘制交易分析图

        参数:
        figsize (Tuple[int, int]): 图形大小
        """
        import matplotlib.pyplot as plt

        if not self.trades:
            print("没有交易记录")
            return

        # 提取交易数据
        trade_dates = [trade['timestamp'] for trade in self.trades if 'profit' in trade]
        profits = [trade['profit'] for trade in self.trades if 'profit' in trade]

        # 创建交易收益Series
        trade_profits = pd.Series(profits, index=trade_dates)

        # 计算累积收益
        cumulative_profits = trade_profits.cumsum()

        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 绘制交易收益
        axes[0, 0].bar(trade_profits.index, trade_profits.values, color=['green' if p > 0 else 'red' for p in trade_profits.values])
        axes[0, 0].set_title('交易收益')
        axes[0, 0].set_xlabel('日期')
        axes[0, 0].set_ylabel('收益')
        axes[0, 0].grid(True)

        # 绘制累积收益
        axes[0, 1].plot(cumulative_profits, label='累积收益')
        axes[0, 1].set_title('累积收益')
        axes[0, 1].set_xlabel('日期')
        axes[0, 1].set_ylabel('累积收益')
        axes[0, 1].grid(True)

        # 绘制收益分布直方图
        axes[1, 0].hist(trade_profits.values, bins=20, color='blue', alpha=0.7)
        axes[1, 0].set_title('收益分布')
        axes[1, 0].set_xlabel('收益')
        axes[1, 0].set_ylabel('频率')
        axes[1, 0].grid(True)

        # 绘制交易类型饼图
        trade_types = [trade['type'] for trade in self.trades]
        type_counts = pd.Series(trade_types).value_counts()
        axes[1, 1].pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('交易类型分布')

        plt.tight_layout()
        plt.show()

    def print_performance_summary(self) -> None:
        """
        打印性能摘要
        """
        if not self.performance_metrics:
            self.calculate_performance_metrics()

        metrics = self.performance_metrics

        print("=" * 50)
        print("策略性能摘要")
        print("=" * 50)
        print(f"初始资金: {self.initial_capital:,.2f}")
        print(f"最终资金: {self.portfolio_value:,.2f}")
        print(f"总收益率: {metrics['total_return']*100:.2f}%")
        print(f"年化收益率: {metrics['annualized_return']*100:.2f}%")
        print(f"最大回撤: {metrics['max_drawdown']*100:.2f}%")
        print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
        print(f"索提诺比率: {metrics['sortino_ratio']:.2f}")
        print(f"卡玛比率: {metrics['calmar_ratio']:.2f}")
        print(f"胜率: {metrics['win_rate']*100:.2f}%")
        print(f"盈亏比: {metrics['profit_loss_ratio']:.2f}")
        print(f"最大连续亏损次数: {metrics['max_consecutive_losses']}")
        print(f"总交易次数: {metrics['total_trades']}")
        print(f"日均交易次数: {metrics['trades_per_day']:.2f}")
        print("=" * 50)