import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    回测引擎类，用于回测因子策略
    """
    
    def __init__(self, initial_capital=1000000.0):
        """
        初始化回测引擎
        
        参数:
        initial_capital (float): 初始资金
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # 持仓 {symbol: quantity}
        self.position_history = []  # 持仓历史
        self.trade_history = []  # 交易历史
        self.equity_curve = []  # 权益曲线
        self.returns = []  # 收益率序列
        self.dates = []  # 日期序列
        
        logger.info(f"初始化回测引擎，初始资金: {initial_capital}")
    
    def run_backtest(self, data, factor, strategy, start_date=None, end_date=None, 
                    commission=0.0003, slippage=0.0001, rebalance_freq='M'):
        """
        运行回测
        
        参数:
        data (pd.DataFrame): 股票数据，包含open, high, low, close, volume等列
        factor (pd.Series): 因子值
        strategy (function): 策略函数，接收因子值和数据，返回目标仓位
        start_date (str): 回测开始日期，格式为'YYYY-MM-DD'
        end_date (str): 回测结束日期，格式为'YYYY-MM-DD'
        commission (float): 手续费率
        slippage (float): 滑点率
        rebalance_freq (str): 再平衡频率，'D'表示每日，'W'表示每周，'M'表示每月
        
        返回:
        dict: 回测结果
        """
        logger.info(f"开始回测，回测区间: {start_date} 至 {end_date}，再平衡频率: {rebalance_freq}")
        
        # 初始化回测状态
        self.current_capital = self.initial_capital
        self.positions = {}
        self.position_history = []
        self.trade_history = []
        self.equity_curve = [self.initial_capital]
        self.returns = [0.0]
        self.dates = []
        
        # 确保数据和因子的索引对齐
        common_index = data.index.intersection(factor.index)
        data = data.loc[common_index]
        factor = factor.loc[common_index]
        
        # 过滤日期范围
        if start_date:
            data = data[data.index >= start_date]
            factor = factor[factor.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
            factor = factor[factor.index <= end_date]
        
        # 确定再平衡日期
        if rebalance_freq == 'D':
            rebalance_dates = data.index
        elif rebalance_freq == 'W':
            rebalance_dates = data.resample('W').last().index
        elif rebalance_freq == 'M':
            rebalance_dates = data.resample('M').last().index
        else:
            raise ValueError(f"不支持的再平衡频率: {rebalance_freq}")
        
        # 遍历每个交易日
        prev_date = None
        for date in data.index:
            current_data = data.loc[date]
            current_factor = factor.loc[date]
            
            # 记录日期
            self.dates.append(date)
            
            # 计算当前持仓市值
            portfolio_value = self.current_capital
            for symbol, quantity in self.positions.items():
                if symbol in current_data:
                    portfolio_value += quantity * current_data[symbol]['close']
            
            # 记录权益曲线
            self.equity_curve.append(portfolio_value)
            
            # 计算收益率
            if prev_date:
                daily_return = (portfolio_value / self.equity_curve[-2]) - 1
                self.returns.append(daily_return)
            
            # 检查是否为再平衡日期
            if date in rebalance_dates:
                # 调用策略函数获取目标仓位
                target_positions = strategy(current_factor, current_data)
                
                # 执行交易以达到目标仓位
                self._execute_trades(target_positions, current_data, date, commission, slippage)
            
            # 记录持仓
            self.position_history.append(self.positions.copy())
            
            prev_date = date
        
        # 计算回测结果
        results = self._calculate_results()
        
        logger.info(f"回测完成，最终资金: {self.equity_curve[-1]:.2f}，总收益率: {results['total_return']:.2%}")
        
        return results
    
    def _execute_trades(self, target_positions, current_data, date, commission, slippage):
        """
        执行交易以达到目标仓位
        
        参数:
        target_positions (dict): 目标仓位 {symbol: quantity}
        current_data (pd.Series): 当前交易日数据
        date (datetime): 交易日期
        commission (float): 手续费率
        slippage (float): 滑点率
        """
        # 计算需要买入和卖出的股票
        for symbol, target_qty in target_positions.items():
            current_qty = self.positions.get(symbol, 0)
            
            # 计算需要交易的数量
            trade_qty = target_qty - current_qty
            
            if trade_qty != 0:
                # 获取交易价格（考虑滑点）
                price = current_data[symbol]['close']
                if trade_qty > 0:  # 买入
                    trade_price = price * (1 + slippage)
                else:  # 卖出
                    trade_price = price * (1 - slippage)
                
                # 计算交易成本
                trade_value = abs(trade_qty) * trade_price
                trade_cost = trade_value * commission
                
                # 执行交易
                if trade_qty > 0:  # 买入
                    if self.current_capital >= (trade_value + trade_cost):
                        self.current_capital -= (trade_value + trade_cost)
                        self.positions[symbol] = self.positions.get(symbol, 0) + trade_qty
                        
                        # 记录交易
                        self.trade_history.append({
                            'date': date,
                            'symbol': symbol,
                            'action': 'buy',
                            'quantity': trade_qty,
                            'price': trade_price,
                            'cost': trade_cost
                        })
                    else:
                        logger.warning(f"资金不足，无法买入 {symbol}，需要 {trade_value + trade_cost}，当前资金 {self.current_capital}")
                else:  # 卖出
                    self.current_capital += (trade_value - trade_cost)
                    self.positions[symbol] = self.positions.get(symbol, 0) + trade_qty  # trade_qty为负数
                    
                    # 如果持仓为0，则从持仓字典中删除
                    if self.positions[symbol] == 0:
                        del self.positions[symbol]
                    
                    # 记录交易
                    self.trade_history.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'sell',
                        'quantity': abs(trade_qty),
                        'price': trade_price,
                        'cost': trade_cost
                    })
    
    def _calculate_results(self):
        """
        计算回测结果
        
        返回:
        dict: 回测结果
        """
        # 转换为pandas Series
        equity_curve = pd.Series(self.equity_curve, index=self.dates)
        returns = pd.Series(self.returns, index=self.dates)
        
        # 计算各种指标
        total_return = (equity_curve[-1] / equity_curve[0]) - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # 计算最大回撤
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        # 计算夏普比率
        risk_free_rate = 0.02  # 假设无风险利率为2%
        sharpe_ratio = (annual_return - risk_free_rate) / (returns.std() * np.sqrt(252))
        
        # 计算索提诺比率
        downside_returns = returns[returns < 0]
        sortino_ratio = (annual_return - risk_free_rate) / (downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 else np.nan
        
        # 计算胜率
        win_trades = sum(1 for trade in self.trade_history if trade['action'] == 'sell' and trade['price'] > trade['price'])
        total_trades = sum(1 for trade in self.trade_history if trade['action'] == 'sell')
        win_rate = win_trades / total_trades if total_trades > 0 else 0
        
        # 汇总结果
        results = {
            'equity_curve': equity_curve,
            'returns': returns,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'win_rate': win_rate,
            'trade_history': self.trade_history,
            'position_history': self.position_history
        }
        
        return results
    
    def plot_results(self, results, figsize=(15, 12)):
        """
        绘制回测结果
        
        参数:
        results (dict): 回测结果
        figsize (tuple): 图表大小
        """
        plt.figure(figsize=figsize)
        
        # 绘制权益曲线
        plt.subplot(3, 1, 1)
        results['equity_curve'].plot()
        plt.title('权益曲线')
        plt.grid(True)
        
        # 绘制收益率
        plt.subplot(3, 1, 2)
        results['returns'].plot()
        plt.title('每日收益率')
        plt.grid(True)
        
        # 绘制回撤
        plt.subplot(3, 1, 3)
        cumulative_returns = (1 + results['returns']).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        drawdown.plot()
        plt.title('回撤')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # 打印回测统计结果
        print(f"总收益率: {results['total_return']:.2%}")
        print(f"年化收益率: {results['annual_return']:.2%}")
        print(f"最大回撤: {results['max_drawdown']:.2%}")
        print(f"夏普比率: {results['sharpe_ratio']:.2f}")
        print(f"索提诺比率: {results['sortino_ratio']:.2f}")
        print(f"胜率: {results['win_rate']:.2%}")