# -*- coding: utf-8 -*-
"""
交易规则设定模块

定义交易执行的规则和约束
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple
from enum import Enum


class OrderType(Enum):
    """订单类型枚举"""
    MARKET = 1  # 市价单
    LIMIT = 2   # 限价单
    STOP = 3    # 止损单
    STOP_LIMIT = 4  # 止损限价单


class OrderDirection(Enum):
    """订单方向枚举"""
    BUY = 1     # 买入
    SELL = -1   # 卖出


class TradingRules:
    """
    交易规则设定

    定义交易执行的规则和约束
    """

    def __init__(self, 
                 max_position: int = 1,
                 position_sizing: str = 'fixed',
                 order_type: OrderType = OrderType.MARKET,
                 max_orders_per_day: int = 10,
                 min_holding_period: int = 5,
                 max_holding_period: int = 100,
                 trading_hours: List[Tuple[str, str]] = None,
                 trading_fee: float = 0.0003,
                 slippage: float = 0.0001):
        """
        初始化交易规则

        参数:
        max_position (int): 最大持仓数量
        position_sizing (str): 仓位大小策略，可选'fixed', 'percent', 'kelly'
        order_type (OrderType): 订单类型
        max_orders_per_day (int): 每日最大订单数量
        min_holding_period (int): 最小持仓周期
        max_holding_period (int): 最大持仓周期
        trading_hours (List[Tuple[str, str]]): 交易时间段列表，每个元素为(开始时间, 结束时间)
        trading_fee (float): 交易费用比例
        slippage (float): 滑点比例
        """
        self.max_position = max_position
        self.position_sizing = position_sizing
        self.order_type = order_type
        self.max_orders_per_day = max_orders_per_day
        self.min_holding_period = min_holding_period
        self.max_holding_period = max_holding_period

        # 默认交易时间为9:30-11:30, 13:00-15:00
        if trading_hours is None:
            self.trading_hours = [('09:30', '11:30'), ('13:00', '15:00')]
        else:
            self.trading_hours = trading_hours

        self.trading_fee = trading_fee
        self.slippage = slippage

        # 交易计数器
        self.order_counter = {}
        self.position_start_time = {}

    def is_trading_time(self, timestamp: pd.Timestamp) -> bool:
        """
        判断是否为交易时间

        参数:
        timestamp (pd.Timestamp): 时间戳

        返回:
        bool: 是否为交易时间
        """
        time_str = timestamp.strftime('%H:%M')

        for start_time, end_time in self.trading_hours:
            if start_time <= time_str <= end_time:
                return True

        return False

    def can_place_order(self, timestamp: pd.Timestamp, symbol: str, 
                       current_position: int, holding_periods: Dict[str, int]) -> bool:
        """
        判断是否可以下单

        参数:
        timestamp (pd.Timestamp): 时间戳
        symbol (str): 交易标的
        current_position (int): 当前持仓
        holding_periods (Dict[str, int]): 持仓周期字典

        返回:
        bool: 是否可以下单
        """
        # 检查是否为交易时间
        if not self.is_trading_time(timestamp):
            return False

        # 检查当日订单数量是否超过限制
        date_str = timestamp.strftime('%Y-%m-%d')
        if date_str not in self.order_counter:
            self.order_counter[date_str] = 0

        if self.order_counter[date_str] >= self.max_orders_per_day:
            return False

        # 检查持仓是否超过最大持仓
        if abs(current_position) >= self.max_position:
            return False

        # 检查最小持仓周期
        if symbol in holding_periods and holding_periods[symbol] < self.min_holding_period:
            return False

        return True

    def calculate_position_size(self, signal: float, price: float, 
                              portfolio_value: float) -> int:
        """
        计算仓位大小

        参数:
        signal (float): 信号强度
        price (float): 当前价格
        portfolio_value (float): 投资组合价值

        返回:
        int: 仓位大小（股数）
        """
        if self.position_sizing == 'fixed':
            # 固定仓位
            return self.max_position if signal > 0 else -self.max_position

        elif self.position_sizing == 'percent':
            # 百分比仓位
            percent = min(1.0, abs(signal))
            position_value = portfolio_value * percent
            shares = int(position_value / price)
            return shares if signal > 0 else -shares

        elif self.position_sizing == 'kelly':
            # 凯利公式仓位
            # 简化版：假设信号强度代表胜率
            win_rate = min(0.9, max(0.1, (abs(signal) + 1) / 2))
            edge = 2 * win_rate - 1  # 简化的边际优势

            if edge <= 0:
                return 0

            # 凯利比例
            kelly_fraction = edge / 1.0  # 假设赔率为1:1

            # 限制最大仓位
            kelly_fraction = min(0.5, kelly_fraction)  # 半凯利

            position_value = portfolio_value * kelly_fraction
            shares = int(position_value / price)
            return shares if signal > 0 else -shares

        else:
            return self.max_position if signal > 0 else -self.max_position

    def apply_trading_costs(self, price: float, direction: OrderDirection) -> float:
        """
        应用交易成本

        参数:
        price (float): 价格
        direction (OrderDirection): 交易方向

        返回:
        float: 考虑交易成本后的价格
        """
        # 应用滑点
        if direction == OrderDirection.BUY:
            execution_price = price * (1 + self.slippage)
        else:
            execution_price = price * (1 - self.slippage)

        # 应用交易费用
        if direction == OrderDirection.BUY:
            execution_price *= (1 + self.trading_fee)
        else:
            execution_price *= (1 - self.trading_fee)

        return execution_price

    def should_close_position(self, symbol: str, current_position: int, 
                            holding_periods: Dict[str, int], 
                            current_price: float, entry_price: float,
                            stop_loss: float = 0.05, take_profit: float = 0.1) -> bool:
        """
        判断是否应该平仓

        参数:
        symbol (str): 交易标的
        current_position (int): 当前持仓
        holding_periods (Dict[str, int]): 持仓周期字典
        current_price (float):