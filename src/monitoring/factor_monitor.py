# -*- coding: utf-8 -*-
"""
因子表现监控模块

监控因子的表现指标
"""

import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Callable
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

from src.monitoring.monitor_base import Monitor
from src.evaluation.factor_evaluation import calculate_ic, calculate_factor_returns

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FactorMonitor(Monitor):
    """
    因子表现监控

    监控因子的IC值、收益率等指标
    """

    def __init__(self, 
                 name: str,
                 factors: Dict[str, pd.DataFrame],
                 returns: Dict[str, pd.DataFrame],
                 ic_threshold: float = 0.02,
                 ic_decay_threshold: float = 0.5,
                 return_threshold: float = 0.0,
                 check_interval: float = 3600.0,
                 save_dir: str = None):
        """
        初始化因子表现监控

        参数:
        name (str): 监控名称
        factors (Dict[str, pd.DataFrame]): 因子数据，格式为 {factor_name: factor_df}
        returns (Dict[str, pd.DataFrame]): 收益率数据，格式为 {symbol: returns_df}
        ic_threshold (float): IC阈值，低于此值触发警报
        ic_decay_threshold (float): IC衰减阈值，衰减超过此比例触发警报
        return_threshold (float): 收益率阈值，低于此值触发警报
        check_interval (float): 检查间隔（秒）
        save_dir (str): 保存目录
        """
        super().__init__(name, check_interval, save_dir)
        self.factors = factors
        self.returns = returns
        self.ic_threshold = ic_threshold
        self.ic_decay_threshold = ic_decay_threshold
        self.return_threshold = return_threshold

        # 初始化监控数据
        self.data = {
            'ic_history': {},
            'return_history': {},
            'alerts': []
        }

    def check(self):
        """
        检查因子表现
        """
        logger.info(f"检查因子表现: {self.name}")

        # 获取当前时间
        current_time = datetime.now()

        # 计算每个因子的IC值
        for factor_name, factor_df in self.factors.items():
            try:
                # 计算IC值
                ic_df = calculate_ic(factor_df, self.returns)

                # 获取最新的IC值
                latest_ic = ic_df.iloc[-1]['ic'] if not ic_df.empty else np.nan

                # 记录IC历史
                if factor_name not in self.data['ic_history']:
                    self.data['ic_history'][factor_name] = []

                self.data['ic_history'][factor_name].append({
                    'timestamp': current_time,
                    'ic': latest_ic
                })

                # 检查IC值是否低于阈值
                if not np.isnan(latest_ic) and abs(latest_ic) < self.ic_threshold:
                    self.trigger_alert({
                        'type': 'low_ic',
                        'factor': factor_name,
                        'ic': latest_ic,
                        'threshold': self.ic_threshold,
                        'message': f"因子 {factor_name} 的IC值 ({latest_ic:.4f}) 低于阈值 ({self.ic_threshold})"
                    })

                # 检查IC值是否衰减
                if len(self.data['ic_history'][factor_name]) > 1:
                    previous_ic = self.data['ic_history'][factor_name][-2]['ic']
                    if not np.isnan(previous_ic) and not np.isnan(latest_ic) and previous_ic != 0:
                        decay_ratio = (previous_ic - latest_ic) / abs(previous_ic)
                        if decay_ratio > self.ic_decay_threshold:
                            self.trigger_alert({
                                'type': 'ic_decay',
                                'factor': factor_name,
                                'previous_ic': previous_ic,
                                'current_ic': latest_ic,
                                'decay_ratio': decay_ratio,
                                'threshold': self.ic_decay_threshold,
                                'message': f"因子 {factor_name} 的IC值从 {previous_ic:.4f} 衰减到 {latest_ic:.4f}，衰减比例 {decay_ratio:.2f} 超过阈值 ({self.ic_decay_threshold})"
                            })
            except Exception as e:
                logger.error(f"计算因子 {factor_name} 的IC值时出错: {e}")

        # 计算每个因子的收益率
        for factor_name, factor_df in self.factors.items():
            try:
                # 计算因子收益率
                returns_df = calculate_factor_returns(factor_df, self.returns)

                # 获取最新的收益率
                latest_return = returns_df.iloc[-1]['return'] if not returns_df.empty else np.nan

                # 记录收益率历史
                if factor_name not in self.data['return_history']:
                    self.data['return_history'][factor_name] = []

                self.data['return_history'][factor_name].append({
                    'timestamp': current_time,
                    'return': latest_return
                })

                # 检查收益率是否低于阈值
                if not np.isnan(latest_return) and latest_return < self.return_threshold:
                    self.trigger_alert({
                        'type': 'low_return',
                        'factor': factor_name,
                        'return': latest_return,
                        'threshold': self.return_threshold,
                        'message': f"因子 {factor_name} 的收益率 ({latest_return:.4f}) 低于阈值 ({self.return_threshold})"
                    })
            except Exception as e:
                logger.error(f"计算因子 {factor_name} 的收益率时出错: {e}")

    def plot_ic_history(self, factor_name: str = None, figsize: tuple = (12, 6)):
        """
        绘制IC历史

        参数:
        factor_name (str): 因子名称，如果为None则绘制所有因子
        figsize (tuple): 图形大小
        """
        plt.figure(figsize=figsize)

        if factor_name:
            # 绘制单个因子的IC历史
            if factor_name in self.data['ic_history']:
                ic_history = self.data['ic_history'][factor_name]
                df = pd.DataFrame(ic_history)
                plt.plot(df['timestamp'], df['ic'], label=factor_name)
                plt.axhline(y=self.ic_threshold, color='r', linestyle='--', label=f'阈值 ({self.ic_threshold})')
                plt.axhline(y=-self.ic_threshold, color='r', linestyle='--')
                plt.title(f"因子 {factor_name} 的IC历史")
            else:
                logger.warning(f"因子 {factor_name} 没有IC历史数据")
        else:
            # 绘制所有因子的IC历史
            for factor_name, ic_history in self.data['ic_history'].items():
                df = pd.DataFrame(ic_history)
                plt.plot(df['timestamp'], df['ic'], label=factor_name)

            plt.axhline(y=self.ic_threshold, color='r', linestyle='--', label=f'阈值 ({self.ic_threshold})')
            plt.axhline(y=-self.ic_threshold, color='r', linestyle='--')
            plt.title("所有因子的IC历史")

        plt.xlabel("时间")
        plt.ylabel("IC值")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_return_history(self, factor_name: str = None, figsize: tuple = (12, 6)):
        """
        绘制收益率历史

        参数:
        factor_name (str): 因子名称，如果为None则绘制所有因子
        figsize (tuple): 图形大小
        """
        plt.figure(figsize=figsize)

        if factor_name:
            # 绘制单个因子的收益率历史
            if factor_name in self.data['return_history']:
                return_history = self.data['return_history'][factor_name]
                df = pd.DataFrame(return_history)
                plt.plot(df['timestamp'], df['return'], label=factor_name)
                plt.axhline(y=self.return_threshold, color='r', linestyle='--', label=f'阈值 ({self.return_threshold})')
                plt.title(f"因子 {factor_name} 的收益率历史")
            else:
                logger.warning(f"因子 {factor_name} 没有收益率历史数据")
        else:
            # 绘制所有因子的收益率历史
            for factor_name, return_history in self.data['return_history'].items():
                df = pd.DataFrame(return_history)
                plt.plot(df['timestamp'], df['return'], label=factor_name)

            plt.axhline(y=self.return_threshold, color='r', linestyle='--', label=f'阈值 ({self.return_threshold})')
            plt.title("所有因子的收益率历史")

        plt.xlabel("时间")
        plt.ylabel("收益率")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()