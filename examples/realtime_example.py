# -*- coding: utf-8 -*-
"""
实时系统示例
"""

import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.realtime.realtime_system import RealtimeSystem
from src.strategy.strategy import Strategy

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleStrategy(Strategy):
    """
    简单策略示例
    """

    def __init__(self, ma_short=5, ma_long=20):
        """
        初始化策略

        参数:
        ma_short (int): 短期均线周期
        ma_long (int): 长期均线周期
        """
        super().__init__()
        self.ma_short = ma_short
        self.ma_long = ma_long

    def generate_signal(self, symbol, factors):
        """
        生成信号

        参数:
        symbol (str): 股票代码
        factors (Dict): 因子数据

        返回:
        int: 信号，1表示买入，-1表示卖出，0表示持仓不变
        """
        # 检查是否有均线因子
        if 'ma_short' not in factors or 'ma_long' not in factors:
            return 0

        ma_short = factors['ma_short']
        ma_long = factors['ma_long']

        # 生成信号
        if ma_short > ma_long:
            return 1  # 买入信号
        elif ma_short < ma_long:
            return -1  # 卖出信号
        else:
            return 0  # 持仓不变


def signal_handler(signal_data):
    """
    信号处理函数

    参数:
    signal_data (Dict): 信号数据
    """
    timestamp = signal_data.get('timestamp')
    data = signal_data.get('data', {})

    for symbol, signal_info in data.items():
        signal = signal_info.get('signal')

        if signal == 1:
            logger.info(f"[{timestamp}] 买入信号: {symbol}")
        elif signal == -1:
            logger.info(f"[{timestamp}] 卖出信号: {symbol}")


def main():
    """
    主函数
    """
    # 创建策略
    strategy = SimpleStrategy(ma_short=5, ma_long=20)

    # 股票代码列表
    symbols = ['000001.SZ', '600000.SH', '300059.SZ']

    # 因子模块列表
    factor_modules = ['src.features.basic_features']

    # 创建实时系统
    system = RealtimeSystem(
        strategy=strategy,
        symbols=symbols,
        factor_modules=factor_modules,
        data_feed_type='simulated',
        data_feed_params={
            'interval': 1.0,
            'replay_speed': 10.0
        },
        calculation_interval=0.5,
        generation_interval=1.0
    )

    # 添加信号处理器
    system.add_signal_handler(signal_handler)

    try:
        # 启动实时系统
        system.start()

        # 运行一段时间
        logger.info("系统将运行60秒...")
        time.sleep(60)
    except KeyboardInterrupt:
        logger.info("用户中断")
    finally:
        # 停止实时系统
        system.stop()


if __name__ == "__main__":
    main()