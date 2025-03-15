# -*- coding: utf-8 -*-
"""
实时系统模块

整合数据接入、因子计算和信号生成
"""

import time
import threading
import queue
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Callable
from datetime import datetime, timedelta

from src.realtime.data_feed import DataFeed, SimulatedDataFeed, WebSocketDataFeed, RESTDataFeed
from src.realtime.factor_calculator import FactorCalculator
from src.realtime.signal_generator import SignalGenerator
from src.strategy.strategy import Strategy

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealtimeSystem:
    """
    实时系统

    整合数据接入、因子计算和信号生成
    """

    def __init__(self, 
                 strategy: Strategy,
                 symbols: List[str],
                 factor_modules: List[str],
                 data_feed_type: str = 'simulated',
                 data_feed_params: Dict = None,
                 calculation_interval: float = 0.1,
                 generation_interval: float = 0.5,
                 buffer_size: int = 1000):
        """
        初始化实时系统

        参数:
        strategy (Strategy): 策略对象
        symbols (List[str]): 股票代码列表
        factor_modules (List[str]): 因子模块列表
        data_feed_type (str): 数据接入类型，可选值：'simulated', 'websocket', 'rest'
        data_feed_params (Dict): 数据接入参数
        calculation_interval (float): 因子计算间隔（秒）
        generation_interval (float): 信号生成间隔（秒）
        buffer_size (int): 缓冲区大小
        """
        self.strategy = strategy
        self.symbols = symbols
        self.factor_modules = factor_modules
        self.data_feed_type = data_feed_type
        self.data_feed_params = data_feed_params or {}
        self.calculation_interval = calculation_interval
        self.generation_interval = generation_interval
        self.buffer_size = buffer_size

        # 创建队列
        self.data_queue = queue.Queue(maxsize=buffer_size)
        self.factor_queue = queue.Queue(maxsize=buffer_size)
        self.signal_queue = queue.Queue(maxsize=buffer_size)

        # 创建数据接入
        self.data_feed = self._create_data_feed()

        # 创建因子计算器
        self.factor_calculator = FactorCalculator(
            factor_modules=factor_modules,
            data_queue=self.data_queue,
            factor_queue=self.factor_queue,
            buffer_size=buffer_size,
            calculation_interval=calculation_interval
        )

        # 创建信号生成器
        self.signal_generator = SignalGenerator(
            strategy=strategy,
            factor_queue=self.factor_queue,
            signal_queue=self.signal_queue,
            buffer_size=buffer_size,
            generation_interval=generation_interval
        )

        # 信号处理器
        self.signal_handlers = []

        # 信号处理线程
        self.signal_thread = None
        self.running = False

    def _create_data_feed(self) -> DataFeed:
        """
        创建数据接入

        返回:
        DataFeed: 数据接入对象
        """
        if self.data_feed_type == 'simulated':
            return SimulatedDataFeed(
                symbols=self.symbols,
                **self.data_feed_params
            )
        elif self.data_feed_type == 'websocket':
            return WebSocketDataFeed(
                symbols=self.symbols,
                **self.data_feed_params
            )
        elif self.data_feed_type == 'rest':
            return RESTDataFeed(
                symbols=self.symbols,
                **self.data_feed_params
            )
        else:
            raise ValueError(f"不支持的数据接入类型: {self.data_feed_type}")

    def add_signal_handler(self, handler: Callable):
        """
        添加信号处理器

        参数:
        handler (Callable): 信号处理函数，接收信号数据作为参数
        """
        self.signal_handlers.append(handler)

    def _process_signals(self):
        """
        处理信号
        """
        while self.running:
            try:
                # 获取信号
                signal_data = self.signal_queue.get(block=True, timeout=0.1)

                # 调用信号处理器
                for handler in self.signal_handlers:
                    try:
                        handler(signal_data)
                    except Exception as e:
                        logger.error(f"处理信号时出错: {e}")
            except queue.Empty:
                # 队列为空，继续等待
                pass
            except Exception as e:
                logger.error(f"处理信号线程出错: {e}")
                time.sleep(1)

    def start(self):
        """
        启动实时系统
        """
        logger.info("启动实时系统...")

        # 设置运行标志
        self.running = True

        # 启动信号处理线程
        self.signal_thread = threading.Thread(target=self._process_signals)
        self.signal_thread.daemon = True
        self.signal_thread.start()

        # 启动各组件
        self.data_feed.start()
        self.factor_calculator.start()
        self.signal_generator.start()

        logger.info("实时系统已启动")

    def stop(self):
        """
        停止实时系统
        """
        logger.info("停止实时系统...")

        # 设置运行标志
        self.running = False

        # 停止各组件
        self.signal_generator.stop()
        self.factor_calculator.stop()
        self.data_feed.stop()

        # 等待信号处理线程结束
        if self.signal_thread and self.signal_thread.is_alive():
            self.signal_thread.join(timeout=5)

        logger.info("实时系统已停止")