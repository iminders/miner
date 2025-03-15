# -*- coding: utf-8 -*-
"""
信号实时生成模块

用于实时生成交易信号
"""

import time
import threading
import queue
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Callable
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    信号生成器

    用于实时生成交易信号
    """

    def __init__(self, 
                 strategy,
                 factor_queue: queue.Queue,
                 signal_queue: queue.Queue = None,
                 buffer_size: int = 100,
                 generation_interval: float = 0.5):
        """
        初始化信号生成器

        参数:
        strategy: 策略对象
        factor_queue (queue.Queue): 因子队列
        signal_queue (queue.Queue): 信号队列，如果为None则创建新队列
        buffer_size (int): 缓冲区大小
        generation_interval (float): 生成间隔（秒）
        """
        self.strategy = strategy
        self.factor_queue = factor_queue
        self.signal_queue = signal_queue or queue.Queue(maxsize=buffer_size)
        self.buffer_size = buffer_size
        self.generation_interval = generation_interval

        # 因子缓冲区，用于存储最近的因子数据
        self.factor_buffer = {}

        # 运行标志
        self.running = False
        self.thread = None

    def start(self):
        """
        启动信号生成
        """
        if self.running:
            logger.warning("信号生成已经在运行")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        logger.info("信号生成已启动")

    def stop(self):
        """
        停止信号生成
        """
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        logger.info("信号生成已停止")

    def _update_factor_buffer(self, factor_data: Dict):
        """
        更新因子缓冲区

        参数:
        factor_data (Dict): 因子数据
        """
        timestamp = factor_data.get('timestamp')
        data = factor_data.get('data', {})

        for symbol, symbol_factors in data.items():
            # 如果缓冲区中没有该股票的数据，则创建
            if symbol not in self.factor_buffer:
                self.factor_buffer[symbol] = []

            # 添加数据
            symbol_factors['timestamp'] = timestamp
            self.factor_buffer[symbol].append(symbol_factors)

            # 限制缓冲区大小
            if len(self.factor_buffer[symbol]) > self.buffer_size:
                self.factor_buffer[symbol] = self.factor_buffer[symbol][-self.buffer_size:]

    def _generate_signals(self) -> Dict:
        """
        生成信号

        返回:
        Dict: 信号数据
        """
        signals = {}

        # 遍历所有股票
        for symbol, factor_list in self.factor_buffer.items():
            if not factor_list:
                continue

            # 获取最新因子
            latest_factors = factor_list[-1]

            try:
                # 调用策略生成信号
                signal = self.strategy.generate_signal(symbol, latest_factors)

                # 存储信号
                if signal is not None:
                    signals[symbol] = {
                        'timestamp': latest_factors.get('timestamp', datetime.now()),
                        'signal': signal,
                        'factors': latest_factors
                    }
            except Exception as e:
                logger.error(f"生成信号时出错: {e}")

        return signals

    def _run(self):
        """
        运行信号生成
        """
        last_generation_time = datetime.now()

        while self.running:
            try:
                # 获取因子数据
                factor_data = self.factor_queue.get(block=True, timeout=0.1)

                # 更新因子缓冲区
                self._update_factor_buffer(factor_data)

                # 检查是否需要生成信号
                current_time = datetime.now()
                if (current_time - last_generation_time).total_seconds() >= self.generation_interval:
                    # 生成信号
                    signals = self._generate_signals()

                    # 将信号放入队列
                    if signals:
                        try:
                            self.signal_queue.put({
                                'type': 'signal_data',
                                'timestamp': current_time,
                                'data': signals
                            }, block=False)
                        except queue.Full:
                            logger.warning("信号队列已满，丢弃数据")

                    # 更新最后生成时间
                    last_generation_time = current_time
            except queue.Empty:
                # 队列为空，继续等待
                pass
            except Exception as e:
                logger.error(f"信号生成时出错: {e}")
                time.sleep(1)  # 出错时暂停一下，避免CPU占用过高