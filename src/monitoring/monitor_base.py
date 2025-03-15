# -*- coding: utf-8 -*-
"""
监控系统基础模块

提供监控系统的基础类和功能
"""

import time
import threading
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Callable
from datetime import datetime, timedelta
import json
import os
import pickle

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Monitor:
    """
    监控基类

    提供监控的基本接口和功能
    """

    def __init__(self, 
                 name: str,
                 check_interval: float = 60.0,
                 save_dir: str = None):
        """
        初始化监控

        参数:
        name (str): 监控名称
        check_interval (float): 检查间隔（秒）
        save_dir (str): 保存目录
        """
        self.name = name
        self.check_interval = check_interval
        self.save_dir = save_dir or os.path.join(os.getcwd(), 'monitor_data')

        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)

        # 监控数据
        self.data = {}

        # 警报处理器
        self.alert_handlers = []

        # 运行标志
        self.running = False
        self.thread = None

    def add_alert_handler(self, handler: Callable):
        """
        添加警报处理器

        参数:
        handler (Callable): 警报处理函数，接收警报数据作为参数
        """
        self.alert_handlers.append(handler)

    def trigger_alert(self, alert_data: Dict):
        """
        触发警报

        参数:
        alert_data (Dict): 警报数据
        """
        # 添加基本信息
        alert_data.update({
            'monitor': self.name,
            'timestamp': datetime.now(),
        })

        # 调用警报处理器
        for handler in self.alert_handlers:
            try:
                handler(alert_data)
            except Exception as e:
                logger.error(f"处理警报时出错: {e}")

    def check(self):
        """
        检查监控指标（子类实现）
        """
        raise NotImplementedError("子类必须实现check方法")

    def save_data(self):
        """
        保存监控数据
        """
        try:
            # 构建文件路径
            file_path = os.path.join(self.save_dir, f"{self.name}_data.pkl")

            # 保存数据
            with open(file_path, 'wb') as f:
                pickle.dump(self.data, f)

            logger.debug(f"监控数据已保存到 {file_path}")
        except Exception as e:
            logger.error(f"保存监控数据时出错: {e}")

    def load_data(self):
        """
        加载监控数据
        """
        try:
            # 构建文件路径
            file_path = os.path.join(self.save_dir, f"{self.name}_data.pkl")

            # 检查文件是否存在
            if not os.path.exists(file_path):
                logger.warning(f"监控数据文件不存在: {file_path}")
                return

            # 加载数据
            with open(file_path, 'rb') as f:
                self.data = pickle.load(f)

            logger.debug(f"监控数据已从 {file_path} 加载")
        except Exception as e:
            logger.error(f"加载监控数据时出错: {e}")

    def _run(self):
        """
        运行监控
        """
        while self.running:
            try:
                # 检查监控指标
                self.check()

                # 保存数据
                self.save_data()

                # 等待下一次检查
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"监控运行时出错: {e}")
                time.sleep(10)  # 出错时暂停一下，避免CPU占用过高

    def start(self):
        """
        启动监控
        """
        if self.running:
            logger.warning(f"监控 {self.name} 已经在运行")
            return

        # 加载历史数据
        self.load_data()

        # 设置运行标志
        self.running = True

        # 启动监控线程
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

        logger.info(f"监控 {self.name} 已启动")

    def stop(self):
        """
        停止监控
        """
        if not self.running:
            logger.warning(f"监控 {self.name} 未在运行")
            return

        # 设置运行标志
        self.running = False

        # 等待线程结束
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

        # 保存数据
        self.save_data()

        logger.info(f"监控 {self.name} 已停止")