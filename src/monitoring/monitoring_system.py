# -*- coding: utf-8 -*-
"""
监控系统模块

整合各种监控和报警
"""

import logging
from typing import Dict, List, Union, Optional, Callable
import time
import threading

from src.monitoring.monitor_base import Monitor
from src.monitoring.factor_monitor import FactorMonitor
from src.monitoring.strategy_monitor import StrategyMonitor
from src.monitoring.anomaly_detector import AnomalyDetector
from src.monitoring.alert_handlers import AlertHandler, ConsoleAlertHandler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MonitoringSystem:
    """
    监控系统

    整合各种监控和报警
    """

    def __init__(self):
        """
        初始化监控系统
        """
        self.monitors = {}
        self.alert_handlers = {}

    def add_monitor(self, monitor: Monitor):
        """
        添加监控

        参数:
        monitor (Monitor): 监控对象
        """
        if monitor.name in self.monitors:
            logger.warning(f"监控 {monitor.name} 已存在，将被替换")

        self.monitors[monitor.name] = monitor
        logger.info(f"已添加监控: {monitor.name}")

    def add_alert_handler(self, handler: AlertHandler):
        """
        添加报警处理器

        参数:
        handler (AlertHandler): 报警处理器
        """
        if handler.name in self.alert_handlers:
            logger.warning(f"报警处理器 {handler.name} 已存在，将被替换")

        self.alert_handlers[handler.name] = handler
        logger.info(f"已添加报警处理器: {handler.name}")

    def register_alert_handlers(self):
        """
        注册报警处理器到所有监控
        """
        for monitor in self.monitors.values():
            for handler in self.alert_handlers.values():
                monitor.add_alert_handler(handler.handle)

    def start_all(self):
        """
        启动所有监控
        """
        # 注册报警处理器
        self.register_alert_handlers()

        # 启动所有监控
        for name, monitor in self.monitors.items():
            try:
                monitor.start()
                logger.info(f"已启动监控: {name}")
            except Exception as e:
                logger.error(f"启动监控 {name} 时出错: {e}")

    def stop_all(self):
        """
        停止所有监控
        """
        for name, monitor in self.monitors.items():
            try:
                monitor.stop()
                logger.info(f"已停止监控: {name}")
            except Exception as e:
                logger.error(f"停止监控 {name} 时出错: {e}")

    def get_monitor(self, name: str) -> Optional[Monitor]:
        """
        获取监控

        参数:
        name (str): 监控名称

        返回:
        Optional[Monitor]: 监控对象
        """
        return self.monitors.get(name)