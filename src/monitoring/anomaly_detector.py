# -*- coding: utf-8 -*-
"""
异常检测与报警模块

检测异常并发送报警
"""

import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Callable
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.monitoring.monitor_base import Monitor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnomalyDetector(Monitor):
    """
    异常检测器

    检测数据中的异常并发送报警
    """

    def __init__(self, 
                 name: str,
                 data_source: Callable,
                 features: List[str],
                 contamination: float = 0.05,
                 check_interval: float = 300.0,
                 save_dir: str = None):
        """
        初始化异常检测器

        参数:
        name (str): 监控名称
        data_source (Callable): 数据源函数，返回DataFrame
        features (List[str]): 特征列表
        contamination (float): 异常比例
        check_interval (float): 检查间隔（秒）
        save_dir (str): 保存目录
        """
        super().__init__(name, check_interval, save_dir)
        self.data_source = data_source
        self.features = features
        self.contamination = contamination

        # 初始化模型
        self.model = None
        self.scaler = StandardScaler()

        # 初始化监控数据
        self.data = {
            'anomalies': [],
            'alerts': []
        }

    def train_model(self, data: pd.DataFrame):
        """
        训练异常检测模型

        参数:
        data (pd.DataFrame): 训练数据
        """
        try:
            # 提取特征
            X = data[self.features].values

            # 标准化
            X_scaled = self.scaler.fit_transform(X)

            # 训练模型
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42
            )
            self.model.fit(X_scaled)

            logger.info(f"异常检测模型已训练: {self.name}")
        except Exception as e:
            logger.error(f"训练异常检测模型时出错: {e}")

    def detect_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        检测异常

        参数:
        data (pd.DataFrame): 数据

        返回:
        pd.DataFrame: 带有异常标记的数据
        """
        if self.model is None:
            logger.warning("模型未训练，先训练模型")
            self.train_model(data)
            return pd.DataFrame()

        try:
            # 提取特征
            X = data[self.features].values

            # 标准化
            X_scaled = self.scaler.transform(X)

            # 预测
            predictions = self.model.predict(X_scaled)

            # 添加异常标记
            data_with_anomalies = data.copy()
            data_with_anomalies['anomaly'] = predictions

            # 筛选异常
            anomalies = data_with_anomalies[data_with_anomalies['anomaly'] == -1]

            return anomalies
        except Exception as e:
            logger.error(f"检测异常时出错: {e}")
            return pd.DataFrame()

    def check(self):
        """
        检查异常
        """
        logger.info(f"检查异常: {self.name}")

        # 获取当前时间
        current_time = datetime.now()

        try:
            # 获取数据
            data = self.data_source()

            # 如果模型未训练，先训练模型
            if self.model is None:
                self.train_model(data)
                return

            # 检测异常
            anomalies = self.detect_anomalies(data)

            # 如果有异常，触发警报
            if not anomalies.empty:
                # 记录异常
                for _, row in anomalies.iterrows():
                    anomaly_record = {
                        'timestamp': current_time,
                        'data': row.to_dict()
                    }
                    self.data['anomalies'].append(anomaly_record)

                # 触发警报
                self.trigger_alert({
                    'type': 'anomaly_detected',
                    'count': len(anomalies),
                    'anomalies': anomalies.to_dict(orient='records'),
                    'message': f"检测到 {len(anomalies)} 个异常"
                })
        except Exception as e:
            logger.error(f"检查异常时出错: {e}")

    def plot_anomalies(self, feature1: str, feature2: str, figsize: tuple = (12, 8)):
        """
        绘制异常散点图

        参数:
        feature1 (str): 特征1
        feature2 (str): 特征2
        figsize (tuple): 图形大小
        """
        if not self.data['anomalies']:
            logger.warning("没有异常数据")
            return

        # 获取所有异常数据
        anomaly_data = []
        for record in self.data['anomalies']:
            anomaly_data.append(record['data'])

        # 转换为DataFrame
        df = pd.DataFrame(anomaly_data)

        # 检查特征是否存在
        if feature1 not in df.columns or feature2 not in df.columns:
            logger.warning(f"特征 {feature1} 或 {feature2} 不存在")
            return

        plt.figure(figsize=figsize)
        plt.scatter(df[feature1], df[feature2], c='red', marker='x', s=100)
        plt.title(f"异常散点图: {feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_anomaly_timeline(self, figsize: tuple = (12, 6)):
        """
        绘制异常时间线

        参数:
        figsize (tuple): 图形大小
        """
        if not self.data['anomalies']:
            logger.warning("没有异常数据")
            return

        # 获取所有异常时间
        timestamps = [record['timestamp'] for record in self.data['anomalies']]
        counts = [1] * len(timestamps)

        plt.figure(figsize=figsize)
        plt.stem(timestamps, counts, linefmt='r-', markerfmt='ro', basefmt='k-')
        plt.title("异常时间线")
        plt.xlabel("时间")
        plt.ylabel("异常")
        plt.grid(True)
        plt.tight_layout()
        plt.show()