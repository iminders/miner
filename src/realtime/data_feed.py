# -*- coding: utf-8 -*-
"""
数据实时接入模块

用于接入实时行情数据
"""

import time
import threading
import queue
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Callable
import websocket
import json
import requests
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataFeed:
    """
    数据实时接入基类

    提供数据接入的基本接口
    """

    def __init__(self, symbols: List[str], queue_size: int = 1000):
        """
        初始化数据接入

        参数:
        symbols (List[str]): 股票代码列表
        queue_size (int): 队列大小
        """
        self.symbols = symbols
        self.data_queue = queue.Queue(maxsize=queue_size)
        self.running = False
        self.thread = None

    def start(self):
        """
        启动数据接入
        """
        if self.running:
            logger.warning("数据接入已经在运行")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        logger.info("数据接入已启动")

    def stop(self):
        """
        停止数据接入
        """
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        logger.info("数据接入已停止")

    def _run(self):
        """
        运行数据接入（子类实现）
        """
        raise NotImplementedError("子类必须实现_run方法")

    def get_data(self, block: bool = True, timeout: Optional[float] = None) -> Dict:
        """
        获取数据

        参数:
        block (bool): 是否阻塞
        timeout (Optional[float]): 超时时间

        返回:
        Dict: 数据
        """
        try:
            return self.data_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None


class SimulatedDataFeed(DataFeed):
    """
    模拟数据接入

    用于测试和开发
    """

    def __init__(self, symbols: List[str], 
                 historical_data: Dict[str, pd.DataFrame] = None,
                 interval: float = 1.0,
                 replay_speed: float = 1.0,
                 add_noise: bool = True,
                 noise_level: float = 0.0001):
        """
        初始化模拟数据接入

        参数:
        symbols (List[str]): 股票代码列表
        historical_data (Dict[str, pd.DataFrame]): 历史数据
        interval (float): 数据间隔（秒）
        replay_speed (float): 回放速度
        add_noise (bool): 是否添加噪声
        noise_level (float): 噪声水平
        """
        super().__init__(symbols)
        self.historical_data = historical_data or {}
        self.interval = interval
        self.replay_speed = replay_speed
        self.add_noise = add_noise
        self.noise_level = noise_level

        # 如果没有提供历史数据，则生成模拟数据
        if not self.historical_data:
            self._generate_simulated_data()

    def _generate_simulated_data(self):
        """
        生成模拟数据
        """
        # 生成时间索引
        now = datetime.now()
        start_time = now - timedelta(days=1)
        timestamps = pd.date_range(start=start_time, end=now, freq=f'{self.interval}S')

        for symbol in self.symbols:
            # 生成随机价格
            initial_price = np.random.uniform(10, 100)
            price_changes = np.random.normal(0, 0.001, size=len(timestamps))
            prices = initial_price * (1 + np.cumsum(price_changes))

            # 生成随机成交量
            volumes = np.random.exponential(1000, size=len(timestamps))

            # 生成买卖盘数据
            bid_prices = prices * (1 - np.random.uniform(0.001, 0.003, size=len(timestamps)))
            ask_prices = prices * (1 + np.random.uniform(0.001, 0.003, size=len(timestamps)))
            bid_volumes = np.random.exponential(500, size=len(timestamps))
            ask_volumes = np.random.exponential(500, size=len(timestamps))

            # 创建DataFrame
            df = pd.DataFrame({
                'timestamp': timestamps,
                'price': prices,
                'volume': volumes,
                'bid_price_1': bid_prices,
                'ask_price_1': ask_prices,
                'bid_volume_1': bid_volumes,
                'ask_volume_1': ask_volumes
            })

            # 设置索引
            df.set_index('timestamp', inplace=True)

            # 存储数据
            self.historical_data[symbol] = df

    def _run(self):
        """
        运行模拟数据接入
        """
        if not self.historical_data:
            logger.error("没有历史数据可供模拟")
            return

        # 获取所有数据的时间索引
        all_timestamps = set()
        for df in self.historical_data.values():
            all_timestamps.update(df.index)

        # 排序时间索引
        sorted_timestamps = sorted(all_timestamps)

        # 模拟实时数据流
        for i, timestamp in enumerate(sorted_timestamps):
            if not self.running:
                break

            # 收集当前时间点的所有数据
            current_data = {}

            for symbol, df in self.historical_data.items():
                if timestamp in df.index:
                    # 获取当前时间点的数据
                    row = df.loc[timestamp].copy()

                    # 添加噪声
                    if self.add_noise:
                        price_noise = np.random.normal(0, self.noise_level)
                        row['price'] *= (1 + price_noise)
                        row['bid_price_1'] *= (1 + price_noise * 0.9)
                        row['ask_price_1'] *= (1 + price_noise * 1.1)

                    # 转换为字典
                    data_dict = row.to_dict()
                    data_dict['symbol'] = symbol
                    data_dict['timestamp'] = timestamp

                    current_data[symbol] = data_dict

            # 将数据放入队列
            if current_data:
                try:
                    self.data_queue.put({
                        'type': 'market_data',
                        'timestamp': timestamp,
                        'data': current_data
                    }, block=False)
                except queue.Full:
                    logger.warning("数据队列已满，丢弃数据")

            # 控制数据发送速度
            if i < len(sorted_timestamps) - 1:
                next_timestamp = sorted_timestamps[i + 1]
                sleep_time = (next_timestamp - timestamp).total_seconds() / self.replay_speed
                if sleep_time > 0:
                    time.sleep(sleep_time)


class WebSocketDataFeed(DataFeed):
    """
    WebSocket数据接入

    通过WebSocket接入实时行情数据
    """

    def __init__(self, symbols: List[str], 
                 ws_url: str,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 reconnect_interval: int = 5,
                 max_reconnect_attempts: int = 10):
        """
        初始化WebSocket数据接入

        参数:
        symbols (List[str]): 股票代码列表
        ws_url (str): WebSocket URL
        api_key (Optional[str]): API Key
        api_secret (Optional[str]): API Secret
        reconnect_interval (int): 重连间隔（秒）
        max_reconnect_attempts (int): 最大重连尝试次数
        """
        super().__init__(symbols)
        self.ws_url = ws_url
        self.api_key = api_key
        self.api_secret = api_secret
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        self.ws = None
        self.reconnect_count = 0

    def _on_message(self, ws, message):
        """
        处理WebSocket消息

        参数:
        ws: WebSocket对象
        message: 消息
        """
        try:
            # 解析消息
            data = json.loads(message)

            # 处理不同类型的消息
            if 'type' in data:
                if data['type'] == 'market_data':
                    # 处理行情数据
                    try:
                        self.data_queue.put(data, block=False)
                    except queue.Full:
                        logger.warning("数据队列已满，丢弃数据")
                elif data['type'] == 'heartbeat':
                    # 处理心跳消息
                    logger.debug("收到心跳消息")
                else:
                    logger.warning(f"未知消息类型: {data['type']}")
            else:
                logger.warning("消息缺少类型字段")
        except json.JSONDecodeError:
            logger.error(f"JSON解析错误: {message}")
        except Exception as e:
            logger.error(f"处理消息时出错: {e}")

    def _on_error(self, ws, error):
        """
        处理WebSocket错误

        参数:
        ws: WebSocket对象
        error: 错误
        """
        logger.error(f"WebSocket错误: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        """
        处理WebSocket关闭

        参数:
        ws: WebSocket对象
        close_status_code: 关闭状态码
        close_msg: 关闭消息
        """
        logger.info(f"WebSocket已关闭: {close_status_code} - {close_msg}")

        # 尝试重连
        if self.running and self.reconnect_count < self.max_reconnect_attempts:
            self.reconnect_count += 1
            logger.info(f"尝试重连 ({self.reconnect_count}/{self.max_reconnect_attempts})...")
            time.sleep(self.reconnect_interval)
            self._connect()
        elif self.reconnect_count >= self.max_reconnect_attempts:
            logger.error("达到最大重连尝试次数，停止重连")
            self.running = False

    def _on_open(self, ws):
        """
        处理WebSocket打开

        参数:
        ws: WebSocket对象
        """
        logger.info("WebSocket已连接")
        self.reconnect_count = 0

        # 订阅行情数据
        subscription = {
            'type': 'subscribe',
            'symbols': self.symbols,
            'channels': ['orderbook', 'trades']
        }

        # 添加认证信息
        if self.api_key and self.api_secret:
            subscription['api_key'] = self.api_key
            # 在实际应用中，这里应该添加签名

        # 发送订阅请求
        ws.send(json.dumps(subscription))

    def _connect(self):
        """
        连接WebSocket
        """
        # 创建WebSocket连接
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )

    def _run(self):
        """
        运行WebSocket数据接入
        """
        # 连接WebSocket
        self._connect()

        # 运行WebSocket
        while self.running:
            try:
                self.ws.run_forever()
                if not self.running:
                    break
                logger.info("WebSocket连接已断开，等待重连...")
                time.sleep(self.reconnect_interval)
            except Exception as e:
                logger.error(f"WebSocket运行时出错: {e}")
                if self.running:
                    time.sleep(self.reconnect_interval)

    def stop(self):
        """
        停止WebSocket数据接入
        """
        super().stop()
        if self.ws:
            self.ws.close()


class RESTDataFeed(DataFeed):
    """
    REST API数据接入

    通过REST API接入实时行情数据
    """

    def __init__(self, symbols: List[str], 
                 api_url: str,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 interval: float = 1.0):
        """
        初始化REST API数据接入

        参数:
        symbols (List[str]): 股票代码列表
        api_url (str): API URL
        api_key (Optional[str]): API Key
        api_secret (Optional[str]): API Secret
        interval (float): 请求间隔（秒）
        """
        super().__init__(symbols)
        self.api_url = api_url
        self.api_key = api_key
        self.api_secret = api_secret
        self.interval = interval
        self.session = requests.Session()

        # 设置请求头
        if self.api_key:
            self.session.headers.update({
                'X-API-Key': self.api_key
            })

    def _fetch_data(self):
        """
        获取数据

        返回:
        Dict: 数据
        """
        try:
            # 构建请求参数
            params = {
                'symbols': ','.join(self.symbols)
            }

            # 发送请求
            response = self.session.get(f"{self.api_url}/market/data", params=params)

            # 检查