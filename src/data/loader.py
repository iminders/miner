import os
import pandas as pd
import numpy as np
import h5py
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class OrderBookLoader:
    """
    加载和处理高频orderbook数据
    """
    def __init__(self, data_dir: str, cache_dir: Optional[str] = None):
        """
        初始化OrderBookLoader

        Args:
            data_dir: 原始数据目录
            cache_dir: 缓存数据目录
        """
        self.data_dir = data_dir
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(data_dir), 'processed')
        os.makedirs(self.cache_dir, exist_ok=True)

    def list_available_dates(self) -> List[str]:
        """
        列出可用的交易日期

        Returns:
            日期列表
        """
        dates = []
        for item in os.listdir(self.data_dir):
            if os.path.isdir(os.path.join(self.data_dir, item)) and item.isdigit() and len(item) == 8:
                dates.append(item)
        return sorted(dates)

    def list_available_stocks(self, date: str) -> List[str]:
        """
        列出指定日期可用的股票代码

        Args:
            date: 交易日期，格式YYYYMMDD

        Returns:
            股票代码列表
        """
        date_dir = os.path.join(self.data_dir, date)
        if not os.path.exists(date_dir):
            logger.warning(f"Date directory {date} does not exist")
            return []

        stocks = []
        for item in os.listdir(date_dir):
            if item.endswith('.h5') or item.endswith('.hdf5'):
                stocks.append(item.split('.')[0])
        return sorted(stocks)

    def load_orderbook(self, date: str, stock_code: str) -> pd.DataFrame:
        """
        加载指定日期和股票的orderbook数据

        Args:
            date: 交易日期，格式YYYYMMDD
            stock_code: 股票代码

        Returns:
            orderbook数据DataFrame
        """
        file_path = os.path.join(self.data_dir, date, f"{stock_code}.h5")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Orderbook file not found: {file_path}")

        # 尝试从缓存加载
        cache_path = os.path.join(self.cache_dir, date, f"{stock_code}_processed.h5")
        if os.path.exists(cache_path):
            logger.info(f"Loading processed data from cache: {cache_path}")
            return pd.read_hdf(cache_path, 'orderbook')

        # 加载原始数据
        logger.info(f"Loading raw orderbook data: {file_path}")
        with h5py.File(file_path, 'r') as f:
            # 假设数据结构为每个时间戳一个组，每个组包含买卖价格和数量
            # 实际实现需要根据您的数据格式调整
            timestamps = list(f.keys())
            data = []

            for ts in timestamps:
                group = f[ts]
                bid_prices = group['bid_prices'][:]
                bid_volumes = group['bid_volumes'][:]
                ask_prices = group['ask_prices'][:]
                ask_volumes = group['ask_volumes'][:]

                row = {
                    'timestamp': int(ts),
                    'datetime': pd.to_datetime(int(ts), unit='ms'),
                    'bid_price_1': bid_prices[0] if len(bid_prices) > 0 else np.nan,
                    'bid_volume_1': bid_volumes[0] if len(bid_volumes) > 0 else 0,
                    'bid_price_2': bid_prices[1] if len(bid_prices) > 1 else np.nan,
                    'bid_volume_2': bid_volumes[1] if len(bid_volumes) > 1 else 0,
                    'bid_price_3': bid_prices[2] if len(bid_prices) > 2 else np.nan,
                    'bid_volume_3': bid_volumes[2] if len(bid_volumes) > 2 else 0,
                    'bid_price_4': bid_prices[3] if len(bid_prices) > 3 else np.nan,
                    'bid_volume_4': bid_volumes[3] if len(bid_volumes) > 3 else 0,
                    'bid_price_5': bid_prices[4] if len(bid_prices) > 4 else np.nan,
                    'bid_volume_5': bid_volumes[4] if len(bid_volumes) > 4 else 0,
                    'ask_price_1': ask_prices[0] if len(ask_prices) > 0 else np.nan,
                    'ask_volume_1': ask_volumes[0] if len(ask_volumes) > 0 else 0,
                    'ask_price_2': ask_prices[1] if len(ask_prices) > 1 else np.nan,
                    'ask_volume_2': ask_volumes[1] if len(ask_volumes) > 1 else 0,
                    'ask_price_3': ask_prices[2] if len(ask_prices) > 2 else np.nan,
                    'ask_volume_3': ask_volumes[2] if len(ask_volumes) > 2 else 0,
                    'ask_price_4': ask_prices[3] if len(ask_prices) > 3 else np.nan,
                    'ask_volume_4': ask_volumes[3] if len(ask_volumes) > 3 else 0,
                    'ask_price_5': ask_prices[4] if len(ask_prices) > 4 else np.nan,
                    'ask_volume_5': ask_volumes[4] if len(ask_volumes) > 4 else 0,
                }
                data.append(row)

        df = pd.DataFrame(data)
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # 缓存处理后的数据
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_hdf(cache_path, 'orderbook', mode='w', format='table')

        return df

    def load_multiple_days(self, dates: List[str], stock_code: str) -> pd.DataFrame:
        """
        加载多个交易日的orderbook数据

        Args:
            dates: 交易日期列表
            stock_code: 股票代码

        Returns:
            合并后的orderbook数据
        """
        dfs = []
        for date in dates:
            try:
                df = self.load_orderbook(date, stock_code)
                df['date'] = date
                dfs.append(df)
            except FileNotFoundError:
                logger.warning(f"Data not found for {stock_code} on {date}")
                continue

        if not dfs:
            raise ValueError(f"No data found for {stock_code} on specified dates")

        return pd.concat(dfs, ignore_index=True)

    def load_multiple_stocks(self, date: str, stock_codes: List[str]) -> Dict[str, pd.DataFrame]:
        """
        加载指定日期的多个股票数据

        Args:
            date: 交易日期
            stock_codes: 股票代码列表

        Returns:
            股票代码到DataFrame的映射
        """
        result = {}
        for code in stock_codes:
            try:
                result[code] = self.load_orderbook(date, code)
            except FileNotFoundError:
                logger.warning(f"Data not found for {code} on {date}")
                continue

        return result