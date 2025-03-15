import pandas as pd
import numpy as np
import os
from pathlib import Path
import h5py
import pyarrow.parquet as pq


class OrderBookLoader:
    """
    加载和预处理orderbook数据的类
    """

    def __init__(self, data_dir, file_format='parquet'):
        """
        初始化OrderBookLoader

        参数:
        data_dir (str): 数据目录路径
        file_format (str): 数据文件格式，支持'parquet'和'h5'
        """
        self.data_dir = Path(data_dir)
        self.file_format = file_format

    def list_data_files(self, date=None, stock_code=None):
        """
        列出符合条件的数据文件

        参数:
        date (str, optional): 日期，格式为'YYYYMMDD'
        stock_code (str, optional): 股票代码

        返回:
        list: 符合条件的文件路径列表
        """
        pattern = ""
        if date:
            pattern += f"*{date}*"
        if stock_code:
            pattern += f"*{stock_code}*"

        if not pattern:
            pattern = "*"

        if self.file_format == 'parquet':
            pattern += ".parquet"
        elif self.file_format == 'h5':
            pattern += ".h5"

        return list(self.data_dir.glob(pattern))

    def load_single_file(self, file_path):
        """
        加载单个数据文件

        参数:
        file_path (str): 文件路径

        返回:
        pd.DataFrame: 加载的数据
        """
        file_path = Path(file_path)

        if self.file_format == 'parquet':
            return pq.read_table(file_path).to_pandas()
        elif self.file_format == 'h5':
            with h5py.File(file_path, 'r') as f:
                # 假设H5文件中的数据存储在'orderbook'组下
                data = f['orderbook'][:]
                columns = f['columns'][:]
                return pd.DataFrame(data, columns=columns)
        else:
            raise ValueError(f"不支持的文件格式: {self.file_format}")

    def load_data(self, date=None, stock_code=None):
        """
        加载指定日期和股票的数据

        参数:
        date (str, optional): 日期，格式为'YYYYMMDD'
        stock_code (str, optional): 股票代码

        返回:
        pd.DataFrame: 加载的数据
        """
        files = self.list_data_files(date, stock_code)

        if not files:
            raise FileNotFoundError(f"未找到符合条件的数据文件: date={date}, stock_code={stock_code}")

        dfs = []
        for file in files:
            df = self.load_single_file(file)
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)

    def clean_data(self, df):
        """
        清洗orderbook数据

        参数:
        df (pd.DataFrame): 原始数据

        返回:
        pd.DataFrame: 清洗后的数据
        """
        # 1. 删除重复行
        df = df.drop_duplicates()

        # 2. 处理缺失值
        # 对于价格列，使用前向填充
        price_cols = [col for col in df.columns if 'price' in col.lower()]
        df[price_cols] = df[price_cols].fillna(method='ffill')

        # 对于数量列，缺失值填充为0
        volume_cols = [col for col in df.columns if 'volume' in col.lower() or 'size' in col.lower() or 'qty' in col.lower()]
        df[volume_cols] = df[volume_cols].fillna(0)

        # 3. 异常值处理
        # 价格异常值检测（例如，价格为负或者超过合理范围）
        for col in price_cols:
            # 计算价格的中位数和标准差
            median = df[col].median()
            std = df[col].std()

            # 将超过中位数±5倍标准差的值视为异常值，替换为NaN，然后使用前向填充
            df.loc[df[col] < median - 5 * std, col] = np.nan
            df.loc[df[col] > median + 5 * std, col] = np.nan
            df[col] = df[col].fillna(method='ffill')

        # 4. 确保时间戳是有序的
        df = df.sort_values(by='timestamp')

        return df

    def align_timestamps(self, df, freq='1S'):
        """
        将数据按指定频率重采样，确保时间戳对齐

        参数:
        df (pd.DataFrame): 原始数据
        freq (str): 重采样频率，默认为1秒

        返回:
        pd.DataFrame: 重采样后的数据
        """
        # 确保timestamp列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # 设置timestamp为索引
        df = df.set_index('timestamp')

        # 按指定频率重采样
        # 对于价格列，使用最后一个值
        price_cols = [col for col in df.columns if 'price' in col.lower()]
        # 对于数量列，使用最后一个值
        volume_cols = [col for col in df.columns if 'volume' in col.lower() or 'size' in col.lower() or 'qty' in col.lower()]

        # 创建一个空的DataFrame来存储重采样结果
        resampled = pd.DataFrame()

        # 对价格列使用最后一个值重采样
        for col in price_cols:
            resampled[col] = df[col].resample(freq).last()

        # 对数量列使用最后一个值重采样
        for col in volume_cols:
            resampled[col] = df[col].resample(freq).last()

        # 重置索引，将timestamp作为列
        resampled = resampled.reset_index()

        return resampled