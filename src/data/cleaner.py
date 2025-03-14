import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class OrderBookCleaner:
    """
    清洗和预处理orderbook数据
    """
    def __init__(self, trading_hours: Optional[List[Tuple[str, str]]] = None):
        """
        初始化OrderBookCleaner
        
        Args:
            trading_hours: 交易时间段列表，每个元素为(开始时间, 结束时间)，格式为"HH:MM:SS"
                           默认为A股交易时间 9:30-11:30, 13:00-15:00
        """
        self.trading_hours = trading_hours or [
            ("09:30:00", "11:30:00"),
            ("13:00:00", "15:00:00")
        ]
    
    def filter_trading_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        过滤非交易时段的数据
        
        Args:
            df: 包含'datetime'列的DataFrame
            
        Returns:
            过滤后的DataFrame
        """
        if 'datetime' not in df.columns:
            raise ValueError("DataFrame must contain 'datetime' column")
        
        # 提取时间部分
        df['time'] = df['datetime'].dt.strftime('%H:%M:%S')
        
        # 创建过滤条件
        mask = pd.Series(False, index=df.index)
        for start, end in self.trading_hours:
            mask |= (df['time'] >= start) & (df['time'] <= end)
        
        # 应用过滤并删除临时列
        filtered_df = df[mask].copy()
        filtered_df.drop('time', axis=1, inplace=True)
        
        logger.info(f"Filtered {len(df) - len(filtered_df)} records outside trading hours")
        return filtered_df
    
    def remove_outliers(self, df: pd.DataFrame, columns: List[str], 
                        method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
        """
        移除异常值
        
        Args:
            df: 输入DataFrame
            columns: 需要检查异常值的列
            method: 异常值检测方法，'iqr'或'zscore'
            threshold: 异常值阈值
            
        Returns:
            清洗后的DataFrame
        """
        df_clean = df.copy()
        total_outliers = 0
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found in DataFrame")
                continue
                
            # 跳过非数值列
            if not np.issubdtype(df[col].dtype, np.number):
                continue
                
            # 计算异常值掩码
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            elif method == 'zscore':
                z_scores = (df[col] - df[col].mean()) / df[col].std()
                outlier_mask = np.abs(z_scores) > threshold
            else:
                raise ValueError(f"Unsupported outlier detection method: {method}")
                
            # 处理异常值
            num_outliers = outlier_mask.sum()
            total_outliers += num_outliers
            df_clean.loc[outlier_mask, col] = np.nan
            
            if num_outliers > 0:
                logger.debug(f"Found {num_outliers} outliers in column {col}")

        logger.info(f"Total outliers removed: {total_outliers}")
        return df_clean
    
    def align_timestamps(self, df: pd.DataFrame, freq: str = '1S') -> pd.DataFrame:
        """
        时间戳对齐和插值处理
        
        Args:
            df: 包含'datetime'索引的DataFrame
            freq: 重采样频率，默认为1秒
            
        Returns:
            对齐后的DataFrame
        """
        if 'datetime' not in df.columns:
            raise ValueError("DataFrame must contain 'datetime column")
        
        # 设置时间索引
        df = df.set_index('datetime').sort_index()
        
        # 创建完整时间范围
        start_time = df.index.min().floor('D') + pd.Timedelta(hours=9, minutes=30)
        end_time = df.index.max().ceil('D') - pd.Timedelta(hours=9)
        new_index = pd.date_range(start=start_time, end=end_time, freq=freq)
        
        # 重采样并填充缺失值
        resampled_df = df.reindex(new_index, method='nearest', tolerance=pd.Timedelta('500ms'))
        
        # 前向填充剩余缺失值
        resampled_df = resampled_df.ffill()
        
        logger.info(f"Aligned timestamps from {len(df)} to {len(resampled_df)} records")
        return resampled_df.reset_index(names='datetime')
    
    def handle_missing_data(self, df: pd.DataFrame, max_gap: str = '10S') -> pd.DataFrame:
        """
        处理缺失数据
        
        Args:
            df: 包含时间戳的DataFrame
            max_gap: 最大允许的时间间隔，超过该间隔不进行填充
            
        Returns:
            处理后的DataFrame
        """
        df = df.set_index('datetime').sort_index()
        
        # 计算时间间隔
        time_gaps = df.index.to_series().diff()
        gap_threshold = pd.Timedelta(max_gap)
        
        # 标记需要删除的长间隔
        long_gaps = time_gaps > gap_threshold
        df_clean = df[~long_gaps]
        
        # 前向填充短期缺失
        df_clean = df_clean.ffill().bfill()
        
        logger.info(f"Removed {long_gaps.sum()} long gaps and filled {df_clean.isna().sum().sum()} missing values")
        return df_clean.reset_index(names='datetime')
    
    def normalize_prices(self, df: pd.DataFrame, price_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        价格数据标准化处理
        
        Args:
            df: 输入DataFrame
            price_cols: 需要标准化的价格列，默认为所有bid_price和ask_price列
            
        Returns:
            标准化后的DataFrame
        """
        df_norm = df.copy()
        
        # 如果未指定价格列，自动检测
        if price_cols is None:
            price_cols = [col for col in df.columns if 'price' in col.lower()]
        
        # 计算中间价
        if 'bid_price_1' in df.columns and 'ask_price_1' in df.columns:
            df_norm['mid_price'] = (df['bid_price_1'] + df['ask_price_1']) / 2
            
            # 使用中间价进行标准化
            for col in price_cols:
                if col in df.columns:
                    df_norm[f'{col}_norm'] = df[col] / df_norm['mid_price'] - 1
        
        logger.info(f"Normalized {len(price_cols)} price columns")
        return df_norm
    
    def calculate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算基本特征
        
        Args:
            df: 输入DataFrame，需包含orderbook数据
            
        Returns:
            添加基本特征后的DataFrame
        """
        df_features = df.copy()
        
        # 检查必要列是否存在
        required_cols = ['bid_price_1', 'ask_price_1', 'bid_volume_1', 'ask_volume_1']
        if not all(col in df.columns for col in required_cols):
            logger.warning("Missing required columns for feature calculation")
            return df
        
        # 1. 价差相关特征
        df_features['spread'] = df['ask_price_1'] - df['bid_price_1']
        df_features['spread_bps'] = df_features['spread'] / ((df['ask_price_1'] + df['bid_price_1'])/2) * 10000
        
        # 2. 订单簿不平衡特征
        df_features['order_imbalance'] = (df['bid_volume_1'] - df['ask_volume_1']) / (df['bid_volume_1'] + df['ask_volume_1'])
        
        # 3. 深度特征
        if all(col in df.columns for col in ['bid_volume_1', 'bid_volume_2', 'bid_volume_3', 'bid_volume_4', 'bid_volume_5']):
            df_features['bid_depth'] = df[['bid_volume_1', 'bid_volume_2', 'bid_volume_3', 'bid_volume_4', 'bid_volume_5']].sum(axis=1)
        
        if all(col in df.columns for col in ['ask_volume_1', 'ask_volume_2', 'ask_volume_3', 'ask_volume_4', 'ask_volume_5']):
            df_features['ask_depth'] = df[['ask_volume_1', 'ask_volume_2', 'ask_volume_3', 'ask_volume_4', 'ask_volume_5']].sum(axis=1)
            
        if 'bid_depth' in df_features.columns and 'ask_depth' in df_features.columns:
            df_features['depth_imbalance'] = (df_features['bid_depth'] - df_features['ask_depth']) / (df_features['bid_depth'] + df_features['ask_depth'])
        
        # 4. 加权中间价
        if all(col in df.columns for col in ['bid_price_1', 'ask_price_1', 'bid_volume_1', 'ask_volume_1']):
            total_vol = df['bid_volume_1'] + df['ask_volume_1']
            df_features['vwap_mid'] = (df['bid_price_1'] * df['ask_volume_1'] + df['ask_price_1'] * df['bid_volume_1']) / total_vol
        
        logger.info("Calculated basic orderbook features")
        return df_features
    
    def process_pipeline(self, df: pd.DataFrame, freq: str = '1S', outlier_threshold: float = 3.0) -> pd.DataFrame:
        """
        完整的数据处理流水线
        
        Args:
            df: 原始orderbook数据
            freq: 重采样频率
            outlier_threshold: 异常值阈值
            
        Returns:
            处理后的DataFrame
        """
        # 1. 过滤交易时段
        df_filtered = self.filter_trading_hours(df)
        
        # 2. 移除异常值
        price_cols = [col for col in df_filtered.columns if 'price' in col.lower()]
        volume_cols = [col for col in df_filtered.columns if 'volume' in col.lower()]
        df_clean = self.remove_outliers(df_filtered, columns=price_cols, threshold=outlier_threshold)
        
        # 3. 时间戳对齐
        df_aligned = self.align_timestamps(df_clean, freq=freq)
        
        # 4. 处理缺失值
        df_filled = self.handle_missing_data(df_aligned)
        
        # 5. 计算基本特征
        df_features = self.calculate_basic_features(df_filled)
        
        # 6. 价格标准化
        df_normalized = self.normalize_prices(df_features)
        
        logger.info(f"Completed full processing pipeline: {len(df)} -> {len(df_normalized)} records")
        return df_normalized