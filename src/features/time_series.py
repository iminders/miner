import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class TimeSeriesFeatures:
    """
    时间序列特征提取器
    """
    
    def __init__(self, window_sizes: Optional[List[int]] = None):
        """
        初始化时间序列特征提取器
        
        Args:
            window_sizes: 滚动窗口大小列表，默认为[5, 10, 20, 50, 100]
        """
        self.window_sizes = window_sizes or [5, 10, 20, 50, 100]
    
    def calculate_momentum_features(self, df: pd.DataFrame, price_col: str = 'mid_price') -> pd.DataFrame:
        """
        计算动量特征
        
        Args:
            df: 输入DataFrame，需包含价格数据
            price_col: 价格列名，默认为'mid_price'
            
        Returns:
            添加动量特征后的DataFrame
        """
        result = df.copy()
        
        if price_col not in df.columns:
            logger.warning(f"Missing {price_col} column for momentum calculation")
            return result
        
        # 计算价格变化率
        result['price_change'] = result[price_col].diff()
        result['price_change_pct'] = result[price_col].pct_change()
        
        # 计算不同窗口的动量指标
        for window in self.window_sizes:
            # 计算过去window个周期的价格变化率
            result[f'momentum_{window}'] = result[price_col].pct_change(window)
            
            # 计算过去window个周期的价格变化率的符号
            result[f'momentum_sign_{window}'] = np.sign(result[f'momentum_{window}'])
            
            # 计算过去window个周期的价格变化率的绝对值
            result[f'momentum_abs_{window}'] = np.abs(result[f'momentum_{window}'])
            
            # 计算过去window个周期的价格变化率的Z-score
            result[f'momentum_zscore_{window}'] = (
                result[f'momentum_{window}'] - 
                result[f'momentum_{window}'].rolling(window=window*3).mean()
            ) / result[f'momentum_{window}'].rolling(window=window*3).std()
        
        logger.info("Calculated momentum features")
        return result
    
    def calculate_volatility_features(self, df: pd.DataFrame, price_col: str = 'mid_price') -> pd.DataFrame:
        """
        计算波动率特征
        
        Args:
            df: 输入DataFrame，需包含价格数据
            price_col: 价格列名，默认为'mid_price'
            
        Returns:
            添加波动率特征后的DataFrame
        """
        result = df.copy()
        
        if price_col not in df.columns:
            logger.warning(f"Missing {price_col} column for volatility calculation")
            return result
        
        # 计算收益率
        if 'price_change_pct' not in result.columns:
            result['price_change_pct'] = result[price_col].pct_change()
        
        # 计算不同窗口的波动率指标
        for window in self.window_sizes:
            # 计算过去window个周期的波动率（标准差）
            result[f'volatility_{window}'] = result['price_change_pct'].rolling(window=window).std()
            
            # 计算过去window个周期的波动率的变化率
            result[f'volatility_change_{window}'] = result[f'volatility_{window}'].pct_change()
            
            # 计算过去window个周期的波动率的Z-score
            result[f'volatility_zscore_{window}'] = (
                result[f'volatility_{window}'] - 
                result[f'volatility_{window}'].rolling(window=window*3).mean()
            ) / result[f'volatility_{window}'].rolling(window=window*3).std()
            
            # 计算过去window个周期的高低价差波动率
            if 'ask_price_1' in df.columns and 'bid_price_1' in df.columns:
                high_low_range = result['ask_price_1'] - result['bid_price_1']
                result[f'high_low_volatility_{window}'] = high_low_range.rolling(window=window).std()
        
        logger.info("Calculated volatility features")
        return result
    
    def calculate_mean_reversion_features(self, df: pd.DataFrame, price_col: str = 'mid_price') -> pd.DataFrame:
        """
        计算均值回归特征
        
        Args:
            df: 输入DataFrame，需包含价格数据
            price_col: 价格列名，默认为'mid_price'
            
        Returns:
            添加均值回归特征后的DataFrame
        """
        result = df.copy()
        
        if price_col not in df.columns:
            logger.warning(f"Missing {price_col} column for mean reversion calculation")
            return result
        
        # 计算不同窗口的均值回归指标
        for window in self.window_sizes:
            # 计算过去window个周期的移动平均
            result[f'ma_{window}'] = result[price_col].rolling(window=window).mean()
            
            # 计算当前价格与移动平均的偏离度
            result[f'price_deviation_{window}'] = (result[price_col] - result[f'ma_{window}']) / result[f'ma_{window}']
            
            # 计算价格偏离度的Z-score
            result[f'price_deviation_zscore_{window}'] = (
                result[f'price_deviation_{window}'] - 
                result[f'price_deviation_{window}'].rolling(window=window*3).mean()
            ) / result[f'price_deviation_{window}'].rolling(window=window*3).std()
            
            # 计算RSI (Relative Strength Index)
            delta = result[price_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            
            # 避免除以零
            loss = loss.replace(0, np.nan)
            
            rs = gain / loss
            result[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        logger.info("Calculated mean reversion features")
        return result
    
    def calculate_jump_features(self, df: pd.DataFrame, price_col: str = 'mid_price', threshold: float = 3.0) -> pd.DataFrame:
        """
        计算价格跳跃特征
        
        Args:
            df: 输入DataFrame，需包含价格数据
            price_col: 价格列名，默认为'mid_price'
            threshold: 判断跳跃的阈值，以标准差的倍数表示
            
        Returns:
            添加价格跳跃特征后的DataFrame
        """
        result = df.copy()
        
        if price_col not in df.columns:
            logger.warning(f"Missing {price_col} column for jump calculation")
            return result
        
        # 计算收益率
        if 'price_change_pct' not in result.columns:
            result['price_change_pct'] = result[price_col].pct_change()
        
        # 计算不同窗口的价格跳跃指标
        for window in self.window_sizes:
            # 计算过去window个周期的收益率标准差
            result[f'return_std_{window}'] = result['price_change_pct'].rolling(window=window).std()
            
            # 计算收益率的Z-score
            result[f'return_zscore_{window}'] = result['price_change_pct'] / result[f'return_std_{window}']
            
            # 判断是否发生价格跳跃
            result[f'jump_{window}'] = np.where(
                np.abs(result[f'return_zscore_{window}']) > threshold,
                1,
                0
            )
            
            # 计算跳跃的方向
            result[f'jump_direction_{window}'] = np.where(
                result[f'jump_{window}'] == 1,
                np.sign(result['price_change_pct']),
                0
            )
            
            # 计算跳跃的大小
            result[f'jump_size_{window}'] = np.where(
                result[f'jump_{window}'] == 1,
                np.abs(result['price_change_pct']),
                0
            )
            
            # 计算过去N个周期内跳跃的累计次数
            result[f'jump_count_{window}'] = result[f'jump_{window}'].rolling(window=window).sum()
        
        logger.info("Calculated jump features")
        return result
    
    def calculate_autocorrelation_features(self, df: pd.DataFrame, price_col: str = 'mid_price', lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """
        计算自相关特征
        
        Args:
            df: 输入DataFrame，需包含价格数据
            price_col: 价格列名，默认为'mid_price'
            lags: 滞后期数列表
            
        Returns:
            添加自相关特征后的DataFrame
        """
        result = df.copy()
        
        if price_col not in df.columns:
            logger.warning(f"Missing {price_col} column for autocorrelation calculation")
            return result
        
        # 计算收益率
        if 'price_change_pct' not in result.columns:
            result['price_change_pct'] = result[price_col].pct_change()
        
        # 计算不同窗口的自相关指标
        for window in self.window_sizes:
            for lag in lags:
                # 计算过去window个周期的收益率自相关
                result[f'autocorr_{lag}_{window}'] = result['price_change_pct'].rolling(window=window).apply(
                    lambda x: x.autocorr(lag=lag) if len(x) > lag + 1 else np.nan
                )
        
        logger.info("Calculated autocorrelation features")
        return result
    
    def extract_all_time_series_features(self, df: pd.DataFrame, price_col: str = 'mid_price') -> pd.DataFrame:
        """
        提取所有时间序列特征
        
        Args:
            df: 输入DataFrame，需包含价格数据
            price_col: 价格列名，默认为'mid_price'
            
        Returns:
            添加所有时间序列特征后的DataFrame
        """
        result = df.copy()
        
        # 按顺序应用所有特征提取方法
        result = self.calculate_momentum_features(result, price_col)
        result = self.calculate_volatility_features(result, price_col)
        result = self.calculate_mean_reversion_features(result, price_col)
        result = self.calculate_jump_features(result, price_col)
        result = self.calculate_autocorrelation_features(result, price_col)
        
        logger.info(f"Extracted all time series features, total columns: {len(result.columns)}")
        return result