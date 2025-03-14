import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class MicrostructureFeatures:
    """
    市场微观结构特征提取器
    """
    
    def __init__(self, window_sizes: Optional[List[int]] = None):
        """
        初始化微观结构特征提取器
        
        Args:
            window_sizes: 滚动窗口大小列表，默认为[5, 10, 20, 50, 100]
        """
        self.window_sizes = window_sizes or [5, 10, 20, 50, 100]
    
    def calculate_effective_spread(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算有效价差 (Effective Spread)
        
        Args:
            df: 输入DataFrame，需包含bid_price_1和ask_price_1
            
        Returns:
            添加有效价差特征后的DataFrame
        """
        result = df.copy()
        
        if 'bid_price_1' not in df.columns or 'ask_price_1' not in df.columns:
            logger.warning("Missing price columns for effective spread calculation")
            return result
        
        # 计算中间价
        if 'mid_price' not in df.columns:
            result['mid_price'] = (df['bid_price_1'] + df['ask_price_1']) / 2
        
        # 计算绝对有效价差
        result['effective_spread'] = df['ask_price_1'] - df['bid_price_1']
        
        # 计算相对有效价差 (bps)
        result['effective_spread_bps'] = result['effective_spread'] / result['mid_price'] * 10000
        
        # 计算有效价差的移动平均
        for window in self.window_sizes:
            result[f'effective_spread_ma_{window}'] = result['effective_spread'].rolling(window).mean()
            result[f'effective_spread_bps_ma_{window}'] = result['effective_spread_bps'].rolling(window).mean()
        
        logger.info("Calculated effective spread features")
        return result
    
    def calculate_order_flow_imbalance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算订单流不平衡 (Order Flow Imbalance)
        
        Args:
            df: 输入DataFrame，需包含bid_volume和ask_volume
            
        Returns:
            添加订单流不平衡特征后的DataFrame
        """
        result = df.copy()
        
        # 检查必要列
        bid_volume_cols = [col for col in df.columns if 'bid_volume' in col.lower()]
        ask_volume_cols = [col for col in df.columns if 'ask_volume' in col.lower()]
        
        if not bid_volume_cols or not ask_volume_cols:
            logger.warning("Missing volume columns for order flow imbalance calculation")
            return result
        
        # 计算一级订单不平衡
        if 'bid_volume_1' in df.columns and 'ask_volume_1' in df.columns:
            result['order_imbalance_1'] = (df['bid_volume_1'] - df['ask_volume_1']) / (df['bid_volume_1'] + df['ask_volume_1'])
            
            # 计算订单不平衡的移动平均和波动率
            for window in self.window_sizes:
                result[f'order_imbalance_1_ma_{window}'] = result['order_imbalance_1'].rolling(window).mean()
                result[f'order_imbalance_1_std_{window}'] = result['order_imbalance_1'].rolling(window).std()
        
        # 计算整体订单不平衡（所有层级）
        if all(f'bid_volume_{i}' in df.columns for i in range(1, 6)) and all(f'ask_volume_{i}' in df.columns for i in range(1, 6)):
            total_bid_volume = sum(df[f'bid_volume_{i}'] for i in range(1, 6))
            total_ask_volume = sum(df[f'ask_volume_{i}'] for i in range(1, 6))
            result['order_imbalance_total'] = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
            
            # 计算整体订单不平衡的移动平均和波动率
            for window in self.window_sizes:
                result[f'order_imbalance_total_ma_{window}'] = result['order_imbalance_total'].rolling(window).mean()
                result[f'order_imbalance_total_std_{window}'] = result['order_imbalance_total'].rolling(window).std()
        
        logger.info("Calculated order flow imbalance features")
        return result
    
    def calculate_price_impact(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算价格冲击 (Price Impact)
        
        Args:
            df: 输入DataFrame，需包含价格和交易量数据
            
        Returns:
            添加价格冲击特征后的DataFrame
        """
        result = df.copy()
        
        # 检查必要列
        if 'mid_price' not in df.columns:
            if 'bid_price_1' in df.columns and 'ask_price_1' in df.columns:
                result['mid_price'] = (df['bid_price_1'] + df['ask_price_1']) / 2
            else:
                logger.warning("Missing price columns for price impact calculation")
                return result
        
        # 计算价格变化
        result['price_change'] = result['mid_price'].diff()
        result['price_change_bps'] = result['price_change'] / result['mid_price'].shift(1) * 10000
        
        # 计算订单不平衡（如果不存在）
        if 'order_imbalance_1' not in df.columns and 'bid_volume_1' in df.columns and 'ask_volume_1' in df.columns:
            result['order_imbalance_1'] = (df['bid_volume_1'] - df['ask_volume_1']) / (df['bid_volume_1'] + df['ask_volume_1'])
        
        # 计算价格冲击系数（价格变化与订单不平衡的比率）
        if 'order_imbalance_1' in result.columns:
            # 避免除以零
            result['price_impact_coef'] = result['price_change_bps'] / result['order_imbalance_1'].replace(0, np.nan)
            
            # 计算价格冲击系数的移动平均
            for window in self.window_sizes:
                result[f'price_impact_coef_ma_{window}'] = result['price_impact_coef'].rolling(window).mean()
        
        # 计算Kyle's lambda（价格变化与交易量的比率）
        if 'total_volume' in df.columns:
            result['kyles_lambda'] = result['price_change_bps'] / result['total_volume'].replace(0, np.nan)
            
            # 计算Kyle's lambda的移动平均
            for window in self.window_sizes:
                result[f'kyles_lambda_ma_{window}'] = result['kyles_lambda'].rolling(window).mean()
        
        logger.info("Calculated price impact features")
        return result
    
    def calculate_probability_informed_trading(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算知情交易概率 (Probability of Informed Trading, PIN)
        
        Args:
            df: 输入DataFrame，需包含订单簿数据
            
        Returns:
            添加PIN相关特征后的DataFrame
        """
        result = df.copy()
        
        # 检查必要列
        if 'bid_volume_1' not in df.columns or 'ask_volume_1' not in df.columns:
            logger.warning("Missing volume columns for PIN calculation")
            return result
        
        # 计算买卖订单流不平衡的绝对值
        result['order_flow_imbalance_abs'] = np.abs(df['bid_volume_1'] - df['ask_volume_1'])
        
        # 计算总订单流
        result['total_order_flow'] = df['bid_volume_1'] + df['ask_volume_1']
        
        # 计算PIN代理变量（不平衡的绝对值除以总订单流）
        result['pin_proxy'] = result['order_flow_imbalance_abs'] / result['total_order_flow']
        
        # 计算PIN代理变量的移动平均
        for window in self.window_sizes:
            result[f'pin_proxy_ma_{window}'] = result['pin_proxy'].rolling(window).mean()
        
        logger.info("Calculated probability of informed trading features")
        return result
    
    def calculate_order_book_pressure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算订单簿压力 (Order Book Pressure)
        
        Args:
            df: 输入DataFrame，需包含订单簿数据
            
        Returns:
            添加订单簿压力特征后的DataFrame
        """
        result = df.copy()
        
        # 检查是否有足够的价格深度
        bid_price_cols = [f'bid_price_{i}' for i in range(1, 6)]
        ask_price_cols = [f'ask_price_{i}' for i in range(1, 6)]
        bid_volume_cols = [f'bid_volume_{i}' for i in range(1, 6)]
        ask_volume_cols = [f'ask_volume_{i}' for i in range(1, 6)]
        
        has_depth = (all(col in df.columns for col in bid_price_cols) and 
                     all(col in df.columns for col in ask_price_cols) and
                     all(col in df.columns for col in bid_volume_cols) and
                     all(col in df.columns for col in ask_volume_cols))
        
        if not has_depth:
            logger.warning("Insufficient orderbook depth for pressure calculation")
            return result
        
        # 计算买方订单簿压力
        bid_pressure = 0
        for i in range(1, 6):
            # 价格差异（相对于最优价格）
            price_diff = (df['bid_price_1'] - df[f'bid_price_{i}']) / df['bid_price_1']
            # 加权订单量（越接近最优价格权重越高）
            weighted_volume = df[f'bid_volume_{i}'] * (1 - price_diff)
            bid_pressure += weighted_volume
        
        result['bid_pressure'] = bid_pressure
        
        # 计算卖方订单簿压力
        ask_pressure = 0
        for i in range(1, 6):
            # 价格差异（相对于最优价格）
            price_diff = (df[f'ask_price_{i}'] - df['ask_price_1']) / df['ask_price_1']
            # 加权订单量（越接近最优价格权重越高）
            weighted_volume = df[f'ask_volume_{i}'] * (1 - price_diff)
            ask_pressure += weighted_volume
        
        result['ask_pressure'] = ask_pressure
        
        # 计算订单簿压力比率
        result['pressure_ratio'] = result['bid_pressure'] / result['ask_pressure']
        
        # 计算订单簿压力不平衡
        result['pressure_imbalance'] = (result['bid_pressure'] - result['ask_pressure']) / (result['bid_pressure'] + result['ask_pressure'])
        
        # 计算订单簿压力的移动平均
        for window in self.window_sizes:
            result[f'pressure_imbalance_ma_{window}'] = result['pressure_imbalance'].rolling(window).mean()
        
        logger.info("Calculated order book pressure features")
        return result
    
    def extract_all_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取所有微观结构特征
        
        Args:
            df: 输入DataFrame，需包含orderbook数据
            
        Returns:
            添加所有微观结构特征后的DataFrame
        """
        result = df.copy()
        
        # 按顺序应用所有特征提取方法
        result = self.calculate_effective_spread(result)
        result = self.calculate_order_flow_imbalance(result)
        result = self.calculate_price_impact(result)
        result = self.calculate_probability_informed_trading(result)
        result = self.calculate_order_book_pressure(result)
        
        logger.info(f"Extracted all microstructure features, total columns: {len(result.columns)}")
        return result