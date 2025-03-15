import pandas as pd
import numpy as np


class OrderBookFeatureExtractor:
    """
    从orderbook数据中提取基础特征的类
    """
    
    def __init__(self, n_levels=10):
        """
        初始化OrderBookFeatureExtractor
        
        参数:
        n_levels (int): 订单簿深度级别数量
        """
        self.n_levels = n_levels
    
    def extract_basic_price_features(self, df):
        """
        提取基本价格特征
        
        参数:
        df (pd.DataFrame): 包含orderbook数据的DataFrame
        
        返回:
        pd.DataFrame: 添加了基本价格特征的DataFrame
        """
        result = df.copy()
        
        # 假设df包含bid_price_1, bid_price_2, ..., ask_price_1, ask_price_2, ...等列
        
        # 1. 计算中间价 (mid price)
        result['mid_price'] = (result['bid_price_1'] + result['ask_price_1']) / 2
        
        # 2. 计算加权中间价 (weighted mid price)
        bid_volume_1 = result['bid_volume_1']
        ask_volume_1 = result['ask_volume_1']
        total_volume = bid_volume_1 + ask_volume_1
        
        # 避免除以零
        total_volume = np.where(total_volume == 0, 1, total_volume)
        
        result['weighted_mid_price'] = (
            result['bid_price_1'] * ask_volume_1 + 
            result['ask_price_1'] * bid_volume_1
        ) / total_volume
        
        # 3. 计算价差 (spread)
        result['spread'] = result['ask_price_1'] - result['bid_price_1']
        
        # 4. 计算相对价差 (relative spread)
        result['relative_spread'] = result['spread'] / result['mid_price']
        
        # 5. 计算价格波动
        result['price_volatility'] = result['mid_price'].rolling(window=60).std()
        
        return result
    
    def extract_volume_features(self, df):
        """
        提取交易量相关特征
        
        参数:
        df (pd.DataFrame): 包含orderbook数据的DataFrame
        
        返回:
        pd.DataFrame: 添加了交易量特征的DataFrame
        """
        result = df.copy()
        
        # 1. 计算买卖盘总量
        bid_cols = [f'bid_volume_{i}' for i in range(1, self.n_levels + 1) if f'bid_volume_{i}' in df.columns]
        ask_cols = [f'ask_volume_{i}' for i in range(1, self.n_levels + 1) if f'ask_volume_{i}' in df.columns]
        
        result['total_bid_volume'] = result[bid_cols].sum(axis=1)
        result['total_ask_volume'] = result[ask_cols].sum(axis=1)
        
        # 2. 计算买卖盘量比
        result['volume_ratio'] = result['total_bid_volume'] / result['total_ask_volume']
        # 处理除以零的情况
        result['volume_ratio'] = result['volume_ratio'].replace([np.inf, -np.inf], np.nan).fillna(1)
        
        # 3. 计算买卖盘不平衡度
        result['volume_imbalance'] = (result['total_bid_volume'] - result['total_ask_volume']) / (result['total_bid_volume'] + result['total_ask_volume'])
        
        # 4. 计算前N档累积量
        for level in range(1, min(5, self.n_levels) + 1):
            bid_cols_level = [f'bid_volume_{i}' for i in range(1, level + 1) if f'bid_volume_{i}' in df.columns]
            ask_cols_level = [f'ask_volume_{i}' for i in range(1, level + 1) if f'ask_volume_{i}' in df.columns]
            
            result[f'cum_bid_volume_{level}'] = result[bid_cols_level].sum(axis=1)
            result[f'cum_ask_volume_{level}'] = result[ask_cols_level].sum(axis=1)
        
        return result
    
    def extract_order_flow_features(self, df):
        """
        提取订单流特征
        
        参数:
        df (pd.DataFrame): 包含orderbook数据的DataFrame
        
        返回:
        pd.DataFrame: 添加了订单流特征的DataFrame
        """
        result = df.copy()
        
        # 计算买卖盘前N档的价格和数量变化
        for level in range(1, min(3, self.n_levels) + 1):
            # 价格变化
            result[f'bid_price_{level}_change'] = result[f'bid_price_{level}'].diff()
            result[f'ask_price_{level}_change'] = result[f'ask_price_{level}'].diff()
            
            # 数量变化
            result[f'bid_volume_{level}_change'] = result[f'bid_volume_{level}'].diff()
            result[f'ask_volume_{level}_change'] = result[f'ask_volume_{level}'].diff()
        
        # 计算订单流失衡指标 (OFI - Order Flow Imbalance)
        # 简化版OFI: 买盘数量变化 - 卖盘数量变化
        result['order_flow_imbalance'] = result['bid_volume_1_change'] - result['ask_volume_1_change']
        
        # 计算加权订单流失衡
        result['weighted_ofi'] = (
            result['bid_volume_1_change'] * result['bid_price_1'] - 
            result['ask_volume_1_change'] * result['ask_price_1']
        )
        
        return result
    
    def extract_liquidity_features(self, df):
        """
        提取流动性特征
        
        参数:
        df (pd.DataFrame): 包含orderbook数据的DataFrame
        
        返回:
        pd.DataFrame: 添加了流动性特征的DataFrame
        """
        result = df.copy()
        
        # 1. 计算市场深度 (Market Depth)
        # 市场深度 = 买盘深度 + 卖盘深度
        bid_depth = 0
        ask_depth = 0
        
        for level in range(1, self.n_levels + 1):
            if f'bid_price_{level}' in df.columns and f'bid_volume_{level}' in df.columns:
                bid_depth += result[f'bid_volume_{level}'] * result[f'bid_price_{level}']
            
            if f'ask_price_{level}' in df.columns and f'ask_volume_{level}' in df.columns:
                ask_depth += result[f'ask_volume_{level}'] * result[f'ask_price_{level}']
        
        result['market_depth'] = bid_depth + ask_depth
        
        # 2. 计算有效价差 (Effective Spread)
        # 有效价差 = 2 * |成交价 - 中间价|
        # 假设我们使用最优买卖价的中间价作为参考
        if 'trade_price' in df.columns:
            result['effective_spread'] = 2 * np.abs(result['trade_price'] - result['mid_price'])
        
        # 3. 计算流动性指标 (Kyle's Lambda)
        # 简化版: 价格变化与成交量变化的比率
        if 'trade_volume' in df.columns:
            result['price_change'] = result['mid_price'].diff()
            result['volume_change'] = result['trade_volume'].diff()
            
            # 避免除以零
            result['volume_change'] = result['volume_change'].replace(0, np.nan)
            
            result['kyles_lambda'] = result['price_change'] / result['volume_change']
            result['kyles_lambda'] = result['kyles_lambda'].replace([np.inf, -np.inf], np.nan)
        
        return result
    
    def extract_all_features(self, df):
        """
        提取所有基础特征
        
        参数:
        df (pd.DataFrame): 包含orderbook数据的DataFrame
        
        返回:
        pd.DataFrame: 添加了所有基础特征的DataFrame
        """
        result = df.copy()
        
        # 依次提取各类特征
        result = self.extract_basic_price_features(result)
        result = self.extract_volume_features(result)
        result = self.extract_order_flow_features(result)
        result = self.extract_liquidity_features(result)
        
        return result