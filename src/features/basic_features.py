# -*- coding: utf-8 -*-
"""
基础特征提取模块

从orderbook数据中提取基本价格和量信息
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union


class BasicFeatureExtractor:
    """
    基础特征提取器
    
    从orderbook数据中提取基本价格和量信息
    """
    
    def __init__(self, levels: int = 5):
        """
        初始化特征提取器
        
        参数:
        levels (int): 提取的订单簿深度级别数量
        """
        self.levels = levels
    
    def extract_mid_price(self, orderbook: pd.DataFrame) -> pd.Series:
        """
        提取中间价
        
        参数:
        orderbook (pd.DataFrame): 订单簿数据，包含bid_price_1, ask_price_1等列
        
        返回:
        pd.Series: 中间价时间序列
        """
        # 计算最优买卖价的中间价
        mid_price = (orderbook['bid_price_1'] + orderbook['ask_price_1']) / 2
        return mid_price
    
    def extract_best_prices(self, orderbook: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        提取最优买卖价
        
        参数:
        orderbook (pd.DataFrame): 订单簿数据
        
        返回:
        Dict[str, pd.Series]: 包含最优买价和卖价的字典
        """
        return {
            'best_bid': orderbook['bid_price_1'],
            'best_ask': orderbook['ask_price_1']
        }
    
    def extract_weighted_mid_price(self, orderbook: pd.DataFrame, levels: int = None) -> pd.Series:
        """
        提取加权中间价
        
        参数:
        orderbook (pd.DataFrame): 订单簿数据
        levels (int): 考虑的深度级别，默认使用初始化时设定的级别
        
        返回:
        pd.Series: 加权中间价时间序列
        """
        if levels is None:
            levels = self.levels
        
        # 限制级别不超过可用数据
        levels = min(levels, self._get_available_levels(orderbook))
        
        # 初始化权重和价格乘积的总和
        weighted_sum_bid = pd.Series(0, index=orderbook.index)
        weighted_sum_ask = pd.Series(0, index=orderbook.index)
        total_bid_volume = pd.Series(0, index=orderbook.index)
        total_ask_volume = pd.Series(0, index=orderbook.index)
        
        # 计算加权和
        for i in range(1, levels + 1):
            bid_price_col = f'bid_price_{i}'
            ask_price_col = f'ask_price_{i}'
            bid_volume_col = f'bid_volume_{i}'
            ask_volume_col = f'ask_volume_{i}'
            
            if all(col in orderbook.columns for col in [bid_price_col, ask_price_col, bid_volume_col, ask_volume_col]):
                weighted_sum_bid += orderbook[bid_price_col] * orderbook[bid_volume_col]
                weighted_sum_ask += orderbook[ask_price_col] * orderbook[ask_volume_col]
                total_bid_volume += orderbook[bid_volume_col]
                total_ask_volume += orderbook[ask_volume_col]
        
        # 计算加权平均价格
        weighted_bid = weighted_sum_bid / total_bid_volume.replace(0, np.nan)
        weighted_ask = weighted_sum_ask / total_ask_volume.replace(0, np.nan)
        
        # 计算加权中间价
        weighted_mid_price = (weighted_bid + weighted_ask) / 2
        
        return weighted_mid_price
    
    def extract_spread(self, orderbook: pd.DataFrame) -> pd.Series:
        """
        提取买卖价差
        
        参数:
        orderbook (pd.DataFrame): 订单簿数据
        
        返回:
        pd.Series: 买卖价差时间序列
        """
        spread = orderbook['ask_price_1'] - orderbook['bid_price_1']
        return spread
    
    def extract_relative_spread(self, orderbook: pd.DataFrame) -> pd.Series:
        """
        提取相对买卖价差（价差/中间价）
        
        参数:
        orderbook (pd.DataFrame): 订单簿数据
        
        返回:
        pd.Series: 相对买卖价差时间序列
        """
        mid_price = self.extract_mid_price(orderbook)
        spread = self.extract_spread(orderbook)
        relative_spread = spread / mid_price
        return relative_spread
    
    def extract_order_book_depth(self, orderbook: pd.DataFrame, levels: int = None) -> Dict[str, pd.Series]:
        """
        提取订单簿深度
        
        参数:
        orderbook (pd.DataFrame): 订单簿数据
        levels (int): 考虑的深度级别，默认使用初始化时设定的级别
        
        返回:
        Dict[str, pd.Series]: 包含买盘深度和卖盘深度的字典
        """
        if levels is None:
            levels = self.levels
        
        # 限制级别不超过可用数据
        levels = min(levels, self._get_available_levels(orderbook))
        
        # 初始化买卖盘总量
        bid_depth = pd.Series(0, index=orderbook.index)
        ask_depth = pd.Series(0, index=orderbook.index)
        
        # 累加各级别的挂单量
        for i in range(1, levels + 1):
            bid_volume_col = f'bid_volume_{i}'
            ask_volume_col = f'ask_volume_{i}'
            
            if bid_volume_col in orderbook.columns:
                bid_depth += orderbook[bid_volume_col]
            
            if ask_volume_col in orderbook.columns:
                ask_depth += orderbook[ask_volume_col]
        
        return {
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'total_depth': bid_depth + ask_depth
        }
    
    def extract_order_book_imbalance(self, orderbook: pd.DataFrame, levels: int = None) -> pd.Series:
        """
        提取订单簿不平衡指标
        
        参数:
        orderbook (pd.DataFrame): 订单簿数据
        levels (int): 考虑的深度级别，默认使用初始化时设定的级别
        
        返回:
        pd.Series: 订单簿不平衡指标时间序列
        """
        if levels is None:
            levels = self.levels
        
        depths = self.extract_order_book_depth(orderbook, levels)
        bid_depth = depths['bid_depth']
        ask_depth = depths['ask_depth']
        
        # 计算不平衡指标: (bid_depth - ask_depth) / (bid_depth + ask_depth)
        # 范围从-1到1，正值表示买盘压力大，负值表示卖盘压力大
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).replace(0, np.nan)
        
        return imbalance
    
    def extract_price_levels(self, orderbook: pd.DataFrame, levels: int = None) -> Dict[str, pd.DataFrame]:
        """
        提取价格级别信息
        
        参数:
        orderbook (pd.DataFrame): 订单簿数据
        levels (int): 考虑的深度级别，默认使用初始化时设定的级别
        
        返回:
        Dict[str, pd.DataFrame]: 包含买盘和卖盘价格级别信息的字典
        """
        if levels is None:
            levels = self.levels
        
        # 限制级别不超过可用数据
        levels = min(levels, self._get_available_levels(orderbook))
        
        # 提取买盘价格级别
        bid_prices = pd.DataFrame(index=orderbook.index)
        bid_volumes = pd.DataFrame(index=orderbook.index)
        
        # 提取卖盘价格级别
        ask_prices = pd.DataFrame(index=orderbook.index)
        ask_volumes = pd.DataFrame(index=orderbook.index)
        
        for i in range(1, levels + 1):
            bid_price_col = f'bid_price_{i}'
            ask_price_col = f'ask_price_{i}'
            bid_volume_col = f'bid_volume_{i}'
            ask_volume_col = f'ask_volume_{i}'
            
            if bid_price_col in orderbook.columns:
                bid_prices[f'level_{i}'] = orderbook[bid_price_col]
                bid_volumes[f'level_{i}'] = orderbook[bid_volume_col]
            
            if ask_price_col in orderbook.columns:
                ask_prices[f'level_{i}'] = orderbook[ask_price_col]
                ask_volumes[f'level_{i}'] = orderbook[ask_volume_col]
        
        return {
            'bid_prices': bid_prices,
            'bid_volumes': bid_volumes,
            'ask_prices': ask_prices,
            'ask_volumes': ask_volumes
        }
    
    def extract_all_basic_features(self, orderbook: pd.DataFrame) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        """
        提取所有基本特征
        
        参数:
        orderbook (pd.DataFrame): 订单簿数据
        
        返回:
        Dict[str, Union[pd.Series, pd.DataFrame]]: 包含所有基本特征的字典
        """
        features = {}
        
        # 提取价格特征
        features['mid_price'] = self.extract_mid_price(orderbook)
        features.update(self.extract_best_prices(orderbook))
        features['weighted_mid_price'] = self.extract_weighted_mid_price(orderbook)
        features['spread'] = self.extract_spread(orderbook)
        features['relative_spread'] = self.extract_relative_spread(orderbook)
        
        # 提取深度特征
        features.update(self.extract_order_book_depth(orderbook))
        features['order_book_imbalance'] = self.extract_order_book_imbalance(orderbook)
        
        # 提取价格级别信息
        price_levels = self.extract_price_levels(orderbook)
        features.update({
            'bid_price_levels': price_levels['bid_prices'],
            'bid_volume_levels': price_levels['bid_volumes'],
            'ask_price_levels': price_levels['ask_prices'],
            'ask_volume_levels': price_levels['ask_volumes']
        })
        
        return features
    
    def _get_available_levels(self, orderbook: pd.DataFrame) -> int:
        """
        获取订单簿中可用的深度级别数量
        
        参数:
        orderbook (pd.DataFrame): 订单簿数据
        
        返回:
        int: 可用的深度级别数量
        """
        # 查找最大可用级别
        max_level = 0
        for i in range(1, 100):  # 假设最多100个级别
            if f'bid_price_{i}' in orderbook.columns and f'ask_price_{i}' in orderbook.columns:
                max_level = i
            else:
                break
        
        return max_level