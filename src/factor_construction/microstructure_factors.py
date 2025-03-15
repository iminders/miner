import pandas as pd
import numpy as np
from numba import jit


class MicrostructureFactors:
    """
    市场微观结构因子构建类
    """
    
    def __init__(self, n_levels=10):
        """
        初始化MicrostructureFactors
        
        参数:
        n_levels (int): 订单簿深度级别数量
        """
        self.n_levels = n_levels
    
    def calculate_order_imbalance_factors(self, df):
        """
        计算订单簿不平衡因子
        
        参数:
        df (pd.DataFrame): 包含orderbook数据的DataFrame
        
        返回:
        pd.DataFrame: 添加了订单簿不平衡因子的DataFrame
        """
        result = df.copy()
        
        # 1. 计算订单簿不平衡因子 (OIR - Order Imbalance Ratio)
        # OIR = (买盘量 - 卖盘量) / (买盘量 + 卖盘量)
        if 'total_bid_volume' in df.columns and 'total_ask_volume' in df.columns:
            total_volume = result['total_bid_volume'] + result['total_ask_volume']
            result['order_imbalance_ratio'] = (result['total_bid_volume'] - result['total_ask_volume']) / total_volume
        else:
            # 如果没有预先计算的总量，则计算前N档的总量
            bid_cols = [f'bid_volume_{i}' for i in range(1, self.n_levels + 1) if f'bid_volume_{i}' in df.columns]
            ask_cols = [f'ask_volume_{i}' for i in range(1, self.n_levels + 1) if f'ask_volume_{i}' in df.columns]
            
            total_bid = result[bid_cols].sum(axis=1)
            total_ask = result[ask_cols].sum(axis=1)
            total_volume = total_bid + total_ask
            
            result['order_imbalance_ratio'] = (total_bid - total_ask) / total_volume
        
        # 2. 计算加权订单簿不平衡因子 (WOIR - Weighted Order Imbalance Ratio)
        # WOIR = Σ(买盘量_i * 买盘价格_i - 卖盘量_i * 卖盘价格_i) / Σ(买盘量_i * 买盘价格_i + 卖盘量_i * 卖盘价格_i)
        weighted_bid = 0
        weighted_ask = 0
        
        for level in range(1, self.n_levels + 1):
            bid_price_col = f'bid_price_{level}'
            bid_volume_col = f'bid_volume_{level}'
            ask_price_col = f'ask_price_{level}'
            ask_volume_col = f'ask_volume_{level}'
            
            if all(col in df.columns for col in [bid_price_col, bid_volume_col, ask_price_col, ask_volume_col]):
                weighted_bid += result[bid_price_col] * result[bid_volume_col]
                weighted_ask += result[ask_price_col] * result[ask_volume_col]
        
        weighted_total = weighted_bid + weighted_ask
        result['weighted_order_imbalance_ratio'] = (weighted_bid - weighted_ask) / weighted_total
        
        # 3. 计算订单簿倾斜度 (Order Book Slope)
        # 订单簿倾斜度 = (最优卖价 - 最优买价) / (最优卖量 + 最优买量)
        if all(col in df.columns for col in ['ask_price_1', 'bid_price_1', 'ask_volume_1', 'bid_volume_1']):
            price_diff = result['ask_price_1'] - result['bid_price_1']
            volume_sum = result['ask_volume_1'] + result['bid_volume_1']
            result['order_book_slope'] = price_diff / volume_sum
            result['order_book_slope'] = result['order_book_slope'].replace([np.inf, -np.inf], np.nan)
        
        return result
    
    def calculate_price_pressure_factors(self, df):
        """
        计算价格压力因子
        
        参数:
        df (pd.DataFrame): 包含orderbook数据的DataFrame
        
        返回:
        pd.DataFrame: 添加了价格压力因子的DataFrame
        """
        result = df.copy()
        
        # 1. 计算买卖盘价格压力
        # 买盘价格压力 = Σ(买盘量_i / (中间价 - 买盘价格_i))
        # 卖盘价格压力 = Σ(卖盘量_i / (卖盘价格_i - 中间价))
        if 'mid_price' in df.columns:
            bid_pressure = 0
            ask_pressure = 0
            
            for level in range(1, self.n_levels + 1):
                bid_price_col = f'bid_price_{level}'
                bid_volume_col = f'bid_volume_{level}'
                ask_price_col = f'ask_price_{level}'
                ask_volume_col = f'ask_volume_{level}'
                
                if all(col in df.columns for col in [bid_price_col, bid_volume_col]):
                    # 避免除以零或负数
                    price_diff = result['mid_price'] - result[bid_price_col]
                    price_diff = price_diff.replace(0, np.nan)
                    bid_pressure += result[bid_volume_col] / price_diff
                
                if all(col in df.columns for col in [ask_price_col, ask_volume_col]):
                    # 避免除以零或负数
                    price_diff = result[ask_price_col] - result['mid_price']
                    price_diff = price_diff.replace(0, np.nan)
                    ask_pressure += result[ask_volume_col] / price_diff
            
            result['bid_price_pressure'] = bid_pressure
            result['ask_price_pressure'] = ask_pressure
            
            # 2. 计算净价格压力
            result['net_price_pressure'] = result['bid_price_pressure'] - result['ask_price_pressure']
            
            # 3. 计算相对价格压力
            total_pressure = result['bid_price_pressure'] + result['ask_price_pressure']
            result['relative_price_pressure'] = (result['bid_price_pressure'] - result['ask_price_pressure']) / total_pressure
            result['relative_price_pressure'] = result['relative_price_pressure'].replace([np.inf, -np.inf], np.nan)
        
        return result
    
    def calculate_liquidity_factors(self, df):
        """
        计算流动性因子
        
        参数:
        df (pd.DataFrame): 包含orderbook数据的DataFrame
        
        返回:
        pd.DataFrame: 添加了流动性因子的DataFrame
        """
        result = df.copy()
        
        # 1. 计算有效价差 (Effective Spread)
        if 'spread' in df.columns and 'mid_price' in df.columns:
            result['relative_effective_spread'] = result['spread'] / result['mid_price']
        
        # 2. 计算市场深度 (Market Depth)
        if 'market_depth' not in df.columns:
            bid_depth = 0
            ask_depth = 0
            
            for level in range(1, self.n_levels + 1):
                bid_price_col = f'bid_price_{level}'
                bid_volume_col = f'bid_volume_{level}'
                ask_price_col = f'ask_price_{level}'
                ask_volume_col = f'ask_volume_{level}'
                
                if all(col in df.columns for col in [bid_price_col, bid_volume_col]):
                    bid_depth += result[bid_volume_col] * result[bid_price_col]
                
                if all(col in df.columns for col in [ask_price_col, ask_volume_col]):
                    ask_depth += result[ask_volume_col] * result[ask_price_col]
            
            result['market_depth'] = bid_depth + ask_depth
        
        # 3. 计算流动性比率 (Liquidity Ratio)
        # 流动性比率 = 价格变化 / 成交量变化
        if 'mid_price' in df.columns and 'trade_volume' in df.columns:
            result['price_change_pct'] = result['mid_price'].pct_change()
            result['volume_change'] = result['trade_volume'].diff()
            
            # 避免除以零
            result['volume_change'] = result['volume_change'].replace(0, np.nan)
            
            result['liquidity_ratio'] = np.abs(result['price_change_pct']) / result['volume_change']
            result['liquidity_ratio'] = result['liquidity_ratio'].replace([np.inf, -np.inf], np.nan)
        
        # 4. 计算Amihud非流动性指标 (Amihud Illiquidity)
        # Amihud = |收益率| / 成交量
        if 'mid_price' in df.columns and 'trade_volume' in df.columns:
            result['return'] = result['mid_price'].pct_change()
            result['amihud_illiquidity'] = np.abs(result['return']) / result['trade_volume']
            result['amihud_illiquidity'] = result['amihud_illiquidity'].replace([np.inf, -np.inf], np.nan)
        
        return result
    
    def calculate_order_flow_toxicity_factors(self, df):
        """
        计算订单流毒性因子
        
        参数:
        df (pd.DataFrame): 包含orderbook数据的DataFrame
        
        返回:
        pd.DataFrame: 添加了订单流毒性因子的DataFrame
        """
        result = df.copy()
        
        # 1. 计算VPIN (Volume-Synchronized Probability of Informed Trading)
        # 简化版VPIN: |买入成交量 - 卖出成交量| / 总成交量
        if 'bid_volume_1_change' in df.columns and 'ask_volume_1_change' in df.columns:
            buy_volume = np.where(result['bid_volume_1_change'] > 0, result['bid_volume_1_change'], 0)
            sell_volume = np.where(result['ask_volume_1_change'] < 0, -result['ask_volume_1_change'], 0)
            
            total_volume = buy_volume + sell_volume
            # 避免除以零
            total_volume = np.where(total_volume == 0, np.nan, total_volume)
            
            result['vpin'] = np.abs(buy_volume - sell_volume) / total_volume
        
        # 2. 计算订单流毒性指标 (OFT - Order Flow Toxicity)
        # 简化版OFT: 价格变化与订单流失衡的相关性
        if 'mid_price' in df.columns and 'order_flow_imbalance' in df.columns:
            # 使用滚动窗口计算相关性
            result['price_change'] = result['mid_price'].diff()
            
            # 使用60秒窗口计算相关性
            result['oft_correlation'] = result['price_change'].rolling(window=60).corr(result['order_flow_imbalance'])
            
            # 计算订单流毒性指标
            result['order_flow_toxicity'] = np.abs(result['oft_correlation'])
        
        return result
    
    def calculate_volatility_factors(self, df):
        """
        计算波动性相关因子
        
        参数:
        df (pd.DataFrame): 包含orderbook数据的DataFrame
        
        返回:
        pd.DataFrame: 添加了波动性因子的DataFrame
        """
        result = df.copy()
        
        # 1. 计算价格波动率
        if 'mid_price' in df.columns:
            # 计算收益率
            result['return'] = result['mid_price'].pct_change()
            
            # 计算不同窗口的波动率
            windows = [30, 60, 300, 600]  # 30秒, 1分钟, 5分钟, 10分钟
            for window in windows:
                result[f'volatility_{window}s'] = result['return'].rolling(window=window).std() * np.sqrt(window)
        
        # 2. 计算价差波动率
        if 'spread' in df.columns:
            windows = [30, 60, 300]
            for window in windows:
                result[f'spread_volatility_{window}s'] = result['spread'].rolling(window=window).std()
        
        # 3. 计算订单簿不平衡波动率
        if 'order_imbalance_ratio' in result.columns:
            windows = [30, 60, 300]
            for window in windows:
                result[f'oir_volatility_{window}s'] = result['order_imbalance_ratio'].rolling(window=window).std()
        
        # 4. 计算价格压力波动率
        if 'net_price_pressure' in result.columns:
            windows = [30, 60, 300]
            for window in windows:
                result[f'price_pressure_volatility_{window}s'] = result['net_price_pressure'].rolling(window=window).std()
        
        # 5. 计算已实现波动率 (Realized Volatility)
        if 'return' in result.columns:
            windows = [60, 300, 600]
            for window in windows:
                result[f'realized_volatility_{window}s'] = np.sqrt(
                    (result['return'] ** 2).rolling(window=window).sum()
                )
        
        return result
    
    def calculate_microstructure_noise_factors(self, df):
        """
        计算微观结构噪声因子
        
        参数:
        df (pd.DataFrame): 包含orderbook数据的DataFrame
        
        返回:
        pd.DataFrame: 添加了微观结构噪声因子的DataFrame
        """
        result = df.copy()
        
        # 1. 计算自相关系数
        if 'return' in result.columns:
            windows = [60, 300]
            lags = [1, 2, 3, 5]
            
            for window in windows:
                for lag in lags:
                    result[f'return_autocorr_{lag}_{window}s'] = result['return'].rolling(window=window).apply(
                        lambda x: x.autocorr(lag) if len(x) > lag else np.nan
                    )
        
        # 2. 计算收益率方差比
        if 'return' in result.columns:
            windows = [60, 300]
            for window in windows:
                # 计算原始收益率的方差
                var_ret = result['return'].rolling(window=window).var()
                
                # 计算聚合收益率的方差
                agg_ret = result['return'].rolling(window=2).sum().rolling(window=window//2).var()
                
                # 计算方差比
                result[f'variance_ratio_{window}s'] = (agg_ret / (2 * var_ret))
                result[f'variance_ratio_{window}s'] = result[f'variance_ratio_{window}s'].replace([np.inf, -np.inf], np.nan)
        
        # 3. 计算Roll隐含价差
        if 'return' in result.columns:
            windows = [60, 300]
            for window in windows:
                # 计算一阶自协方差
                def roll_cov(x):
                    if len(x) <= 1:
                        return np.nan
                    return np.cov(x[:-1], x[1:])[0, 1]
                
                autocov = result['return'].rolling(window=window).apply(roll_cov)
                
                # 计算Roll隐含价差
                result[f'roll_implicit_spread_{window}s'] = 2 * np.sqrt(-autocov)
                # 当自协方差为正时，Roll模型不适用，设为NaN
                result[f'roll_implicit_spread_{window}s'] = np.where(autocov >= 0, np.nan, result[f'roll_implicit_spread_{window}s'])
        
        return result
    
    def calculate_information_flow_factors(self, df):
        """
        计算信息流因子
        
        参数:
        df (pd.DataFrame): 包含orderbook数据的DataFrame
        
        返回:
        pd.DataFrame: 添加了信息流因子的DataFrame
        """
        result = df.copy()
        
        # 1. 计算价格发现效率 (Price Discovery Efficiency)
        if 'mid_price' in df.columns:
            windows = [60, 300]
            for window in windows:
                # 计算价格变化
                price_change = result['mid_price'].diff()
                
                # 计算价格变化的自相关性
                result[f'price_efficiency_{window}s'] = 1 - np.abs(
                    price_change.rolling(window=window).apply(
                        lambda x: x.autocorr(1) if len(x) > 1 else np.nan
                    )
                )
        
        # 2. 计算信息比率 (Information Ratio)
        if 'return' in result.columns:
            windows = [60, 300, 600]
            for window in windows:
                # 计算累积收益率
                cum_return = result['return'].rolling(window=window).sum()
                
                # 使用波动率作为风险度量
                vol = result['return'].rolling(window=window).std()
                vol = vol.replace(0, np.nan)  # 避免除以零
                
                # 计算信息比率
                result[f'information_ratio_{window}s'] = cum_return / vol
                result[f'information_ratio_{window}s'] = result[f'information_ratio_{window}s'].replace([np.inf, -np.inf], np.nan)
        
        return result
    
    def calculate_advanced_factors(self, df):
        """
        计算高级微观结构因子
        
        参数:
        df (pd.DataFrame): 包含orderbook数据的DataFrame
        
        返回:
        pd.DataFrame: 添加了高级微观结构因子的DataFrame
        """
        result = df.copy()
        
        # 1. 计算订单簿形状因子 (Order Book Shape)
        # 订单簿形状 = 买盘斜率 / 卖盘斜率
        bid_slopes = []
        ask_slopes = []
        
        for level in range(1, min(5, self.n_levels)):
            if level + 1 <= self.n_levels:
                bid_price_col1 = f'bid_price_{level}'
                bid_price_col2 = f'bid_price_{level+1}'
                bid_volume_col1 = f'bid_volume_{level}'
                bid_volume_col2 = f'bid_volume_{level+1}'
                
                ask_price_col1 = f'ask_price_{level}'
                ask_price_col2 = f'ask_price_{level+1}'
                ask_volume_col1 = f'ask_volume_{level}'
                ask_volume_col2 = f'ask_volume_{level+1}'
                
                if all(col in df.columns for col in [bid_price_col1, bid_price_col2, bid_volume_col1, bid_volume_col2]):
                    price_diff = result[bid_price_col1] - result[bid_price_col2]
                    volume_diff = result[bid_volume_col2] - result[bid_volume_col1]
                    
                    # 避免除以零
                    price_diff = price_diff.replace(0, np.nan)
                    
                    bid_slope = volume_diff / price_diff
                    bid_slopes.append(bid_slope)
                
                if all(col in df.columns for col in [ask_price_col1, ask_price_col2, ask_volume_col1, ask_volume_col2]):
                    price_diff = result[ask_price_col2] - result[ask_price_col1]
                    volume_diff = result[ask_volume_col2] - result[ask_volume_col1]
                    
                    # 避免除以零
                    price_diff = price_diff.replace(0, np.nan)
                    
                    ask_slope = volume_diff / price_diff
                    ask_slopes.append(ask_slope)
        
        if bid_slopes and ask_slopes:
            avg_bid_slope = sum(bid_slopes) / len(bid_slopes)
            avg_ask_slope = sum(ask_slopes) / len(ask_slopes)
            
            # 避免除以零
            avg_ask_slope = avg_ask_slope.replace(0, np.nan)
            
            result['order_book_shape'] = avg_bid_slope / avg_ask_slope
            result['order_book_shape'] = result['order_book_shape'].replace([np.inf, -np.inf], np.nan)
        
        # 2. 计算订单簿弹性 (Order Book Resilience)
        # 订单簿弹性 = 价格冲击后恢复速度
        if 'mid_price' in df.columns:
            windows = [60, 300]
            for window in windows:
                # 计算价格偏离移动平均的程度
                ma = result['mid_price'].rolling(window=window).mean()
                deviation = (result['mid_price'] - ma) / ma
                
                # 计算偏离的自相关性（负相关表示更快的均值回归）
                result[f'order_book_resilience_{window}s'] = -deviation.rolling(window=window).apply(
                    lambda x: x.autocorr(1) if len(x) > 1 else np.nan
                )
        
        return result
    
    def calculate_all_factors(self, df):
        """
        计算所有微观结构因子
        
        参数:
        df (pd.DataFrame): 包含orderbook数据的DataFrame
        
        返回:
        pd.DataFrame: 添加了所有微观结构因子的DataFrame
        """
        result = df.copy()
        
        # 预处理：计算中间价和价差（如果不存在）
        if 'mid_price' not in result.columns and 'bid_price_1' in result.columns and 'ask_price_1' in result.columns:
            result['mid_price'] = (result['bid_price_1'] + result['ask_price_1']) / 2
        
        if 'spread' not in result.columns and 'bid_price_1' in result.columns and 'ask_price_1' in result.columns:
            result['spread'] = result['ask_price_1'] - result['bid_price_1']
        
        # 依次计算各类因子
        result = self.calculate_order_imbalance_factors(result)
        result = self.calculate_price_pressure_factors(result)
        result = self.calculate_liquidity_factors(result)
        result = self.calculate_order_flow_toxicity_factors(result)
        result = self.calculate_volatility_factors(result)
        result = self.calculate_microstructure_noise_factors(result)
        result = self.calculate_information_flow_factors(result)
        result = self.calculate_advanced_factors(result)
        
        return result