# -*- coding: utf-8 -*-
"""
微观结构因子模块

从orderbook数据中提取市场微观结构因子
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union
from src.features.basic_features import BasicFeatureExtractor


class MicrostructureFactorExtractor:
    """
    微观结构因子提取器

    从orderbook数据中提取市场微观结构因子
    """

    def __init__(self, levels: int = 5):
        """
        初始化因子提取器

        参数:
        levels (int): 提取的订单簿深度级别数量
        """
        self.levels = levels
        self.basic_extractor = BasicFeatureExtractor(levels=levels)

    def extract_order_book_pressure(self, orderbook: pd.DataFrame, window: int = 20) -> Dict[str, pd.Series]:
        """
        提取订单簿压力因子

        参数:
        orderbook (pd.DataFrame): 订单簿数据
        window (int): 滚动窗口大小

        返回:
        Dict[str, pd.Series]: 包含买盘压力和卖盘压力的字典
        """
        # 获取订单簿深度
        depths = self.basic_extractor.extract_order_book_depth(orderbook, self.levels)
        bid_depth = depths['bid_depth']
        ask_depth = depths['ask_depth']

        # 计算滚动平均深度
        rolling_bid_depth = bid_depth.rolling(window=window, min_periods=1).mean()
        rolling_ask_depth = ask_depth.rolling(window=window, min_periods=1).mean()

        # 计算相对于滚动平均的压力
        bid_pressure = bid_depth / rolling_bid_depth - 1
        ask_pressure = ask_depth / rolling_ask_depth - 1

        # 计算净压力（买盘压力 - 卖盘压力）
        net_pressure = bid_pressure - ask_pressure

        return {
            'bid_pressure': bid_pressure,
            'ask_pressure': ask_pressure,
            'net_pressure': net_pressure
        }

    def extract_effective_spread(self, orderbook: pd.DataFrame) -> pd.Series:
        """
        提取有效价差

        参数:
        orderbook (pd.DataFrame): 订单簿数据

        返回:
        pd.Series: 有效价差时间序列
        """
        # 获取基本价格信息
        mid_price = self.basic_extractor.extract_mid_price(orderbook)
        best_prices = self.basic_extractor.extract_best_prices(orderbook)

        # 获取深度信息
        depths = self.basic_extractor.extract_order_book_depth(orderbook, 1)
        bid_depth_1 = orderbook['bid_volume_1']
        ask_depth_1 = orderbook['ask_volume_1']

        # 计算总交易量
        total_volume = bid_depth_1 + ask_depth_1

        # 计算加权平均成交价格（假设按照挂单量比例成交）
        weighted_execution_price = (best_prices['best_bid'] * ask_depth_1 + best_prices['best_ask'] * bid_depth_1) / total_volume

        # 计算有效价差（加权成交价与中间价的偏差）
        effective_spread = 2 * abs(weighted_execution_price - mid_price) / mid_price

        return effective_spread

    def extract_market_depth_factor(self, orderbook: pd.DataFrame, price_levels: int = None) -> pd.Series:
        """
        提取市场深度因子

        参数:
        orderbook (pd.DataFrame): 订单簿数据
        price_levels (int): 考虑的价格级别数量

        返回:
        pd.Series: 市场深度因子时间序列
        """
        if price_levels is None:
            price_levels = self.levels

        # 获取价格级别信息
        price_level_data = self.basic_extractor.extract_price_levels(orderbook, price_levels)

        # 获取中间价
        mid_price = self.basic_extractor.extract_mid_price(orderbook)

        # 计算买盘深度加权价格距离
        bid_depth_distance = pd.Series(0.0, index=orderbook.index)
        for i in range(1, price_levels + 1):
            level_col = f'level_{i}'
            if level_col in price_level_data['bid_prices'].columns:
                price_distance = (mid_price - price_level_data['bid_prices'][level_col]) / mid_price
                volume = price_level_data['bid_volumes'][level_col]
                bid_depth_distance += price_distance * volume

        # 计算卖盘深度加权价格距离
        ask_depth_distance = pd.Series(0.0, index=orderbook.index)
        for i in range(1, price_levels + 1):
            level_col = f'level_{i}'
            if level_col in price_level_data['ask_prices'].columns:
                price_distance = (price_level_data['ask_prices'][level_col] - mid_price) / mid_price
                volume = price_level_data['ask_volumes'][level_col]
                ask_depth_distance += price_distance * volume

        # 获取总深度
        depths = self.basic_extractor.extract_order_book_depth(orderbook, price_levels)
        bid_total_volume = depths['bid_depth']
        ask_total_volume = depths['ask_depth']

        # 计算加权平均距离
        bid_avg_distance = bid_depth_distance / bid_total_volume
        ask_avg_distance = ask_depth_distance / ask_total_volume

        # 计算市场深度因子（买卖盘平均距离的调和平均）
        market_depth_factor = 2 / (1/bid_avg_distance + 1/ask_avg_distance)

        return market_depth_factor

    def extract_order_flow_toxicity(self, orderbook: pd.DataFrame, window: int = 50) -> pd.Series:
        """
        提取订单流毒性因子

        参数:
        orderbook (pd.DataFrame): 订单簿数据
        window (int): 滚动窗口大小

        返回:
        pd.Series: 订单流毒性因子时间序列
        """
        # 获取订单簿不平衡指标
        imbalance = self.basic_extractor.extract_order_book_imbalance(orderbook)

        # 计算中间价
        mid_price = self.basic_extractor.extract_mid_price(orderbook)

        # 计算中间价收益率
        mid_price_returns = mid_price.pct_change()

        # 计算不平衡指标与后续收益率的相关性（滚动窗口）
        # 这里使用简化的方法：计算不平衡指标与后续收益率的乘积的滚动和
        # 正值表示不平衡指标对价格变动有预测性，负值表示预测性较弱
        imbalance_shifted = imbalance.shift(1)  # 使用前一时刻的不平衡指标
        predictive_power = (imbalance_shifted * mid_price_returns).rolling(window=window, min_periods=10).mean()

        # 计算订单流毒性（预测能力的绝对值）
        # 毒性高表示信息不对称程度高，不平衡指标对价格变动的预测能力强
        toxicity = predictive_power.abs()

        return toxicity

    def extract_kyle_lambda(self, orderbook: pd.DataFrame, trades: pd.DataFrame, window: int = 50) -> pd.Series:
        """
        提取Kyle's Lambda（价格影响因子）

        参数:
        orderbook (pd.DataFrame): 订单簿数据
        trades (pd.DataFrame): 成交数据，包含volume和direction列（买为1，卖为-1）
        window (int): 滚动窗口大小

        返回:
        pd.Series: Kyle's Lambda时间序列
        """
        # 计算中间价
        mid_price = self.basic_extractor.extract_mid_price(orderbook)

        # 计算中间价收益率
        mid_price_returns = mid_price.pct_change()

        # 计算净成交量（考虑方向）
        if 'direction' in trades.columns and 'volume' in trades.columns:
            net_volume = trades['volume'] * trades['direction']

            # 将净成交量重采样到与orderbook相同的时间索引
            resampled_net_volume = net_volume.reindex(mid_price.index, method='ffill').fillna(0)

            # 计算Kyle's Lambda（价格变动与净成交量的比率）
            # 使用滚动回归的方法
            rolling_cov = mid_price_returns.rolling(window=window, min_periods=10).cov(resampled_net_volume)
            rolling_var = resampled_net_volume.rolling(window=window, min_periods=10).var()

            # 避免除以零
            kyle_lambda = rolling_cov / rolling_var.replace(0, np.nan)

            return kyle_lambda
        else:
            raise ValueError("成交数据必须包含volume和direction列")

    def extract_amihud_illiquidity(self, orderbook: pd.DataFrame, trades: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        提取Amihud非流动性因子

        参数:
        orderbook (pd.DataFrame): 订单簿数据
        trades (pd.DataFrame): 成交数据，包含volume列
        window (int): 滚动窗口大小

        返回:
        pd.Series: Amihud非流动性因子时间序列
        """
        # 计算中间价
        mid_price = self.basic_extractor.extract_mid_price(orderbook)

        # 计算中间价收益率的绝对值
        abs_returns = mid_price.pct_change().abs()

        # 将成交量重采样到与orderbook相同的时间索引
        if 'volume' in trades.columns:
            resampled_volume = trades['volume'].reindex(mid_price.index, method='ffill').fillna(0)

            # 计算Amihud非流动性因子（收益率绝对值与成交量的比率）
            # 使用滚动窗口计算平均值
            amihud_illiquidity = (abs_returns / resampled_volume.replace(0, np.nan)).rolling(window=window, min_periods=5).mean()

            return amihud_illiquidity
        else:
            raise ValueError("成交数据必须包含volume列")

    def extract_vpin(self, orderbook: pd.DataFrame, trades: pd.DataFrame, bucket_size: int = 50, num_buckets: int = 50) -> pd.Series:
        """
        提取成交量同步概率信息交易指标 (VPIN - Volume-Synchronized Probability of Informed Trading)

        参数:
        orderbook (pd.DataFrame): 订单簿数据
        trades (pd.DataFrame): 成交数据，包含volume和price列
        bucket_size (int): 每个桶的成交量大小（按总成交量的百分比）
        num_buckets (int): 计算VPIN的桶数量

        返回:
        pd.Series: VPIN时间序列
        """
        if 'volume' not in trades.columns or 'price' not in trades.columns:
            raise ValueError("成交数据必须包含volume和price列")

        # 计算中间价
        mid_price = self.basic_extractor.extract_mid_price(orderbook)

        # 将成交价格重采样到与orderbook相同的时间索引
        resampled_price = trades['price'].reindex(mid_price.index, method='ffill')

        # 将成交量重采样到与orderbook相同的时间索引
        resampled_volume = trades['volume'].reindex(mid_price.index, method='ffill').fillna(0)

        # 计算总成交量
        total_volume = resampled_volume.sum()

        # 计算每个桶的目标成交量
        bucket_target = total_volume / bucket_size

        # 初始化桶
        buckets = []
        current_bucket_volume = 0
        current_bucket_buy_volume = 0
        current_bucket_sell_volume = 0
        last_price = None

        # 填充桶
        for idx, (price, volume) in enumerate(zip(resampled_price, resampled_volume)):
            if pd.isna(price) or volume == 0:
                continue

            # 确定交易方向（根据价格变动）
            if last_price is not None:
                if price > last_price:
                    # 买方发起
                    current_bucket_buy_volume += volume
                elif price < last_price:
                    # 卖方发起
                    current_bucket_sell_volume += volume
                else:
                    # 价格不变，平分
                    current_bucket_buy_volume += volume / 2
                    current_bucket_sell_volume += volume / 2
            else:
                # 第一个交易，平分
                current_bucket_buy_volume += volume / 2
                current_bucket_sell_volume += volume / 2

            last_price = price
            current_bucket_volume += volume

            # 检查桶是否已满
            if current_bucket_volume >= bucket_target:
                buckets.append((current_bucket_buy_volume, current_bucket_sell_volume))
                current_bucket_volume = 0
                current_bucket_buy_volume = 0
                current_bucket_sell_volume = 0

        # 添加最后一个未满的桶
        if current_bucket_volume > 0:
            buckets.append((current_bucket_buy_volume, current_bucket_sell_volume))

        # 计算VPIN
        vpin_values = []
        for i in range(len(buckets) - num_buckets + 1):
            window_buckets = buckets[i:i+num_buckets]
            total_buy = sum(b[0] for b in window_buckets)
            total_sell = sum(b[1] for b in window_buckets)
            total = total_buy + total_sell

            # 计算买卖不平衡的绝对值之和
            imbalance_sum = sum(abs(b[0] - b[1]) for b in window_buckets)

            # 计算VPIN
            vpin = imbalance_sum / total if total > 0 else np.nan
            vpin_values.append(vpin)

        # 创建VPIN时间序列
        # 注意：VPIN的时间索引是近似的，因为它基于成交量桶而不是时间
        vpin_index = mid_price.index[num_buckets-1:]
        if len(vpin_values) < len(vpin_index):
            vpin_index = vpin_index[:len(vpin_values)]

        vpin_series = pd.Series(vpin_values, index=vpin_index[:len(vpin_values)])

        # 将VPIN值填充到所有时间点
        full_vpin = vpin_series.reindex(mid_price.index, method='ffill')

        return full_vpin

    def extract_order_book_slope(self, orderbook: pd.DataFrame, levels: int = None) -> Dict[str, pd.Series]:
        """
        提取订单簿斜率因子

        参数:
        orderbook (pd.DataFrame): 订单簿数据
        levels (int): 考虑的深度级别，默认使用初始化时设定的级别

        返回:
        Dict[str, pd.Series]: 包含买盘斜率和卖盘斜率的字典
        """
        if levels is None:
            levels = self.levels

        # 获取价格级别信息
        price_level_data = self.basic_extractor.extract_price_levels(orderbook, levels)

        # 获取中间价
        mid_price = self.basic_extractor.extract_mid_price(orderbook)

        # 初始化斜率计算的数据
        bid_slopes = pd.Series(0.0, index=orderbook.index)
        ask_slopes = pd.Series(0.0, index=orderbook.index)

        # 计算买盘斜率（使用线性回归）
        for idx in orderbook.index:
            bid_prices = []
            bid_volumes = []
            ask_prices = []
            ask_volumes = []

            # 收集买盘数据点
            for i in range(1, levels + 1):
                level_col = f'level_{i}'
                if level_col in price_level_data['bid_prices'].columns:
                    price = price_level_data['bid_prices'].loc[idx, level_col]
                    volume = price_level_data['bid_volumes'].loc[idx, level_col]

                    if not pd.isna(price) and not pd.isna(volume) and volume > 0:
                        # 计算相对价格（相对于中间价的百分比）
                        rel_price = (price - mid_price[idx]) / mid_price[idx]
                        bid_prices.append(rel_price)
                        bid_volumes.append(volume)

            # 收集卖盘数据点
            for i in range(1, levels + 1):
                level_col = f'level_{i}'
                if level_col in price_level_data['ask_prices'].columns:
                    price = price_level_data['ask_prices'].loc[idx, level_col]
                    volume = price_level_data['ask_volumes'].loc[idx, level_col]

                    if not pd.isna(price) and not pd.isna(volume) and volume > 0:
                        # 计算相对价格（相对于中间价的百分比）
                        rel_price = (price - mid_price[idx]) / mid_price[idx]
                        ask_prices.append(rel_price)
                        ask_volumes.append(volume)

            # 计算买盘斜率（如果有足够的数据点）
            if len(bid_prices) >= 2:
                bid_slope, _, _, _, _ = np.polyfit(bid_prices, bid_volumes, 1, full=True)
                bid_slopes[idx] = bid_slope[0]

            # 计算卖盘斜率（如果有足够的数据点）
            if len(ask_prices) >= 2:
                ask_slope, _, _, _, _ = np.polyfit(ask_prices, ask_volumes, 1, full=True)
                ask_slopes[idx] = ask_slope[0]

        # 计算买卖盘斜率比
        slope_ratio = bid_slopes / ask_slopes.replace(0, np.nan)

        return {
            'bid_slope': bid_slopes,
            'ask_slope': ask_slopes,
            'slope_ratio': slope_ratio
        }

    def extract_all_microstructure_factors(self, orderbook: pd.DataFrame, trades: pd.DataFrame = None) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        """
        提取所有微观结构因子

        参数:
        orderbook (pd.DataFrame): 订单簿数据
        trades (pd.DataFrame): 成交数据，可选

        返回:
        Dict[str, Union[pd.Series, pd.DataFrame]]: 包含所有微观结构因子的字典
        """
        factors = {}

        # 提取不需要成交数据的因子
        factors.update(self.extract_order_book_pressure(orderbook))
        factors['effective_spread'] = self.extract_effective_spread(orderbook)
        factors['market_depth'] = self.extract_market_depth_factor(orderbook)
        factors['order_flow_toxicity'] = self.extract_order_flow_toxicity(orderbook)
        factors.update(self.extract_order_book_slope(orderbook))

        # 提取需要成交数据的因子
        if trades is not None:
            try:
                factors['kyle_lambda'] = self.extract_kyle_lambda(orderbook, trades)
            except ValueError as e:
                print(f"无法计算Kyle's Lambda: {e}")

            try:
                factors['amihud_illiquidity'] = self.extract_amihud_illiquidity(orderbook, trades)
            except ValueError as e:
                print(f"无法计算Amihud非流动性因子: {e}")

            try:
                factors['vpin'] = self.extract_vpin(orderbook, trades)
            except ValueError as e:
                print(f"无法计算VPIN: {e}")

        return factors