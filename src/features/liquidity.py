import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class LiquidityFeatures:
    """
    流动性特征提取器
    """

    def __init__(self, window_sizes: Optional[List[int]] = None):
        """
        初始化流动性特征提取器

        Args:
            window_sizes: 滚动窗口大小列表，默认为[5, 10, 20, 50, 100]
        """
        self.window_sizes = window_sizes or [5, 10, 20, 50, 100]

    def calculate_amihud_illiquidity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算Amihud非流动性指标

        Args:
            df: 输入DataFrame，需包含价格和交易量数据

        Returns:
            添加Amihud非流动性特征后的DataFrame
        """
        result = df.copy()

        # 检查必要列
        if 'mid_price' not in df.columns:
            if 'bid_price_1' in df.columns and 'ask_price_1' in df.columns:
                result['mid_price'] = (df['bid_price_1'] + df['ask_price_1']) / 2
            else:
                logger.warning("Missing price columns for Amihud illiquidity calculation")
                return result

        # 计算价格变化的绝对值
        result['abs_price_change'] = np.abs(result['mid_price'].pct_change())

        # 检查交易量列
        volume_col = None
        for col in ['total_volume', 'bid_volume_1', 'ask_volume_1']:
            if col in df.columns:
                volume_col = col
                break

        if volume_col is None:
            logger.warning("Missing volume column for Amihud illiquidity calculation")
            return result

        # 计算Amihud非流动性指标
        result['amihud_illiquidity'] = result['abs_price_change'] / result[volume_col]

        # 计算移动平均
        for window in self.window_sizes:
            result[f'amihud_illiquidity_ma_{window}'] = result['amihud_illiquidity'].rolling(window).mean()

        logger.info("Calculated Amihud illiquidity features")
        return result

    def calculate_market_depth(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算市场深度指标

        Args:
            df: 输入DataFrame，需包含订单簿数据

        Returns:
            添加市场深度特征后的DataFrame
        """
        result = df.copy()

        # 检查必要列
        bid_volume_cols = [f'bid_volume_{i}' for i in range(1, 6)]
        ask_volume_cols = [f'ask_volume_{i}' for i in range(1, 6)]

        # 计算买方深度
        if all(col in df.columns for col in bid_volume_cols):
            result['bid_depth'] = df[bid_volume_cols].sum(axis=1)
        elif 'bid_volume_1' in df.columns:
            result['bid_depth'] = df['bid_volume_1']

        # 计算卖方深度
        if all(col in df.columns for col in ask_volume_cols):
            result['ask_depth'] = df[ask_volume_cols].sum(axis=1)
        elif 'ask_volume_1' in df.columns:
            result['ask_depth'] = df['ask_volume_1']

        # 计算总深度
        if 'bid_depth' in result.columns and 'ask_depth' in result.columns:
            result['total_depth'] = result['bid_depth'] + result['ask_depth']

            # 计算深度不平衡
            result['depth_imbalance'] = (result['bid_depth'] - result