import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

class FactorBuilder:
    """
    因子构建器，用于将特征转换为预测因子
    """
    
    def __init__(self, 
                 prediction_horizons: Optional[List[int]] = None,
                 scaler_type: str = 'robust'):
        """
        初始化因子构建器
        
        Args:
            prediction_horizons: 预测时间范围列表（单位：样本数），默认为[1, 5, 10, 20, 50]
            scaler_type: 标准化方法，'standard'或'robust'
        """
        self.prediction_horizons = prediction_horizons or [1, 5, 10, 20, 50]
        self.scaler_type = scaler_type
        self.scaler = RobustScaler() if scaler_type == 'robust' else StandardScaler()
        self.pca = None
        self.feature_importance = {}
    
    def create_return_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建未来收益率标签
        
        Args:
            df: 输入DataFrame，需包含价格数据
            
        Returns:
            添加未来收益率标签后的DataFrame
        """
        result = df.copy()
        
        # 检查必要列
        if 'mid_price' not in df.columns:
            if 'bid_price_1' in df.columns and 'ask_price_1' in df.columns:
                result['mid_price'] = (df['bid_price_1'] + df['ask_price_1']) / 2
            else:
                logger.warning("Missing price columns for return label creation")
                return result
        
        # 计算未来收益率
        for horizon in self.prediction_horizons:
            # 绝对收益率
            result[f'future_return_{horizon}'] = result['mid_price'].shift(-horizon) / result['mid_price'] - 1
            
            # 收益率符号（用于分类）
            result[f'future_return_sign_{horizon}'] = np.sign(result[f'future_return_{horizon}'])
            
            # 波动率归一化收益率
            rolling_std = result['mid_price'].pct_change().rolling(horizon * 5).std()
            result[f'future_return_zscore_{horizon}'] = result[f'future_return_{horizon}'] / rolling_std
        
        logger.info(f"Created return labels for {len(self.prediction_horizons)} horizons")
        return result
    
    def normalize_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        标准化特征
        
        Args:
            df: 输入DataFrame
            feature_cols: 需要标准化的特征列
            
        Returns:
            标准化后的DataFrame
        """
        result = df.copy()
        
        # 过滤有效的特征列
        valid_cols = [col for col in feature_cols if col in df.columns]
        if not valid_cols:
            logger.warning("No valid feature columns for normalization")
            return result
        
        # 提取特征矩阵
        X = result[valid_cols].values
        
        # 处理缺失值
        X = np.nan_to_num(X, nan=0)
        
        # 拟合并转换
        X_scaled = self.scaler.fit_transform(X)
        
        # 更新DataFrame
        for i, col in enumerate(valid_cols):
            result[f'{col}_norm'] = X_scaled[:, i]
        
        logger.info(f"Normalized {len(valid_cols)} features")
        return result
    
    def create_pca_factors(self, df: pd.DataFrame, feature_cols: List[str], n_components: int = 5) -> pd.DataFrame:
        """
        使用PCA创建因子
        
        Args:
            df: 输入DataFrame
            feature_cols: 用于PCA的特征列
            n_components: PCA组件数量
            
        Returns:
            添加PCA因子后的DataFrame
        """
        result = df.copy()
        
        # 过滤有效的特征列
        valid_cols = [col for col in feature_cols if col in df.columns]
        if not valid_cols:
            logger.warning("No valid feature columns for PCA")
            return result
        
        # 提取特征矩阵
        X = result[valid_cols].values
        
        # 处理缺失值
        X = np.nan_to_num(X, nan=0)
        
        # 拟合PCA
        self.pca = PCA(n_components=min(n_components, len(valid_cols)))
        X_pca = self.pca.fit_transform(X)
        
        # 添加PCA因子
        for i in range(X_pca.shape[1]):
            result[f'pca_factor_{i+1}'] = X_pca[:, i]
        
        # 记录特征重要性
        for i, col in enumerate(valid_cols):
            self.feature_importance[col] = np.sum(np.abs(self.pca.components_[:, i]))
        
        logger.info(f"Created {X_pca.shape[1]} PCA factors, explained variance: {np.sum(self.pca.explained_variance_ratio_):.4f}")
        return result
    
    def create_momentum_factors(self, df: pd.DataFrame, price_col: str = 'mid_price') -> pd.DataFrame:
        """
        创建动量因子
        
        Args:
            df: 输入DataFrame
            price_col: 价格列名
            
        Returns:
            添加动量因子后的DataFrame
        """
        result = df.copy()
        
        # 检查价格列
        if price_col not in df.columns:
            logger.warning(f"Price column {price_col} not found for momentum factor creation")
            return result
        
        # 计算不同时间窗口的动量
        momentum_windows = [5, 10, 20, 50, 100, 200]
        for window in momentum_windows:
            # 价格动量
            result[f'momentum_{window}'] = result[price_col] / result[price_col].shift(window) - 1
            
            # 波动率归一化动量
            rolling_std = result[price_col].pct_change().rolling(window).std()
            result[f'momentum_zscore_{window}'] = result[f'momentum_{window}'] / rolling_std
        
        logger.info(f"Created momentum factors for {len(momentum_windows)} windows")
        return result
    
    def create_technical_factors(self, df: pd.DataFrame, price_col: str = 'mid_price') -> pd.DataFrame:
        """
        创建技术指标因子
        
        Args:
            df: 输入DataFrame
            price_col: 价格列名
            
        Returns:
            添加技术指标因子后的DataFrame
        """
        result = df.copy()
        
        # 检查价格列
        if price_col not in df.columns:
            logger.warning(f"Price column {price_col} not found for technical factor creation")
            return result
        
        # 计算移动平均
        ma_windows = [5, 10, 20, 50, 100]
        for window in ma_windows:
            # 简单移动平均
            result[f'sma_{window}'] = result[price_col].rolling(window).mean()
            
            # 指数移动平均
            result[f'ema_{window}'] = result[price_col].ewm(span=window).mean()
            
            # 相对强弱指标 (RSI)
            delta = result[price_col].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window).mean()
            avg_loss = loss.rolling(window).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            result[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        # 计算MACD
        result['ema_12'] = result[price_col].ewm(span=12).mean()
        result['ema_26'] = result[price_col].ewm(span=26).mean()
        result['macd'] = result['ema_12'] - result['ema_26']
        result['macd_signal'] = result['macd'].ewm(span=9).mean()
        result['macd_hist'] = result['macd'] - result['macd_signal']
        
        # 计算布林带
        for window in [20, 50]:
            result[f'bollinger_mid_{window}'] = result[price_col].rolling(window).mean()
            result[f'bollinger_std_{window}'] = result[price_col].rolling(window).std()
            result[f'bollinger_upper_{window}'] = result[f'bollinger_mid_{window}'] + 2 * result[f'bollinger_std_{window}']
            result[f'bollinger_lower_{window}'] = result[f'bollinger_mid_{window}'] - 2 * result[f'bollinger_std_{window}']
            result[f'bollinger_pct_b_{window}'] = (result[price_col] - result[f'bollinger_lower_{window}']) / (result[f'bollinger_upper_{window}'] - result[f'bollinger_lower_{window}'])
        
        logger.info("Created technical indicator factors")
        return result
    
    def build_all_factors(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        构建所有因子
        
        Args:
            df: 输入DataFrame
            feature_cols: 用于构建因子的特征列，默认为None（自动检测）
            
        Returns:
            添加所有因子后的DataFrame
        """
        result = df.copy()
        
        # 如果未指定特征列，自动检测
        if feature_cols is None:
            # 排除日期、价格和标签列
            exclude_patterns = ['datetime', 'date', 'time', 'future_return', 'pca_factor', 'norm']
            feature_cols = [col for col in df.columns if not any(pattern in col.lower() for pattern in exclude_patterns)]
        
        # 创建标签
        result = self.create_return_labels(result)
        
        # 标准化特征
        result = self.normalize_features(result, feature_cols)
        
        # 创建PCA因子
        norm_cols = [col for col in result.columns if col.endswith('_norm')]
        result = self.create_pca_factors(result, norm_cols)
        
        # 创建动量因子
        result = self.create_momentum_factors(result)
        
        # 创建技术指标因子
        result = self.create_technical_factors(result)
        
        logger.info(f"Built all factors, total columns: {len(result.columns)}")
        return result