import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

class MLFeatureEngineering:
    """
    机器学习特征工程
    """
    
    def __init__(self):
        """
        初始化机器学习特征工程
        """
        self.scaler = StandardScaler()
        self.pca = None
    
    def create_interaction_features(self, df: pd.DataFrame, feature_pairs: Optional[List[Tuple[str, str]]] = None) -> pd.DataFrame:
        """
        创建特征交互项
        
        Args:
            df: 输入DataFrame，包含基础特征
            feature_pairs: 需要交互的特征对列表，如果为None则自动选择
            
        Returns:
            添加了特征交互项的DataFrame
        """
        result = df.copy()
        
        # 如果没有指定特征对，则选择一些重要的特征进行交互
        if feature_pairs is None:
            # 选择一些可能有交互关系的特征
            potential_features = []
            
            # 价格类特征
            price_features = [col for col in df.columns if 'price' in col.lower() and 'change' not in col.lower()]
            potential_features.extend(price_features[:3] if len(price_features) > 3 else price_features)
            
            # 交易量类特征
            volume_features = [col for col in df.columns if 'volume' in col.lower() and 'change' not in col.lower()]
            potential_features.extend(volume_features[:3] if len(volume_features) > 3 else volume_features)
            
            # 订单流类特征
            flow_features = [col for col in df.columns if 'flow' in col.lower() or 'imbalance' in col.lower()]
            potential_features.extend(flow_features[:3] if len(flow_features) > 3 else flow_features)
            
            # 波动率类特征
            volatility_features = [col for col in df.columns if 'volatility' in col.lower()]
            potential_features.extend(volatility_features[:3] if len(volatility_features) > 3 else volatility_features)
            
            # 生成特征对
            feature_pairs = []
            for i in range(len(potential_features)):
                for j in range(i+1, len(potential_features)):
                    feature_pairs.append((potential_features[i], potential_features[j]))
        
        # 创建特征交互项
        for feature1, feature2 in feature_pairs:
            if feature1 in df.columns and feature2 in df.columns:
                # 乘积
                result[f'{feature1}_{feature2}_product'] = result[feature1] * result[feature2]
                
                # 比率 (避免除以零)
                result[f'{feature1}_{feature2}_ratio'] = result[feature1] / result[feature2].replace(0, np.nan)
                
                # 差值
                result[f'{feature1}_{feature2}_diff'] = result[feature1] - result[feature2]
        
        logger.info(f"Created {len(feature_pairs) * 3} interaction features")
        return result
    
    def create_nonlinear_transformations(self, df: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        创建非线性变换特征
        
        Args:
            df: 输入DataFrame，包含基础特征
            features: 需要进行非线性变换的特征列表，如果为None则自动选择
            
        Returns:
            添加了非线性变换特征的DataFrame
        """
        result = df.copy()
        
        # 如果没有指定特征，则选择数值型特征
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # 排除时间戳和ID类特征
            features = [col for col in features if not any(x in col.lower() for x in ['timestamp', 'id', 'date', 'time'])]
            
            # 限制特征数量，避免维度爆炸
            features = features[:20] if len(features) > 20 else features
        
        # 创建非线性变换特征
        nonlinear_features_count = 0
        for feature in features:
            if feature in df.columns:
                # 平方
                result[f'{feature}_squared'] = result[feature] ** 2
                nonlinear_features_count += 1
                
                # 平方根 (对于非负值)
                if (result[feature] >= 0).all():
                    result[f'{feature}_sqrt'] = np.sqrt(result[feature])
                    nonlinear_features_count += 1
                
                # 对数变换 (对于正值)
                if (result[feature] > 0).all():
                    result[f'{feature}_log'] = np.log(result[feature])
                    nonlinear_features_count += 1
                
                # 倒数 (避免除以零)
                result[f'{feature}_inverse'] = 1 / result[feature].replace(0, np.nan)
                nonlinear_features_count += 1
        
        logger.info(f"Created {nonlinear_features_count} nonlinear transformation features")
        return result
    
    def create_polynomial_features(self, df: pd.DataFrame, features: Optional[List[str]] = None, degree: int = 2) -> pd.DataFrame:
        """
        创建多项式特征
        
        Args:
            df: 输入DataFrame，包含基础特征
            features: 需要进行多项式变换的特征列表，如果为None则自动选择
            degree: 多项式的次数
            
        Returns:
            添加了多项式特征的DataFrame
        """
        # 如果没有指定特征，则选择数值型特征
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # 排除时间戳和ID类特征
            features = [col for col in features if not any(x in col.lower() for x in ['timestamp', 'id', 'date', 'time'])]
            
            # 限制特征数量，避免维度爆炸
            features = features[:10] if len(features) > 10 else features
        
        # 创建多项式特征
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        
        # 提取特征子集
        X = df[features].fillna(0)  # 填充缺失值，避免错误
        
        # 转换特征
        poly_features = poly.fit_transform(X)
        
        # 获取特征名称
        feature_names = poly.get_feature_names_out(features)
        
        # 创建多项式特征DataFrame
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
        
        # 只保留交互项和高次项，排除原始特征
        interaction_features = [col for col in poly_df.columns if '_' in col]
        
        # 合并结果
        result = pd.concat([df, poly_df[interaction_features]], axis=1)
        
        logger.info(f"Created {len(interaction_features)} polynomial features")
        return result
    
    def create_time_series_features(self, df: pd.DataFrame, features: Optional[List[str]] = None, windows: List[int] = [5, 10, 30, 60]) -> pd.DataFrame:
        """
        创建时间序列特征
        
        Args:
            df: 输入DataFrame，包含基础特征
            features: 需要进行时间序列变换的特征列表，如果为None则自动选择
            windows: 滚动窗口大小列表
            
        Returns:
            添加了时间序列特征的DataFrame
        """
        result = df.copy()
        
        # 如果没有指定特征，则选择数值型特征
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # 排除时间戳和ID类特征
            features = [col for col in features if not any(x in col.lower() for x in ['timestamp', 'id', 'date', 'time'])]
            
            # 限制特征数量，避免维度爆炸
            features = features[:10] if len(features) > 10 else features
        
        # 创建时间序列特征
        time_series_features_count = 0
        for feature in features:
            if feature in df.columns:
                for window in windows:
                    # 移动平均
                    result[f'{feature}_ma_{window}'] = result[feature].rolling(window=window).mean()
                    time_series_features_count += 1
                    
                    # 移动标准差
                    result[f'{feature}_std_{window}'] = result[feature].rolling(window=window).std()
                    time_series_features_count += 1
                    
                    # 移动最大值
                    result[f'{feature}_max_{window}'] = result[feature].rolling(window=window).max()
                    time_series_features_count += 1
                    
                    # 移动最小值
                    result[f'{feature}_min_{window}'] = result[feature].rolling(window=window).min()
                    time_series_features_count += 1
                    
                    # 移动中位数
                    result[f'{feature}_median_{window}'] = result[feature].rolling(window=window).median()
                    time_series_features_count += 1
                    
                    # 移动偏度
                    result[f'{feature}_skew_{window}'] = result[feature].rolling(window=window).skew()
                    time_series_features_count += 1
                    
                    # 移动峰度
                    result[f'{feature}_kurt_{window}'] = result[feature].rolling(window=window).kurt()
                    time_series_features_count += 1
        
        logger.info(f"Created {time_series_features_count} time series features")
        return result
    
    def reduce_dimensions(self, df: pd.DataFrame, features: Optional[List[str]] = None, n_components: int = 10) -> pd.DataFrame:
        """
        使用PCA进行降维
        
        Args:
            df: 输入DataFrame，包含特征
            features: 需要进行降维的特征列表，如果为None则自动选择
            n_components: PCA组件数量
            
        Returns:
            添加了PCA特征的DataFrame
        """
        result = df.copy()
        
        # 如果没有指定特征，则选择数值型特征
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # 排除时间戳和ID类特征
            features = [col for col in features if not any(x in col.lower() for x in ['timestamp', 'id', 'date', 'time'])]
        
        # 提取特征子集
        X = df[features].fillna(0)  # 填充缺失值，避免错误
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 执行PCA
        self.pca = PCA(n_components=min(n_components, len(features)))
        pca_features = self.pca.fit_transform(X_scaled)
        
        # 创建PCA特征DataFrame
        pca_df = pd.DataFrame(
            pca_features,
            columns=[f'pca_component_{i+1}' for i in range(pca_features.shape[1])],
            index=df.index
        )
        
        # 合并结果
        result = pd.concat([result, pca_df], axis=1)
        
        logger.info(f"Created {pca_features.shape[1]} PCA components, explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
        return result
    
    def create_all_features(self, df: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        创建所有机器学习特征
        
        Args:
            df: 输入DataFrame，包含基础特征
            features: 需要进行特征工程的特征列表，如果为None则自动选择
            
        Returns:
            添加了所有机器学习特征的DataFrame
        """
        result = df.copy()
        
        logger.info("Starting ML feature engineering process")
        
        # 依次创建各类特征
        result = self.create_interaction_features(result, feature_pairs=None)
        result = self.create_nonlinear_transformations(result, features=features)
        result = self.create_polynomial_features(result, features=features)
        result = self.create_time_series_features(result, features=features)
        
        # 降维通常在最后进行，可以根据需要选择是否执行
        # result = self.reduce_dimensions(result, features=None)
        
        logger.info(f"ML feature engineering completed, total features: {len(result.columns)}")
        return result