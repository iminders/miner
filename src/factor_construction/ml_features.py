import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA


class MLFeatureEngineering:
    """
    机器学习特征工程类
    """
    
    def __init__(self):
        """
        初始化MLFeatureEngineering
        """
        self.scaler = StandardScaler()
        self.pca = None
    
    def create_interaction_features(self, df, feature_pairs=None):
        """
        创建特征交互项
        
        参数:
        df (pd.DataFrame): 包含基础特征的DataFrame
        feature_pairs (list): 需要交互的特征对列表，如果为None则自动选择
        
        返回:
        pd.DataFrame: 添加了特征交互项的DataFrame
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
                result[feature2] = result[feature2].replace(0, np.nan)
                result[f'{feature1}_{feature2}_ratio'] = result[feature1] / result[feature2]
                
                # 差值
                result[f'{feature1}_{feature2}_diff'] = result[feature1] - result[feature2]
        
        return result
    
    def create_nonlinear_transformations(self, df, features=None):
        """
        创建非线性变换特征
        
        参数:
        df (pd.DataFrame): 包含基础特征的DataFrame
        features (list): 需要进行非线性变换的特征列表，如果为None则自动选择
        
        返回:
        pd.DataFrame: 添加了非线性变换特征的DataFrame
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
        for feature in features:
            if feature in df.columns:
                # 平方
                result[f'{feature}_squared'] = result[feature] ** 2
                
                # 平方根 (对于非负值)
                if (result[feature] >= 0).all():
                    result[f'{feature}_sqrt'] = np.sqrt(result[feature])
                
                # 对数变换 (对于正值)
                if (result[feature] > 0).all():
                    result[f'{feature}_log'] = np.log(result[feature])
                
                # 倒数 (避免除以零)
                result[feature] = result[feature].replace(0, np.nan)
                result[f'{feature}_inverse'] = 1 / result[feature]
        
        return result
    
    def create_polynomial_features(self, df, features=None, degree=2):
        """
        创建多项式特征
        
        参数:
        df (pd.DataFrame): 包含基础特征的DataFrame
        features (list): 需要进行多项式变换的特征列表，如果为None则自动选择
        degree (int): 多项式的次数
        
        返回:
        pd.DataFrame: 添加了多项式特征的DataFrame
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
        
        return result
    
    def create_time_series_features(self, df, features=None, windows=[5, 10, 30, 60]):
        """
        创建时间序列特征
        
        参数:
        df (pd.DataFrame): 包含基础特征的DataFrame
        features (list): 需要进行时间序列变换的特征列表，如果为None则自动选择
        windows (list): 滚动窗口大小列表，单位为秒
        
        返回:
        pd.DataFrame: 添加了时间序列特征的DataFrame
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
        for feature in features:
            if feature in df.columns:
                for window in windows:
                    # 移动平均
                    result[f'{feature}_ma_{window}s'] = result[feature].rolling(window=window).mean()
                    
                    # 移动标准差
                    result[f'{feature}_std_{window}s'] = result[feature].rolling(window=window).std()
                    
                    # 移动最大值
                    result[f'{feature}_max_{window}s'] = result[feature].rolling(window=window).max()
                    
                    # 移动最小值
                    result[f'{feature}_min_{window}s'] = result[feature].rolling(window=window).min()
                    
                    # 移动中位数
                    result[f'{feature}_median_{window}s'] = result[feature].rolling(window=window).median()
                    
                    # 移动偏度
                    result[f'{feature}_skew_{window}s'] = result[feature].rolling(window=window).skew()
                    
                    # 移动峰度
                    result[f'{feature}_kurt_{window}s'] = result[feature].rolling(window=window).kurt()
                    
                    # 移动自相关
                    result[f'{feature}_autocorr_{window}s'] = result[feature].rolling(window=window).apply(
                        lambda x: x.autocorr(1) if len(x) > 1 else np.nan
                    )
        
        return result
    
    def reduce_dimensions(self, df, features=None, n_components=10):
        """
        使用PCA进行降维
        
        参数:
        df (pd.DataFrame): 包含特征的DataFrame
        features (list): 需要进行降维的特征列表，如果为None则自动选择
        n_components (int): PCA组件数量
        
        返回:
        pd.DataFrame: 添加了PCA特征的DataFrame
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
        
        return result
    
    def create_all_features(self, df, features=None):
        """
        创建所有机器学习特征
        
        参数:
        df (pd.DataFrame): 包含基础特征的DataFrame
        features (list): 需要进行特征工程的特征列表，如果为None则自动选择
        
        返回:
        pd.DataFrame: 添加了所有机器学习特征的DataFrame
        """
        result = df.copy()
        
        # 依次创建各类特征
        result = self.create_interaction_features(result, feature_pairs=None)
        result = self.create_nonlinear_transformations(result, features=features)
        result = self.create_polynomial_features(result, features=features)
        result = self.create_time_series_features(result, features=features)
        
        # 降维通常在最后进行，可以根据需要选择是否执行
        # result = self.reduce_dimensions(result, features=None)
        
        return result