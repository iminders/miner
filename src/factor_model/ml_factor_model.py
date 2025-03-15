import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import os

logger = logging.getLogger(__name__)

class MLFactorModel:
    """
    机器学习因子模型类，用于使用机器学习方法构建因子模型
    """
    
    def __init__(self):
        """
        初始化机器学习因子模型
        """
        self.factors = None
        self.returns = None
        self.model = None
        self.scaler = None
        self.feature_importance = None
        self.model_type = None
        self.test_predictions = None
        self.test_actual = None
        
    def load_data(self, factors, returns):
        """
        加载因子数据和收益率数据
        
        参数:
        factors (pd.DataFrame): 因子数据，每列为一个因子
        returns (pd.Series): 目标收益率数据
        """
        # 确保因子和收益率的索引对齐
        common_index = factors.index.intersection(returns.index)
        self.factors = factors.loc[common_index]
        self.returns = returns.loc[common_index]
        
        logger.info(f"加载了{len(self.factors)}行数据，{self.factors.shape[1]}个因子")
    
    def train_model(self, model_type='xgboost', test_size=0.3, random_state=42, **model_params):
        """
        训练机器学习模型
        
        参数:
        model_type (str): 模型类型，可选 'linear', 'ridge', 'lasso', 'elastic_net', 'random_forest', 'gbdt', 'svr', 'xgboost', 'lightgbm'
        test_size (float): 测试集比例
        random_state (int): 随机种子
        model_params (dict): 模型参数
        
        返回:
        dict: 模型评估结果
        """
        # 保存模型类型
        self.model_type = model_type
        
        # 数据预处理
        X = self.factors.copy()
        y = self.returns.copy()
        
        # 处理缺失值
        X = X.fillna(X.mean())
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # 标准化特征
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 选择模型
        if model_type == 'linear':
            self.model = LinearRegression(**model_params)
        elif model_type == 'ridge':
            self.model = Ridge(**model_params)
        elif model_type == 'lasso':
            self.model = Lasso(**model_params)
        elif model_type == 'elastic_net':
            self.model = ElasticNet(**model_params)
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(**model_params)
        elif model_type == 'gbdt':
            self.model = GradientBoostingRegressor(**model_params)
        elif model_type == 'svr':
            self.model = SVR(**model_params)
        elif model_type == 'xgboost':
            self.model = xgb.XGBRegressor(**model_params)
        elif model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor(**model_params)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 训练模型
        logger.info(f"使用{model_type}模型训练...")
        self.model.fit(X_train_scaled, y_train)
        
        # 预测
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        # 保存测试集预测结果
        self.test_predictions = test_pred
        self.test_actual = y_test
        
        # 计算评估指标
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # 计算方向准确率
        train_direction_accuracy = np.mean((y_train > 0) == (train_pred > 0))
        test_direction_accuracy = np.mean((y_test > 0) == (test_pred > 0))
        
        # 计算特征重要性
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.Series(self.model.feature_importances_, index=X.columns)
        elif hasattr(self.model, 'coef_'):
            self.feature_importance = pd.Series(np.abs(self.model.coef_), index=X.columns)
        else:
            self.feature_importance = None
        
        # 汇总结果
        results = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_direction_accuracy': train_direction_accuracy,
            'test_direction_accuracy': test_direction_accuracy,
            'feature_importance': self.feature_importance
        }
        
        logger.info(f"模型训练完成，测试集R²: {test_r2:.4f}, 测试集方向准确率: {test_direction_accuracy:.4f}")
        
        return results
    
    def cross_validate(self, n_splits=5, model_type='xgboost', **model_params):
        """
        使用时间序列交叉验证评估模型
        
        参数:
        n_splits (int): 折数
        model_type (str): 模型类型
        model_params (dict): 模型参数
        
        返回:
        dict: 交叉验证结果
        """
        # 数据预处理
        X = self.factors.copy()
        y = self.returns.copy()
        
        # 处理缺失值
        X = X.fillna(X.mean())
        
        # 初始化时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # 初始化结果列表
        train_mse_list = []
        test_mse_list = []
        train_r2_list = []
        test_r2_list = []
        train_direction_accuracy_list = []
        test_direction_accuracy_list = []
        
        # 交叉验证
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # 标准化特征
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 选择模型
            if model_type == 'linear':
                model = LinearRegression(**model_params)
            elif model_type == 'ridge':
                model = Ridge(**model_params)
            elif model_type == 'lasso':
                model = Lasso(**model_params)
            elif model_type == 'elastic_net':
                model = ElasticNet(**model_params)
            elif model_type == 'random_forest':
                model = RandomForestRegressor(**model_params)
            elif model_type == 'gbdt':
                model = GradientBoostingRegressor(**model_params)
            elif model_type == 'svr':
                model = SVR(**model_params)
            elif model_type == 'xgboost':
                model = xgb.XGBRegressor(**model_params)
            elif model_type == 'lightgbm':
                model = lgb.LGBMRegressor(**model_params)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            # 训练模型
            model.fit(X_train_scaled, y_train)
            
            # 预测
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            
            # 计算评估指标
            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            # 计算方向准确率
            train_direction_accuracy = np.mean((y_train > 0) == (train_pred > 0))
            test_direction_accuracy = np.mean((y_test > 0) == (test_pred > 0))
            
            # 记录结果
            train_mse_list.append(train_mse)
            test_mse_list.append(test_mse)
            train_r2_list.append(train_r2)
            test_r2_list.append(test_r2)
            train_direction_accuracy_list.append(train_direction_accuracy)
            test_direction_accuracy_list.append(test_direction_accuracy)
        
        # 计算平均结果
        avg_train_mse = np.mean(train_mse_list)
        avg_test_mse = np.mean(test_mse_list)
        avg_train_r2 = np.mean(train_r2_list)
        avg_test_r2 = np.mean(test_r2_list)
        avg_train_direction_accuracy = np.mean(train_direction_accuracy_list)
        avg_test_direction_accuracy = np.mean(test_direction_accuracy_list)
        
        # 计算标准差
        std_train_mse = np.std(train_mse_list)
        std_test_mse = np.std(test_mse_list)
        std_train_r2 = np.std(train_r2_list)
        std_test_r2 = np.std(test_r2_list)
        std_train_direction_accuracy = np.std(train_direction_accuracy_list)
        std_test_direction_accuracy = np.std(test_direction_accuracy_list)
        
        # 汇总结果
        results = {
            'avg_train_mse': avg_train_mse,
            'avg_test_mse': avg_test_mse,
            'avg_train_r2': avg_train_r2,
            'avg_test_r2': avg_test_r2,
            'avg_train_direction_accuracy': avg_train_direction_accuracy,
            'avg_test_direction_accuracy': avg_test_direction_accuracy,
            'std_train_mse': std_train_mse,
            'std_test_mse': std_test_mse,
            'std_train_r2': std_train_r2,
            'std_test_r2': std_test_r2,
            'std_train_direction_accuracy': std_train_direction_accuracy,
            'std_test_direction_accuracy': std_test_direction_accuracy
        }
        
        logger.info(f"交叉验证完成，平均测试集R²: {avg_test_r2:.4f} ± {std_test_r2:.4f}, 平均测试集方向准确率: {avg_test_direction_accuracy:.4f} ± {std_test_direction_accuracy:.4f}")
        
        return results
    
    def predict(self, factors):
        """
        使用训练好的模型进行预测
        
        参数:
        factors (pd.DataFrame): 因子数据
        
        返回:
        np.array: 预测的收益率
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train_model方法")
        
        # 数据预处理
        X = factors.copy()
        
        # 处理缺失值
        X = X.fillna(X.mean())
        
        # 标准化特征
        X_scaled = self.scaler.transform(X)
        
        # 预测
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def plot_feature_importance(self, top_n=10, figsize=(12, 8)):
        """
        绘制特征重要性
        
        参数:
        top_n (int): 显示前N个重要特征
        figsize (tuple): 图表大小
        """
        if self.feature_importance is None:
            logger.warning("模型没有特征重要性属性")
            return
        
        plt.figure(figsize=figsize)
        
        # 获取前N个重要特征
        top_features = self.feature_importance.sort_values(ascending=False).head(top_n)
        
        # 绘制条形图
        top_features.plot(kind='bar')
        
        plt.title(f'{self.model_type}模型特征重要性 (Top {top_n})', fontsize=16)
        plt.xlabel('特征', fontsize=12)
        plt.ylabel('重要性', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, figsize=(12, 8)):
        """
        绘制测试集预测结果
        
        参数:
        figsize (tuple): 图表大小
        """
        if self.test_predictions is None or self.test_actual is None:
            logger.warning("没有测试集预测结果，请先调用train_model方法")
            return
        
        plt.figure(figsize=figsize)
        
        # 创建索引
        index = range(len(self.test_actual))
        
        # 绘制散点图
        plt.scatter(index, self.test_actual, label='实际值', alpha=0.7)
        plt.scatter(index, self.test_predictions, label='预测值', alpha=0.7)
        
        # 绘制回归线
        z = np.polyfit(self.test_actual, self.test_predictions, 1)
        p = np.poly1d(z)
        plt.plot(index, p(self.test_actual), "r--", label=f'拟合线: y={z[0]:.4f}x+{z[1]:.4f}')
        
        plt.title(f'{self.model_type}模型预测结果', fontsize=16)
        plt.xlabel('样本', fontsize=12)
        plt.ylabel('收益率', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # 绘制预测值与实际值的对比图
        plt.figure(figsize=figsize)
        plt.scatter(self.test_actual, self.test_predictions, alpha=0.7)
        
        # 添加对角线
        min_val = min(min(self.test_actual), min(self.test_predictions))
        max_val = max(max(self.test_actual), max(self.test_predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='对角线')
        
        # 添加回归线
        plt.plot(self.test_actual, p(self.test_actual), "r--", label=f'拟合线: y={z[0]:.4f}x+{z[1]:.4f}')
        
        plt.title(f'{self.model_type}模型预测值与实际值对比', fontsize=16)
        plt.xlabel('实际值', fontsize=12)
        plt.ylabel('预测值', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_residuals(self, figsize=(12, 8)):
        """
        绘制残差分析图
        
        参数:
        figsize (tuple): 图表大小
        """
        if self.test_predictions is None or self.test_actual is None:
            logger.warning("没有测试集预测结果，请先调用train_model方法")
            return
        
        # 计算残差
        residuals = self.test_actual - self.test_predictions
        
        plt.figure(figsize=figsize)
        
        # 绘制残差散点图
        plt.subplot(2, 1, 1)
        plt.scatter(self.test_predictions, residuals, alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('残差分析', fontsize=16)
        plt.xlabel('预测值', fontsize=12)
        plt.ylabel('残差', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 绘制残差直方图
        plt.subplot(2, 1, 2)
        plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('残差分布', fontsize=16)
        plt.xlabel('残差', fontsize=12)
        plt.ylabel('频数', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 打印残差统计信息
        print(f"残差均值: {np.mean(residuals):.6f}")
        print(f"残差标准差: {np.std(residuals):.6f}")
        print(f"残差中位数: {np.median(residuals):.6f}")
        print(f"残差最小值: {np.min(residuals):.6f}")
        print(f"残差最大值: {np.max(residuals):.6f}")
    
    def save_model(self, filepath):
        """
        保存模型
        
        参数:
        filepath (str): 模型保存路径
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train_model方法")
        
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存模型
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type,
            'feature_names': self.factors.columns.tolist() if self.factors is not None else None
        }
        
        joblib.dump(model_data, filepath)
        
        logger.info(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """
        加载模型
        
        参数:
        filepath (str): 模型加载路径
        """
        # 检查文件是否存在
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"找不到模型文件: {filepath}")
        
        # 加载模型
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_importance = model_data['feature_importance']
        self.model_type = model_data['model_type']
        
        logger.info(f"模型已从{filepath}加载，模型类型: {self.model_type}")
        
        return self
    
    def hyperparameter_tuning(self, param_grid, model_type='xgboost', cv=5, scoring='neg_mean_squared_error'):
        """
        超参数调优
        
        参数:
        param_grid (dict): 参数网格
        model_type (str): 模型类型
        cv (int): 交叉验证折数
        scoring (str): 评分标准
        
        返回:
        dict: 最佳参数和结果
        """
        from sklearn.model_selection import GridSearchCV
        
        logger.info(f"开始{model_type}模型的超参数调优...")
        
        # 数据预处理
        X = self.factors.copy()
        y = self.returns.copy()
        
        # 处理缺失值
        X = X.fillna(X.mean())
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 选择基础模型
        if model_type == 'linear':
            base_model = LinearRegression()
        elif model_type == 'ridge':
            base_model = Ridge()
        elif model_type == 'lasso':
            base_model = Lasso()
        elif model_type == 'elastic_net':
            base_model = ElasticNet()
        elif model_type == 'random_forest':
            base_model = RandomForestRegressor()
        elif model_type == 'gbdt':
            base_model = GradientBoostingRegressor()
        elif model_type == 'svr':
            base_model = SVR()
        elif model_type == 'xgboost':
            base_model = xgb.XGBRegressor()
        elif model_type == 'lightgbm':
            base_model = lgb.LGBMRegressor()
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 创建网格搜索
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        # 执行网格搜索
        grid_search.fit(X_scaled, y)
        
        # 获取最佳参数和结果
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        logger.info(f"超参数调优完成，最佳{scoring}分数: {best_score:.4f}")
        logger.info(f"最佳参数: {best_params}")
        
        # 使用最佳参数训练模型
        self.train_model(model_type=model_type, **best_params)
        
        # 返回结果
        return {
            'best_params': best_params,
            'best_score': best_score,
            'cv_results': grid_search.cv_results_
        }