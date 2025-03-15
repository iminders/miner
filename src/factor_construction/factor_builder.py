import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib

from src.factor_construction.microstructure_factors import MicrostructureFactors
from src.factor_construction.time_series_factors import TimeSeriesFactors
from src.factor_construction.ml_features import MLFeatureEngineering

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FactorBuilder:
    """
    因子构建器，用于组合特征生成预测因子
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        初始化因子构建器
        
        Args:
            output_dir: 输出目录，用于保存模型和结果
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.models = {}
        
        # 创建输出目录（如果不存在）
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def build_linear_factor(self, df: pd.DataFrame, features: List[str], target_col: str, model_name: str = 'linear') -> Dict:
        """
        构建线性因子模型
        
        Args:
            df: 输入DataFrame，包含特征和目标变量
            features: 特征列表
            target_col: 目标变量列名
            model_name: 模型名称
            
        Returns:
            模型评估结果
        """
        logger.info(f"构建线性因子模型: {model_name}")
        
        # 准备数据
        X = df[features]
        y = df[target_col]
        
        # 创建时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=5)
        
        # 创建模型
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ])
        
        # 交叉验证
        cv_scores = cross_val_score(pipeline, X, y, cv=tscv, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-cv_scores)
        
        # 在整个数据集上训练模型
        pipeline.fit(X, y)
        
        # 预测
        y_pred = pipeline.predict(X)
        
        # 计算评估指标
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        # 计算特征重要性（系数）
        coefficients = pipeline.named_steps['model'].coef_
        
        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Coefficient': coefficients
        }).sort_values('Coefficient', ascending=False)
        
        # 绘制特征重要性图表
        if self.output_dir:
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
            plt.title(f'Feature Coefficients for {model_name}')
            plt.xlabel('Coefficient')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{model_name}_coefficients.png', dpi=300, bbox_inches='tight')
        
        # 保存模型
        if self.output_dir:
            model_path = self.output_dir / f'{model_name}.joblib'
            joblib.dump(pipeline, model_path)
            logger.info(f"模型已保存至 {model_path}")
        
        # 保存模型到内存
        self.models[model_name] = pipeline
        
        # 返回评估结果
        result = {
            'model_name': model_name,
            'model_type': 'linear',
            'rmse': rmse,
            'r2': r2,
            'cv_rmse_mean': rmse_scores.mean(),
            'cv_rmse_std': rmse_scores.std(),
            'feature_importance': feature_importance
        }
        
        logger.info(f"线性因子模型 {model_name} 构建完成，RMSE: {rmse:.6f}, R²: {r2:.6f}")
        
        return result
    
    def build_ridge_factor(self, df: pd.DataFrame, features: List[str], target_col: str, model_name: str = 'ridge', alpha: float = 1.0) -> Dict:
        """
        构建岭回归因子模型
        
        Args:
            df: 输入DataFrame，包含特征和目标变量
            features: 特征列表
            target_col: 目标变量列名
            model_name: 模型名称
            alpha: 正则化强度
            
        Returns:
            模型评估结果
        """
        logger.info(f"构建岭回归因子模型: {model_name}")
        
        # 准备数据
        X = df[features]
        y = df[target_col]
        
        # 创建时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=5)
        
        # 创建模型
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=alpha))
        ])
        
        # 交叉验证
        cv_scores = cross_val_score(pipeline, X, y, cv=tscv, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-cv_scores)
        
        # 在整个数据集上训练模型
        pipeline.fit(X, y)
        
        # 预测
        y_pred = pipeline.predict(X)
        
        # 计算评估指标
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        # 计算特征重要性（系数）
        coefficients = pipeline.named_steps['model'].coef_
        
        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Coefficient': coefficients
        }).sort_values('Coefficient', ascending=False)
        
        # 绘制特征重要性图表
        if self.output_dir:
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
            plt.title(f'Feature Coefficients for {model_name}')
            plt.xlabel('Coefficient')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{model_name}_coefficients.png', dpi=300, bbox_inches='tight')
        
        # 保存模型
        if self.output_dir:
            model_path = self.output_dir / f'{model_name}.joblib'
            joblib.dump(pipeline, model_path)
            logger.info(f"模型已保存至 {model_path}")
        
        # 保存模型到内存
        self.models[model_name] = pipeline
        
        # 返回评估结果
        result = {
            'model_name': model_name,
            'model_type': 'ridge',
            'alpha': alpha,
            'rmse': rmse,
            'r2': r2,
            'cv_rmse_mean': rmse_scores.mean(),
            'cv_rmse_std': rmse_scores.std(),
            'feature_importance': feature_importance
        }
        
        logger.info(f"岭回归因子模型 {model_name} 构建完成，RMSE: {rmse:.6f}, R²: {r2:.6f}")
        
        return result
    
    def build_lasso_factor(self, df: pd.DataFrame, features: List[str], target_col: str, model_name: str = 'lasso', alpha: float = 0.01) -> Dict:
        """
        构建Lasso回归因子模型
        
        Args:
            df: 输入DataFrame，包含特征和目标变量
            features: 特征列表
            target_col: 目标变量列名
            model_name: 模型名称
            alpha: 正则化强度
            
        Returns:
            模型评估结果
        """
        logger.info(f"构建Lasso回归因子模型: {model_name}")
        
        # 准备数据
        X = df[features]
        y = df[target_col]
        
        # 创建时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=5)
        
        # 创建模型
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', Lasso(alpha=alpha))
        ])
        
        # 交叉验证
        cv_scores = cross_val_score(pipeline, X, y, cv=tscv, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-cv_scores)
        
        # 在整个数据集上训练模型
        pipeline.fit(X, y)
        
        # 预测
        y_pred = pipeline.predict(X)
        
        # 计算评估指标
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        # 计算特征重要性（系数）
        coefficients = pipeline.named_steps['model'].coef_
        
        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Coefficient': coefficients
        }).sort_values('Coefficient', ascending=False)
        
        # 绘制特征重要性图表
        if self.output_dir:
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
            plt.title(f'Feature Coefficients for {model_name}')
            plt.xlabel('Coefficient')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{model_name}_coefficients.png', dpi=300, bbox_inches='tight')
        
        # 保存模型
        if self.output_dir:
            model_path = self.output_dir / f'{model_name}.joblib'
            joblib.dump(pipeline, model_path)
            logger.info(f"模型已保存至 {model_path}")
        
        # 保存模型到内存
        self.models[model_name] = pipeline
        
        # 返回评估结果
        result = {
            'model_name': model_name,
            'model_type': 'lasso',
            'alpha': alpha,
            'rmse': rmse,
            'r2': r2,
            'cv_rmse_mean': rmse_scores.mean(),
            'cv_rmse_std': rmse_scores.std(),
            'feature_importance': feature_importance
        }
        
        logger.info(f"Lasso回归因子模型 {model_name} 构建完成，RMSE: {rmse:.6f}, R²: {r2:.6f}")
        
        return result
    
    def build_elastic_net_factor(self, df: pd.DataFrame, features: List[str], target_col: str, model_name: str = 'elastic_net', alpha: float = 0.01, l1_ratio: float = 0.5) -> Dict:
        """
        构建ElasticNet回归因子模型
        
        Args:
            df: 输入DataFrame，包含特征和目标变量
            features: 特征列表
            target_col: 目标变量列名
            model_name: 模型名称
            alpha: 正则化强度
            l1_ratio: L1正则化比例
            
        Returns:
            模型评估结果
        """
        logger.info(f"构建ElasticNet回归因子模型: {model_name}")
        
        # 准备数据
        X = df[features]
        y = df[target_col]
        
        # 创建时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=5)
        
        # 创建模型
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', ElasticNet(alpha=alpha, l1_ratio=l1_ratio))
        ])
        
        # 交叉验证
        cv_scores = cross_val_score(pipeline, X, y, cv=tscv, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-cv_scores)
        
        # 在整个数据集上训练模型
        pipeline.fit(X, y)
        
        # 预测
        y_pred = pipeline.predict(X)
        
        # 计算评估指标
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        # 计算特征重要性（系数）
        coefficients = pipeline.named_steps['model'].coef_
        
        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Coefficient': coefficients
        }).sort_values('Coefficient', ascending=False)
        
        # 绘制特征重要性图表
        if self.output_dir:
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
            plt.title(f'Feature Coefficients for {model_name}')
            plt.xlabel('Coefficient')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{model_name}_coefficients.png', dpi=300, bbox_inches='tight')
        
        # 保存模型
        if self.output_dir:
            model_path = self.output_dir / f'{model_name}.joblib'
            joblib.dump(pipeline, model_path)
            logger.info(f"模型已保存至 {model_path}")
        
        # 保存模型到内存
        self.models[model_name] = pipeline
        
        # 返回评估结果
        result = {
            'model_name': model_name,
            'model_type': 'elastic_net',
            'alpha': alpha,
            'l1_ratio': l1_ratio,
            'rmse': rmse,
            'r2': r2,
            'cv_rmse_mean': rmse_scores.mean(),
            'cv_rmse_std': rmse_scores.std(),
            'feature_importance': feature_importance
        }
        
        logger.info(f"ElasticNet回归因子模型 {model_name} 构建完成，RMSE: {rmse:.6f}, R²: {r2:.6f}")
        
        return result
    
    def build_random_forest_factor(self, df: pd.DataFrame, features: List[str], target_col: str, model_name: str = 'random_forest', n_estimators: int = 100, max_depth: int = None) -> Dict:
        """
        构建随机森林因子模型
        
        Args:
            df: 输入DataFrame，包含特征和目标变量
            features: 特征列表
            target_col: 目标变量列名
            model_name: 模型名称
            n_estimators: 树的数量
            max_depth: 树的最大深度
            
        Returns:
            模型评估结果
        """
        logger.info(f"构建随机森林因子模型: {model_name}")
        
        # 准备数据
        X = df[features]
        y = df[target_col]
        
        # 创建时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=5)
        
        # 创建模型
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42))
        ])
        
        # 交叉验证
        cv_scores = cross_val_score(pipeline, X, y, cv=tscv, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-cv_scores)
        
        # 在整个数据集上训练模型
        pipeline.fit(X, y)
        
        # 预测
        y_pred = pipeline.predict(X)
        
        # 计算评估指标
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        # 计算特征重要性
        importances = pipeline.named_steps['model'].feature_importances_
        
        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # 绘制特征重要性图表
        if self.output_dir:
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
            plt.title(f'Feature Importance for {model_name}')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{model_name}_importance.png', dpi=300, bbox_inches='tight')
        
        # 保存模型
        if self.output_dir:
            model_path = self.output_dir / f'{model_name}.joblib'
            joblib.dump(pipeline, model_path)
            logger.info(f"模型已保存至 {model_path}")
        
        # 保存模型到内存
        self.models[model_name] = pipeline
        
        # 返回评估结果
        result = {
            'model_name': model_name,
            'model_type': 'random_forest',
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'rmse': rmse,
            'r2': r2,
            'cv_rmse_mean': rmse_scores.mean(),
            'cv_rmse_std': rmse_scores.std(),
            'feature_importance': feature_importance
        }
        
        logger.info(f"随机森林因子模型 {model_name} 构建完成，RMSE: {rmse:.6f}, R²: {r2:.6f}")
        
        return result
    
    def build_gradient_boosting_factor(self, df: pd.DataFrame, features: List[str], target_col: str, model_name: str = 'gradient_boosting', n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3) -> Dict:
        """
        构建梯度提升树因子模型
        
        Args:
            df: 输入DataFrame，包含特征和目标变量
            features: 特征列表
            target_col: 目标变量列名
            model_name: 模型名称
            n_estimators: 树的数量
            learning_rate: 学习率
            max_depth: 树的最大深度
            
        Returns:
            模型评估结果
        """
        logger.info(f"构建梯度提升树因子模型: {model_name}")
        
        # 准备数据
        X = df[features]
        y = df[target_col]
        
        # 创建时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=5)
        
        # 创建模型
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42))
        ])
        
        # 交叉验证
        cv_scores = cross_val_score(pipeline, X, y, cv=tscv, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-cv_scores)
        
        # 在整个数据集上训练模型
        pipeline.fit(X, y)
        
        # 预测
        y_pred = pipeline.predict(X)
        
        # 计算评估指标
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        # 计算特征重要性
        importances = pipeline.named_steps['model'].feature_importances_
        
        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # 绘制特征重要性图表
        if self.output_dir:
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
            plt.title(f'Feature Importance for {model_name}')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{model_name}_importance.png', dpi=300, bbox_inches='tight')
        
        # 保存模型
        if self.output_dir:
            model_path = self.output_dir / f'{model_name}.joblib'
            joblib.dump(pipeline, model_path)
            logger.info(f"模型已保存至 {model_path}")
        
        # 保存模型到内存
        self.models[model_name] = pipeline
        
        # 返回评估结果
        result = {
            'model_name': model_name,
            'model_type': 'gradient_boosting',
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'rmse': rmse,
            'r2': r2,
            'cv_rmse_mean': rmse_scores.mean(),
            'cv_rmse_std': rmse_scores.std(),
            'feature_importance': feature_importance
        }
        
        logger.info(f"梯度提升树因子模型 {model_name} 构建完成，RMSE: {rmse:.6f}, R²: {r2:.6f}")
        
        return result
    
    def build_ensemble_factor(self, df: pd.DataFrame, features: List[str], target_col: str, model_names: List[str], weights: Optional[List[float]] = None, ensemble_name: str = 'ensemble') -> Dict:
        """
        构建集成因子模型
        
        Args:
            df: 输入DataFrame，包含特征和目标变量
            features: 特征列表
            target_col: 目标变量列名
            model_names: 要集成的模型名称列表
            weights: 模型权重列表，如果为None则使用等权重
            ensemble_name: 集成模型名称
            
        Returns:
            集成模型评估结果
        """
        logger.info(f"构建集成因子模型: {ensemble_name}")
        
        # 检查所有模型是否都已构建
        for model_name in model_names:
            if model_name not in self.models:
                raise ValueError(f"模型 {model_name} 尚未构建")
        
        # 准备数据
        X = df[features]
        y = df[target_col]
        
        # 如果没有指定权重，则使用等权重
        if weights is None:
            weights = [1.0 / len(model_names)] * len(model_names)
        
        # 确保权重和模型数量一致
        if len(weights) != len(model_names):
            raise ValueError(f"权重数量 ({len(weights)}) 与模型数量 ({len(model_names)}) 不一致")
        
        # 归一化权重
        weights = np.array(weights) / sum(weights)
        
        # 获取每个模型的预测
        predictions = []
        for model_name in model_names:
            model = self.models[model_name]
            pred = model.predict(X)
            predictions.append(pred)
        
        # 加权平均预测
        y_pred = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            y_pred += weights[i] * pred
        
        # 计算评估指标
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        # 创建集成模型的特征重要性
        feature_importance = pd.DataFrame({'Feature': features})
        
        # 合并各个模型的特征重要性
        for i, model_name in enumerate(model_names):
            model_result = self.get_model_result(model_name)
            if model_result and 'feature_importance' in model_result:
                model_fi = model_result['feature_importance']
                if 'Coefficient' in model_fi.columns:
                    feature_importance[f'{model_name}_coef'] = 0.0
                    for _, row in model_fi.iterrows():
                        if row['Feature'] in feature_importance['Feature'].values:
                            idx = feature_importance.index[feature_importance['Feature'] == row['Feature']][0]
                            feature_importance.at[idx, f'{model_name}_coef'] = row['Coefficient']
                elif 'Importance' in model_fi.columns:
                    feature_importance[f'{model_name}_imp'] = 0.0
                    for _, row in model_fi.iterrows():
                        if row['Feature'] in feature_importance['Feature'].values:
                            idx = feature_importance.index[feature_importance['Feature'] == row['Feature']][0]
                            feature_importance.at[idx, f'{model_name}_imp'] = row['Importance']
        
        # 计算加权平均特征重要性
        importance_cols = [col for col in feature_importance.columns if col.endswith('_coef') or col.endswith('_imp')]
        if importance_cols:
            feature_importance['Ensemble_Importance'] = 0.0
            for i, col in enumerate(importance_cols):
                model_name = col.split('_')[0]
                model_idx = model_names.index(model_name)
                feature_importance['Ensemble_Importance'] += weights[model_idx] * feature_importance[col]
            
            # 按重要性排序
            feature_importance = feature_importance.sort_values('Ensemble_Importance', ascending=False)
        
        # 绘制特征重要性图表
        if self.output_dir and 'Ensemble_Importance' in feature_importance.columns:
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Ensemble_Importance', y='Feature', data=feature_importance.head(20))
            plt.title(f'Feature Importance for {ensemble_name}')
            plt.xlabel('Ensemble Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{ensemble_name}_importance.png', dpi=300, bbox_inches='tight')
        
        # 创建集成模型结果
        result = {
            'model_name': ensemble_name,
            'model_type': 'ensemble',
            'base_models': model_names,
            'weights': weights.tolist(),
            'rmse': rmse,
            'r2': r2,
            'feature_importance': feature_importance
        }
        
        logger.info(f"集成因子模型 {ensemble_name} 构建完成，RMSE: {rmse:.6f}, R²: {r2:.6f}")
        
        return result
    
    def get_model_result(self, model_name: str) -> Optional[Dict]:
        """
        获取模型评估结果
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型评估结果，如果模型不存在则返回None
        """
        if hasattr(self, 'model_results') and model_name in self.model_results:
            return self.model_results[model_name]
        return None
    
    def evaluate_model(self, df: pd.DataFrame, features: List[str], target_col: str, model_name: str, test_size: float = 0.2) -> Dict:
        """
        评估模型在测试集上的性能
        
        Args:
            df: 输入DataFrame，包含特征和目标变量
            features: 特征列表
            target_col: 目标变量列名
            model_name: 模型名称
            test_size: 测试集比例
            
        Returns:
            模型评估结果
        """
        if model_name not in self.models: