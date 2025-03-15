import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    特征选择器，用于选择最重要的特征
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        初始化FeatureSelector

        Args:
            output_dir: 输出目录，用于保存特征重要性图表
        """
        self.output_dir = Path(output_dir) if output_dir else None

        # 创建输出目录（如果不存在）
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def select_by_correlation(self, df: pd.DataFrame, target_col: str, top_n: int = 20) -> List[str]:
        """
        根据与目标变量的相关性选择特征

        Args:
            df: 输入DataFrame，包含特征和目标变量
            target_col: 目标变量列名
            top_n: 选择的特征数量

        Returns:
            选择的特征列表
        """
        logger.info(f"根据与 {target_col} 的相关性选择特征...")

        # 计算相关性
        correlations = df.corr()[target_col].abs().sort_values(ascending=False)

        # 排除目标变量自身
        correlations = correlations.drop(target_col)

        # 选择前N个特征
        selected_features = correlations.head(top_n).index.tolist()

        # 绘制相关性图表
        if self.output_dir:
            plt.figure(figsize=(12, 8))
            sns.barplot(x=correlations.head(top_n).values, y=correlations.head(top_n).index)
            plt.title(f'Top {top_n} Features by Correlation with {target_col}')
            plt.xlabel('Absolute Correlation')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'correlation_importance.png', dpi=300, bbox_inches='tight')

        logger.info(f"根据相关性选择了 {len(selected_features)} 个特征")
        return selected_features

    def select_by_mutual_info(self, df: pd.DataFrame, target_col: str, top_n: int = 20) -> List[str]:
        """
        根据与目标变量的互信息选择特征

        Args:
            df: 输入DataFrame，包含特征和目标变量
            target_col: 目标变量列名
            top_n: 选择的特征数量

        Returns:
            选择的特征列表
        """
        logger.info(f"根据与 {target_col} 的互信息选择特征...")

        # 准备数据
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # 计算互信息
        selector = SelectKBest(mutual_info_regression, k=top_n)
        selector.fit(X, y)

        # 获取特征得分
        scores = selector.scores_

        # 创建特征得分DataFrame
        feature_scores = pd.DataFrame({
            'Feature': X.columns,
            'Score': scores
        }).sort_values('Score', ascending=False)

        # 选择前N个特征
        selected_features = feature_scores.head(top_n)['Feature'].tolist()

        # 绘制互信息图表
        if self.output_dir:
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Score', y='Feature', data=feature_scores.head(top_n))
            plt.title(f'Top {top_n} Features by Mutual Information with {target_col}')
            plt.xlabel('Mutual Information Score')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'mutual_info_importance.png', dpi=300, bbox_inches='tight')

        logger.info(f"根据互信息选择了 {len(selected_features)} 个特征")
        return selected_features

    def select_by_random_forest(self, df: pd.DataFrame, target_col: str, top_n: int = 20) -> List[str]:
        """
        根据随机森林特征重要性选择特征

        Args:
            df: 输入DataFrame，包含特征和目标变量
            target_col: 目标变量列名
            top_n: 选择的特征数量

        Returns:
            选择的特征列表
        """
        logger.info(f"根据随机森林特征重要性选择特征...")

        # 准备数据
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # 训练随机森林模型
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)

        # 获取特征重要性
        importances = rf.feature_importances_

        # 创建特征重要性DataFrame
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        # 选择前N个特征
        selected_features = feature_importances.head(top_n)['Feature'].tolist()

        # 绘制特征重要性图表
        if self.output_dir:
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importances.head(top_n))
            plt.title(f'Top {top_n} Features by Random Forest Importance for {target_col}')
            plt.xlabel('Feature Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'random_forest_importance.png', dpi=300, bbox_inches='tight')

        logger.info(f"根据随机森林特征重要性选择了 {len(selected_features)} 个特征")
        return selected_features

    def select_by_lasso(self, df: pd.DataFrame, target_col: str, top_n: int = 20) -> List[str]:
        """
        根据Lasso回归系数选择特征

        Args:
            df: 输入DataFrame，包含特征和目标变量
            target_col: 目标变量列名
            top_n: 选择的特征数量

        Returns:
            选择的特征列表
        """
        logger.info(f"根据Lasso回归系数选择特征...")

        # 准备数据
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # 创建Lasso回归模型
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lasso', Lasso(alpha=0.01, random_state=42))
        ])

        # 训练模型
        pipeline.fit(X, y)

        # 获取特征系数
        coefficients = pipeline.named_steps['lasso'].coef_

        # 创建特征系数DataFrame
        feature_coefficients = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': np.abs(coefficients)
        }).sort_values('Coefficient', ascending=False)

        # 选择前N个特征
        selected_features = feature_coefficients.head(top_n)['Feature'].tolist()

        # 绘制特征系数图表
        if self.output_dir:
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Coefficient', y='Feature', data=feature_coefficients.head(top_n))
            plt.title(f'Top {top_n} Features by Lasso Coefficient for {target_col}')
            plt.xlabel('Absolute Coefficient')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'lasso_importance.png', dpi=300, bbox_inches='tight')

        logger.info(f"根据Lasso回归系数选择了 {len(selected_features)} 个特征")
        return selected_features

    def select_by_ensemble(self, df: pd.DataFrame, target_col: str, top_n: int = 20) -> List[str]:
        """
        综合多种方法选择特征

        Args:
            df: 输入DataFrame，包含特征和目标变量
            target_col: 目标变量列名
            top_n: 选择的特征数量

        Returns:
            选择的特征列表
        """
        logger.info(f"使用综合方法选择特征...")

        # 使用各种方法选择特征
        corr_features = self.select_by_correlation(df, target_col, top_n=top_n)
        mi_features = self.select_by_mutual_info(df, target_col, top_n=top_n)
        rf_features = self.select_by_random_forest(df, target_col, top_n=top_n)
        lasso_features = self.select_by_lasso(df, target_col, top_n=top_n)

        # 统计每个特征被选中的次数
        all_features = list(set(corr_features + mi_features + rf_features + lasso_features))
        feature_counts = {}

        for feature in all_features:
            count = 0
            if feature in corr_features:
                count += 1
            if feature in mi_features:
                count += 1
            if feature in rf_features:
                count += 1
            if feature in lasso_features:
                count += 1
            feature_counts[feature] = count

        # 按被选中次数排序
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)

        # 选择被多种方法共同选中的特征
        ensemble_features = [feature for feature, count in sorted_features if count >= 2]

        # 如果选择的特征不足top_n个，则从剩余特征中补充
        if len(ensemble_features) < top_n:
            remaining_features = [feature for feature, count in sorted_features if count == 1]
            ensemble_features.extend(remaining_features[:top_n - len(ensemble_features)])

        # 限制为top_n个特征
        selected_features = ensemble_features[:top_n]

        # 绘制特征选择频率图表
        if self.output_dir:
            feature_freq_df = pd.DataFrame({
                'Feature': [feature for feature, _ in sorted_features[:top_n]],
                'Frequency': [count for _, count in sorted_features[:top_n]]
            })

            plt.figure(figsize=(12, 8))
            sns.barplot(x='Frequency', y='Feature', data=feature_freq_df)
            plt.title(f'Top {top_n} Features by Selection Frequency for {target_col}')
            plt.xlabel('Selection Frequency')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'ensemble_importance.png', dpi=300, bbox_inches='tight')

        logger.info(f"使用综合方法选择了 {len(selected_features)} 个特征")
        return selected_features

    def evaluate_feature_set(self, df: pd.DataFrame, features: List[str], target_col: str) -> float:
        """
        评估特征集的预测性能

        Args:
            df: 输入DataFrame，包含特征和目标变量
            features: 特征列表
            target_col: 目标变量列名

        Returns:
            交叉验证得分
        """
        # 准备数据
        X = df[features]
        y = df[target_col]

        # 创建模型
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        # 交叉验证
        scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')

        # 计算RMSE
        rmse_scores = np.sqrt(-scores)
        mean_rmse = rmse_scores.mean()

        return mean_rmse

    def select_features(self, df: pd.DataFrame, target_col: str, method: str = 'ensemble', top_n: int = 20) -> List[str]:
        """
        根据指定方法选择特征

        Args:
            df: 输入DataFrame，包含特征和目标变量
            target_col: 目标变量列名
            method: 特征选择方法，可选值为'correlation', 'mutual_info', 'random_forest', 'lasso', 'ensemble'
            top_n: 选择的特征数量

        Returns:
            选择的特征列表
        """
        # 移除非数值型列
        numeric_df = df.select_dtypes(include=[np.number])

        # 确保目标变量在DataFrame中
        if target_col not in numeric_df.columns:
            raise ValueError(f"目标变量 {target_col} 不在数值型列中")

        # 根据指定方法选择特征
        if method == 'correlation':
            selected_features = self.select_by_correlation(numeric_df, target_col, top_n)
        elif method == 'mutual_info':
            selected_features = self.select_by_mutual_info(numeric_df, target_col, top_n)
        elif method == 'random_forest':
            selected_features = self.select_by_random_forest(numeric_df, target_col, top_n)
        elif method == 'lasso':
            selected_features = self.select_by_lasso(numeric_df, target_col, top_n)
        elif method == 'ensemble':
            selected_features = self.select_by_ensemble(numeric_df, target_col, top_n)
        else:
            raise ValueError(f"不支持的特征选择方法: {method}")

        # 评估选择的特征集
        rmse = self.evaluate_feature_set(numeric_df, selected_features, target_col)
        logger.info(f"选择的特征集RMSE: {rmse:.6f}")

        return selected_features

    def select_from_file(self, file_path: str, target_col: str, method: str = 'ensemble', top_n: int = 20) -> List[str]:
        """
        从文件加载数据并选择特征

        Args:
            file_path: 数据文件路径
            target_col: 目标变量列名
            method: 特征选择方法
            top_n: 选择的特征数量

        Returns:
            选择的特征列表
        """
        logger.info(f"从 {file_path} 加载数据...")

        # 加载数据
        file_path = Path(file_path)
        if file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        elif file_path.suffix == '.h5':
            df = pd.read_hdf(file_path, key='feature_data')
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")

        logger.info(f"数据加载完成，共 {len(df)} 行，{len(df.columns)} 列")

        # 选择特征
        selected_features = self.select_features(df, target_col, method, top_n)

        # 保存选择的特征列表
        if self.output_dir:
            feature_list_path = self.output_dir / 'selected_features.txt'
            with open(feature_list_path, 'w') as f:
                for feature in selected_features:
                    f.write(f"{feature}\n")
            logger.info(f"选择的特征列表已保存至 {feature_list_path}")

        return selected_features