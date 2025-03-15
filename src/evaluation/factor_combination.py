# -*- coding: utf-8 -*-
"""
多因子组合模块

实现因子相关性分析、因子聚类、多因子模型构建和权重优化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import cvxpy as cp


class FactorCombiner:
    """
    多因子组合器

    实现因子相关性分析、因子聚类、多因子模型构建和权重优化
    """

    def __init__(self):
        """
        初始化多因子组合器
        """
        self.scaler = StandardScaler()

    def analyze_factor_correlation(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        分析因子相关性

        参数:
        factors (pd.DataFrame): 因子数据框，每列为一个因子

        返回:
        pd.DataFrame: 因子相关性矩阵
        """
        # 计算因子相关性矩阵
        corr_matrix = factors.corr()
        return corr_matrix

    def plot_correlation_heatmap(self, corr_matrix: pd.DataFrame, title: str = "因子相关性热图") -> None:
        """
        绘制相关性热图

        参数:
        corr_matrix (pd.DataFrame): 相关性矩阵
        title (str): 图表标题
        """
        plt.figure(figsize=(12, 10))
        plt.matshow(corr_matrix, fignum=1)
        plt.colorbar()
        plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
        plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
        plt.title(title)
        plt.tight_layout()
        plt.savefig('factor_correlation_heatmap.png')
        plt.show()

    def cluster_factors(self, factors: pd.DataFrame, n_clusters: int = None, 
                       method: str = 'ward', threshold: float = 1.0) -> Dict:
        """
        对因子进行聚类

        参数:
        factors (pd.DataFrame): 因子数据框，每列为一个因子
        n_clusters (int): 聚类数量，如果为None则使用阈值自动确定
        method (str): 聚类方法，可选'ward', 'complete', 'average', 'single'
        threshold (float): 聚类距离阈值，仅当n_clusters为None时使用

        返回:
        Dict: 包含聚类结果的字典
        """
        # 计算相关性矩阵
        corr_matrix = self.analyze_factor_correlation(factors)

        # 将相关性转换为距离（1 - 相关性的绝对值）
        distance_matrix = 1 - np.abs(corr_matrix)

        # 计算层次聚类
        condensed_dist = pdist(distance_matrix)
        Z = linkage(condensed_dist, method=method)

        # 根据阈值或指定的聚类数量确定聚类
        if n_clusters is None:
            clusters = fcluster(Z, threshold, criterion='distance')
        else:
            clusters = fcluster(Z, n_clusters, criterion='maxclust')

        # 将聚类结果组织成字典
        cluster_dict = {}
        for i, cluster_id in enumerate(clusters):
            factor_name = factors.columns[i]
            if cluster_id not in cluster_dict:
                cluster_dict[cluster_id] = []
            cluster_dict[cluster_id].append(factor_name)

        # 创建结果字典
        result = {
            'linkage': Z,
            'clusters': clusters,
            'cluster_dict': cluster_dict,
            'n_clusters': len(cluster_dict)
        }

        return result

    def plot_dendrogram(self, cluster_result: Dict, title: str = "因子聚类树状图") -> None:
        """
        绘制聚类树状图

        参数:
        cluster_result (Dict): 聚类结果字典
        title (str): 图表标题
        """
        plt.figure(figsize=(15, 8))
        dendrogram(
            cluster_result['linkage'],
            labels=cluster_result['clusters'],
            leaf_rotation=90.,
            leaf_font_size=10.
        )
        plt.title(title)
        plt.xlabel('因子')
        plt.ylabel('距离')
        plt.tight_layout()
        plt.savefig('factor_dendrogram.png')
        plt.show()

    def select_representative_factors(self, factors: pd.DataFrame, target: pd.Series, 
                                     cluster_result: Dict) -> pd.DataFrame:
        """
        从每个聚类中选择代表性因子

        参数:
        factors (pd.DataFrame): 因子数据框，每列为一个因子
        target (pd.Series): 目标变量
        cluster_result (Dict): 聚类结果字典

        返回:
        pd.DataFrame: 选择的代表性因子
        """
        # 确保因子和目标具有相同的索引
        common_index = factors.index.intersection(target.index)
        factors_aligned = factors.loc[common_index]
        target_aligned = target.loc[common_index]

        # 移除NaN值
        mask = ~(factors_aligned.isna().any(axis=1) | target_aligned.isna())
        factors_clean = factors_aligned[mask]
        target_clean = target_aligned[mask]

        if len(factors_clean) == 0:
            return pd.DataFrame()

        # 从每个聚类中选择与目标相关性最高的因子
        selected_factors = []

        for cluster_id, factor_names in cluster_result['cluster_dict'].items():
            # 计算每个因子与目标的相关性
            correlations = {}
            for factor_name in factor_names:
                if factor_name in factors_clean.columns:
                    corr = factors_clean[factor_name].corr(target_clean)
                    correlations[factor_name] = abs(corr)  # 使用相关性的绝对值

            # 选择相关性最高的因子
            if correlations:
                best_factor = max(correlations.items(), key=lambda x: x[1])[0]
                selected_factors.append(best_factor)

        # 创建选择的因子数据框
        selected_df = factors_clean[selected_factors]

        return selected_df

    def build_factor_model(self, factors: pd.DataFrame, target: pd.Series, 
                          model_type: str = 'linear', alpha: float = 1.0) -> Dict:
        """
        构建多因子模型

        参数:
        factors (pd.DataFrame): 因子数据框，每列为一个因子
        target (pd.Series): 目标变量
        model_type (str): 模型类型，可选'linear', 'ridge', 'lasso', 'elastic_net'
        alpha (float): 正则化参数，仅用于ridge, lasso和elastic_net

        返回:
        Dict: 包含模型和评估结果的字典
        """
        # 确保因子和目标具有相同的索引
        common_index = factors.index.intersection(target.index)
        factors_aligned = factors.loc[common_index]
        target_aligned = target.loc[common_index]

        # 移除NaN值
        mask = ~(factors_aligned.isna().any(axis=1) | target_aligned.isna())
        factors_clean = factors_aligned[mask]
        target_clean = target_aligned[mask]

        if len(factors_clean) == 0:
            return {}

        # 标准化因子
        X = self.scaler.fit_transform(factors_clean)
        y = target_clean.values

        # 选择模型类型
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'ridge':
            model = Ridge(alpha=alpha)
        elif model_type == 'lasso':
            model = Lasso(alpha=alpha)
        elif model_type == 'elastic_net':
            model = ElasticNet(alpha=alpha, l1_ratio=0.5)
        else:
            raise ValueError("不支持的模型类型，请选择'linear', 'ridge', 'lasso'或'elastic_net'")

        # 拟合模型
        model.fit(X, y)

        # 预测
        y_pred = model.predict(X)

        # 评估模型
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        # 获取系数
        if hasattr(model, 'coef_'):
            coefficients = pd.Series(model.coef_, index=factors_clean.columns)
        else:
            coefficients = pd.Series(index=factors_clean.columns)

        # 创建结果字典
        result = {
            'model': model,
            'scaler': self.scaler,
            'mse': mse,
            'r2': r2,
            'coefficients': coefficients,
            'y_pred': pd.Series(y_pred, index=factors_clean.index)
        }

        return result

    def optimize_weights(self, factors: pd.DataFrame, target: pd.Series, 
                        method: str = 'equal', ic_series: Dict[str, pd.Series] = None,
                        risk_aversion: float = 1.0) -> pd.Series:
        """
        优化因子权重

        参数:
        factors (pd.DataFrame): 因子数据框，每列为一个因子
        target (pd.Series): 目标变量
        method (str): 权重优化方法，可选'equal', 'ic', 'regression', 'mean_variance'
        ic_series (Dict[str, pd.Series]): 每个因子的IC时间序列，仅用于'ic'方法
        risk_aversion (float): 风险厌恶系数，仅用于'mean_variance'方法

        返回:
        pd.Series: 优化后的因子权重
        """
        # 确保因子和目标具有相同的索引
        common_index = factors.index.intersection(target.index)
        factors_aligned = factors.loc[common_index]
        target_aligned = target.loc[common_index]

        # 移除NaN值
        mask = ~(factors_aligned.isna().any(axis=1) | target_aligned.isna())
        factors_clean = factors_aligned[mask]
        target_clean = target_aligned[mask]

        if len(factors_clean) == 0:
            return pd.Series()

        # 初始化权重
        weights = pd.Series(index=factors_clean.columns)

        # 根据不同方法优化权重
        if method == 'equal':
            # 等权重
            weights[:] = 1.0 / len(weights)

        elif method == 'ic':
            # 基于IC值的权重
            if ic_series is None:
                raise ValueError("使用'ic'方法时必须提供ic_series参数")

            # 计算每个因子的平均IC
            ic_means = {}
            for factor_name, ic in ic_series.items():
                if factor_name in weights.index:
                    ic_means[factor_name] = ic.mean()

            # 将IC转换为权重
            total_ic = sum(abs(ic) for ic in ic_means.values())
            if total_ic > 0:
                for factor_name, ic in ic_means.items():
                    weights[factor_name] = abs(ic) / total_ic
            else:
                weights[:] = 1.0 / len(weights)

        elif method == 'regression':
            # 基于回归系数的权重
            model_result = self.build_factor_model(factors_clean, target_clean)
            if 'coefficients' in model_result:
                coeffs = model_result['coefficients']
                total_coeff = sum(abs(c) for c in coeffs)
                if total_coeff > 0:
                    for factor_name, coeff in coeffs.items():
                        weights[factor_name] = abs(coeff) / total_coeff
                else:
                    weights[:] = 1.0 / len(weights)
            else:
                weights[:] = 1.0 / len(weights)

        elif method == 'mean_variance':
            # 均值-方差优化
            # 标准化因子
            X = self.scaler.fit_transform(factors_clean)

            # 计算因子收益率（假设因子与目标之间存在线性关系）
            factor_returns = pd.DataFrame(X, index=factors_clean.index, columns=factors_clean.columns)

            # 计算因子收益率的协方差矩阵
            cov_matrix = factor_returns.cov().values

            # 计算因子的预期收益率（使用与目标的相关性作为代理）
            expected_returns = np.array([factor_returns[col].corr(target_clean) for col in factor_returns.columns])

            # 使用cvxpy求解均值-方差优化问题
            n = len(expected_returns)
            w = cp.Variable(n)

            # 目标函数：最大化收益率 - 风险厌恶系数 * 方差
            objective = cp.Maximize(expected_returns @ w - risk_aversion * cp.quad_form(w, cov_matrix))

            # 约束条件
            constraints = [
                cp.sum(w) == 1,  # 权重和为1
                w >= 0  # 权重非负
            ]

            # 求解问题
            problem = cp.Problem(objective, constraints)
            try:
                problem.solve()

                # 获取最优权重
                if w.value is not None:
                    for i, factor_name in enumerate(factors_clean.columns):
                        weights[factor_name] = w.value[i]
                else:
                    weights[:] = 1.0 / len(weights)
            except:
                # 如果优化失败，使用等权重
                weights[:] = 1.0 / len(weights)
        else:
            raise ValueError("不支持的权重优化方法，请选择'equal', 'ic', 'regression'或'mean_variance'")

        return weights

    def combine_factors(self, factors: pd.DataFrame, weights: pd.Series = None) -> pd.Series:
        """
        根据权重组合因子

        参数:
        factors (pd.DataFrame): 因子数据框，每列为一个因子
        weights (pd.Series): 因子权重，如果为None则使用等权重

        返回:
        pd.Series: 组合因子
        """
        # 如果没有提供权重，使用等权重
        if weights is None:
            weights = pd.Series(1.0 / factors.shape[1], index=factors.columns)

        # 确保权重和因子具有相同的列
        common_columns = factors.columns.intersection(weights.index)
        if len(common_columns) == 0:
            return pd.Series(index=factors.index)

        # 标准化因子
        normalized_factors = pd.DataFrame(index=factors.index)
        for col in common_columns:
            factor = factors[col]
            normalized_factors[col] = (factor - factor.mean()) / factor.std()

        # 计算加权组合因子
        combined_factor = pd.Series(0, index=factors.index)
        for col in common_columns:
            combined_factor += weights[col] * normalized_factors[col]

        return combined_factor

    def evaluate_combined_factor(self, combined_factor: pd.Series, target: pd.Series, 
                                evaluator=None) -> Dict:
        """
        评估组合因子

        参数:
        combined_factor (pd.Series): 组合因子
        target (pd.Series): 目标变量
        evaluator: 因子评估器，如果为None则仅计算相关性

        返回:
        Dict: 评估结果
        """
        # 确保因子和目标具有相同的索引
        common_index = combined_factor.index.intersection(target.index)
        factor_aligned = combined_factor.loc[common_index]
        target_aligned = target.loc[common_index]

        # 移除NaN值
        mask = ~(factor_aligned.isna() | target_aligned.isna())
        factor_clean = factor_aligned[mask]
        target_clean = target_aligned[mask]

        if len(factor_clean) == 0:
            return {}

        # 计算相关性
        correlation = factor_clean.corr(target_clean)

        # 创建结果字典
        result = {
            'correlation': correlation
        }

        # 如果提供了评估器，使用评估器进行全面评估
        if evaluator is not None:
            evaluation = evaluator.evaluate_factor(factor_clean, target_clean)
            result.update(evaluation)

        return result

    def perform_pca_combination(self, factors: pd.DataFrame, n_components: int = None, 
                               variance_threshold: float = 0.95) -> Dict:
        """
        使用PCA组合因子

        参数:
        factors (pd.DataFrame): 因子数据框，每列为一个因子
        n_components (int): 主成分数量，如果为None则使用方差阈值自动确定
        variance_threshold (float): 方差解释阈值，仅当n_components为None时使用

        返回:
        Dict: 包含PCA结果的字典
        """
        # 移除NaN值
        factors_clean = factors.dropna()

        if len(factors_clean) == 0:
            return {}

        # 标准化因子
        X = self.scaler.fit_transform(factors_clean)

        # 设置PCA
        if n_components is None:
            pca = PCA(n_components=variance_threshold, svd_solver='full')
        else:
            pca = PCA(n_components=n_components)

        # 应用PCA
        pca_result = pca.fit_transform(X)

        # 创建主成分DataFrame
        pca_df = pd.DataFrame(
            pca_result,
            index=factors_clean.index,
            columns=[f'PC{i+1}' for i in range(pca_result.shape[1])]
        )

        # 创建结果字典
        result = {
            'pca': pca,
            'scaler': self.scaler,
            'components': pca.components_,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
            'n_components': pca_result.shape[1],
            'pca_factors': pca_df
        }

        return result

    def plot_pca_results(self, pca_result: Dict, title: str = "PCA分析结果") -> None:
        """
        绘制PCA结果

        参数:
        pca_result (Dict): PCA结果字典
        title (str): 图表标题
        """
        # 创建图形
        plt.figure(figsize=(15, 10))

        # 1. 绘制解释方差比例
        plt.subplot(2, 2, 1)
        plt.bar(range(1, len(pca_result['explained_variance_ratio']) + 1), 
                pca_result['explained_variance_ratio'])
        plt.xlabel('主成分')
        plt.ylabel('解释方差比例')
        plt.title('各主成分解释方差比例')
        plt.grid(True, alpha=0.3)

        # 2. 绘制累积解释方差
        plt.subplot(2, 2, 2)
        plt.plot(range(1, len(pca_result['cumulative_variance_ratio']) + 1), 
                pca_result['cumulative_variance_ratio'], 'o-')
        plt.axhline(y=0.95, color='r', linestyle='--')
        plt.xlabel('主成分数量')
        plt.ylabel('累积解释方差比例')
        plt.title('累积解释方差比例')
        plt.grid(True, alpha=0.3)

        # 3. 绘制前两个主成分的因子载荷
        plt.subplot(2, 2, 3)
        components = pd.DataFrame(
            pca_result['components'][:2, :],
            columns=pca_result['pca_factors'].columns,
            index=['PC1', 'PC2']
        )
        plt.scatter(components.iloc[0], components.iloc[1])

        # 添加因子标签
        for i, txt in enumerate(components.columns):
            plt.annotate(txt, (components.iloc[0, i], components.iloc[1, i]))

        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('第一主成分')
        plt.ylabel('第二主成分')
        plt.title('因子载荷散点图')
        plt.grid(True, alpha=0.3)

        # 4. 绘制前两个主成分的散点图
        if len(pca_result['pca_factors'].columns) >= 2:
            plt.subplot(2, 2, 4)
            plt.scatter(pca_result['pca_factors']['PC1'], pca_result['pca_factors']['PC2'], alpha=0.5)
            plt.xlabel('第一主成分')
            plt.ylabel('第二主成分')
            plt.title('样本在前两个主成分上的分布')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.9)
        plt.savefig('pca_results.png')
        plt.show()

    def build_multi_factor_model(self, factors: pd.DataFrame, target: pd.Series, 
                                method: str = 'regression', **kwargs) -> Dict:
        """
        构建多因子模型

        参数:
        factors (pd.DataFrame): 因子数据框，每列为一个因子
        target (pd.Series): 目标变量
        method (str): 模型构建方法，可选'regression', 'pca', 'cluster'
        **kwargs: 其他参数

        返回:
        Dict: 包含模型结果的字典
        """
        result = {}

        if method == 'regression':
            # 使用回归模型
            model_type = kwargs.get('model_type', 'linear')
            alpha = kwargs.get('alpha', 1.0)

            result = self.build_factor_model(factors, target, model_type, alpha)

            # 使用模型系数作为权重
            if 'coefficients' in result:
                weights = result['coefficients'].copy()
                # 标准化权重
                total_weight = sum(abs(w) for w in weights)
                if total_weight > 0:
                    weights = weights.abs() / total_weight

                # 组合因子
                combined_factor = self.combine_factors(factors, weights)
                result['weights'] = weights
                result['combined_factor'] = combined_factor

        elif method == 'pca':
            # 使用PCA
            n_components = kwargs.get('n_components', None)
            variance_threshold = kwargs.get('variance_threshold', 0.95)

            # 执行PCA
            pca_result = self.perform_pca_combination(factors, n_components, variance_threshold)
            result.update(pca_result)

            if 'pca_factors' in pca_result:
                # 使用第一主成分作为组合因子
                result['combined_factor'] = pca_result['pca_factors']['PC1']

        elif method == 'cluster':
            # 使用聚类
            n_clusters = kwargs.get('n_clusters', None)
            cluster_method = kwargs.get('cluster_method', 'ward')
            threshold = kwargs.get('threshold', 1.0)

            # 执行聚类
            cluster_result = self.cluster_factors(factors, n_clusters, cluster_method, threshold)
            result['cluster_result'] = cluster_result

            # 从每个聚类中选择代表性因子
            selected_factors = self.select_representative_factors(factors, target, cluster_result)
            result['selected_factors'] = selected_factors

            # 优化权重
            weight_method = kwargs.get('weight_method', 'equal')
            ic_series = kwargs.get('ic_series', None)
            risk_aversion = kwargs.get('risk_aversion', 1.0)

            weights = self.optimize_weights(selected_factors, target, weight_method, ic_series, risk_aversion)
            result['weights'] = weights

            # 组合因子
            combined_factor = self.combine_factors(selected_factors, weights)
            result['combined_factor'] = combined_factor

        else:
            raise ValueError("不支持的模型构建方法，请选择'regression', 'pca'或'cluster'")

        return result