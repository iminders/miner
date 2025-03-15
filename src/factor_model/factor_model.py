import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from ..factor_evaluation.factor_evaluator import FactorEvaluator
from ..factor_evaluation.factor_selector import FactorSelector

logger = logging.getLogger(__name__)

class FactorModel:
    """
    因子模型类，整合因子计算、评估和选择的功能
    """

    def __init__(self):
        """
        初始化因子模型
        """
        self.factor_evaluator = FactorEvaluator()
        self.factor_selector = FactorSelector()
        self.factors_df = None
        self.returns = None
        self.selected_factors = None
        self.factor_weights = None
        self.combined_factor = None
        self.evaluation_results = {}

    def load_data(self, factors_df, returns):
        """
        加载因子数据和收益率数据

        参数:
        factors_df (pd.DataFrame): 包含因子数据的DataFrame
        returns (pd.Series): 收益率序列
        """
        self.factors_df = factors_df
        self.returns = returns
        logger.info(f"加载了{len(factors_df.columns)}个因子和{len(returns)}个收益率数据点")

    def evaluate_factors(self, factor_names=None, periods=[1, 5, 10, 20], quantiles=5, method='pearson'):
        """
        评估因子

        参数:
        factor_names (list): 要评估的因子名称列表，如果为None则评估所有因子
        periods (list): 未来收益率周期列表
        quantiles (int): 分层数量
        method (str): 相关系数计算方法，可选 'pearson', 'spearman'

        返回:
        dict: 包含各因子评估结果的字典
        """
        if self.factors_df is None or self.returns is None:
            raise ValueError("请先加载因子数据和收益率数据")

        if factor_names is None:
            factor_names = self.factors_df.columns.tolist()

        logger.info(f"评估{len(factor_names)}个因子")

        # 评估每个因子
        for factor_name in factor_names:
            self.evaluation_results[factor_name] = self.factor_evaluator.evaluate_factor(
                self.factors_df, factor_name, self.returns, periods=periods, quantiles=quantiles, method=method
            )

        return self.evaluation_results

    def compare_factors(self, metric='ic_ir'):
        """
        比较因子性能

        参数:
        metric (str): 用于比较的指标，可选 'ic', 'ic_ir', 'long_short_return'

        返回:
        pd.Series: 各因子的性能指标
        """
        if not self.evaluation_results:
            raise ValueError("请先评估因子")

        comparison = self.factor_evaluator.compare_factors(self.evaluation_results, metric)

        return comparison

    def select_factors(self, method='threshold', **kwargs):
        """
        选择因子

        参数:
        method (str): 因子选择方法，可选 'threshold', 'rank', 'correlation', 'lasso', 'random_forest', 'forward_selection', 'backward_elimination', 'combined'
        **kwargs: 传递给选择方法的参数

        返回:
        list: 选中的因子列表
        """
        if self.factors_df is None or self.returns is None:
            raise ValueError("请先加载因子数据和收益率数据")

        logger.info(f"使用{method}方法选择因子")

        # 根据不同方法选择因子
        if method == 'threshold':
            if not self.evaluation_results:
                raise ValueError("请先评估因子")
            self.selected_factors = self.factor_selector.select_by_threshold(
                self.evaluation_results, **kwargs
            )

        elif method == 'rank':
            if not self.evaluation_results:
                raise ValueError("请先评估因子")
            self.selected_factors = self.factor_selector.select_by_rank(
                self.evaluation_results, **kwargs
            )

        elif method == 'correlation':
            factor_names = kwargs.pop('factor_names', self.factors_df.columns.tolist())
            self.selected_factors = self.factor_selector.select_by_correlation(
                self.factors_df, factor_names, **kwargs
            )

        elif method == 'lasso':
            factor_names = kwargs.pop('factor_names', self.factors_df.columns.tolist())
            self.selected_factors = self.factor_selector.select_by_lasso(
                self.factors_df, factor_names, self.returns, **kwargs
            )

        elif method == 'random_forest':
            factor_names = kwargs.pop('factor_names', self.factors_df.columns.tolist())
            self.selected_factors = self.factor_selector.select_by_random_forest(
                self.factors_df, factor_names, self.returns, **kwargs
            )

        elif method == 'forward_selection':
            factor_names = kwargs.pop('factor_names', self.factors_df.columns.tolist())
            self.selected_factors = self.factor_selector.select_by_forward_selection(
                self.factors_df, factor_names, self.returns, **kwargs
            )

        elif method == 'backward_elimination':
            factor_names = kwargs.pop('factor_names', self.factors_df.columns.tolist())
            self.selected_factors = self.factor_selector.select_by_backward_elimination(
                self.factors_df, factor_names, self.returns, **kwargs
            )

        elif method == 'combined':
            factor_names = kwargs.pop('factor_names', self.factors_df.columns.tolist())
            evaluation_results = kwargs.pop('evaluation_results', self.evaluation_results if self.evaluation_results else None)
            self.selected_factors = self.factor_selector.select_by_combined_method(
                self.factors_df, factor_names, self.returns, evaluation_results, **kwargs
            )

        else:
            raise ValueError(f"不支持的因子选择方法: {method}")

        return self.selected_factors

    def optimize_weights(self, factor_names=None, method='equal_weight', **kwargs):
        """
        优化因子权重

        参数:
        factor_names (list): 要优化权重的因子名称列表，如果为None则使用已选择的因子
        method (str): 权重优化方法，可选 'equal_weight', 'regression', 'ic_weight'
        **kwargs: 传递给优化方法的参数

        返回:
        pd.Series: 因子权重
        """
        if self.factors_df is None or self.returns is None:
            raise ValueError("请先加载因子数据和收益率数据")

        if factor_names is None:
            if self.selected_factors is None:
                raise ValueError("请先选择因子或提供因子名称列表")
            factor_names = self.selected_factors

        logger.info(f"使用{method}方法优化{len(factor_names)}个因子的权重")

        self.factor_weights = self.factor_selector.optimize_factor_weights(
            self.factors_df, factor_names, self.returns, method=method, **kwargs
        )

        return self.factor_weights

    def generate_combined_factor(self, weights=None):
        """
        生成组合因子

        参数:
        weights (pd.Series): 因子权重，如果为None则使用已优化的权重

        返回:
        pd.Series: 组合因子
        """
        if self.factors_df is None:
            raise ValueError("请先加载因子数据")

        if weights is None:
            if self.factor_weights is None:
                raise ValueError("请先优化因子权重或提供权重")
            weights = self.factor_weights

        logger.info(f"生成{len(weights)}个因子的组合因子")

        self.combined_factor = self.factor_selector.generate_combined_factor(
            self.factors_df, weights
        )

        return self.combined_factor

    def evaluate_combined_factor(self, test_size=0.3, model_type='linear'):
        """
        评估组合因子的预测性能

        参数:
        test_size (float): 测试集比例
        model_type (str): 模型类型，可选 'linear', 'ridge', 'lasso', 'elastic_net', 'random_forest'

        返回:
        dict: 包含评估结果的字典
        """
        if self.factors_df is None or self.returns is None:
            raise ValueError("请先加载因子数据和收益率数据")

        if self.selected_factors is None:
            raise ValueError("请先选择因子")

        logger.info(f"评估{len(self.selected_factors)}个因子的组合性能")

        evaluation_results = self.factor_selector.evaluate_factor_combination(
            self.factors_df, self.selected_factors, self.returns, test_size, model_type
        )

        return evaluation_results

    def cross_validate_combined_factor(self, n_splits=5, model_type='linear'):
        """
        使用时间序列交叉验证评估组合因子

        参数:
        n_splits (int): 交叉验证折数
        model_type (str): 模型类型，可选 'linear', 'ridge', 'lasso', 'elastic_net', 'random_forest'

        返回:
        dict: 包含交叉验证结果的字典
        """
        if self.factors_df is None or self.returns is None:
            raise ValueError("请先加载因子数据和收益率数据")

        if self.selected_factors is None:
            raise ValueError("请先选择因子")

        logger.info(f"使用{n_splits}折时间序列交叉验证评估{len(self.selected_factors)}个因子的组合性能")

        cv_results = self.factor_selector.cross_validate_factor_combination(
            self.factors_df, self.selected_factors, self.returns, n_splits, model_type
        )

        return cv_results

    def plot_factor_correlation(self, factor_names=None):
        """
        绘制因子相关性热图

        参数:
        factor_names (list): 要绘制的因子名称列表，如果为None则使用已选择的因子
        """
        if self.factors_df is None:
            raise ValueError("请先加载因子数据")

        if factor_names is None:
            if self.selected_factors is None:
                raise ValueError("请先选择因子或提供因子名称列表")
            factor_names = self.selected_factors

        # 计算相关性矩阵
        correlation_matrix = self.factor_evaluator.calculate_factor_correlation(self.factors_df, factor_names)

        # 绘制热图
        self.factor_evaluator.plot_factor_correlation(correlation_matrix, title='因子相关性矩阵')

    def plot_factor_ic_time_series(self, factor_name):
        """
        绘制因子IC时间序列图

        参数:
        factor_name (str): 因子名称
        """
        if not self.evaluation_results or factor_name not in self.evaluation_results:
            raise ValueError(f"请先评估因子: {factor_name}")

        # 获取IC时间序列
        ic_series = self.evaluation_results[factor_name]['ic_series']

        # 绘制IC时间序列图
        self.factor_evaluator.plot_ic_time_series(ic_series, title=f'{factor_name} IC时间序列')

    def plot_factor_ic_decay(self, factor_name):
        """
        绘制因子IC衰减图

        参数:
        factor_name (str): 因子名称
        """
        if not self.evaluation_results or factor_name not in self.evaluation_results:
            raise ValueError(f"请先评估因子: {factor_name}")

        # 获取IC衰减
        ic_decay = self.evaluation_results[factor_name]['ic_decay']

        # 绘制IC衰减图
        self.factor_evaluator.plot_ic_decay(ic_decay, title=f'{factor_name} IC衰减')

    def plot_factor_quantile_returns(self, factor_name):
        """
        绘制因子分层收益率图

        参数:
        factor_name (str): 因子名称
        """
        if not self.evaluation_results or factor_name not in self.evaluation_results:
            raise ValueError(f"请先评估因子: {factor_name}")

        # 获取分层收益率
        quantile_returns = self.evaluation_results[factor_name]['quantile_returns']

        # 绘制分层收益率图
        self.factor_evaluator.plot_quantile_returns(quantile_returns, title=f'{factor_name} 分层收益率')

    def plot_combined_factor_performance(self, test_size=0.3):
        """
        绘制组合因子性能图

        参数:
        test_size (float): 测试集比例
        """
        if self.combined_factor is None:
            raise ValueError("请先生成组合因子")

        if self.returns is None:
            raise ValueError("请先加载收益率数据")

        # 计算未来1期收益率
        forward_returns = self.returns.shift(-1)

        # 确保数据对齐
        common_index = self.combined_factor.index.intersection(forward_returns.index)
        factor = self.combined_factor.loc[common_index]
        returns = forward_returns.loc[common_index]

        # 划分训练集和测试集
        split_idx = int(len(factor) * (1 - test_size))
        factor_train, factor_test = factor.iloc[:split_idx], factor.iloc[split_idx:]
        returns_train, returns_test = returns.iloc[:split_idx], returns.iloc[split_idx:]

        # 绘制散点图
        plt.figure(figsize=(16, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(factor_train, returns_train, alpha=0.5)
        plt.title('训练集: 组合因子 vs 未来收益率')
        plt.xlabel('组合因子值')
        plt.ylabel('未来收益率')
        plt.grid(True, alpha=0.3)

        # 添加回归线
        z = np.polyfit(factor_train, returns_train, 1)
        p = np.poly1d(z)
        plt.plot(factor_train, p(factor_train), "r--", alpha=0.8)

        plt.subplot(1, 2, 2)
        plt.scatter(factor_test, returns_test, alpha=0.5)
        plt.title('测试集: 组合因子 vs 未来收益率')
        plt.xlabel('组合因子值')
        plt.ylabel('未来收益率')
        plt.grid(True, alpha=0.3)

        # 添加回归线
        z = np.polyfit(factor_test, returns_test, 1)
        p = np.poly1d(z)
        plt.plot(factor_test, p(factor_test), "r--", alpha=0.8)

        plt.tight_layout()
        plt.show()

    def save_model(self, filepath):
        """
        保存模型

        参数:
        filepath (str): 保存路径
        """
        import pickle

        model_data = {
            'selected_factors': self.selected_factors,
            'factor_weights': self.factor_weights,
            'evaluation_results': self.evaluation_results
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"模型已保存到: {filepath}")

    def load_model(self, filepath):
        """
        加载模型

        参数:
        filepath (str): 加载路径
        """
        import pickle

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.selected_factors = model_data['selected_factors']
        self.factor_weights = model_data['factor_weights']
        self.evaluation_results = model_data['evaluation_results']

        logger.info(f"模型已从{filepath}加载")
        logger.info(f"加载了{len(self.selected_factors)}个选中因子和权重")