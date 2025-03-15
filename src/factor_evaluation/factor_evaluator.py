import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)

class FactorEvaluator:
    """
    因子评估类，用于评估因子的有效性
    """
    
    def __init__(self):
        """
        初始化因子评估器
        """
        pass
    
    def calculate_ic(self, factor_data, forward_returns, method='pearson'):
        """
        计算因子的信息系数(IC)
        
        参数:
        factor_data (pd.Series): 因子数据
        forward_returns (pd.Series): 未来收益率
        method (str): 相关系数计算方法，可选 'pearson', 'spearman'
        
        返回:
        float: 信息系数
        """
        if method == 'pearson':
            ic, p_value = stats.pearsonr(factor_data.dropna(), forward_returns.dropna())
        elif method == 'spearman':
            ic, p_value = stats.spearmanr(factor_data.dropna(), forward_returns.dropna())
        else:
            raise ValueError(f"不支持的相关系数计算方法: {method}")
        
        return ic, p_value
    
    def calculate_ic_series(self, factor_df, factor_name, forward_returns, method='pearson'):
        """
        计算因子的信息系数(IC)时间序列
        
        参数:
        factor_df (pd.DataFrame): 包含因子数据的DataFrame
        factor_name (str): 因子名称
        forward_returns (pd.Series): 未来收益率
        method (str): 相关系数计算方法，可选 'pearson', 'spearman'
        
        返回:
        pd.Series: 信息系数时间序列
        """
        ic_series = pd.Series(index=factor_df.index)
        p_value_series = pd.Series(index=factor_df.index)
        
        for date in factor_df.index:
            if date in forward_returns.index:
                factor_value = factor_df.loc[date, factor_name]
                return_value = forward_returns.loc[date]
                
                if not np.isnan(factor_value) and not np.isnan(return_value):
                    ic_series.loc[date], p_value_series.loc[date] = self.calculate_ic(
                        pd.Series([factor_value]), 
                        pd.Series([return_value]), 
                        method
                    )
        
        return ic_series, p_value_series
    
    def calculate_ic_decay(self, factor_df, factor_name, returns, periods=[1, 5, 10, 20], method='pearson'):
        """
        计算因子的IC衰减
        
        参数:
        factor_df (pd.DataFrame): 包含因子数据的DataFrame
        factor_name (str): 因子名称
        returns (pd.Series): 收益率序列
        periods (list): 未来收益率周期列表
        method (str): 相关系数计算方法，可选 'pearson', 'spearman'
        
        返回:
        pd.Series: 不同周期的IC值
        """
        ic_decay = {}
        
        for period in periods:
            # 计算未来period期的收益率
            forward_returns = returns.shift(-period)
            
            # 计算IC
            ic, p_value = self.calculate_ic(factor_df[factor_name], forward_returns, method)
            ic_decay[period] = ic
        
        return pd.Series(ic_decay)
    
    def calculate_factor_returns(self, factor_data, forward_returns, quantiles=5):
        """
        计算分层组合收益率
        
        参数:
        factor_data (pd.Series): 因子数据
        forward_returns (pd.Series): 未来收益率
        quantiles (int): 分层数量
        
        返回:
        pd.Series: 各分位数的平均收益率
        """
        # 确保数据对齐
        valid_data = pd.concat([factor_data, forward_returns], axis=1).dropna()
        factor_data = valid_data.iloc[:, 0]
        forward_returns = valid_data.iloc[:, 1]
        
        # 按因子值分组
        labels = range(1, quantiles + 1)
        quantile_labels = pd.qcut(factor_data, quantiles, labels=labels)
        
        # 计算各组的平均收益率
        quantile_returns = pd.Series(index=labels)
        for quantile in labels:
            quantile_returns[quantile] = forward_returns[quantile_labels == quantile].mean()
        
        return quantile_returns
    
    def calculate_factor_turnover(self, factor_df, factor_name, quantiles=5):
        """
        计算因子换手率
        
        参数:
        factor_df (pd.DataFrame): 包含因子数据的DataFrame
        factor_name (str): 因子名称
        quantiles (int): 分层数量
        
        返回:
        float: 因子换手率
        """
        # 获取因子数据
        factor_data = factor_df[factor_name]
        
        # 计算每个时间点的分位数
        quantile_data = pd.DataFrame(index=factor_data.index)
        quantile_data['quantile'] = pd.qcut(factor_data, quantiles, labels=False) + 1
        
        # 计算换手率
        turnover = 0
        for i in range(1, len(quantile_data)):
            prev_quantile = quantile_data.iloc[i-1]['quantile']
            curr_quantile = quantile_data.iloc[i]['quantile']
            
            # 如果分位数发生变化，则视为换手
            if prev_quantile != curr_quantile:
                turnover += 1
        
        # 计算平均换手率
        avg_turnover = turnover / (len(quantile_data) - 1) if len(quantile_data) > 1 else 0
        
        return avg_turnover
    
    def calculate_factor_exposure(self, factor_data, risk_factors):
        """
        计算因子对风险因子的暴露度
        
        参数:
        factor_data (pd.Series): 因子数据
        risk_factors (pd.DataFrame): 风险因子数据
        
        返回:
        pd.Series: 因子对各风险因子的暴露度
        """
        # 确保数据对齐
        valid_data = pd.concat([factor_data, risk_factors], axis=1).dropna()
        factor_data = valid_data.iloc[:, 0]
        risk_factors = valid_data.iloc[:, 1:]
        
        # 使用线性回归计算暴露度
        model = LinearRegression()
        model.fit(risk_factors, factor_data)
        
        # 获取回归系数
        exposures = pd.Series(model.coef_, index=risk_factors.columns)
        
        return exposures
    
    def plot_ic_time_series(self, ic_series, title='Information Coefficient Time Series'):
        """
        绘制IC时间序列图
        
        参数:
        ic_series (pd.Series): IC时间序列
        title (str): 图表标题
        """
        plt.figure(figsize=(12, 6))
        ic_series.plot()
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.axhline(y=ic_series.mean(), color='g', linestyle='--', alpha=0.7, label=f'Mean IC: {ic_series.mean():.4f}')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Information Coefficient')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_ic_decay(self, ic_decay, title='IC Decay'):
        """
        绘制IC衰减图
        
        参数:
        ic_decay (pd.Series): 不同周期的IC值
        title (str): 图表标题
        """
        plt.figure(figsize=(10, 6))
        ic_decay.plot(kind='bar')
        plt.title(title)
        plt.xlabel('Forward Periods')
        plt.ylabel('Information Coefficient')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_quantile_returns(self, quantile_returns, title='Quantile Returns'):
        """
        绘制分层收益率图
        
        参数:
        quantile_returns (pd.Series): 各分位数的平均收益率
        title (str): 图表标题
        """
        plt.figure(figsize=(10, 6))
        quantile_returns.plot(kind='bar')
        plt.title(title)
        plt.xlabel('Quantile')
        plt.ylabel('Average Return')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_factor_exposures(self, exposures, title='Factor Exposures to Risk Factors'):
        """
        绘制因子暴露度图
        
        参数:
        exposures (pd.Series): 因子对各风险因子的暴露度
        title (str): 图表标题
        """
        plt.figure(figsize=(12, 6))
        exposures.plot(kind='bar')
        plt.title(title)
        plt.xlabel('Risk Factor')
        plt.ylabel('Exposure')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def evaluate_factor(self, factor_df, factor_name, returns, risk_factors=None, periods=[1, 5, 10, 20], quantiles=5, method='pearson'):
        """
        全面评估因子
        
        参数:
        factor_df (pd.DataFrame): 包含因子数据的DataFrame
        factor_name (str): 因子名称
        returns (pd.Series): 收益率序列
        risk_factors (pd.DataFrame): 风险因子数据
        periods (list): 未来收益率周期列表
        quantiles (int): 分层数量
        method (str): 相关系数计算方法，可选 'pearson', 'spearman'
        
        返回:
        dict: 包含各项评估指标的字典
        """
        logger.info(f"评估因子: {factor_name}")
        
        # 计算1期未来收益率
        forward_returns = returns.shift(-1)
        
        # 1. 计算IC
        ic, p_value = self.calculate_ic(factor_df[factor_name], forward_returns, method)
        logger.info(f"IC: {ic:.4f}, p-value: {p_value:.4f}")
        
        # 2. 计算IC时间序列
        ic_series, p_value_series = self.calculate_ic_series(factor_df, factor_name, forward_returns, method)
        
        # 3. 计算IC衰减
        ic_decay = self.calculate_ic_decay(factor_df, factor_name, returns, periods, method)
        
        # 4. 计算分层收益率
        quantile_returns = self.calculate_factor_returns(factor_df[factor_name], forward_returns, quantiles)
        
        # 5. 计算因子换手率
        turnover = self.calculate_factor_turnover(factor_df, factor_name, quantiles)
        logger.info(f"因子换手率: {turnover:.4f}")
        
        # 6. 计算因子对风险因子的暴露度
        exposures = None
        if risk_factors is not None:
            exposures = self.calculate_factor_exposure(factor_df[factor_name], risk_factors)
        
        # 7. 计算IC的统计特性
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        ic_positive_pct = (ic_series > 0).mean()
        
        logger.info(f"IC均值: {ic_mean:.4f}, IC标准差: {ic_std:.4f}, IC IR: {ic_ir:.4f}, IC>0占比: {ic_positive_pct:.4f}")
        
        # 8. 计算多空组合收益率
        long_short_return = quantile_returns.iloc[-1] - quantile_returns.iloc[0]
        logger.info(f"多空组合收益率: {long_short_return:.4f}")
        
        # 返回评估结果
        evaluation_results = {
            'factor_name': factor_name,
            'ic': ic,
            'p_value': p_value,
            'ic_series': ic_series,
            'p_value_series': p_value_series,
            'ic_decay': ic_decay,
            'quantile_returns': quantile_returns,
            'turnover': turnover,
            'exposures': exposures,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ic_ir': ic_ir,
            'ic_positive_pct': ic_positive_pct,
            'long_short_return': long_short_return
        }
        
        return evaluation_results
    
    def evaluate_multiple_factors(self, factor_df, factor_names, returns, risk_factors=None, periods=[1, 5, 10, 20], quantiles=5, method='pearson'):
        """
        评估多个因子
        
        参数:
        factor_df (pd.DataFrame): 包含因子数据的DataFrame
        factor_names (list): 因子名称列表
        returns (pd.Series): 收益率序列
        risk_factors (pd.DataFrame): 风险因子数据
        periods (list): 未来收益率周期列表
        quantiles (int): 分层数量
        method (str): 相关系数计算方法，可选 'pearson', 'spearman'
        
        返回:
        dict: 包含各因子评估结果的字典
        """
        logger.info(f"评估{len(factor_names)}个因子")
        
        evaluation_results = {}
        for factor_name in factor_names:
            evaluation_results[factor_name] = self.evaluate_factor(
                factor_df, factor_name, returns