import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import scipy.optimize as sco

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """
    投资组合优化类，用于优化资产配置
    """
    
    def __init__(self):
        """
        初始化投资组合优化器
        """
        pass
    
    def optimize_portfolio(self, returns, method='mean_variance', risk_free_rate=0.02, target_return=None, target_risk=None, constraints=None):
        """
        优化投资组合
        
        参数:
        returns (pd.DataFrame): 资产收益率数据，每列为一个资产
        method (str): 优化方法，可选 'mean_variance', 'min_variance', 'max_sharpe', 'risk_parity'
        risk_free_rate (float): 无风险利率
        target_return (float): 目标收益率，仅在method='mean_variance'时使用
        target_risk (float): 目标风险，仅在method='mean_variance'时使用
        constraints (dict): 约束条件，例如 {'min_weight': 0.0, 'max_weight': 0.3}
        
        返回:
        dict: 优化结果
        """
        logger.info(f"使用{method}方法优化投资组合")
        
        # 计算资产的预期收益率和协方差矩阵
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # 设置约束条件
        if constraints is None:
            constraints = {'min_weight': 0.0, 'max_weight': 1.0}
        
        min_weight = constraints.get('min_weight', 0.0)
        max_weight = constraints.get('max_weight', 1.0)
        
        # 资产数量
        num_assets = len(returns.columns)
        
        # 根据不同方法优化投资组合
        if method == 'mean_variance':
            weights = self._optimize_mean_variance(mean_returns, cov_matrix, num_assets, target_return, target_risk, min_weight, max_weight)
        elif method == 'min_variance':
            weights = self._optimize_min_variance(cov_matrix, num_assets, min_weight, max_weight)
        elif method == 'max_sharpe':
            weights = self._optimize_max_sharpe(mean_returns, cov_matrix, risk_free_rate, num_assets, min_weight, max_weight)
        elif method == 'risk_parity':
            weights = self._optimize_risk_parity(cov_matrix, num_assets, min_weight, max_weight)
        else:
            raise ValueError(f"不支持的优化方法: {method}")
        
        # 计算投资组合的预期收益率和风险
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # 计算夏普比率
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev
        
        # 计算各资产的风险贡献
        risk_contribution = self._calculate_risk_contribution(weights, cov_matrix)
        
        # 汇总结果
        result = {
            'weights': pd.Series(weights, index=returns.columns),
            'return': portfolio_return,
            'risk': portfolio_std_dev,
            'sharpe_ratio': sharpe_ratio,
            'risk_contribution': pd.Series(risk_contribution, index=returns.columns)
        }
        
        logger.info(f"投资组合优化完成，预期收益率: {portfolio_return:.4f}, 风险: {portfolio_std_dev:.4f}, 夏普比率: {sharpe_ratio:.4f}")
        
        return result
    
    def _optimize_mean_variance(self, mean_returns, cov_matrix, num_assets, target_return=None, target_risk=None, min_weight=0.0, max_weight=1.0):
        """
        均值方差优化
        
        参数:
        mean_returns (pd.Series): 资产预期收益率
        cov_matrix (pd.DataFrame): 资产协方差矩阵
        num_assets (int): 资产数量
        target_return (float): 目标收益率
        target_risk (float): 目标风险
        min_weight (float): 最小权重
        max_weight (float): 最大权重
        
        返回:
        np.array: 优化后的权重
        """
        # 初始权重
        init_weights = np.array([1.0 / num_assets] * num_assets)
        
        # 权重约束
        bounds = tuple((min_weight, max_weight) for _ in range(num_assets))
        
        # 权重和为1的约束
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # 如果指定了目标收益率，添加收益率约束
        if target_return is not None:
            constraints = (
                constraints,
                {'type': 'eq', 'fun': lambda x: np.sum(mean_returns * x) - target_return}
            )
        
        # 如果指定了目标风险，添加风险约束
        if target_risk is not None:
            constraints = (
                constraints,
                {'type': 'eq', 'fun': lambda x: np.sqrt(np.dot(x.T, np.dot(cov_matrix, x))) - target_risk}
            )
        
        # 定义目标函数：最小化投资组合风险
        def objective(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # 优化
        result = sco.minimize(objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if not result['success']:
            logger.warning(f"优化失败: {result['message']}")
        
        return result['x']
    
    def _optimize_min_variance(self, cov_matrix, num_assets, min_weight=0.0, max_weight=1.0):
        """
        最小方差优化
        
        参数:
        cov_matrix (pd.DataFrame): 资产协方差矩阵
        num_assets (int): 资产数量
        min_weight (float): 最小权重
        max_weight (float): 最大权重
        
        返回:
        np.array: 优化后的权重
        """
        # 初始权重
        init_weights = np.array([1.0 / num_assets] * num_assets)
        
        # 权重约束
        bounds = tuple((min_weight, max_weight) for _ in range(num_assets))
        
        # 权重和为1的约束
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # 定义目标函数：最小化投资组合方差
        def objective(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        # 优化
        result = sco.minimize(objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if not result['success']:
            logger.warning(f"优化失败: {result['message']}")
        
        return result['x']
    
    def _optimize_max_sharpe(self, mean_returns, cov_matrix, risk_free_rate, num_assets, min_weight=0.0, max_weight=1.0):
        """
        最大夏普比率优化
        
        参数:
        mean_returns (pd.Series): 资产预期收益率
        cov_matrix (pd.DataFrame): 资产协方差矩阵
        risk_free_rate (float): 无风险利率
        num_assets (int): 资产数量
        min_weight (float): 最小权重
        max_weight (float): 最大权重
        
        返回:
        np.array: 优化后的权重
        """
        # 初始权重
        init_weights = np.array([1.0 / num_assets] * num_assets)
        
        # 权重约束
        bounds = tuple((min_weight, max_weight) for _ in range(num_assets))
        
        # 权重和为1的约束
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # 定义目标函数：最大化夏普比率（最小化负夏普比率）
        def objective(weights):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev
            return -sharpe_ratio  # 最小化负夏普比率
        
        # 优化
        result = sco.minimize(objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if not result['success']:
            logger.warning(f"优化失败: {result['message']}")
        
        return result['x']
    
    def _optimize_risk_parity(self, cov_matrix, num_assets, min_weight=0.0, max_weight=1.0):
        """
        风险平价优化
        
        参数:
        cov_matrix (pd.DataFrame): 资产协方差矩阵
        num_assets (int): 资产数量
        min_weight (float): 最小权重
        max_weight (float): 最大权重
        
        返回:
        np.array: 优化后的权重
        """
        # 初始权重
        init_weights = np.array([1.0 / num_assets] * num_assets)
        
        # 权重约束
        bounds = tuple((min_weight, max_weight) for _ in range(num_assets))
        
        # 权重和为1的约束
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # 定义目标函数：最小化风险贡献的方差
        def objective(weights):
            # 计算投资组合风险
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # 计算每个资产的风险贡献
            risk_contribution = np.array([])
            for i in range(len(weights)):
                asset_risk_contribution = weights[i] * np.dot(cov_matrix[i], weights) / portfolio_risk
                risk_contribution = np.append(risk_contribution, asset_risk_contribution)
            
            # 计算风险贡献的方差
            target_risk_contribution = portfolio_risk / num_assets
            risk_contribution_variance = np.sum((risk_contribution - target_risk_contribution) ** 2)
            
            return risk_contribution_variance
        
        # 优化
        result = sco.minimize(objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if not result['success']:
            logger.warning(f"优化失败: {result['message']}")
        
        return result['x']
    
    def _calculate_risk_contribution(self, weights, cov_matrix):
        """
        计算各资产的风险贡献
        
        参数:
        weights (np.array): 资产权重
        cov_matrix (pd.DataFrame): 资产协方差矩阵
        
        返回:
        np.array: 各资产的风险贡献
        """
        # 计算投资组合风险
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # 计算每个资产的风险贡献
        risk_contribution = np.array([])
        for i in range(len(weights)):
            asset_risk_contribution = weights[i] * np.dot(cov_matrix.iloc[i], weights) / portfolio_risk
            risk_contribution = np.append(risk_contribution, asset_risk_contribution)
        
        return risk_contribution
    
    def generate_efficient_frontier(self, returns, num_portfolios=100, risk_free_rate=0.02, min_weight=0.0, max_weight=1.0):
        """
        生成有效前沿
        
        参数:
        returns (pd.DataFrame): 资产收益率数据，每列为一个资产
        num_portfolios (int): 投资组合数量
        risk_free_rate (float): 无风险利率
        min_weight (float): 最小权重
        max_weight (float): 最大权重
        
        返回:
        pd.DataFrame: 有效前沿数据
        """
        logger.info(f"生成有效前沿，投资组合数量: {num_portfolios}")
        
        # 计算资产的预期收益率和协方差矩阵
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # 资产数量
        num_assets = len(returns.columns)
        
        # 生成随机权重
        weights_record = []
        returns_record = []
        risks_record = []
        sharpe_record = []
        
        for i in range(num_portfolios):
            # 生成随机权重
            weights = np.random.random(num_assets)
            weights = weights / np.sum(weights)
            
            # 确保权重在约束范围内
            weights = np.clip(weights, min_weight, max_weight)
            weights = weights / np.sum(weights)
            
            # 计算投资组合的预期收益率和风险
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # 计算夏普比率
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev
            
            # 记录结果
            weights_record.append(weights)
            returns_record.append(portfolio_return)
            risks_record.append(portfolio_std_dev)
            sharpe_record.append(sharpe_ratio)
        
        # 创建结果DataFrame
        results = pd.DataFrame({
            'return': returns_record,
            'risk': risks_record,
            'sharpe': sharpe_record
        })
        
        # 添加权重列
        for i, asset in enumerate(returns.columns):
            results[asset] = [weights[i] for weights in weights_record]
        
        # 计算最小方差投资组合
        min_var_weights = self._optimize_min_variance(cov_matrix, num_assets, min_weight, max_weight)
        min_var_return = np.sum(mean_returns * min_var_weights)
        min_var_risk = np.sqrt(np.dot(min_var_weights.T, np.dot(cov_matrix, min_var_weights)))
        min_var_sharpe = (min_var_return - risk_free_rate) / min_var_risk
        
        # 计算最大夏普比率投资组合
        max_sharpe_weights = self._optimize_max_sharpe(mean_returns, cov_matrix, risk_free_rate, num_assets, min_weight, max_weight)
        max_sharpe_return = np.sum(mean_returns * max_sharpe_weights)
        max_sharpe_risk = np.sqrt(np.dot(max_sharpe_weights.T, np.dot(cov_matrix, max_sharpe_weights)))
        max_sharpe_sharpe = (max_sharpe_return - risk_free_rate) / max_sharpe_risk
        
        # 添加最小方差和最大夏普比率投资组合
        min_var_portfolio = {
            'return': min_var_return,
            'risk': min_var_risk,
            'sharpe': min_var_sharpe
        }
        for i, asset in enumerate(returns.columns):
            min_var_portfolio[asset] = min_var_weights[i]
        
        max_sharpe_portfolio = {
            'return': max_sharpe_return,
            'risk': max_sharpe_risk,
            'sharpe': max_sharpe_sharpe
        }
        for i, asset in enumerate(returns.columns):
            max_sharpe_portfolio[asset] = max_sharpe_weights[i]
        
        # 添加到结果中
        results = pd.concat([results, pd.DataFrame([min_var_portfolio], index=['Min Variance'])])
        results = pd.concat([results, pd.DataFrame([max_sharpe_portfolio], index=['Max Sharpe'])])
        
        logger.info(f"有效前沿生成完成，包含{len(results)}个投资组合")
        
        return results
    
        def plot_efficient_frontier(self, results, risk_free_rate=0.02, figsize=(12, 8)):
            """
            绘制有效前沿
            
            参数:
            results (pd.DataFrame): 有效前沿数据
            risk_free_rate (float): 无风险利率
            figsize (tuple): 图表大小
            """
            plt.figure(figsize=figsize)
            
            # 绘制随机投资组合
            plt.scatter(results['risk'], results['return'], 
                           c=results['sharpe'], cmap='viridis', 
                           marker='o', s=10, alpha=0.5)
            
            # 绘制最小方差投资组合
            plt.scatter(results.loc['Min Variance', 'risk'], 
                           results.loc['Min Variance', 'return'], 
                           marker='*', color='r', s=300, label='最小方差')
            
            # 绘制最大夏普比率投资组合
            plt.scatter(results.loc['Max Sharpe', 'risk'], 
                           results.loc['Max Sharpe', 'return'], 
                           marker='*', color='g', s=300, label='最大夏普比率')
            
            # 绘制资本市场线
            x_min, x_max = plt.xlim()
            y_min = risk_free_rate
            y_max = risk_free_rate + results.loc['Max Sharpe', 'sharpe'] * x_max
            plt.plot([0, x_max], [risk_free_rate, y_max], 'k--', label='资本市场线')
            
            # 添加标题和标签
            plt.title('投资组合有效前沿', fontsize=16)
            plt.xlabel('风险 (标准差)', fontsize=12)
            plt.ylabel('预期收益率', fontsize=12)
            plt.colorbar(label='夏普比率')
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        def plot_weights(self, weights, title='投资组合权重分配', figsize=(10, 6)):
            """
            绘制投资组合权重分配
            
            参数:
            weights (pd.Series): 资产权重
            title (str): 图表标题
            figsize (tuple): 图表大小
            """
            plt.figure(figsize=figsize)
            
            # 绘制权重条形图
            weights.sort_values(ascending=False).plot(kind='bar')
            
            # 添加标题和标签
            plt.title(title, fontsize=16)
            plt.xlabel('资产', fontsize=12)
            plt.ylabel('权重', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # 添加权重值标签
            for i, v in enumerate(weights.sort_values(ascending=False)):
                plt.text(i, v + 0.01, f'{v:.2%}', ha='center')
            
            plt.tight_layout()
            plt.show()
        
        def plot_risk_contribution(self, risk_contribution, title='风险贡献分布', figsize=(10, 6)):
            """
            绘制风险贡献分布
            
            参数:
            risk_contribution (pd.Series): 风险贡献
            title (str): 图表标题
            figsize (tuple): 图表大小
            """
            plt.figure(figsize=figsize)
            
            # 计算风险贡献百分比
            risk_contribution_pct = risk_contribution / risk_contribution.sum()
            
            # 绘制风险贡献饼图
            plt.pie(risk_contribution_pct, labels=risk_contribution_pct.index, 
                       autopct='%1.1f%%', startangle=90, shadow=True)
            
            # 添加标题
            plt.title(title, fontsize=16)
            
            plt.axis('equal')  # 确保饼图是圆的
            plt.tight_layout()
            plt.show()
        
        def calculate_portfolio_performance(self, weights, returns):
            """
            计算投资组合的历史表现
            
            参数:
            weights (pd.Series): 资产权重
            returns (pd.DataFrame): 资产收益率数据，每列为一个资产
            
            返回:
            pd.Series: 投资组合历史收益率
            """
            # 确保权重和收益率的资产对齐
            common_assets = set(weights.index).intersection(set(returns.columns))
            if len(common_assets) != len(weights):
                logger.warning(f"权重和收益率数据的资产不完全匹配，只使用共同的{len(common_assets)}个资产")
            
            # 计算投资组合收益率
            portfolio_returns = (returns[weights.index] * weights).sum(axis=1)
            
            return portfolio_returns
        
        def calculate_portfolio_statistics(self, portfolio_returns, risk_free_rate=0.02):
            """
            计算投资组合统计指标
            
            参数:
            portfolio_returns (pd.Series): 投资组合历史收益率
            risk_free_rate (float): 无风险利率（年化）
            
            返回:
            dict: 投资组合统计指标
            """
            # 转换无风险利率为与收益率相同的频率
            if isinstance(portfolio_returns.index, pd.DatetimeIndex):
                # 估计收益率的频率
                if len(portfolio_returns) > 1:
                    time_diff = (portfolio_returns.index[-1] - portfolio_returns.index[0]).days
                    periods_per_year = len(portfolio_returns) / (time_diff / 365)
                else:
                    # 默认假设为日频数据
                    periods_per_year = 252
            else:
                # 默认假设为日频数据
                periods_per_year = 252
            
            # 转换无风险利率
            rf_rate_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
            
            # 计算累积收益率
            cumulative_return = (1 + portfolio_returns).prod() - 1
            
            # 计算年化收益率
            annual_return = (1 + cumulative_return) ** (periods_per_year / len(portfolio_returns)) - 1
            
            # 计算波动率（年化）
            volatility = portfolio_returns.std() * np.sqrt(periods_per_year)
            
            # 计算夏普比率
            sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility != 0 else np.nan
            
            # 计算最大回撤
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns / running_max) - 1
            max_drawdown = drawdown.min()
            
            # 计算索提诺比率
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 0
            sortino_ratio = (annual_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else np.nan
            
            # 计算卡玛比率
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
            
            # 汇总结果
            stats = {
                'cumulative_return': cumulative_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'periods_per_year': periods_per_year
            }
            
            return stats
        
        def plot_portfolio_performance(self, portfolio_returns, benchmark_returns=None, figsize=(15, 10)):
            """
            绘制投资组合表现
            
            参数:
            portfolio_returns (pd.Series): 投资组合历史收益率
            benchmark_returns (pd.Series): 基准收益率，可选
            figsize (tuple): 图表大小
            """
            plt.figure(figsize=figsize)
            
            # 计算累积收益率
            portfolio_cum_returns = (1 + portfolio_returns).cumprod() - 1
            
            # 绘制投资组合累积收益率
            plt.subplot(2, 1, 1)
            portfolio_cum_returns.plot(label='投资组合')
            
            # 如果有基准，也绘制基准累积收益率
            if benchmark_returns is not None:
                benchmark_cum_returns = (1 + benchmark_returns).cumprod() - 1
                benchmark_cum_returns.plot(label='基准')
            
            plt.title('累积收益率', fontsize=16)
            plt.xlabel('')
            plt.ylabel('累积收益率', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # 绘制回撤
            plt.subplot(2, 1, 2)
            
            # 计算投资组合回撤
            portfolio_cum_returns = (1 + portfolio_returns).cumprod()
            portfolio_running_max = portfolio_cum_returns.cummax()
            portfolio_drawdown = (portfolio_cum_returns / portfolio_running_max) - 1
            portfolio_drawdown.plot(label='投资组合回撤')
            
            # 如果有基准，也绘制基准回撤
            if benchmark_returns is not None:
                benchmark_cum_returns = (1 + benchmark_returns).cumprod()
                benchmark_running_max = benchmark_cum_returns.cummax()
                benchmark_drawdown = (benchmark_cum_returns / benchmark_running_max) - 1
                benchmark_drawdown.plot(label='基准回撤')
            
            plt.title('回撤', fontsize=16)
            plt.xlabel('')
            plt.ylabel('回撤', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            
            # 计算并打印统计指标
            portfolio_stats = self.calculate_portfolio_statistics(portfolio_returns)
            
            print(f"累积收益率: {portfolio_stats['cumulative_return']:.2%}")
            print(f"年化收益率: {portfolio_stats['annual_return']:.2%}")
            print(f"波动率: {portfolio_stats['volatility']:.2%}")
            print(f"夏普比率: {portfolio_stats['sharpe_ratio']:.2f}")
            print(f"最大回撤: {portfolio_stats['max_drawdown']:.2%}")
            print(f"索提诺比率: {portfolio_stats['sortino_ratio']:.2f}")
            print(f"卡玛比率: {portfolio_stats['calmar_ratio']:.2f}")
            
            if benchmark_returns is not None:
                benchmark_stats = self.calculate_portfolio_statistics(benchmark_returns)
                
                print("\n基准统计:")
                print(f"累积收益率: {benchmark_stats['cumulative_return']:.2%}")
                print(f"年化收益率: {benchmark_stats['annual_return']:.2%}")
                print(f"波动率: {benchmark_stats['volatility']:.2%}")
                print(f"夏普比率: {benchmark_stats['sharpe_ratio']:.2f}")
                print(f"最大回撤: {benchmark_stats['max_drawdown']:.2%}")