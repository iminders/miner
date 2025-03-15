import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)

class FactorSelector:
    """
    因子选择类，用于选择最佳因子组合
    """
    
    def __init__(self):
        """
        初始化因子选择器
        """
        pass
    
    def select_by_threshold(self, evaluation_results, metric='ic_ir', threshold=0.1):
        """
        根据阈值选择因子
        
        参数:
        evaluation_results (dict): 包含各因子评估结果的字典
        metric (str): 用于选择的指标，可选 'ic', 'ic_ir', 'long_short_return'
        threshold (float): 阈值
        
        返回:
        list: 选中的因子列表
        """
        logger.info(f"使用{metric}指标和阈值{threshold}选择因子")
        
        selected_factors = []
        for factor_name, results in evaluation_results.items():
            if results[metric] > threshold:
                selected_factors.append(factor_name)
        
        logger.info(f"选中{len(selected_factors)}个因子")
        
        return selected_factors
    
    def select_by_rank(self, evaluation_results, metric='ic_ir', top_n=10):
        """
        根据排名选择因子
        
        参数:
        evaluation_results (dict): 包含各因子评估结果的字典
        metric (str): 用于选择的指标，可选 'ic', 'ic_ir', 'long_short_return'
        top_n (int): 选择排名前top_n的因子
        
        返回:
        list: 选中的因子列表
        """
        logger.info(f"使用{metric}指标选择排名前{top_n}的因子")
        
        # 提取指标值
        metric_values = {}
        for factor_name, results in evaluation_results.items():
            metric_values[factor_name] = results[metric]
        
        # 按指标值排序
        sorted_factors = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
        
        # 选择前top_n个因子
        selected_factors = [factor[0] for factor in sorted_factors[:top_n]]
        
        logger.info(f"选中{len(selected_factors)}个因子")
        
        return selected_factors
    
    def select_by_correlation(self, factor_df, factor_names, threshold=0.7):
        """
        根据相关性选择因子，去除高度相关的因子
        
        参数:
        factor_df (pd.DataFrame): 包含因子数据的DataFrame
        factor_names (list): 因子名称列表
        threshold (float): 相关性阈值
        
        返回:
        list: 选中的因子列表
        """
        logger.info(f"使用相关性阈值{threshold}选择因子")
        
        # 提取因子数据
        factors_data = factor_df[factor_names]
        
        # 计算相关性矩阵
        correlation_matrix = factors_data.corr().abs()
        
        # 创建上三角矩阵
        upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        
        # 找出高度相关的列
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        # 保留的列
        selected_factors = [col for col in factor_names if col not in to_drop]
        
        logger.info(f"选中{len(selected_factors)}个因子")
        
        return selected_factors
    
    def select_by_lasso(self, factor_df, factor_names, returns, alpha=0.01, test_size=0.3):
        """
        使用Lasso回归选择因子
        
        参数:
        factor_df (pd.DataFrame): 包含因子数据的DataFrame
        factor_names (list): 因子名称列表
        returns (pd.Series): 收益率序列
        alpha (float): Lasso正则化参数
        test_size (float): 测试集比例
        
        返回:
        list: 选中的因子列表
        """
        logger.info(f"使用Lasso回归(alpha={alpha})选择因子")
        
        # 提取因子数据
        X = factor_df[factor_names].dropna()
        y = returns.shift(-1).loc[X.index].dropna()  # 使用下一期收益率作为目标变量
        
        # 确保X和y的索引对齐
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        # 划分训练集和测试集
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # 训练Lasso模型
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_train, y_train)
        
        # 获取非零系数的特征
        selected_indices = np.where(lasso.coef_ != 0)[0]
        selected_factors = [factor_names[i] for i in selected_indices]
        
        # 评估模型
        y_pred = lasso.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Lasso模型MSE: {mse:.6f}, R²: {r2:.6f}")
        logger.info(f"选中{len(selected_factors)}个因子")
        
        return selected_factors
    
    def select_by_random_forest(self, factor_df, factor_names, returns, n_estimators=100, test_size=0.3, importance_threshold=0.01):
        """
        使用随机森林选择因子
        
        参数:
        factor_df (pd.DataFrame): 包含因子数据的DataFrame
        factor_names (list): 因子名称列表
        returns (pd.Series): 收益率序列
        n_estimators (int): 随机森林中树的数量
        test_size (float): 测试集比例
        importance_threshold (float): 特征重要性阈值
        
        返回:
        list: 选中的因子列表
        """
        logger.info(f"使用随机森林(n_estimators={n_estimators})选择因子")
        
        # 提取因子数据
        X = factor_df[factor_names].dropna()
        y = returns.shift(-1).loc[X.index].dropna()  # 使用下一期收益率作为目标变量
        
        # 确保X和y的索引对齐
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        # 划分训练集和测试集
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # 训练随机森林模型
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        rf.fit(X_train, y_train)
        
        # 获取特征重要性
        importances = rf.feature_importances_
        
        # 选择重要性大于阈值的特征
        selected_indices = np.where(importances > importance_threshold)[0]
        selected_factors = [factor_names[i] for i in selected_indices]
        
        # 评估模型
        y_pred = rf.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"随机森林模型MSE: {mse:.6f}, R²: {r2:.6f}")
        logger.info(f"选中{len(selected_factors)}个因子")
        
        return selected_factors
    
    def select_by_forward_selection(self, factor_df, factor_names, returns, max_features=10, test_size=0.3):
        """
        使用前向选择法选择因子
        
        参数:
        factor_df (pd.DataFrame): 包含因子数据的DataFrame
        factor_names (list): 因子名称列表
        returns (pd.Series): 收益率序列
        max_features (int): 最大选择的特征数量
        test_size (float): 测试集比例
        
        返回:
        list: 选中的因子列表
        """
        logger.info(f"使用前向选择法选择最多{max_features}个因子")
        
        # 提取因子数据
        X = factor_df[factor_names].dropna()
        y = returns.shift(-1).loc[X.index].dropna()  # 使用下一期收益率作为目标变量
        
        # 确保X和y的索引对齐
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        # 划分训练集和测试集
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # 初始化
        selected_factors = []
        remaining_factors = factor_names.copy()
        current_score = 0
        
        # 前向选择
        for i in range(min(max_features, len(factor_names))):
            best_factor = None
            best_score = current_score
            
            # 尝试添加每个剩余因子
            for factor in remaining_factors:
                # 构建当前特征集
                current_factors = selected_factors + [factor]
                
                # 训练模型
                model = LinearRegression()
                model.fit(X_train[current_factors], y_train)
                
                # 评估模型
                y_pred = model.predict(X_test[current_factors])
                score = r2_score(y_test, y_pred)
                
                # 更新最佳因子
                if score > best_score:
                    best_score = score
                    best_factor = factor
            
            # 如果没有改进，则停止
            if best_factor is None:
                break
            
            # 添加最佳因子
            selected_factors.append(best_factor)
            remaining_factors.remove(best_factor)
            current_score = best_score
            
            logger.info(f"添加因子: {best_factor}, R²: {current_score:.6f}")
        
        logger.info(f"选中{len(selected_factors)}个因子")
        
        return selected_factors
    
    def select_by_backward_elimination(self, factor_df, factor_names, returns, test_size=0.3, threshold=0.001):
        """
        使用后向消除法选择因子
        
        参数:
        factor_df (pd.DataFrame): 包含因子数据的DataFrame
        factor_names (list): 因子名称列表
        returns (pd.Series): 收益率序列
        test_size (float): 测试集比例
        threshold (float): 性能下降阈值，低于此值的特征将被保留
        
        返回:
        list: 选中的因子列表
        """
        logger.info(f"使用后向消除法选择因子")
        
        # 提取因子数据
        X = factor_df[factor_names].dropna()
        y = returns.shift(-1).loc[X.index].dropna()  # 使用下一期收益率作为目标变量
        
        # 确保X和y的索引对齐
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        # 划分训练集和测试集
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # 初始化
        selected_factors = factor_names.copy()
        
        # 使用所有特征训练模型
        model = LinearRegression()
        model.fit(X_train[selected_factors], y_train)
        
        # 评估基准模型
        y_pred = model.predict(X_test[selected_factors])
        current_score = r2_score(y_test, y_pred)
        
        logger.info(f"初始模型R²: {current_score:.6f}")
        
        # 后向消除
        improvement = True
        while improvement and len(selected_factors) > 1:
            improvement = False
            worst_factor = None
            best_score = current_score
            
            # 尝试移除每个因子
            for factor in selected_factors:
                # 构建当前特征集
                current_factors = [f for f in selected_factors if f != factor]
                
                # 训练模型
                model = LinearRegression()
                model.fit(X_train[current_factors], y_train)
                
                # 评估模型
                y_pred = model.predict(X_test[current_factors])
                score = r2_score(y_test, y_pred)
                
                # 如果性能下降不超过阈值，则可以移除该因子
                if score > best_score - threshold:
                    best_score = score
                    worst_factor = factor
                    improvement = True
            
            # 移除最不重要的因子
            if improvement:
                selected_factors.remove(worst_factor)
                current_score = best_score
                logger.info(f"移除因子: {worst_factor}, R²: {current_score:.6f}")
        
        logger.info(f"选中{len(selected_factors)}个因子")
        
        return selected_factors
    
    def select_by_combined_method(self, factor_df, factor_names, returns, evaluation_results=None, 
                                 ic_threshold=0.05, correlation_threshold=0.7, lasso_alpha=0.01, 
                                 test_size=0.3, max_features=20):
        """
        使用组合方法选择因子
        
        参数:
        factor_df (pd.DataFrame): 包含因子数据的DataFrame
        factor_names (list): 因子名称列表
        returns (pd.Series): 收益率序列
        evaluation_results (dict): 包含各因子评估结果的字典
        ic_threshold (float): IC阈值
        correlation_threshold (float): 相关性阈值
        lasso_alpha (float): Lasso正则化参数
        test_size (float): 测试集比例
        max_features (int): 最大选择的特征数量
        
        返回:
        list: 选中的因子列表
        """
        logger.info("使用组合方法选择因子")
        
        # 步骤1: 根据IC筛选因子
        if evaluation_results is not None:
            factors_by_ic = self.select_by_threshold(evaluation_results, metric='ic_ir', threshold=ic_threshold)
            logger.info(f"根据IC筛选出{len(factors_by_ic)}个因子")
        else:
            factors_by_ic = factor_names
            logger.info("跳过IC筛选步骤")
        
        # 步骤2: 去除高度相关的因子
        factors_by_corr = self.select_by_correlation(factor_df, factors_by_ic, threshold=correlation_threshold)
        logger.info(f"去除高度相关因子后剩余{len(factors_by_corr)}个因子")
        
        # 步骤3: 使用Lasso进一步筛选因子
        if len(factors_by_corr) > max_features:
            factors_by_lasso = self.select_by_lasso(factor_df, factors_by_corr, returns, alpha=lasso_alpha, test_size=test_size)
            logger.info(f"使用Lasso筛选出{len(factors_by_lasso)}个因子")
        else:
            factors_by_lasso = factors_by_corr
            logger.info("跳过Lasso筛选步骤")
        
        # 步骤4: 使用前向选择法选择最终因子集
        if len(factors_by_lasso) > max_features:
            final_factors = self.select_by_forward_selection(factor_df, factors_by_lasso, returns, max_features=max_features, test_size=test_size)
        else:
            final_factors = factors_by_lasso
        
        logger.info(f"最终选中{len(final_factors)}个因子")
        
        return final_factors
    
    def evaluate_factor_combination(self, factor_df, factor_names, returns, test_size=0.3, model_type='linear'):
        """
        评估因子组合的预测性能
        
        参数:
        factor_df (pd.DataFrame): 包含因子数据的DataFrame
        factor_names (list): 因子名称列表
        returns (pd.Series): 收益率序列
        test_size (float): 测试集比例
        model_type (str): 模型类型，可选 'linear', 'ridge', 'lasso', 'elastic_net', 'random_forest'
        
        返回:
        dict: 包含评估结果的字典
        """
        logger.info(f"评估{len(factor_names)}个因子的组合性能")
        
        # 提取因子数据
        X = factor_df[factor_names].dropna()
        y = returns.shift(-1).loc[X.index].dropna()  # 使用下一期收益率作为目标变量
        
        # 确保X和y的索引对齐
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        # 划分训练集和测试集
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # 选择模型
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'ridge':
            model = Ridge(alpha=0.1)
        elif model_type == 'lasso':
            model = Lasso(alpha=0.01)
        elif model_type == 'elastic_net':
            model = ElasticNet(alpha=0.01, l1_ratio=0.5)
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 评估模型
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # 计算方向准确率
        train_direction_accuracy = np.mean((y_train > 0) == (y_pred_train > 0))
        test_direction_accuracy = np.mean((y_test > 0) == (y_pred_test > 0))
        
        # 如果是线性模型，获取系数
        if model_type in ['linear', 'ridge', 'lasso', 'elastic_net']:
            coefficients = pd.Series(model.coef_, index=factor_names)
        else:
            coefficients = pd.Series(model.feature_importances_, index=factor_names)
        
        # 记录评估结果
        logger.info(f"训练集MSE: {train_mse:.6f}, R²: {train_r2:.6f}, 方向准确率: {train_direction_accuracy:.4f}")
        logger.info(f"测试集MSE: {test_mse:.6f}, R²: {test_r2:.6f}, 方向准确率: {test_direction_accuracy:.4f}")
        
        # 返回评估结果
        evaluation_results = {
            'model_type': model_type,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_direction_accuracy': train_direction_accuracy,
            'test_direction_accuracy': test_direction_accuracy,
            'coefficients': coefficients,
            'model': model
        }
        
        return evaluation_results
    
    def cross_validate_factor_combination(self, factor_df, factor_names, returns, n_splits=5, model_type='linear'):
        """
        使用时间序列交叉验证评估因子组合
        
        参数:
        factor_df (pd.DataFrame): 包含因子数据的DataFrame
        factor_names (list): 因子名称列表
        returns (pd.Series): 收益率序列
        n_splits (int): 交叉验证折数
        model_type (str): 模型类型，可选 'linear', 'ridge', 'lasso', 'elastic_net', 'random_forest'
        
        返回:
        dict: 包含交叉验证结果的字典
        """
        logger.info(f"使用{n_splits}折时间序列交叉验证评估{len(factor_names)}个因子的组合性能")
        
        # 提取因子数据
        X = factor_df[factor_names].dropna()
        y = returns.shift(-1).loc[X.index].dropna()  # 使用下一期收益率作为目标变量
        
        # 确保X和y的索引对齐
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        # 初始化时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # 选择模型
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'ridge':
            model = Ridge(alpha=0.1)
        elif model_type == 'lasso':
            model = Lasso(alpha=0.01)
        elif model_type == 'elastic_net':
            model = ElasticNet(alpha=0.01, l1_ratio=0.5)
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 初始化结果列表
        train_mse_list = []
        test_mse_list = []
        train_r2_list = []
        test_r2_list = []
        train_direction_accuracy_list = []
        test_direction_accuracy_list = []
        
        # 进行交叉验证
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 评估模型
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # 计算指标
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            train_direction_accuracy = np.mean((y_train > 0) == (y_pred_train > 0))
            test_direction_accuracy = np.mean((y_test > 0) == (y_pred_test > 0))
            
            # 添加到结果列表
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
        
        # 记录评估结果
        logger.info(f"交叉验证平均训练集MSE: {avg_train_mse:.6f} ± {std_train_mse:.6f}")
        logger.info(f"交叉验证平均测试集MSE: {avg_test_mse:.6f} ± {std_test_mse:.6f}")
        logger.info(f"交叉验证平均训练集R²: {avg_train_r2:.6f} ± {std_train_r2:.6f}")
        logger.info(f"交叉验证平均测试集R²: {avg_test_r2:.6f} ± {std_test_r2:.6f}")
        logger.info(f"交叉验证平均训练集方向准确率: {avg_train_direction_accuracy:.4f} ± {std_train_direction_accuracy:.4f}")
        logger.info(f"交叉验证平均测试集方向准确率: {avg_test_direction_accuracy:.4f} ± {std_test_direction_accuracy:.4f}")
        
        # 返回评估结果
        cv_results = {
            'model_type': model_type,
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
            'std_test_direction_accuracy': std_test_direction_accuracy,
            'train_mse_list': train_mse_list,
            'test_mse_list': test_mse_list,
            'train_r2_list': train_r2_list,
            'test_r2_list': test_r2_list,
            'train_direction_accuracy_list': train_direction_accuracy_list,
            'test_direction_accuracy_list': test_direction_accuracy_list
        }
        
        return cv_results
    
    def optimize_factor_weights(self, factor_df, factor_names, returns, test_size=0.3, method='equal_weight'):
        """
        优化因子权重
        
        参数:
        factor_df (pd.DataFrame): 包含因子数据的DataFrame
        factor_names (list): 因子名称列表
        returns (pd.Series): 收益率序列
        test_size (float): 测试集比例
        method (str): 权重优化方法，可选 'equal_weight', 'regression', 'ic_weight'
        
        返回:
        pd.Series: 因子权重
        """
        logger.info(f"使用{method}方法优化{len(factor_names)}个因子的权重")
        
        # 提取因子数据
        X = factor_df[factor_names].dropna()
        y = returns.shift(-1).loc[X.index].dropna()  # 使用下一期收益率作为目标变量
        
        # 确保X和y的索引对齐
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        # 划分训练集和测试集
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # 根据不同方法计算权重
        if method == 'equal_weight':
            # 等权重
            weights = pd.Series(1.0 / len(factor_names), index=factor_names)
        
        elif method == 'regression':
            # 使用线性回归系数作为权重
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # 获取系数并标准化
            weights = pd.Series(model.coef_, index=factor_names)
            weights = weights / weights.abs().sum()  # 标准化权重
        
        elif method == 'ic_weight':
            # 使用IC值作为权重
            ic_values = {}
            for factor in factor_names:
                # 计算IC
                ic, _ = stats.pearsonr(X_train[factor], y_train)
                ic_values[factor] = abs(ic)  # 使用IC的绝对值
            
            # 创建权重Series并标准化
            weights = pd.Series(ic_values)
            weights = weights / weights.sum()  # 标准化权重
        
        else:
            raise ValueError(f"不支持的权重优化方法: {method}")
        
        # 评估加权组合的性能
        weighted_factor_train = (X_train * weights).sum(axis=1)
        weighted_factor_test = (X_test * weights).sum(axis=1)
        
        # 计算加权因子与收益率的相关性
        train_corr, _ = stats.pearsonr(weighted_factor_train, y_train)
        test_corr, _ = stats.pearsonr(weighted_factor_test, y_test)
        
        # 计算方向准确率
        train_direction_accuracy = np.mean((y_train > 0) == (weighted_factor_train > 0))
        test_direction_accuracy = np.mean((y_test > 0) == (weighted_factor_test > 0))
        
        logger.info(f"训练集相关性: {train_corr:.4f}, 方向准确率: {train_direction_accuracy:.4f}")
        logger.info(f"测试集相关性: {test_corr:.4f}, 方向准确率: {test_direction_accuracy:.4f}")
        
        return weights
    
    def generate_combined_factor(self, factor_df, weights):
        """
        根据权重生成组合因子
        
        参数:
        factor_df (pd.DataFrame): 包含因子数据的DataFrame
        weights (pd.Series): 因子权重
        
        返回:
        pd.Series: 组合因子
        """
        logger.info(f"生成{len(weights)}个因子的组合因子")
        
        # 提取因子数据
        factors_data = factor_df[weights.index]
        
        # 计算加权组合
        combined_factor = (factors_data * weights).sum(axis=1)
        
        return combined_factor