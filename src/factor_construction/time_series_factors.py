import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import talib
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TimeSeriesFactors:
    """
    时间序列因子构建类
    包含趋势因子、动量因子、波动率因子、周期性因子等
    """

    def __init__(self):
        """初始化时间序列因子构建器"""
        pass

    def calculate_trend_factors(self, df, price_col='close', windows=[5, 10, 20, 60]):
        """
        计算趋势相关因子

        参数:
        df (pd.DataFrame): 输入数据
        price_col (str): 价格列名
        windows (list): 窗口大小列表

        返回:
        pd.DataFrame: 添加了趋势因子的DataFrame
        """
        logger.info("计算趋势因子")
        result = df.copy()

        # 1. 计算移动平均
        for window in windows:
            result[f'ma_{window}'] = result[price_col].rolling(window=window).mean()

            # 计算价格相对于移动平均的偏离度
            result[f'ma_deviation_{window}'] = (result[price_col] - result[f'ma_{window}']) / result[f'ma_{window}']

        # 2. 计算指数移动平均
        for window in windows:
            result[f'ema_{window}'] = result[price_col].ewm(span=window, adjust=False).mean()

            # 计算价格相对于指数移动平均的偏离度
            result[f'ema_deviation_{window}'] = (result[price_col] - result[f'ema_{window}']) / result[f'ema_{window}']

        # 3. 计算移动平均交叉
        for i, fast_window in enumerate(windows[:-1]):
            for slow_window in windows[i+1:]:
                # 快速移动平均与慢速移动平均的差值
                result[f'ma_cross_{fast_window}_{slow_window}'] = result[f'ma_{fast_window}'] - result[f'ma_{slow_window}']

                # 快速指数移动平均与慢速指数移动平均的差值
                result[f'ema_cross_{fast_window}_{slow_window}'] = result[f'ema_{fast_window}'] - result[f'ema_{slow_window}']

        # 4. 计算线性回归斜率
        for window in windows:
            # 使用talib计算线性回归斜率
            result[f'linear_reg_slope_{window}'] = talib.LINEARREG_SLOPE(result[price_col].values, timeperiod=window)

            # 使用talib计算线性回归截距
            result[f'linear_reg_intercept_{window}'] = talib.LINEARREG_INTERCEPT(result[price_col].values, timeperiod=window)

            # 使用talib计算线性回归预测值
            result[f'linear_reg_angle_{window}'] = talib.LINEARREG_ANGLE(result[price_col].values, timeperiod=window)

        # 5. 计算趋势强度指标
        for window in windows:
            # 计算价格变化方向
            price_direction = result[price_col].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

            # 计算趋势一致性（同向变化的比例）
            result[f'trend_consistency_{window}'] = price_direction.rolling(window=window).apply(
                lambda x: abs(x.sum()) / window
            )

        return result

    def calculate_momentum_factors(self, df, price_col='close', volume_col='volume', windows=[5, 10, 20, 60]):
        """
        计算动量相关因子

        参数:
        df (pd.DataFrame): 输入数据
        price_col (str): 价格列名
        volume_col (str): 成交量列名
        windows (list): 窗口大小列表

        返回:
        pd.DataFrame: 添加了动量因子的DataFrame
        """
        logger.info("计算动量因子")
        result = df.copy()

        # 1. 计算价格动量
        for window in windows:
            # 计算过去n期的收益率
            result[f'return_{window}'] = result[price_col].pct_change(periods=window)

            # 计算过去n期的对数收益率
            result[f'log_return_{window}'] = np.log(result[price_col] / result[price_col].shift(window))

        # 2. 计算相对强弱指标(RSI)
        for window in windows:
            result[f'rsi_{window}'] = talib.RSI(result[price_col].values, timeperiod=window)

        # 3. 计算随机指标(KDJ)
        for window in windows:
            if all(col in result.columns for col in ['high', 'low']):
                # 计算随机指标K值
                result[f'stoch_k_{window}'] = talib.STOCH(
                    result['high'].values, 
                    result['low'].values, 
                    result[price_col].values, 
                    fastk_period=window, 
                    slowk_period=3, 
                    slowk_matype=0, 
                    slowd_period=3, 
                    slowd_matype=0
                )[0]

                # 计算随机指标D值
                result[f'stoch_d_{window}'] = talib.STOCH(
                    result['high'].values, 
                    result['low'].values, 
                    result[price_col].values, 
                    fastk_period=window, 
                    slowk_period=3, 
                    slowk_matype=0, 
                    slowd_period=3, 
                    slowd_matype=0
                )[1]

        # 4. 计算MACD
        # MACD使用固定的参数：快速EMA=12, 慢速EMA=26, 信号线=9
        result['macd'], result['macd_signal'], result['macd_hist'] = talib.MACD(
            result[price_col].values, 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9
        )

        # 5. 计算成交量加权动量
        if volume_col in result.columns:
            for window in windows:
                # 计算成交量加权价格
                result[f'volume_weighted_price_{window}'] = (result[price_col] * result[volume_col]).rolling(window=window).sum() / result[volume_col].rolling(window=window).sum()

                # 计算成交量加权动量
                result[f'volume_weighted_momentum_{window}'] = result[price_col] / result[f'volume_weighted_price_{window}'] - 1

        # 6. 计算动量加速度
        for window in windows:
            if f'return_{window}' in result.columns:
                # 计算动量变化率
                result[f'momentum_acceleration_{window}'] = result[f'return_{window}'].diff()

        return result

    def calculate_volatility_factors(self, df, price_col='close', windows=[5, 10, 20, 60]):
        """
        计算波动率相关因子

        参数:
        df (pd.DataFrame): 输入数据
        price_col (str): 价格列名
        windows (list): 窗口大小列表

        返回:
        pd.DataFrame: 添加了波动率因子的DataFrame
        """
        logger.info("计算波动率因子")
        result = df.copy()

        # 计算收益率
        result['return'] = result[price_col].pct_change()

        # 1. 计算历史波动率
        for window in windows:
            # 计算滚动标准差
            result[f'volatility_{window}'] = result['return'].rolling(window=window).std() * np.sqrt(252)  # 年化

        # 2. 计算平均真实范围(ATR)
        if all(col in result.columns for col in ['high', 'low']):
            for window in windows:
                result[f'atr_{window}'] = talib.ATR(
                    result['high'].values, 
                    result['low'].values, 
                    result[price_col].values, 
                    timeperiod=window
                )

                # 计算相对ATR
                result[f'relative_atr_{window}'] = result[f'atr_{window}'] / result[price_col]

        # 3. 计算布林带
        for window in windows:
            # 计算布林带上轨、中轨、下轨
            result[f'bollinger_upper_{window}'] = result[price_col].rolling(window=window).mean() + 2 * result[price_col].rolling(window=window).std()
            result[f'bollinger_middle_{window}'] = result[price_col].rolling(window=window).mean()
            result[f'bollinger_lower_{window}'] = result[price_col].rolling(window=window).mean() - 2 * result[price_col].rolling(window=window).std()

            # 计算布林带宽度
            result[f'bollinger_bandwidth_{window}'] = (result[f'bollinger_upper_{window}'] - result[f'bollinger_lower_{window}']) / result[f'bollinger_middle_{window}']

            # 计算价格相对于布林带的位置
            result[f'bollinger_position_{window}'] = (result[price_col] - result[f'bollinger_lower_{window}']) / (result[f'bollinger_upper_{window}'] - result[f'bollinger_lower_{window}'])

        # 4. 计算Parkinson波动率
        if all(col in result.columns for col in ['high', 'low']):
            for window in windows:
                # Parkinson波动率基于高低价计算
                high_low_ratio = np.log(result['high'] / result['low'])
                result[f'parkinson_volatility_{window}'] = np.sqrt(
                    1 / (4 * np.log(2)) * high_low_ratio.pow(2).rolling(window=window).mean() * 252
                )

        # 5. 计算Garman-Klass波动率
        if all(col in result.columns for col in ['open', 'high', 'low', 'close']):
            for window in windows:
                # Garman-Klass波动率基于开高低收价计算
                log_hl = np.log(result['high'] / result['low']).pow(2)
                log_co = np.log(result['close'] / result['open']).pow(2)

                result[f'garman_klass_volatility_{window}'] = np.sqrt(
                    (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(window=window).mean() * 252
                )

        # 6. 计算Yang-Zhang波动率
        if all(col in result.columns for col in ['open', 'high', 'low', 'close']):
            for window in windows:
                # Yang-Zhang波动率结合了开盘价、收盘价、最高价和最低价
                log_ho = np.log(result['high'] / result['open'])
                log_lo = np.log(result['low'] / result['open'])
                log_co = np.log(result['close'] / result['open'])

                # 计算开盘波动率
                open_volatility = np.log(result['open'] / result['close'].shift(1)).pow(2).rolling(window=window).mean()

                # 计算日内波动率
                k = 0.34 / (1.34 + (window + 1) / (window - 1))
                close_volatility = log_co.pow(2).rolling(window=window).mean()
                rogers_satchell = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(window=window).mean()

                # 组合计算Yang-Zhang波动率
                result[f'yang_zhang_volatility_{window}'] = np.sqrt(
                    (open_volatility + k * close_volatility + (1 - k) * rogers_satchell) * 252
                )

        return result

    def calculate_mean_reversion_factors(self, df, price_col='close', windows=[5, 10, 20, 60]):
        """
        计算均值回归相关因子

        参数:
        df (pd.DataFrame): 输入数据
        price_col (str): 价格列名
        windows (list): 窗口大小列表

        返回:
        pd.DataFrame: 添加了均值回归因子的DataFrame
        """
        logger.info("计算均值回归因子")
        result = df.copy()

        # 1. 计算Z-Score
        for window in windows:
            # 计算滚动均值和标准差
            rolling_mean = result[price_col].rolling(window=window).mean()
            rolling_std = result[price_col].rolling(window=window).std()

            # 计算Z-Score
            result[f'zscore_{window}'] = (result[price_col] - rolling_mean) / rolling_std

        # 2. 计算均值回归强度
        for window in windows:
            if f'zscore_{window}' in result.columns:
                # 计算Z-Score的自相关系数
                result[f'mean_reversion_strength_{window}'] = result[f'zscore_{window}'].rolling(window=window).apply(
                    lambda x: x.autocorr(1) if len(x) > 1 else np.nan
                )

                # 负的自相关系数表示更强的均值回归趋势
                result[f'mean_reversion_strength_{window}'] = -result[f'mean_reversion_strength_{window}']

        # 3. 计算Hurst指数
        for window in windows:
            if window >= 10:  # Hurst指数需要足够长的时间序列
                # 定义计算Hurst指数的函数
                def hurst_exponent(ts, max_lag=20):
                    """计算Hurst指数"""
                    lags = range(2, min(max_lag, len(ts) // 4))
                    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
                    reg = np.polyfit(np.log(lags), np.log(tau), 1)
                    return reg[0]  # 斜率即为Hurst指数

                # 应用到滚动窗口
                result[f'hurst_exponent_{window}'] = result[price_col].rolling(window=window).apply(
                    lambda x: hurst_exponent(x) if len(x) > 10 else np.nan,
                    raw=True
                )

        # 4. 计算半衰期
        for window in windows:
            if window >= 10:
                # 定义计算半衰期的函数
                def calculate_half_life(ts):
                    """计算均值回归半衰期"""
                    # 计算价格偏离移动平均的程度
                    ma = ts.mean()
                    y = ts - ma
                    # 计算自回归系数
                    lag_y = y.shift(1).dropna()
                    y = y.iloc[1:]
                    if len(y) <= 1 or len(lag_y) <= 1:
                        return np.nan
                    beta = np.polyfit(lag_y, y, 1)[0]
                    # 计算半衰期
                    if beta >= 1 or beta <= 0:
                        return np.nan  # 不存在均值回归或不稳定
                    half_life = -np.log(2) / np.log(beta)
                    return half_life

                # 应用到滚动窗口
                result[f'half_life_{window}'] = result[price_col].rolling(window=window).apply(
                    lambda x: calculate_half_life(x) if len(x) > 10 else np.nan,
                    raw=False
                )

        # 5. 计算ADF检验统计量
        for window in windows:
            if window >= 20:  # ADF检验需要足够长的时间序列
                # 定义计算ADF检验的函数
                def adf_test(ts):
                    """计算ADF检验统计量"""
                    try:
                        result = adfuller(ts, regression='c', autolag='AIC')
                        return result[0]  # 返回ADF统计量
                    except:
                        return np.nan

                # 应用到滚动窗口
                result[f'adf_stat_{window}'] = result[price_col].rolling(window=window).apply(
                    lambda x: adf_test(x) if len(x) > 20 else np.nan,
                    raw=True
                )

        return result

    def calculate_seasonality_factors(self, df, price_col='close', volume_col='volume', windows=[60, 120, 252]):
        """
        计算季节性相关因子

        参数:
        df (pd.DataFrame): 输入数据
        price_col (str): 价格列名
        volume_col (str): 成交量列名
        windows (list): 窗口大小列表

        返回:
        pd.DataFrame: 添加了季节性因子的DataFrame
        """
        logger.info("计算季节性因子")
        result = df.copy()

        # 确保数据有日期索引
        if not isinstance(result.index, pd.DatetimeIndex):
            logger.warning("数据索引不是日期类型，无法计算某些季节性因子")
            has_date_index = False
        else:
            has_date_index = True

        # 1. 计算日内季节性
        if has_date_index:
            # 提取小时
            result['hour'] = result.index.hour

            # 计算每小时的平均收益率
            hourly_returns = result.groupby('hour')[price_col].pct_change().groupby(result['hour']).mean()

            # 映射回原始数据
            result['hourly_seasonality'] = result['hour'].map(hourly_returns)

        # 2. 计算日间季节性
        if has_date_index:
            # 提取星期几
            result['weekday'] = result.index.weekday

            # 计算每个星期几的平均收益率
            daily_returns = result.groupby('weekday')[price_col].pct_change().groupby(result['weekday']).mean()

            # 映射回原始数据
            result['daily_seasonality'] = result['weekday'].map(daily_returns)

        # 3. 计算月度季节性
        if has_date_index:
            # 提取月份
            result['month'] = result.index.month

            # 计算每个月的平均收益率
            monthly_returns = result.groupby('month')[price_col].pct_change().groupby(result['month']).mean()

            # 映射回原始数据
            result['monthly_seasonality'] = result['month'].map(monthly_returns)

        # 4. 使用季节性分解
        for window in windows:
            if window >= 30:  # 季节性分解需要足够长的时间序列
                try:
                    # 对价格进行季节性分解
                    decomposition = seasonal_decompose(
                        result[price_col].iloc[-window:], 
                        model='additive', 
                        period=min(30, window // 3)  # 假设周期为30天或窗口的1/3
                    )

                    # 提取季节性成分
                    seasonal = decomposition.seasonal
                    trend = decomposition.trend
                    residual = decomposition.resid

                    # 将最后一个值作为当前的季节性因子
                    result.loc[result.index[-1], f'seasonal_component_{window}'] = seasonal.iloc[-1]
                    result.loc[result.index[-1], f'trend_component_{window}'] = trend.iloc[-1]
                    result.loc[result.index[-1], f'residual_component_{window}'] = residual.iloc[-1]

                    # 计算季节性强度
                    result.loc[result.index[-1], f'seasonality_strength_{window}'] = np.std(seasonal) / np.std(residual)
                except:
                    logger.warning(f"无法计算窗口大小为{window}的季节性分解")

        # 5. 计算周期性
        for window in windows:
            if window >= 30:
                try:
                    # 计算自相关函数
                    acf_values = acf(result[price_col].iloc[-window:], nlags=window//2)

                    # 找出自相关函数的峰值
                    peaks = []
                    for i in range(1, len(acf_values)-1):
                        if acf_values[i] > acf_values[i-1] and acf_values[i] > acf_values[i+1]:
                            peaks.append((i, acf_values[i]))

                    # 按自相关值排序
                    peaks.sort(key=lambda x: x[1], reverse=True)

                    # 提取最显著的周期
                    if peaks:
                        result.loc[result.index[-1], f'dominant_cycle_{window}'] = peaks[0][0]
                        result.loc[result.index[-1], f'dominant_cycle_strength_{window}'] = peaks[0][1]
                except:
                    logger.warning(f"无法计算窗口大小为{window}的周期性")

        return result

    def calculate_correlation_factors(self, df, price_col='close', ref_cols=None, windows=[20, 60, 120]):
        """
        计算相关性因子

        参数:
        df (pd.DataFrame): 输入数据
        price_col (str): 价格列名
        ref_cols (list): 参考列名列表，用于计算相关性
        windows (list): 窗口大小列表

        返回:
        pd.DataFrame: 添加了相关性因子的DataFrame
        """
        logger.info("计算相关性因子")
        result = df.copy()

        # 如果没有指定参考列，则使用所有数值列
        if ref_cols is None:
            ref_cols = [col for col in df.columns if col != price_col and np.issubdtype(df[col].dtype, np.number)]

        # 计算收益率
        result[f'{price_col}_return'] = result[price_col].pct_change()

        # 为每个参考列计算相关性
        for ref_col in ref_cols:
            # 计算参考列的收益率
            result[f'{ref_col}_return'] = result[ref_col].pct_change()

            # 计算不同窗口的相关性
            for window in windows:
                result[f'correlation_{price_col}_{ref_col}_{window}'] = result[f'{price_col}_return'].rolling(window=window).corr(result[f'{ref_col}_return'])

        # 计算平均相关性
        for window in windows:
            corr_cols = [col for col in result.columns if col.startswith(f'correlation_{price_col}_') and col.endswith(f'_{window}')]
            if corr_cols:
                result[f'avg_correlation_{window}'] = result[corr_cols].mean(axis=1)

        return result

    def calculate_arima_factors(self, df, price_col='close', windows=[60, 120]):
        """
        计算ARIMA模型因子

        参数:
        df (pd.DataFrame): 输入数据
        price_col (str): 价格列名
        windows (list): 窗口大小列表

        返回:
        pd.DataFrame: 添加了ARIMA模型因子的DataFrame
        """
        logger.info("计算ARIMA模型因子")
        result = df.copy()

        # 计算收益率
        result['return'] = result[price_col].pct_change()

        # 对每个窗口拟合ARIMA模型
        for window in windows:
            if len(result) >= window:
                try:
                    # 使用最近的数据拟合ARIMA模型
                    model = ARIMA(result['return'].iloc[-window:].dropna(), order=(1, 0, 1))
                    model_fit = model.fit()

                    # 提取模型参数
                    params = model_fit.params
                    result.loc[result.index[-1], f'arima_ar1_{window}'] = params.get('ar.L1', np.nan)
                    result.loc[result.index[-1], f'arima_ma1_{window}'] = params.get('ma.L1', np.nan)
                    result.loc[result.index[-1], f'arima_const_{window}'] = params.get('const', np.nan)

                    # 计算预测值
                    forecast = model_fit.forecast(steps=1)
                    result.loc[result.index[-1], f'arima_forecast_{window}'] = forecast[0]

                    # 计算预测误差
                    if len(result) > 1:
                        prev_forecast = result.loc[result.index[-2], f'arima_forecast_{window}']
                        if not np.isnan(prev_forecast):
                            actual = result.loc[result.index[-1], 'return']
                            result.loc[result.index[-1], f'arima_error_{window}'] = actual - prev_forecast
                except:
                    logger.warning(f"无法计算窗口大小为{window}的ARIMA模型")

        return result

    def calculate_nonlinear_factors(self, df, price_col='close', windows=[20, 60, 120]):
        """
        计算非线性因子

        参数:
        df (pd.DataFrame): 输入数据
        price_col (str): 价格列名
        windows (list): 窗口大小列表

        返回:
        pd.DataFrame: 添加了非线性因子的DataFrame
        """
        logger.info("计算非线性因子")
        result = df.copy()

        # 计算收益率
        result['return'] = result[price_col].pct_change()

        # 1. 计算收益率的偏度
        for window in windows:
            result[f'skewness_{window}'] = result['return'].rolling(window=window).skew()

        # 2. 计算收益率的峰度
        for window in windows:
            result[f'kurtosis_{window}'] = result['return'].rolling(window=window).kurt()

        # 3. 计算收益率的极值比率
        for window in windows:
            def extreme_value_ratio(x):
                if len(x) <= 1:
                    return np.nan
                pos_returns = x[x > 0]
                neg_returns = x[x < 0]
                if len(pos_returns) == 0 or len(neg_returns) == 0:
                    return np.nan
                return np.max(pos_returns) / abs(np.min(neg_returns))

            result[f'extreme_value_ratio_{window}'] = result['return'].rolling(window=window).apply(
                extreme_value_ratio, raw=True
            )

        # 4. 计算收益率的正负比例
        for window in windows:
            def pos_neg_ratio(x):
                if len(x) <= 1:
                    return np.nan
                pos_count = np.sum(x > 0)
                neg_count = np.sum(x < 0)
                if neg_count == 0:
                    return np.inf
                return pos_count / neg_count

            result[f'pos_neg_ratio_{window}'] = result['return'].rolling(window=window).apply(
                pos_neg_ratio, raw=True
            )

        # 5. 计算收益率的熵
        for window in windows:
            def entropy(x, bins=10):
                if len(x) <= bins:
                    return np.nan
                hist, _ = np.histogram(x, bins=bins)
                hist = hist / np.sum(hist)
                return -np.sum(hist * np.log(hist + 1e-10))

            result[f'entropy_{window}'] = result['return'].rolling(window=window).apply(
                entropy, raw=True
            )

        # 6. 计算收益率的Lyapunov指数（混沌度量）
        for window in windows:
            if window >= 30:  # Lyapunov指数需要足够长的时间序列
                def lyapunov_exponent(x, lag=1, iterations=50):
                    if len(x) <= lag + iterations:
                        return np.nan

                    # 构建时间延迟向量
                    vectors = np.array([x[i:i+iterations] for i in range(len(x)-iterations)])

                    # 计算相邻向量的距离变化率
                    distances = []
                    for i in range(len(vectors)-lag):
                        d0 = np.linalg.norm(vectors[i+lag] - vectors[i])
                        d1 = np.linalg.norm(vectors[i+lag+1] - vectors[i+1])
                        if d0 > 0:
                            distances.append(np.log(d1/d0))

                    if not distances:
                        return np.nan

                    # Lyapunov指数是距离变化率的平均值
                    return np.mean(distances)

                try:
                    result[f'lyapunov_exponent_{window}'] = result['return'].rolling(window=window).apply(
                        lyapunov_exponent, raw=True
                    )
                except:
                    logger.warning(f"无法计算窗口大小为{window}的Lyapunov指数")

        return result

    def calculate_pattern_factors(self, df, price_col='close'):
        """
        计算价格模式因子

        参数:
        df (pd.DataFrame): 输入数据
        price_col (str): 价格列名

        返回:
        pd.DataFrame: 添加了价格模式因子的DataFrame
        """
        logger.info("计算价格模式因子")
        result = df.copy()

        # 确保有开高低收价格
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in result.columns for col in required_cols):
            logger.warning("缺少计算价格模式所需的列（开高低收价格）")
            return result

        # 1. 计算蜡烛图模式
        # 十字星
        result['doji'] = talib.CDLDOJI(
            result['open'].values, 
            result['high'].values, 
            result['low'].values, 
            result['close'].values
        )

        # 锤子线
        result['hammer'] = talib.CDLHAMMER(
            result['open'].values, 
            result['high'].values, 
            result['low'].values, 
            result['close'].values
        )

        # 上吊线
        result['hanging_man'] = talib.CDLHANGINGMAN(
            result['open'].values, 
            result['high'].values, 
            result['low'].values, 
            result['close'].values
        )

        # 吞噬形态
        result['engulfing'] = talib.CDLENGULFING(
            result['open'].values, 
            result['high'].values, 
            result['low'].values, 
            result['close'].values
        )

        # 黄昏之星
        result['evening_star'] = talib.CDLEVENINGSTAR(
            result['open'].values, 
            result['high'].values, 
            result['low'].values, 
            result['close'].values
        )

        # 启明之星
        result['morning_star'] = talib.CDLMORNINGSTAR(
            result['open'].values, 
            result['high'].values, 
            result['low'].values, 
            result['close'].values
        )

        # 三只乌鸦
        result['three_black_crows'] = talib.CDL3BLACKCROWS(
            result['open'].values, 
            result['high'].values, 
            result['low'].values, 
            result['close'].values
        )

        # 三兵推进
        result['three_advancing_soldiers'] = talib.CDL3WHITESOLDIERS(
            result['open'].values, 
            result['high'].values, 
            result['low'].values, 
            result['close'].values
        )

        # 2. 计算其他技术指标模式
        # 计算MACD金叉和死叉
        if all(col in result.columns for col in ['macd', 'macd_signal']):
            result['macd_cross_over'] = ((result['macd'] > result['macd_signal']) & 
                                        (result['macd'].shift(1) <= result['macd_signal'].shift(1))).astype(int)
            result['macd_cross_under'] = ((result['macd'] < result['macd_signal']) & 
                                         (result['macd'].shift(1) >= result['macd_signal'].shift(1))).astype(int)

        # 计算RSI超买超卖
        for window in [14]:  # 使用标准的14天RSI
            if f'rsi_{window}' in result.columns:
                result[f'rsi_overbought_{window}'] = (result[f'rsi_{window}'] > 70).astype(int)
                result[f'rsi_oversold_{window}'] = (result[f'rsi_{window}'] < 30).astype(int)

        # 计算布林带突破
        for window in [20]:  # 使用标准的20天布林带
            if all(col in result.columns for col in [f'bollinger_upper_{window}', f'bollinger_lower_{window}']):
                result[f'bollinger_breakout_up_{window}'] = (result[price_col] > result[f'bollinger_upper_{window}']).astype(int)
                result[f'bollinger_breakout_down_{window}'] = (result[price_col] < result[f'bollinger_lower_{window}']).astype(int)

        return result

    def calculate_volume_factors(self, df, price_col='close', volume_col='volume', windows=[5, 10, 20, 60]):
        """
        计算成交量相关因子

        参数:
        df (pd.DataFrame): 输入数据
        price_col (str): 价格列名
        volume_col (str): 成交量列名
        windows (list): 窗口大小列表

        返回:
        pd.DataFrame: 添加了成交量因子的DataFrame
        """
        logger.info("计算成交量因子")
        result = df.copy()

        if volume_col not in result.columns:
            logger.warning(f"数据中缺少成交量列 {volume_col}")
            return result

        # 1. 计算成交量变化率
        result['volume_change'] = result[volume_col].pct_change()

        # 2. 计算相对成交量（相对于移动平均）
        for window in windows:
            result[f'relative_volume_{window}'] = result[volume_col] / result[volume_col].rolling(window=window).mean()

        # 3. 计算成交量趋势
        for window in windows:
            result[f'volume_trend_{window}'] = talib.LINEARREG_SLOPE(result[volume_col].values, timeperiod=window)

        # 4. 计算价量相关性
        for window in windows:
            result[f'price_volume_corr_{window}'] = result[price_col].pct_change().rolling(window=window).corr(result[volume_col].pct_change())

        # 5. 计算成交量振荡器
        for window in windows:
            # 计算成交量移动平均
            result[f'volume_ma_{window}'] = result[volume_col].rolling(window=window).mean()

            # 计算成交量振荡器（当前成交量与移动平均的差）
            result[f'volume_oscillator_{window}'] = (result[volume_col] - result[f'volume_ma_{window}']) / result[f'volume_ma_{window}']

        # 6. 计算累积/分布线 (A/D Line)
        if all(col in result.columns for col in ['high', 'low']):
            # 计算货币流量乘数
            mfm = ((result['close'] - result['low']) - (result['high'] - result['close'])) / (result['high'] - result['low'])
            mfm = mfm.replace([np.inf, -np.inf], 0)

            # 计算货币流量量
            mfv = mfm * result[volume_col]

            # 计算累积/分布线
            result['ad_line'] = mfv.cumsum()

            # 计算Chaikin振荡器
            result['chaikin_oscillator'] = result['ad_line'].ewm(span=3).mean() - result['ad_line'].ewm(span=10).mean()

        # 7. 计算能量潮指标 (OBV)
        result['obv'] = np.nan

        # 初始值
        if len(result) > 0:
            result.loc[result.index[0], 'obv'] = result.loc[result.index[0], volume_col]

        # 计算OBV
        for i in range(1, len(result)):
            if result.loc[result.index[i], price_col] > result.loc[result.index[i-1], price_col]:
                result.loc[result.index[i], 'obv'] = result.loc[result.index[i-1], 'obv'] + result.loc[result.index[i], volume_col]
            elif result.loc[result.index[i], price_col] < result.loc[result.index[i-1], price_col]:
                result.loc[result.index[i], 'obv'] = result.loc[result.index[i-1], 'obv'] - result.loc[result.index[i], volume_col]
            else:
                result.loc[result.index[i], 'obv'] = result.loc[result.index[i-1], 'obv']

        # 计算OBV的移动平均和斜率
        for window in windows:
            result[f'obv_ma_{window}'] = result['obv'].rolling(window=window).mean()
            result[f'obv_slope_{window}'] = talib.LINEARREG_SLOPE(result['obv'].values, timeperiod=window)

        return result

    def calculate_all_factors(self, df, price_col='close', volume_col='volume', windows=[5, 10, 20, 60]):
        """
        计算所有时间序列因子

        参数:
        df (pd.DataFrame): 输入数据
        price_col (str): 价格列名
        volume_col (str): 成交量列名
        windows (list): 窗口大小列表

        返回:
        pd.DataFrame: 添加了所有时间序列因子的DataFrame
        """
        logger.info("计算所有时间序列因子")

        # 依次计算各类因子
        result = df.copy()
        result = self.calculate_trend_factors(result, price_col, windows)
        result = self.calculate_momentum_factors(result, price_col, volume_col, windows)
        result = self.calculate_volatility_factors(result, price_col, windows)
        result = self.calculate_mean_reversion_factors(result, price_col, windows)

        # 季节性因子需要更长的窗口
        long_windows = [w for w in windows if w >= 60] or [60, 120, 252]
        result = self.calculate_seasonality_factors(result, price_col, volume_col, long_windows)

        # 相关性因子
        result = self.calculate_correlation_factors(result, price_col, None, windows)

        # ARIMA模型因子需要更长的窗口
        arima_windows = [w for w in windows if w >= 60] or [60, 120]
        result = self.calculate_arima_factors(result, price_col, arima_windows)

        # 非线性因子
        result = self.calculate_nonlinear_factors(result, price_col, windows)

        # 价格模式因子
        result = self.calculate_pattern_factors(result, price_col)

        # 成交量因子
        result = self.calculate_volume_factors(result, price_col, volume_col, windows)

        return result

    def calculate_nonlinear_factors(self, df, price_col='close', windows=[20, 60, 120]):
        """
        计算非线性因子

        参数:
        df (pd.DataFrame): 输入数据
        price_col (str): 价格列名
        windows (list): 窗口大小列表

        返回:
        pd.DataFrame: 添加了非线性因子的DataFrame
        """
        logger.info("计算非线性因子")
        result = df.copy()

        # 计算收益率
        result['return'] = result[price_col].pct_change()

        # 1. 计算收益率的偏度
        for window in windows:
            result[f'skewness_{window}'] = result['return'].rolling(window=window).skew()

        # 2. 计算收益率的峰度
        for window in windows:
            result[f'kurtosis_{window}'] = result['return'].rolling(window=window).kurt()

        # 3. 计算收益率的极值比率
        for window in windows:
            def extreme_value_ratio(x):
                if len(x) <= 1:
                    return np.nan
                pos_returns = x[x > 0]
                neg_returns = x[x < 0]
                if len(pos_returns) == 0 or len(neg_returns) == 0:
                    return np.nan
                return np.max(pos_returns) / abs(np.min(neg_returns))

            result[f'extreme_value_ratio_{window}'] = result['return'].rolling(window=window).apply(
                extreme_value_ratio, raw=True
            )

        # 4. 计算收益率的正负比例
        for window in windows:
            def pos_neg_ratio(x):
                if len(x) <= 1:
                    return np.nan
                pos_count = np.sum(x > 0)
                neg_count = np.sum(x < 0)
                if neg_count == 0:
                    return np.inf
                return pos_count / neg_count

            result[f'pos_neg_ratio_{window}'] = result['return'].rolling(window=window).apply(
                pos_neg_ratio, raw=True
            )

        # 5. 计算收益率的熵
        for window in windows:
            def entropy(x, bins=10):
                if len(x) <= bins:
                    return np.nan
                hist, _ = np.histogram(x, bins=bins)
                hist = hist / np.sum(hist)
                return -np.sum(hist * np.log(hist + 1e-10))

            result[f'entropy_{window}'] = result['return'].rolling(window=window).apply(
                entropy, raw=True
            )

        # 6. 计算收益率的Lyapunov指数（混沌度量）
        for window in windows:
            if window >= 30:  # Lyapunov指数需要足够长的时间序列
                def lyapunov_exponent(x, lag=1, iterations=50):
                    if len(x) <= lag + iterations:
                        return np.nan

                    # 构建时间延迟向量
                    vectors = np.array([x[i:i+iterations] for i in range(len(x)-iterations)])

                    # 计算相邻向量的距离变化率
                    distances = []
                    for i in range(len(vectors)-lag):
                        d0 = np.linalg.norm(vectors[i+lag] - vectors[i])
                        d1 = np.linalg.norm(vectors[i+lag+1] - vectors[i+1])
                        if d0 > 0:
                            distances.append(np.log(d1/d0))

                    if not distances:
                        return np.nan

                    # Lyapunov指数是距离变化率的平均值
                    return np.mean(distances)

                try:
                    result[f'lyapunov_exponent_{window}'] = result['return'].rolling(window=window).apply(
                        lyapunov_exponent, raw=True
                    )
                except:
                    logger.warning(f"无法计算窗口大小为{window}的Lyapunov指数")

        return result

    def calculate_pattern_factors(self, df, price_col='close'):
        """
        计算价格模式因子

        参数:
        df (pd.DataFrame): 输入数据
        price_col (str): 价格列名

        返回:
        pd.DataFrame: 添加了价格模式因子的DataFrame
        """
        logger.info("计算价格模式因子")
        result = df.copy()

        # 确保有开高低收价格
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in result.columns for col in required_cols):
            logger.warning("缺少计算价格模式所需的列（开高低收价格）")
            return result

        # 1. 计算蜡烛图模式
        # 十字星
        result['doji'] = talib.CDLDOJI(
            result['open'].values, 
            result['high'].values, 
            result['low'].values, 
            result['close'].values
        )

        # 锤子线
        result['hammer'] = talib.CDLHAMMER(
            result['open'].values, 
            result['high'].values, 
            result['low'].values, 
            result['close'].values
        )

        # 上吊线
        result['hanging_man'] = talib.CDLHANGINGMAN(
            result['open'].values, 
            result['high'].values, 
            result['low'].values, 
            result['close'].values
        )

        # 吞噬形态
        result['engulfing'] = talib.CDLENGULFING(
            result['open'].values, 
            result['high'].values, 
            result['low'].values, 
            result['close'].values
        )

        # 黄昏之星
        result['evening_star'] = talib.CDLEVENINGSTAR(
            result['open'].values, 
            result['high'].values, 
            result['low'].values, 
            result['close'].values
        )

        # 启明之星
        result['morning_star'] = talib.CDLMORNINGSTAR(
            result['open'].values, 
            result['high'].values, 
            result['low'].values, 
            result['close'].values
        )

        # 三只乌鸦
        result['three_black_crows'] = talib.CDL3BLACKCROWS(
            result['open'].values, 
            result['high'].values, 
            result['low'].values, 
            result['close'].values
        )

        # 三兵推进
        result['three_advancing_soldiers'] = talib.CDL3WHITESOLDIERS(
            result['open'].values, 
            result['high'].values, 
            result['low'].values, 
            result['close'].values
        )

        # 2. 计算其他技术指标模式
        # 计算MACD金叉和死叉
        if all(col in result.columns for col in ['macd', 'macd_signal']):
            result['macd_cross_over'] = ((result['macd'] > result['macd_signal']) & 
                                        (result['macd'].shift(1) <= result['macd_signal'].shift(1))).astype(int)
            result['macd_cross_under'] = ((result['macd'] < result['macd_signal']) & 
                                         (result['macd'].shift(1) >= result['macd_signal'].shift(1))).astype(int)

        # 计算RSI超买超卖
        for window in [14]:  # 使用标准的14天RSI
            if f'rsi_{window}' in result.columns:
                result[f'rsi_overbought_{window}'] = (result[f'rsi_{window}'] > 70).astype(int)
                result[f'rsi_oversold_{window}'] = (result[f'rsi_{window}'] < 30).astype(int)

        # 计算布林带突破
        for window in [20]:  # 使用标准的20天布林带
            if all(col in result.columns for col in [f'bollinger_upper_{window}', f'bollinger_lower_{window}']):
                result[f'bollinger_breakout_up_{window}'] = (result[price_col] > result[f'bollinger_upper_{window}']).astype(int)
                result[f'bollinger_breakout_down_{window}'] = (result[price_col] < result[f'bollinger_lower_{window}']).astype(int)

        return result

    def calculate_volume_factors(self, df, price_col='close', volume_col='volume', windows=[5, 10, 20, 60]):
        """
        计算成交量相关因子

        参数:
        df (pd.DataFrame): 输入数据
        price_col (str): 价格列名
        volume_col (str): 成交量列名
        windows (list): 窗口大小列表

        返回:
        pd.DataFrame: 添加了成交量因子的DataFrame
        """
        logger.info("计算成交量因子")
        result = df.copy()

        if volume_col not in result.columns:
            logger.warning(f"数据中缺少成交量列 {volume_col}")
            return result

        # 1. 计算成交量变化率
        result['volume_change'] = result[volume_col].pct_change()

        # 2. 计算相对成交量（相对于移动平均）
        for window in windows:
            result[f'relative_volume_{window}'] = result[volume_col] / result[volume_col].rolling(window=window).mean()

        # 3. 计算成交量趋势
        for window in windows:
            result[f'volume_trend_{window}'] = talib.LINEARREG_SLOPE(result[volume_col].values, timeperiod=window)

        # 4. 计算价量相关性
        for window in windows:
            result[f'price_volume_corr_{window}'] = result[price_col].pct_change().rolling(window=window).corr(result[volume_col].pct_change())

        # 5. 计算成交量振荡器
        for window in windows:
            # 计算成交量移动平均
            result[f'volume_ma_{window}'] = result[volume_col].rolling(window=window).mean()

            # 计算成交量振荡器（当前成交量与移动平均的差）
            result[f'volume_oscillator_{window}'] = (result[volume_col] - result[f'volume_ma_{window}']) / result[f'volume_ma_{window}']

        # 6. 计算累积/分布线 (A/D Line)
        if all(col in result.columns for col in ['high', 'low']):
            # 计算货币流量乘数
            mfm = ((result['close'] - result['low']) - (result['high'] - result['close'])) / (result['high'] - result['low'])
            mfm = mfm.replace([np.inf, -np.inf], 0)

            # 计算货币流量量
            mfv = mfm * result[volume_col]

            # 计算累积/分布线
            result['ad_line'] = mfv.cumsum()

            # 计算Chaikin振荡器
            result['chaikin_oscillator'] = result['ad_line'].ewm(span=3).mean() - result['ad_line'].ewm(span=10).mean()

        # 7. 计算能量潮指标 (OBV)
        result['obv'] = np.nan

        # 初始值
        if len(result) > 0:
            result.loc[result.index[0], 'obv'] = result.loc[result.index[0], volume_col]

        # 计算OBV
        for i in range(1, len(result)):
            if result.loc[result.index[i], price_col] > result.loc[result.index[i-1], price_col]:
                result.loc[result.index[i], 'obv'] = result.loc[result.index[i-1], 'obv'] + result.loc[result.index[i], volume_col]
            elif result.loc[result.index[i], price_col] < result.loc[result.index[i-1], price_col]:
                result.loc[result.index[i], 'obv'] = result.loc[result.index[i-1], 'obv'] - result.loc[result.index[i], volume_col]
            else:
                result.loc[result.index[i], 'obv'] = result.loc[result.index[i-1], 'obv']

        # 计算OBV的移动平均和斜率
        for window in windows:
            result[f'obv_ma_{window}'] = result['obv'].rolling(window=window).mean()
            result[f'obv_slope_{window}'] = talib.LINEARREG_SLOPE(result['obv'].values, timeperiod=window)

        return result

    def calculate_all_factors(self, df, price_col='close', volume_col='volume', windows=[5, 10, 20, 60]):
        """
        计算所有时间序列因子

        参数:
        df (pd.DataFrame): 输入数据
        price_col (str): 价格列名
        volume_col (str): 成交量列名
        windows (list): 窗口大小列表

        返回:
        pd.DataFrame: 添加了所有时间序列因子的DataFrame
        """
        logger.info("计算所有时间序列因子")

        # 依次计算各类因子
        result = df.copy()
        result = self.calculate_trend_factors(result, price_col, windows)
        result = self.calculate_momentum_factors(result, price_col, volume_col, windows)
        result = self.calculate_volatility_factors(result, price_col, windows)
        result = self.calculate_mean_reversion_factors(result, price_col, windows)

        # 季节性因子需要更长的窗口
        long_windows = [w for w in windows if w >= 60] or [60, 120, 252]
        result = self.calculate_seasonality_factors(result, price_col, volume_col, long_windows)

        # 相关性因子
        result = self.calculate_correlation_factors(result, price_col, None, windows)

        # ARIMA模型因子需要更长的窗口
        arima_windows = [w for w in windows if w >= 60] or [60, 120]
        result = self.calculate_arima_factors(result, price_col, arima_windows)

        # 非线性因子
        result = self.calculate_nonlinear_factors(result, price_col, windows)

        # 价格模式因子
        result = self.calculate_pattern_factors(result, price_col)

        # 成交量因子
        result = self.calculate_volume_factors(result, price_col, volume_col, windows)

        return result

    def calculate_all_factors(self, df, price_col='close', volume_col='volume', windows=[5, 10, 20, 60]):
        """
        计算所有时间序列因子

        参数:
        df (pd.DataFrame): 输入数据
        price_col (str): 价格列名
        volume_col (str): 成交量列名
        windows (list): 窗口大小列表

        返回:
        pd.DataFrame: 添加了所有时间序列因子的DataFrame
        """
        logger.info("计算所有时间序列因子")

        # 依次计算各类因子
        result = df.copy()
        result = self.calculate_trend_factors(result, price_col, windows)
        result = self.calculate_momentum_factors(result, price_col, volume_col, windows)
        result = self.calculate_volatility_factors(result, price_col, windows)
        result = self.calculate_mean_reversion_factors(result, price_col, windows)

        # 季节性因子需要更长的窗口
        long_windows = [w for w in windows if w >= 60] or [60, 120, 252]
        result = self.calculate_seasonality_factors(result, price_col, volume_col, long_windows)

        # 相关性因子
        result = self.calculate_correlation_factors(result, price_col, None, windows)

        # ARIMA模型因子需要更长的窗口
        arima_windows = [w for w in windows if w >= 60] or [60, 120]
        result = self.calculate_arima_factors(result, price_col, arima_windows)

        # 非线性因子
        result = self.calculate_nonlinear_factors(result, price_col, windows)

        # 价格模式因子
        result = self.calculate_pattern_factors(result, price_col)

        # 成交量因子
        result = self.calculate_volume_factors(result, price_col, volume_col, windows)

        return result

    def filter_factors(self, df, min_periods=10, correlation_threshold=0.7, variance_threshold=0.01):
        """
        过滤因子，去除缺失值过多、方差过小或高度相关的因子

        参数:
        df (pd.DataFrame): 包含因子的DataFrame
        min_periods (int): 最小非缺失值数量
        correlation_threshold (float): 相关性阈值，高于此值的因子将被视为高度相关
        variance_threshold (float): 方差阈值，低于此值的因子将被视为低信息量

        返回:
        pd.DataFrame: 过滤后的因子DataFrame
        list: 保留的因子列表
        """
        logger.info("过滤因子")
        result = df.copy()

        # 1. 去除缺失值过多的因子
        non_missing_counts = result.count()
        valid_columns = non_missing_counts[non_missing_counts >= min_periods].index.tolist()
        result = result[valid_columns]

        # 2. 去除方差过小的因子
        numeric_cols = [col for col in result.columns if np.issubdtype(result[col].dtype, np.number)]
        variances = result[numeric_cols].var()
        high_variance_cols = variances[variances >= variance_threshold].index.tolist()
        result = result[high_variance_cols]

        # 3. 去除高度相关的因子
        # 计算相关性矩阵
        corr_matrix = result.corr().abs()

        # 创建上三角矩阵
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # 找出高度相关的列
        to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]

        # 保留的列
        to_keep = [col for col in result.columns if col not in to_drop]

        logger.info(f"原始因子数量: {len(df.columns)}")
        logger.info(f"过滤后因子数量: {len(to_keep)}")

        return result[to_keep], to_keep

    def normalize_factors(self, df, method='zscore', winsorize_limits=(0.05, 0.05)):
        """
        标准化因子

        参数:
        df (pd.DataFrame): 包含因子的DataFrame
        method (str): 标准化方法，可选 'zscore', 'minmax', 'rank'
        winsorize_limits (tuple): winsorize截断上下限

        返回:
        pd.DataFrame: 标准化后的因子DataFrame
        """
        logger.info(f"使用{method}方法标准化因子")
        result = df.copy()

        # 获取数值型列
        numeric_cols = [col for col in result.columns if np.issubdtype(result[col].dtype, np.number)]

        for col in numeric_cols:
            # 跳过全是NaN的列
            if result[col].isna().all():
                continue

            # Winsorize处理极端值
            series = result[col].copy()
            if not series.isna().all():
                lower_limit = series.quantile(winsorize_limits[0])
                upper_limit = series.quantile(1 - winsorize_limits[1])
                series = series.clip(lower=lower_limit, upper=upper_limit)

            # 根据选择的方法进行标准化
            if method == 'zscore':
                # Z-score标准化
                mean = series.mean()
                std = series.std()
                if std > 0:
                    result[col] = (series - mean) / std
            elif method == 'minmax':
                # Min-Max标准化
                min_val = series.min()
                max_val = series.max()
                if max_val > min_val:
                    result[col] = (series - min_val) / (max_val - min_val)
            elif method == 'rank':
                # 排序标准化
                result[col] = series.rank(pct=True)

        return result

    def calculate_factor_exposures(self, df, factor_cols, target_col, windows=[20, 60]):
        """
        计算因子暴露度（因子与目标变量的关系）

        参数:
        df (pd.DataFrame): 包含因子和目标变量的DataFrame
        factor_cols (list): 因子列名列表
        target_col (str): 目标变量列名
        windows (list): 窗口大小列表

        返回:
        pd.DataFrame: 包含因子暴露度的DataFrame
        """
        logger.info("计算因子暴露度")
        result = pd.DataFrame(index=df.index)

        # 计算目标变量的未来收益率
        result['future_return'] = df[target_col].pct_change(1).shift(-1)

        # 对每个因子计算滚动相关性和回归系数
        for factor in factor_cols:
            for window in windows:
                if window < len(df):
                    # 计算因子与未来收益率的滚动相关性
                    result[f'{factor}_corr_{window}'] = df[factor].rolling(window=window).corr(result['future_return'])

                    # 计算因子对未来收益率的滚动回归系数
                    def rolling_regression(x):
                        if len(x) <= 1:
                            return np.nan
                        y = result['future_return'].iloc[x.index]
                        if y.isna().any() or x.isna().any():
                            return np.nan
                        try:
                            return np.polyfit(x, y, 1)[0]
                        except:
                            return np.nan

                    result[f'{factor}_beta_{window}'] = df[factor].rolling(window=window).apply(
                        rolling_regression, raw=False
                    )

        return result