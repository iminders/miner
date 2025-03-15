import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def load_stock_data(filepath, date_format='%Y-%m-%d'):
    """
    加载股票数据

    参数:
    filepath (str): 数据文件路径
    date_format (str): 日期格式

    返回:
    pd.DataFrame: 股票数据
    """
    logger.info(f"从{filepath}加载股票数据")

    # 检查文件是否存在
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到文件: {filepath}")

    # 根据文件扩展名确定加载方法
    file_ext = os.path.splitext(filepath)[1].lower()

    if file_ext == '.csv':
        # 加载CSV文件
        df = pd.read_csv(filepath)
    elif file_ext in ['.xlsx', '.xls']:
        # 加载Excel文件
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}")

    # 检查必要的列是否存在
    required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"数据缺少必要的列: {missing_columns}")

    # 将日期列转换为日期时间格式
    df['date'] = pd.to_datetime(df['date'], format=date_format)

    # 设置日期为索引
    df.set_index('date', inplace=True)

    # 按日期排序
    df.sort_index(inplace=True)

    logger.info(f"加载了{len(df)}行数据，日期范围: {df.index.min().strftime('%Y-%m-%d')} 至 {df.index.max().strftime('%Y-%m-%d')}")

    return df

def load_multiple_stocks_data(directory, pattern='*.csv', date_format='%Y-%m-%d'):
    """
    加载多只股票的数据

    参数:
    directory (str): 数据目录
    pattern (str): 文件匹配模式
    date_format (str): 日期格式

    返回:
    dict: 股票代码到数据的映射
    """
    import glob

    logger.info(f"从{directory}加载多只股票数据")

    # 检查目录是否存在
    if not os.path.exists(directory):
        raise FileNotFoundError(f"找不到目录: {directory}")

    # 获取匹配的文件列表
    file_pattern = os.path.join(directory, pattern)
    files = glob.glob(file_pattern)

    if not files:
        raise FileNotFoundError(f"在{directory}中没有找到匹配{pattern}的文件")

    # 加载每个文件
    stocks_data = {}
    for file in files:
        # 从文件名中提取股票代码
        stock_code = os.path.splitext(os.path.basename(file))[0]

        try:
            # 加载数据
            df = load_stock_data(file, date_format)
            stocks_data[stock_code] = df
        except Exception as e:
            logger.error(f"加载{file}时出错: {str(e)}")

    logger.info(f"加载了{len(stocks_data)}只股票的数据")

    return stocks_data

def generate_synthetic_data(n_samples=1000, start_date='2010-01-01', freq='D', seed=42):
    """
    生成合成股票数据用于测试

    参数:
    n_samples (int): 样本数量
    start_date (str): 起始日期
    freq (str): 频率，'D'表示日频，'M'表示月频
    seed (int): 随机种子

    返回:
    pd.DataFrame: 合成股票数据
    """
    logger.info(f"生成{n_samples}个样本的合成股票数据")

    # 设置随机种子
    np.random.seed(seed)

    # 生成日期索引
    dates = pd.date_range(start=start_date, periods=n_samples, freq=freq)

    # 生成价格数据
    initial_price = 100.0
    returns = np.random.normal(0.0005, 0.015, n_samples)
    prices = initial_price * (1 + returns).cumprod()

    # 生成OHLC数据
    daily_volatility = 0.01
    high = prices * (1 + np.random.uniform(0, daily_volatility, n_samples))
    low = prices * (1 - np.random.uniform(0, daily_volatility, n_samples))
    open_prices = prices * (1 + np.random.normal(0, daily_volatility/2, n_samples))

    # 确保high > low
    for i in range(n_samples):
        if high[i] < low[i]:
            high[i], low[i] = low[i], high[i]

    # 生成成交量数据
    volume = np.random.lognormal(10, 1, n_samples) * 1000

    # 创建DataFrame
    df = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume
    }, index=dates)

    logger.info(f"生成了合成数据，日期范围: {df.index.min().strftime('%Y-%m-%d')} 至 {df.index.max().strftime('%Y-%m-%d')}")

    return df