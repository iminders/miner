# -*- coding: utf-8 -*-
"""
监控系统示例
"""

import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.monitoring.monitoring_system import MonitoringSystem
from src.monitoring.factor_monitor import FactorMonitor
from src.monitoring.strategy_monitor import StrategyMonitor
from src.monitoring.anomaly_detector import AnomalyDetector
from src.monitoring.alert_handlers import ConsoleAlertHandler, FileAlertHandler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_sample_data():
    """
    生成示例数据

    返回:
    tuple: (factors, returns)
    """
    # 生成日期范围
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

    # 生成股票代码
    symbols = ['000001.SZ', '600000.SH', '300059.SZ']

    # 生成因子数据
    factors = {}
    for factor_name in ['momentum', 'value', 'size']:
        factor_data = {}
        for symbol in symbols:
            # 生成随机因子值
            factor_values = np.random.randn(len(dates))

            # 添加一些趋势和噪声
            if factor_name == 'momentum':
                # 动量因子随时间衰减
                trend = np.linspace(0.5, -0.2, len(dates))
                factor_values += trend
            elif factor_name == 'value':
                # 价值因子相对稳定
                factor_values += np.sin(np.linspace(0, 4*np.pi, len(dates))) * 0.2

            # 创建DataFrame
            df = pd.DataFrame({
                'date': dates,
                'symbol': symbol,
                'value': factor_values
            })

            factor_data[symbol] = df

        # 合并所有股票的数据
        factor_df = pd.concat(factor_data.values())
        factors[factor_name] = factor_df

    # 生成收益率数据
    returns = {}
    for symbol in symbols:
        # 生成随机收益率
        return_values = np.random.randn(len(dates)) * 0.01  # 1%的日波动率

        # 添加一些趋势
        if symbol == '000001.SZ':
            # 上升趋势
            trend = np.linspace(0, 0.1, len(dates)) / len(dates)
            return_values += trend
        elif symbol == '600000.SH':
            # 下降趋势
            trend = np.linspace(0.1, 0, len(dates)) / len(dates)
            return_values += trend

        # 创建DataFrame
        df = pd.DataFrame({
            'date': dates,
            'return': return_values
        })
        df.set_index('date', inplace=True)

        returns[symbol] = df

    return factors, returns


def generate_strategy_returns():
    """
    生成策略收益率

    返回:
    tuple: (strategy_returns, benchmark_returns)
    """
    # 生成日期范围
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

    # 生成策略收益率
    strategy_returns = np.random.randn(len(dates)) * 0.01  # 1%的日波动率

    # 添加上升趋势
    trend = np.linspace(0, 0.2, len(dates)) / len(dates)
    strategy_returns += trend

    # 添加一些回撤
    strategy_returns[100:150] -= 0.02  # 第100-150天有一次回撤

    # 创建Series
    strategy_returns = pd.Series(strategy_returns, index=dates)

    # 生成基准收益率
    benchmark_returns = np.random.randn(len(dates)) * 0.008  # 0.8%的日波动率

    # 添加较小的上升趋势
    trend = np.linspace(0, 0.1, len(dates)) / len(dates)
    benchmark_returns += trend

    # 创建Series
    benchmark_returns = pd.Series(benchmark_returns, index=dates)

    return strategy_returns, benchmark_returns


def generate_anomaly_data():
    """
    生成异常检测数据

    返回:
    pd.DataFrame: 数据
    """
    # 生成正常数据
    n_samples = 1000
    X1 = np.random.randn(n_samples, 2) * 0.5 + np.array([2, 2])

    # 生成一些异常点
    n_outliers = 50
    X2 = np.random.uniform(low=-1, high=5, size=(n_outliers, 2))

    # 合并数据
    X = np.vstack([X1, X2])

    # 创建DataFrame
    df = pd.DataFrame(X, columns=['feature1', 'feature2'])

    # 添加时间戳
    now = datetime.now()
    timestamps = [now - timedelta(minutes=i) for i in range(n_samples + n_outliers)]
    df['timestamp'] = timestamps

    return df


def main():
    """
    主函数
    """
    # 创建监控系统
    system = MonitoringSystem()

    # 添加报警处理器
    system.add_alert_handler(ConsoleAlertHandler())
    system.add_alert_handler(FileAlertHandler(file_path='/Users/liuwen/Documents/work/repos/github.com/iminders/miner/logs/alerts.log'))

    # 生成示例数据
    factors, returns = generate_sample_data()
    strategy_returns, benchmark_returns = generate_strategy_returns()

    # 创建因子监控
    factor_monitor = FactorMonitor(
        name='factor_monitor',
        factors=factors,
        returns=returns,
        ic_threshold=0.02,
        ic_decay_threshold=0.5,
        return_threshold=0.0,
        check_interval=10.0  # 为了演示，设置较短的检查间隔
    )
    system.add_monitor(factor_monitor)

    # 创建策略监控
    strategy_monitor = StrategyMonitor(
        name='strategy_monitor',
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns,
        return_threshold=-0.01,
        drawdown_threshold=0.1,
        sharpe_threshold=0.5,
        volatility_threshold=0.02,
        check_interval=15.0  # 为了演示，设置较短的检查间隔
    )
    system.add_monitor(strategy_monitor)

    # 创建异常检测器
    anomaly_detector = AnomalyDetector(
        name='anomaly_detector',
        data_source=generate_anomaly_data,
        features=['feature1', 'feature2'],
        contamination=0.05,
        check_interval=20.0  # 为了演示，设置较短的检查间隔
    )
    system.add_monitor(anomaly_detector)

    try:
        # 启动所有监控
        system.start_all()

        # 运行一段时间
        logger.info("监控系统将运行60秒...")
        time.sleep(60)

        # 获取并展示一些监控结果
        factor_monitor = system.get_monitor('factor_monitor')
        if factor_monitor:
            logger.info("因子监控结果:")
            for factor_name, ic_history in factor_monitor.data['ic_history'].items():
                logger.info(f"  因子 {factor_name} 的IC历史: {len(ic_history)} 条记录")

        strategy_monitor = system.get_monitor('strategy_monitor')
        if strategy_monitor:
            logger.info("策略监控结果:")
            logger.info(f"  指标历史: {len(strategy_monitor.data['metrics_history'])} 条记录")

        anomaly_detector = system.get_monitor('anomaly_detector')
        if anomaly_detector:
            logger.info("异常检测结果:")
            logger.info(f"  检测到的异常: {len(anomaly_detector.data['anomalies'])} 条记录")

    except KeyboardInterrupt:
        logger.info("用户中断")
    finally:
        # 停止所有监控
        system.stop_all()


if __name__ == "__main__":
    main()