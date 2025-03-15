# Miner - 高频交易数据挖掘工具

Miner是一个用于高频交易数据挖掘的Python工具包，专注于从订单簿和成交数据中提取有价值的特征和因子。

## 项目结构
miner/
├── README.md                      # 项目说明文档
├── TODO.md                        # 待办事项列表
├── examples/                      # 示例代码目录
│   ├── backtest_example.py        # 回测系统使用示例
│   ├── factor_example.py          # 因子计算示例
│   ├── monitoring_example.py      # 监控系统示例
│   └── realtime_example.py        # 实时交易系统示例
├── requirements.txt               # 项目依赖包列表
└── src/                           # 源代码目录
├── backtest/                  # 回测系统模块
│   ├── backtest_engine.py     # 回测引擎
│   ├── performance_analyzer.py # 性能分析器
│   └── portfolio.py           # 投资组合管理
├── data/                      # 数据处理模块
│   ├── data_loader.py         # 数据加载器
│   ├── data_processor.py      # 数据处理器
│   └── data_source.py         # 数据源接口
├── evaluation/                # 评估模块
│   ├── factor_evaluation.py   # 因子评估
│   └── performance_metrics.py # 性能指标计算
├── features/                  # 特征工程模块
│   ├── basic_features.py      # 基础特征提取
│   ├── feature_extractor.py   # 特征提取器
│   ├── microstructure.py      # 市场微观结构特征
│   ├── ml_features.py         # 机器学习特征
│   └── time_series.py         # 时间序列特征
├── monitoring/                # 监控与报警系统
│   ├── alert_handlers.py      # 报警处理器
│   ├── anomaly_detector.py    # 异常检测器
│   ├── factor_monitor.py      # 因子表现监控
│   ├── monitor_base.py        # 监控基础类
│   ├── monitoring_system.py   # 监控系统整合
│   └── strategy_monitor.py    # 策略表现监控
├── realtime/                  # 实时交易系统
│   ├── data_feed.py           # 实时数据馈送
│   ├── factor_calculator.py   # 实时因子计算
│   ├── realtime_system.py     # 实时系统整合
│   └── signal_generator.py    # 信号生成器
└── strategy/                  # 策略模块
├── portfolio_optimizer.py # 投资组合优化
├── risk_manager.py        # 风险管理
└── strategy.py            # 策略基类


## 模块说明

### 1. 回测系统 (backtest)
- **backtest_engine.py**: 实现回测引擎，支持历史数据回放和策略评估
- **performance_analyzer.py**: 分析策略回测结果，计算各种性能指标
- **portfolio.py**: 管理投资组合，包括持仓、交易和资金管理

### 2. 数据处理 (data)
- **data_loader.py**: 从各种来源加载数据，支持多种数据格式
- **data_processor.py**: 数据预处理，包括清洗、标准化和特征变换
- **data_source.py**: 定义数据源接口，支持实时和历史数据获取

### 3. 评估模块 (evaluation)
- **factor_evaluation.py**: 评估因子的预测能力和稳定性
- **performance_metrics.py**: 计算各种性能指标，如夏普比率、最大回撤等

### 4. 特征工程 (features)
- **basic_features.py**: 从订单簿数据中提取基本价格和量信息
- **feature_extractor.py**: 整合各类特征提取器的主类
- **microstructure.py**: 提取市场微观结构特征，如有效价差、订单流不平衡等
- **ml_features.py**: 生成适用于机器学习的高级特征
- **time_series.py**: 提取时间序列特征，如移动平均、波动率等

### 5. 监控与报警系统 (monitoring)
- **alert_handlers.py**: 处理各种报警，支持控制台、文件、邮件和Webhook等方式
- **anomaly_detector.py**: 检测数据中的异常并发送报警
- **factor_monitor.py**: 监控因子的IC值、收益率等指标
- **monitor_base.py**: 监控系统的基础类，提供通用功能
- **monitoring_system.py**: 整合各种监控和报警功能
- **strategy_monitor.py**: 监控策略的收益率、回撤、夏普比率等指标

### 6. 实时交易系统 (realtime)
- **data_feed.py**: 提供实时市场数据
- **factor_calculator.py**: 实时计算因子值
- **realtime_system.py**: 整合实时交易系统的各个组件
- **signal_generator.py**: 根据因子值生成交易信号

### 7. 策略模块 (strategy)
- **portfolio_optimizer.py**: 优化投资组合权重
- **risk_manager.py**: 管理交易风险，包括仓位控制和止损策略
- **strategy.py**: 策略基类，定义策略接口和通用功能

## 功能特点

### 1. 基础特征提取

- 中间价格、买卖价差、订单簿深度等基本指标
- 订单簿不平衡度、订单簿形状特征
- 价格波动性指标

### 2. 微观结构因子

- 有效价差、实现价差
- 市场深度、订单簿压力
- Kyle's Lambda (价格影响因子)
- 订单流毒性 (Order Flow Toxicity)
- 订单簿斜率和曲率

### 3. 时序特征因子

- 动量因子 (不同时间窗口)
- 波动率因子
- 均值回归因子
- 价格跳跃因子
- 趋势因子

### 4. 机器学习特征工程

- 特征交互项
- 非线性变换
- 时间滞后特征
- 滚动窗口特征
- 降维和特征选择

### 5. 因子评估

- 信息系数(IC)计算
- 因子分组收益率
- 因子换手率和稳定性
- 因子衰减特性
- 因子暴露度和中性化

## 安装

```bash
git clone https://github.com/iminders/miner.git
cd miner
pip install -e .
```

## 使用示例

基础特征提取

```python
from src.features.basic_features import BasicFeatureExtractor

# 初始化特征提取器
extractor = BasicFeatureExtractor()

# 提取基础特征
features = extractor.extract_all_basic_features(orderbook)

```
微观结构特征提取
```python
from src.features.microstructure_features import MicrostructureFeatureExtractor

# 初始化特征提取器
extractor = MicrostructureFeatureExtractor()

# 提取微观结构特征
features = extractor.extract_all_microstructure_features(orderbook)

```

因子评估

```python
from src.evaluation.factor_evaluation import FactorEvaluator

# 初始化因子评估器
evaluator = FactorEvaluator()

# 评估因子
evaluation_results = evaluator.evaluate_factor(factor_data, returns_data)

# 可视化评估结果
evaluator.plot_factor_evaluation(results, 'MyFactor')
```
