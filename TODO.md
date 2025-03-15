# A股高频因子挖掘项目

基于每秒级别的orderbook数据，构建一个A股高频因子挖掘系统，用于发现和验证可能具有预测能力的交易信号。

## 项目概述

本项目旨在利用高频orderbook数据挖掘有效的交易因子，构建量化交易策略，并通过严格的回测验证其有效性。

## 项目阶段

### 第一阶段：数据准备与预处理（2-3周）

1. **数据整理**
   - [x] 建立数据库存储结构 - `src/data/database.py`
   - [x] 清洗和标准化orderbook数据 - `src/data/preprocessing.py`
   - [x] 处理缺失值和异常值 - `src/data/preprocessing.py`
   - [x] 时间戳对齐和同步 - `src/data/preprocessing.py`

2. **基础特征提取**
   - [x] 提取基本价格信息（中间价、最优买卖价等）- `src/features/basic_features.py`
   - [x] 计算基础量价指标（买卖盘深度、挂单量比等）- `src/features/basic_features.py`
   - [x] 计算订单流失衡指标 - `src/features/basic_features.py`

### 第二阶段：因子构建（3-4周）

1. **微观结构因子**
   - [x] 订单簿不平衡因子 - `src/features/microstructure_factors.py`
   - [x] 价格压力因子 - `src/features/microstructure_factors.py`
   - [x] 流动性因子（如有效价差、市场深度等）- `src/features/microstructure_factors.py`
   - [x] 订单流毒性因子 - `src/features/microstructure_factors.py`

2. **时序特征因子**
   - [x] 价格动量因子 - `src/features/time_series_factors.py`
   - [x] 波动率因子 - `src/features/time_series_factors.py`
   - [x] 均值回归因子 - `src/features/time_series_factors.py`
   - [x] 价格跳跃因子 - `src/features/time_series_factors.py`

3. **机器学习特征工程**
   - [x] 特征组合与交互 - `src/features/ml_features.py`
   - [x] 非线性变换 - `src/features/ml_features.py`
   - [x] 时间序列特征（自回归特征、滚动窗口特征等）- `src/features/ml_features.py`

### 第三阶段：因子评估与选择（2-3周）

1. **单因子测试**
   - [x] IC分析（信息系数）- `src/evaluation/factor_evaluation.py`
   - [x] 因子收益率分析 - `src/evaluation/factor_evaluation.py`
   - [x] 因子稳定性分析 - `src/evaluation/factor_evaluation.py`
   - [x] 因子衰减分析 - `src/evaluation/factor_evaluation.py`

2. **多因子组合**
   - [x] 因子相关性分析 - `src/evaluation/factor_combination.py`
   - [x] 因子聚类 - `src/evaluation/factor_combination.py`
   - [x] 多因子模型构建 - `src/evaluation/factor_combination.py`
   - [x] 权重优化 - `src/evaluation/factor_combination.py`

### 第四阶段：策略开发与回测（3-4周）

1. **策略设计**
   - [x] 信号生成逻辑
   - [x] 交易规则设定
   - [x] 风险控制参数

2. **回测系统**
   - [x] 构建高频回测框架 - `src/backtest/backtest_engine.py`
   - [x] 模拟真实交易环境（滑点、手续费等）- `src/backtest/backtest_engine.py`
   - [x] 性能评估指标计算 - `src/backtest/performance_metrics.py`

3. **策略优化**
   - [x] 参数优化 - `src/optimization/parameter_optimization.py`
   - [x] 交易时机优化 - `src/optimization/timing_optimization.py`
   - [x] 止盈止损优化 - `src/optimization/stop_optimization.py`

### 第五阶段：系统实现与部署（2-3周）

1. **实时计算框架**
   - [x] 数据实时接入 - `src/realtime/data_feed.py`
   - [x] 因子实时计算 - `src/realtime/factor_calculator.py`
   - [x] 信号实时生成 - `src/realtime/signal_generator.py`
   - [x] 实时系统整合 - `src/realtime/realtime_system.py`

5. **监控与报警系统**
   - [x] 因子表现监控 - `src/monitoring/factor_monitor.py`
   - [x] 策略表现监控 - `src/monitoring/strategy_monitor.py`
   - [x] 异常检测与报警 - `src/monitoring/anomaly_detector.py`
   - [x] 报警处理器 - `src/monitoring/alert_handlers.py`
   - [x] 监控系统整合 - `src/monitoring/monitoring_system.py`

## 技术栈选择

1. **数据处理与存储**
   - [x] Python (pandas, numpy) - 所有模块
   - [ ] ClickHouse/TimescaleDB（时序数据库）
   - [ ] HDF5/Parquet（高效文件存储）

2. **因子计算**
   - [x] NumPy, SciPy - 所有特征和因子模块
   - [ ] Numba（高性能计算）
   - [ ] Dask/Ray（分布式计算）

3. **机器学习**
   - [x] Scikit-learn - `src/features/ml_features.py`, `src/evaluation/factor_combination.py`
   - [ ] LightGBM/XGBoost
   - [ ] PyTorch/TensorFlow（深度学习模型）

4. **回测与可视化**
   - [ ] Backtrader/Zipline（回测框架）
   - [x] Matplotlib, Seaborn, Plotly - `src/evaluation/factor_evaluation.py`, `src/evaluation/factor_combination.py`
   - [ ] Dash/Streamlit（交互式仪表盘）

## 项目风险与挑战

1. **数据质量问题**
   - [ ] 高频数据噪声大
   - [ ] 可能存在数据缺失或异常
   - [ ] 解决方案：强健的数据清洗流程，异常检测算法

2. **计算性能挑战**
   - [ ] 高频数据量大，计算密集
   - [ ] 解决方案：优化代码，使用并行计算，考虑GPU加速

3. **过拟合风险**
   - [ ] 高频数据易导致过拟合
   - [ ] 解决方案：严格的样本外测试，时间序列交叉验证

4. **市场微观结构变化**
   - [ ] 交易规则和市场结构可能变化
   - [ ] 解决方案：定期重新训练模型，设计自适应因子

## 项目里程碑

1. **第1个月**：
   - [x] 完成数据预处理和基础特征提取 - `src/data/preprocessing.py`, `src/features/basic_features.py`
2. **第2-3个月**：
   - [x] 完成因子构建和初步评估 - `src/features/*`, `src/evaluation/factor_evaluation.py`
3. **第4个月**：
   - [x] 完成多因子模型和策略设计 - `src/evaluation/factor_combination.py`
4. **第5个月**：
   - [ ] 完成回测系统和策略优化
5. **第6个月**：
   - [ ] 完成实时系统部署和监控框架

## 后续扩展方向

1. [ ] 深度学习模型探索（CNN, RNN, Transformer等）
2. [ ] 强化学习交易策略研究
3. [ ] 多市场、多品种因子迁移性研究
4. [ ] 高频套利策略开发
