import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

from src.features.microstructure import MicrostructureFeatures
from src.features.time_series import TimeSeriesFeatures
from src.features.ml_features import MLFeatureEngineering

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    特征提取的主类，整合各类特征
    """
    
    def __init__(self, window_sizes: Optional[List[int]] = None):
        """
        初始化FeatureExtractor
        
        Args:
            window_sizes: 滚动窗口大小列表，默认为[5, 10, 20, 50, 100]
        """
        self.window_sizes = window_sizes or [5, 10, 20, 50, 100]
        
        # 初始化各类特征提取器
        self.microstructure_features = MicrostructureFeatures(window_sizes)
        self.time_series_features = TimeSeriesFeatures(window_sizes)
        self.ml_features = MLFeatureEngineering()
    
    def extract_features(self, df: pd.DataFrame, include_ml_features: bool = True) -> pd.DataFrame:
        """
        提取所有特征
        
        Args:
            df: 输入DataFrame，包含原始数据
            include_ml_features: 是否包含机器学习特征
            
        Returns:
            添加了所有特征的DataFrame
        """
        logger.info("开始提取特征...")
        
        # 提取微观结构特征
        logger.info("提取微观结构特征...")
        df_micro = self.microstructure_features.calculate_effective_spread(df)
        df_micro = self.microstructure_features.calculate_order_flow_imbalance(df_micro)
        df_micro = self.microstructure_features.calculate_price_impact(df_micro)
        logger.info(f"微观结构特征提取完成，共添加 {len(df_micro.columns) - len(df.columns)} 个特征")
        
        # 提取时间序列特征
        logger.info("提取时间序列特征...")
        df_ts = self.time_series_features.extract_all_time_series_features(df_micro)
        logger.info(f"时间序列特征提取完成，共添加 {len(df_ts.columns) - len(df_micro.columns)} 个特征")
        
        # 提取机器学习特征
        if include_ml_features:
            logger.info("提取机器学习特征...")
            df_ml = self.ml_features.create_all_features(df_ts)
            logger.info(f"机器学习特征提取完成，共添加 {len(df_ml.columns) - len(df_ts.columns)} 个特征")
            result = df_ml
        else:
            result = df_ts
        
        logger.info(f"特征提取完成，总共 {len(result.columns) - len(df.columns)} 个特征")
        
        return result
    
    def extract_and_save(self, input_file: str, output_dir: str, include_ml_features: bool = True) -> str:
        """
        从文件加载数据，提取特征并保存结果
        
        Args:
            input_file: 输入文件路径
            output_dir: 输出目录
            include_ml_features: 是否包含机器学习特征
            
        Returns:
            输出文件路径
        """
        input_path = Path(input_file)
        output_dir = Path(output_dir)
        
        # 创建输出目录（如果不存在）
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载数据
        logger.info(f"从 {input_path} 加载数据...")
        
        if input_path.suffix == '.parquet':
            df = pd.read_parquet(input_path)
        elif input_path.suffix == '.h5':
            df = pd.read_hdf(input_path, key='raw_data')
        elif input_path.suffix == '.csv':
            df = pd.read_csv(input_path)
        else:
            raise ValueError(f"不支持的文件格式: {input_path.suffix}")
        
        logger.info(f"数据加载完成，共 {len(df)} 行")
        
        # 提取特征
        df_features = self.extract_features(df, include_ml_features)
        
        # 构建输出文件名
        output_filename = f"{input_path.stem}_features{input_path.suffix}"
        output_path = output_dir / output_filename
        
        # 保存结果
        logger.info(f"保存特征到 {output_path}...")
        
        if output_path.suffix == '.parquet':
            df_features.to_parquet(output_path)
        elif output_path.suffix == '.h5':
            df_features.to_hdf(output_path, key='feature_data')
        elif output_path.suffix == '.csv':
            df_features.to_csv(output_path, index=False)
        
        logger.info(f"特征已保存至 {output_path}")
        
        return str(output_path)
    
    def extract_batch(self, input_dir: str, output_dir: str, file_pattern: str = '*.parquet', include_ml_features: bool = True) -> List[str]:
        """
        批量处理目录中的文件，提取特征并保存结果
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            file_pattern: 文件匹配模式
            include_ml_features: 是否包含机器学习特征
            
        Returns:
            输出文件路径列表
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # 创建输出目录（如果不存在）
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 查找匹配的文件
        input_files = list(input_dir.glob(file_pattern))
        logger.info(f"找到 {len(input_files)} 个文件需要处理")
        
        output_paths = []
        
        # 处理每个文件
        for input_file in input_files:
            try:
                output_path = self.extract_and_save(input_file, output_dir, include_ml_features)
                output_paths.append(output_path)
            except Exception as e:
                logger.error(f"处理文件 {input_file} 时出错: {str(e)}")
        
        logger.info(f"批量处理完成，共处理 {len(output_paths)} 个文件")
        
        return output_paths