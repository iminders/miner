import pandas as pd
import numpy as np
from pathlib import Path
import logging

from src.data_processing.orderbook_loader import OrderBookLoader
from src.data_processing.feature_extractor import OrderBookFeatureExtractor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    数据处理的主类，整合数据加载和特征提取
    """

    def __init__(self, data_dir, output_dir, file_format='parquet', n_levels=10):
        """
        初始化DataProcessor

        参数:
        data_dir (str): 原始数据目录
        output_dir (str): 处理后数据输出目录
        file_format (str): 数据文件格式
        n_levels (int): 订单簿深度级别
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.file_format = file_format
        self.n_levels = n_levels

        # 创建输出目录（如果不存在）
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化数据加载器和特征提取器
        self.loader = OrderBookLoader(data_dir, file_format)
        self.extractor = OrderBookFeatureExtractor(n_levels)

    def process_single_day(self, date, stock_code=None):
        """
        处理单日数据

        参数:
        date (str): 日期，格式为'YYYYMMDD'
        stock_code (str, optional): 股票代码

        返回:
        pd.DataFrame: 处理后的数据
        """
        logger.info(f"开始处理日期 {date} 的数据")

        try:
            # 加载数据
            df = self.loader.load_data(date, stock_code)
            logger.info(f"成功加载数据，共 {len(df)} 行")

            # 清洗数据
            df_clean = self.loader.clean_data(df)
            logger.info(f"数据清洗完成，清洗后 {len(df_clean)} 行")

            # 时间戳对齐
            df_aligned = self.loader.align_timestamps(df_clean)
            logger.info(f"时间戳对齐完成，对齐后 {len(df_aligned)} 行")

            # 提取特征
            df_features = self.extractor.extract_all_features(df_aligned)
            logger.info(f"特征提取完成，共提取 {len(df_features.columns) - len(df_aligned.columns)} 个特征")

            return df_features

        except Exception as e:
            logger.error(f"处理数据时出错: {str(e)}")
            raise

    def process_and_save(self, date, stock_code=None):
        """
        处理数据并保存结果

        参数:
        date (str): 日期，格式为'YYYYMMDD'
        stock_code (str, optional): 股票代码

        返回:
        str: 输出文件路径
        """
        # 处理数据
        df_processed = self.process_single_day(date, stock_code)

        # 构建输出文件名
        filename = f"{date}"
        if stock_code:
            filename += f"_{stock_code}"

        if self.file_format == 'parquet':
            output_path = self.output_dir / f"{filename}_processed.parquet"
            df_processed.to_parquet(output_path)
        elif self.file_format == 'h5':
            output_path = self.output_dir / f"{filename}_processed.h5"
            df_processed.to_hdf(output_path, key='processed_data')
        else:
            output_path = self.output_dir / f"{filename}_processed.csv"
            df_processed.to_csv(output_path, index=False)

        logger.info(f"处理后的数据已保存至 {output_path}")

        return str(output_path)

    def process_date_range(self, start_date, end_date, stock_codes=None):
        """
        处理日期范围内的数据

        参数:
        start_date (str): 开始日期，格式为'YYYYMMDD'
        end_date (str): 结束日期，格式为'YYYYMMDD'
        stock_codes (list, optional): 股票代码列表

        返回:
        list: 输出文件路径列表
        """
        # 将日期字符串转换为datetime对象
        start = pd.to_datetime(start_date, format='%Y%m%d')
        end = pd.to_datetime(end_date, format='%Y%m%d')

        # 生成日期范围
        date_range = pd.date_range(start=start, end=end, freq='D')
        date_strs = [date.strftime('%Y%m%d') for date in date_range]

        output_paths = []

        # 处理每一天的数据
        for date in date_strs:
            if stock_codes:
                # 如果提供了股票代码列表，则处理每个股票
                for stock_code in stock_codes:
                    try:
                        output_path = self.process_and_save(date, stock_code)
                        output_paths.append(output_path)
                    except FileNotFoundError:
                        logger.warning(f"未找到日期 {date} 股票 {stock_code} 的数据文件")
                    except Exception as e:
                        logger.error(f"处理日期 {date} 股票 {stock_code} 的数据时出错: {str(e)}")
            else:
                # 如果没有提供股票代码列表，则处理所有股票的合并数据
                try:
                    output_path = self.process_and_save(date)
                    output_paths.append(output_path)
                except FileNotFoundError:
                    logger.warning(f"未找到日期 {date} 的数据文件")
                except Exception as e:
                    logger.error(f"处理日期 {date} 的数据时出错: {str(e)}")

        return output_paths