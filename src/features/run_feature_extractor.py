import argparse
import logging
from pathlib import Path

from src.features.feature_extractor import FeatureExtractor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='高频交易数据特征提取工具')

    parser.add_argument('--window_sizes', type=int, nargs='+', default=[5, 10, 20, 50, 100],
                        help='滚动窗口大小列表 (默认: 5 10 20 50 100)')

    # 处理模式
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input_file', type=str,
                       help='输入文件路径')
    group.add_argument('--input_dir', type=str,
                       help='输入目录路径')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录路径')

    parser.add_argument('--file_pattern', type=str, default='*.parquet',
                        help='文件匹配模式 (默认: *.parquet)')

    parser.add_argument('--no_ml_features', action='store_true',
                        help='不包含机器学习特征')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 创建特征提取器
    feature_extractor = FeatureExtractor(window_sizes=args.window_sizes)

    # 根据命令行参数选择处理模式
    if args.input_file:
        # 处理单个文件
        try:
            output_path = feature_extractor.extract_and_save(
                args.input_file,
                args.output_dir,
                include_ml_features=not args.no_ml_features
            )
            logger.info(f"特征已提取并保存至: {output_path}")
        except Exception as e:
            logger.error(f"处理出错: {str(e)}")
    else:
        # 批量处理目录中的文件
        try:
            output_paths = feature_extractor.extract_batch(
                args.input_dir,
                args.output_dir,
                file_pattern=args.file_pattern,
                include_ml_features=not args.no_ml_features
            )
            logger.info(f"共处理 {len(output_paths)} 个文件")
        except Exception as e:
            logger.error(f"处理出错: {str(e)}")


if __name__ == "__main__":
    main()