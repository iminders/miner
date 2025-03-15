import argparse
import logging
from pathlib import Path

from src.features.feature_selector import FeatureSelector

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='特征选择工具')

    parser.add_argument('--input_file', type=str, required=True,
                        help='输入文件路径')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录路径')

    parser.add_argument('--target_col', type=str, required=True,
                        help='目标变量列名')

    parser.add_argument('--method', type=str, default='ensemble',
                        choices=['correlation', 'mutual_info', 'random_forest', 'lasso', 'ensemble'],
                        help='特征选择方法 (默认: ensemble)')

    parser.add_argument('--top_n', type=int, default=20,
                        help='选择的特征数量 (默认: 20)')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 创建特征选择器
    feature_selector = FeatureSelector(output_dir=args.output_dir)

    # 选择特征
    try:
        selected_features = feature_selector.select_from_file(
            file_path=args.input_file,
            target_col=args.target_col,
            method=args.method,
            top_n=args.top_n
        )

        logger.info(f"选择了 {len(selected_features)} 个特征:")
        for i, feature in enumerate(selected_features):
            logger.info(f"{i+1}. {feature}")

    except Exception as e:
        logger.error(f"特征选择出错: {str(e)}")


if __name__ == "__main__":
    main()