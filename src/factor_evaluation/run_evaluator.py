import argparse
import logging
from pathlib import Path

from src.factor_evaluation.factor_evaluator import FactorEvaluator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='A股高频因子评估工具')
    
    parser.add_argument('--input_file', type=str, required=True,
                        help='输入文件路径')
    
    parser.add_argument('--output_dir', type=str, required=True,
                        help='评估结果输出目录')
    
    parser.add_argument('--price_col', type=str, default='mid_price',
                        help='价格列名 (默认: mid_price)')
    
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 5, 10, 30, 60],
                        help='未来收益率的时间跨度列表，单位为秒 (默认: 1 5 10 30 60)')
    
    parser.add_argument('--n_quantiles', type=int, default=5,
                        help='分位数数量 (默认: 5)')
    
    parser.add_argument('--method', type=str, default='spearman', choices=['pearson', 'spearman'],
                        help='相关系数计算方法 (默认: spearman)')
    
    parser.add_argument('--rolling_window', type=int, default=None,
                        help='滚动窗口大小 (默认: None)')
    
    parser.add_argument('--top_factors', type=int, default=None,
                        help='评估排名靠前的因子数量 (默认: 全部)')
    
    parser.add_argument('--factor_cols', type=str, nargs='*', default=None,
                        help='指定要评估的因子列名列表 (默认: 自动选择)')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建因子评估器
    evaluator = FactorEvaluator(output_dir=args.output_dir)
    
    # 评估因子
    try:
        evaluation_results = evaluator.evaluate_from_file(
            file_path=args.input_file,
            factor_cols=args.factor_cols,
            price_col=args.price_col,
            horizons=args.horizons,
            n_quantiles=args.n_quantiles,
            method=args.method,
            rolling_window=args.rolling_window,
            top_factors=args.top_factors
        )
        
        # 输出评估结果摘要
        for horizon, summary_df in evaluation_results.items():
            if not summary_df.empty:
                top_factors = summary_df.head(5)
                logger.info(f"\n{horizon} 评估结果 (Top 5):\n{top_factors[['因子名称', 'IC均值', 'IC信息比率', '多空组合收益']]}")
    
    except Exception as e:
        logger.error(f"评估出错: {str(e)}")


if __name__ == "__main__":
    main()