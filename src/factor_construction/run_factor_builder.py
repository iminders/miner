import argparse
import logging
from pathlib import Path
import pandas as pd
import json

from src.factor_construction.factor_builder import FactorBuilder

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='因子构建工具')

    parser.add_argument('--input_file', type=str, required=True,
                        help='输入文件路径')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录路径')

    parser.add_argument('--target_col', type=str, required=True,
                        help='目标变量列名')

    parser.add_argument('--features', type=str, nargs='+',
                        help='特征列表，如果不指定则使用所有数值型列')

    parser.add_argument('--model_type', type=str, default='ensemble',
                        choices=['linear', 'ridge', 'lasso', 'elastic_net', 'random_forest',
                                'gradient_boosting', 'ensemble'],
                        help='模型类型')

    parser.add_argument('--model_name', type=str, default=None,
                        help='模型名称，如果不指定则使用模型类型作为名称')

    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集比例')

    parser.add_argument('--tune_hyperparams', action='store_true',
                        help='是否调优超参数')

    parser.add_argument('--ensemble_models', type=str, nargs='+',
                        help='集成模型列表，仅当model_type为ensemble时有效')

    parser.add_argument('--ensemble_weights', type=float, nargs='+',
                        help='集成模型权重列表，仅当model_type为ensemble时有效')

    parser.add_argument('--build_microstructure', action='store_true',
                        help='是否构建市场微观结构因子')

    parser.add_argument('--build_time_series', action='store_true',
                        help='是否构建时间序列因子')

    parser.add_argument('--build_ml_features', action='store_true',
                        help='是否构建机器学习特征')

    parser.add_argument('--price_col', type=str, default='close',
                        help='价格列名')

    parser.add_argument('--volume_col', type=str, default='volume',
                        help='成交量列名')

    parser.add_argument('--window_sizes', type=int, nargs='+', default=[5, 10, 20, 60],
                        help='窗口大小列表，用于构建机器学习特征')

    parser.add_argument('--save_factor', action='store_true',
                        help='是否保存因子')

    parser.add_argument('--factor_name', type=str, default=None,
                        help='因子名称，如果不指定则使用模型名称')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建因子构建器
    factor_builder = FactorBuilder(output_dir=args.output_dir)

    # 读取输入数据
    logger.info(f"读取输入文件: {args.input_file}")
    if args.input_file.endswith('.csv'):
        df = pd.read_csv(args.input_file)
    elif args.input_file.endswith('.parquet'):
        df = pd.read_parquet(args.input_file)
    else:
        raise ValueError(f"不支持的文件格式: {args.input_file}")

    logger.info(f"数据集大小: {df.shape}")

    # 构建市场微观结构因子
    if args.build_microstructure:
        logger.info("构建市场微观结构因子")
        df = factor_builder.build_microstructure_factors(df, args.price_col, args.volume_col)
        logger.info(f"构建后数据集大小: {df.shape}")

    # 构建时间序列因子
    if args.build_time_series:
        logger.info("构建时间序列因子")
        df = factor_builder.build_time_series_factors(df, args.price_col, args.volume_col)
        logger.info(f"构建后数据集大小: {df.shape}")

    # 构建机器学习特征
    if args.build_ml_features:
        logger.info("构建机器学习特征")
        df = factor_builder.build_ml_features(df, args.target_col, args.window_sizes)
        logger.info(f"构建后数据集大小: {df.shape}")

    # 处理缺失值
    df = df.dropna()
    logger.info(f"处理缺失值后数据集大小: {df.shape}")

    # 确定特征列表
    if args.features:
        features = args.features
    else:
        # 使用所有数值型列作为特征，排除目标变量
        features = df.select_dtypes(include=['number']).columns.tolist()
        if args.target_col in features:
            features.remove(args.target_col)

    logger.info(f"特征数量: {len(features)}")
    logger.info(f"特征列表: {features[:10]}...")

    # 确定模型名称
    model_name = args.model_name if args.model_name else args.model_type

    # 构建因子模型
    if args.model_type == 'linear':
        result = factor_builder.build_linear_factor(df, features, args.target_col, model_name)
    elif args.model_type == 'ridge':
        result = factor_builder.build_ridge_factor(df, features, args.target_col, model_name)
    elif args.model_type == 'lasso':
        result = factor_builder.build_lasso_factor(df, features, args.target_col, model_name)
    elif args.model_type == 'elastic_net':
        result = factor_builder.build_elastic_net_factor(df, features, args.target_col, model_name)
    elif args.model_type == 'random_forest':
        result = factor_builder.build_random_forest_factor(df, features, args.target_col, model_name)
    elif args.model_type == 'gradient_boosting':
        result = factor_builder.build_gradient_boosting_factor(df, features, args.target_col, model_name)
    elif args.model_type == 'ensemble':
        if not args.ensemble_models:
            raise ValueError("使用集成模型时必须指定ensemble_models参数")

        # 构建基础模型（如果尚未构建）
        for base_model in args.ensemble_models:
            if base_model not in factor_builder.models:
                logger.info(f"构建基础模型: {base_model}")
                if base_model.startswith('linear'):
                    factor_builder.build_linear_factor(df, features, args.target_col, base_model)
                elif base_model.startswith('ridge'):
                    factor_builder.build_ridge_factor(df, features, args.target_col, base_model)
                elif base_model.startswith('lasso'):
                    factor_builder.build_lasso_factor(df, features, args.target_col, base_model)
                elif base_model.startswith('elastic_net'):
                    factor_builder.build_elastic_net_factor(df, features, args.target_col, base_model)
                elif base_model.startswith('random_forest'):
                    factor_builder.build_random_forest_factor(df, features, args.target_col, base_model)
                elif base_model.startswith('gradient_boosting'):
                    factor_builder.build_gradient_boosting_factor(df, features, args.target_col, base_model)

        # 构建集成模型
        result = factor_builder.build_ensemble_factor(df, features, args.target_col, args.ensemble_models, args.ensemble_weights, model_name)

    # 评估模型
    eval_result = factor_builder.evaluate_model(df, features, args.target_col, model_name, args.test_size)

    # 合并结果
    result.update(eval_result)

    # 保存结果
    result_path = output_dir / f'{model_name}_result.json'
    with open(result_path, 'w') as f:
        # 移除不可序列化的对象
        serializable_result = {k: v for k, v in result.items() if k not in ['feature_importance']}
        json.dump(serializable_result, f, indent=4)

    # 保存特征重要性
    if 'feature_importance' in result:
        fi_path = output_dir / f'{model_name}_feature_importance.csv'
        result['feature_importance'].to_csv(fi_path, index=False)

    # 生成并保存因子
    if args.save_factor:
        factor_name = args.factor_name if args.factor_name else f'{model_name}_factor'
        factor_df = factor_builder.generate_factor(df, features, model_name, factor_name)
        factor_path = output_dir / f'{factor_name}.parquet'
        factor_builder.save_factor(factor_df, factor_name, factor_path)

    logger.info(f"因子构建完成，结果已保存至 {output_dir}")


if __name__ == '__main__':
    main()