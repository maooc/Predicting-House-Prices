"""Evaluation script for California House Price Prediction model."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.pipeline import get_feature_importance

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_model(model_path: Path) -> object:
    """
    加载训练好的模型。
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        加载的Pipeline
    """
    logger.info(f"Loading model from {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    pipeline = joblib.load(model_path)
    logger.info("Model loaded successfully")
    return pipeline


def evaluate_model(
    pipeline: object, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> Dict[str, float]:
    """
    评估模型性能。
    
    Args:
        pipeline: 训练好的Pipeline
        X_test: 测试特征
        y_test: 测试目标
        
    Returns:
        评估指标字典
    """
    logger.info("Evaluating model...")
    
    # 预测
    y_pred = pipeline.predict(X_test)
    
    # 计算指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 计算MAPE
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    metrics = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "mape": float(mape),
    }
    
    logger.info("Evaluation Results:")
    logger.info(f"  MSE:  {mse:.4f}")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  MAE:  {mae:.4f}")
    logger.info(f"  R²:   {r2:.4f}")
    logger.info(f"  MAPE: {mape:.2f}%")
    
    return metrics


def analyze_predictions(
    y_test: pd.Series, 
    y_pred: np.ndarray,
    output_dir: Path
) -> None:
    """
    分析预测结果并保存。
    
    Args:
        y_test: 真实值
        y_pred: 预测值
        output_dir: 输出目录
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建预测对比DataFrame
    comparison_df = pd.DataFrame({
        "actual": y_test.values,
        "predicted": y_pred,
        "error": y_test.values - y_pred,
        "abs_error": np.abs(y_test.values - y_pred),
        "pct_error": np.abs((y_test.values - y_pred) / y_test.values) * 100,
    })
    
    # 保存预测结果
    output_path = output_dir / "predictions.csv"
    comparison_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    
    # 统计信息
    logger.info("\nPrediction Statistics:")
    logger.info(f"  Mean Actual:    {comparison_df['actual'].mean():.4f}")
    logger.info(f"  Mean Predicted: {comparison_df['predicted'].mean():.4f}")
    logger.info(f"  Mean Error:     {comparison_df['error'].mean():.4f}")
    logger.info(f"  Max Abs Error:  {comparison_df['abs_error'].max():.4f}")


def save_feature_importance(
    pipeline: object, 
    output_dir: Path
) -> None:
    """
    保存特征重要性。
    
    Args:
        pipeline: 训练好的Pipeline
        output_dir: 输出目录
    """
    importance_df = get_feature_importance(pipeline)
    
    if not importance_df.empty:
        output_path = output_dir / "feature_importance.csv"
        importance_df.to_csv(output_path, index=False)
        logger.info(f"Feature importance saved to {output_path}")
        
        logger.info("\nTop 10 Important Features:")
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")


def main():
    """主评估流程。"""
    parser = argparse.ArgumentParser(description="Evaluate California House Price Prediction model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/house_price_pipeline.pkl",
        help="Path to trained model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size ratio",
    )
    
    args = parser.parse_args()
    
    # 加载模型
    model_path = Path(args.model_path)
    pipeline = load_model(model_path)
    
    # 加载数据
    logger.info("Loading test data...")
    from sklearn.model_selection import train_test_split
    from src.train import load_data
    
    X, y = load_data()
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    
    # 评估
    metrics = evaluate_model(pipeline, X_test, y_test)
    
    # 预测
    y_pred = pipeline.predict(X_test)
    
    # 分析预测结果
    output_dir = Path(args.output_dir)
    analyze_predictions(y_test, y_pred, output_dir)
    
    # 保存特征重要性
    save_feature_importance(pipeline, output_dir)
    
    # 保存评估指标
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    logger.info("\nEvaluation completed!")


if __name__ == "__main__":
    main()
