"""Training script for California House Price Prediction model."""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from src.pipeline import build_model_pipeline, get_feature_importance

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ],
)
logger = logging.getLogger(__name__)


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    加载加州房价数据集。
    
    Returns:
        (特征DataFrame, 目标Series) 的元组
    """
    logger.info("Loading California Housing dataset...")
    housing = fetch_california_housing()
    
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name="MedHouseVal")
    
    logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Features: {list(X.columns)}")
    logger.info(f"Target range: [{y.min():.3f}, {y.max():.3f}]")
    
    return X, y


def split_data(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> tuple:
    """
    划分训练集和测试集。
    
    Args:
        X: 特征DataFrame
        y: 目标Series
        test_size: 测试集比例
        random_state: 随机种子
        
    Returns:
        (X_train, X_test, y_train, y_test) 的元组
    """
    logger.info(f"Splitting data with test_size={test_size}, random_state={random_state}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "random_forest",
    model_params: Optional[Dict] = None,
) -> object:
    """
    训练模型Pipeline。
    
    Args:
        X_train: 训练特征
        y_train: 训练目标
        model_type: 模型类型
        model_params: 模型超参数
        
    Returns:
        训练好的Pipeline
    """
    logger.info(f"Starting model training with {model_type}...")
    
    # 构建Pipeline
    pipeline = build_model_pipeline(model_type=model_type, model_params=model_params)
    
    # 训练
    logger.info("Fitting pipeline...")
    pipeline.fit(X_train, y_train)
    
    logger.info("Model training completed")
    
    return pipeline


def save_model(
    pipeline: object, 
    model_path: Path, 
    metrics: Optional[Dict] = None
) -> None:
    """
    保存训练好的模型和相关元数据。
    
    Args:
        pipeline: 训练好的Pipeline
        model_path: 模型保存路径
        metrics: 训练指标字典
    """
    # 确保目录存在
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存Pipeline
    logger.info(f"Saving model to {model_path}")
    joblib.dump(pipeline, model_path)
    
    # 保存元数据
    if metrics:
        metadata_path = model_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")


def main():
    """主训练流程。"""
    parser = argparse.ArgumentParser(description="Train California House Price Prediction model")
    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=["random_forest", "xgboost", "lightgbm"],
        help="Model type to use",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size ratio",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/house_price_pipeline.pkl",
        help="Output model path",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of estimators",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=20,
        help="Maximum tree depth",
    )
    
    args = parser.parse_args()
    
    # 创建日志目录
    Path("logs").mkdir(exist_ok=True)
    
    # 加载数据
    X, y = load_data()
    
    # 划分数据
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=args.test_size)
    
    # 准备模型参数
    model_params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
    }
    
    # 训练模型
    pipeline = train_model(
        X_train, y_train, 
        model_type=args.model_type, 
        model_params=model_params
    )
    
    # 获取特征重要性
    importance_df = get_feature_importance(pipeline)
    logger.info("Top 10 feature importances:")
    for _, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # 保存模型
    model_path = Path(args.output)
    metrics = {
        "model_type": args.model_type,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "timestamp": datetime.now().isoformat(),
    }
    save_model(pipeline, model_path, metrics)
    
    logger.info("Training completed successfully!")
    logger.info(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
