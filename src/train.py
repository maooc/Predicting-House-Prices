import logging
import logging.config
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.pipeline import create_full_pipeline

# 配置日志
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "detailed",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": "training.log",
            "level": "DEBUG",
            "formatter": "json",
        },
    },
    "loggers": {
        "src": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": True,
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """加载加州房价数据集"""
    logger.info("加载加州房价数据集...")
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name=housing.target_names[0])

    logger.info(f"数据集形状: X={X.shape}, y={y.shape}")
    logger.info(f"特征名称: {list(housing.feature_names)}")

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """划分训练集和测试集"""
    logger.info(f"划分数据集: test_size={test_size}, random_state={random_state}")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def evaluate_model(
    model,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    dataset_name: str = "test",
) -> dict:
    """评估模型性能"""
    logger.info(f"评估模型在 {dataset_name} 集上的性能...")
    y_pred = model.predict(X)

    metrics = {
        f"{dataset_name}_mse": float(mean_squared_error(y, y_pred)),
        f"{dataset_name}_rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        f"{dataset_name}_mae": float(mean_absolute_error(y, y_pred)),
        f"{dataset_name}_r2": float(r2_score(y, y_pred)),
    }

    logger.info(f"{dataset_name}集评估结果:")
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")

    return metrics


def save_model(model, file_path: str | Path) -> None:
    """保存模型到文件"""
    file_path = Path(file_path)
    logger.info(f"保存模型到: {file_path.resolve()}")
    joblib.dump(model, file_path)
    logger.info("模型保存成功!")


def save_metrics(metrics: dict, file_path: str | Path) -> None:
    """保存评估指标到JSON文件"""
    file_path = Path(file_path)
    logger.info(f"保存评估指标到: {file_path.resolve()}")
    with open(file_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("评估指标保存成功!")


def main(
    model_path: str = "house_price_pipeline.joblib",
    metrics_path: str = "metrics.json",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """主训练函数"""
    logger.info("=" * 60)
    logger.info("开始房价预测模型训练流程")
    logger.info("=" * 60)

    try:
        # 1. 加载数据
        X, y = load_data()

        # 2. 划分数据集
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=test_size, random_state=random_state
        )

        # 3. 创建Pipeline
        logger.info("创建ML Pipeline...")
        model_params = {
            "n_estimators": 300,
            "max_depth": 7,
            "learning_rate": 0.08,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": random_state,
        }
        logger.info(f"模型超参数: {model_params}")

        pipeline = create_full_pipeline(
            model_params=model_params,
            random_state=random_state,
        )

        # 4. 训练模型
        logger.info("开始训练模型...")
        pipeline.fit(X_train, y_train)
        logger.info("模型训练完成!")

        # 5. 评估模型
        train_metrics = evaluate_model(pipeline, X_train, y_train, "train")
        test_metrics = evaluate_model(pipeline, X_test, y_test, "test")

        # 合并指标
        all_metrics = {**train_metrics, **test_metrics}

        # 6. 保存模型和指标
        save_model(pipeline, model_path)
        save_metrics(all_metrics, metrics_path)

        logger.info("=" * 60)
        logger.info("训练流程完成!")
        logger.info("=" * 60)

        return all_metrics

    except Exception as e:
        logger.error(f"训练流程出现错误: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
