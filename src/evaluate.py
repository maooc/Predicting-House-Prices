import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_and_split_data(
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """加载并划分数据集"""
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name=housing.target_names[0])

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def load_model(model_path: str | Path) -> Any:
    """加载训练好的模型"""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    logger.info(f"加载模型: {model_path.resolve()}")
    return joblib.load(model_path)


def calculate_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> Dict[str, float]:
    """计算回归评估指标"""
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def cross_validation_evaluation(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
) -> Dict[str, float]:
    """交叉验证评估"""
    logger.info(f"执行 {cv}-折 交叉验证...")

    scoring_metrics = {
        "cv_neg_mse": "neg_mean_squared_error",
        "cv_neg_mae": "neg_mean_absolute_error",
        "cv_r2": "r2",
    }

    results = {}
    for metric_name, scorer in scoring_metrics.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
        if "neg" in metric_name:
            scores = -scores
            metric_name = metric_name.replace("neg_", "")

        results[f"{metric_name}_mean"] = float(np.mean(scores))
        results[f"{metric_name}_std"] = float(np.std(scores))

        logger.info(f"  {metric_name}: {np.mean(scores):.4f} (±{np.std(scores):.4f})")

    return results


def residual_analysis(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    save_path: str | Path | None = None,
) -> None:
    """残差分析可视化"""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Residual Analysis", fontsize=16, y=0.95)

    # 残差分布图
    sns.histplot(residuals, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Residual Distribution")
    axes[0, 0].set_xlabel("Residual")

    # Q-Q图
    import scipy.stats as stats
    stats.probplot(residuals, plot=axes[0, 1])
    axes[0, 1].set_title("Q-Q Plot")

    # 残差vs预测值
    axes[1, 0].scatter(y_pred, residuals, alpha=0.5)
    axes[1, 0].axhline(y=0, color="r", linestyle="-")
    axes[1, 0].set_title("Residuals vs Predicted Values")
    axes[1, 0].set_xlabel("Predicted Values")
    axes[1, 0].set_ylabel("Residuals")

    # 实际值vs预测值散点图
    axes[1, 1].scatter(y_pred, y_true, alpha=0.5)
    axes[1, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    axes[1, 1].set_title("Actual vs Predicted Values")
    axes[1, 1].set_xlabel("Predicted Values")
    axes[1, 1].set_ylabel("Actual Values")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"残差分析图已保存: {save_path}")

    plt.close()


def feature_importance_analysis(
    model,
    feature_names: list[str],
    save_path: str | Path | None = None,
    top_n: int = 15,
) -> None:
    """特征重要性分析"""
    try:
        # 尝试从Pipeline中提取模型
        if hasattr(model, "named_steps"):
            xgb_model = model.named_steps["model"]
        else:
            xgb_model = model

        if not hasattr(xgb_model, "feature_importances_"):
            logger.warning("模型不支持特征重要性分析")
            return

        # 获取特征重要性
        importances = xgb_model.feature_importances_

        # 注意：由于预处理中有one-hot编码，特征数量可能不等于原始特征名数量
        # 这里简化处理，使用占位符
        n_features = len(importances)
        display_names = [f"feature_{i}" for i in range(n_features)]
        if len(feature_names) <= n_features:
            display_names[: len(feature_names)] = feature_names

        # 排序
        indices = np.argsort(importances)[::-1][:top_n]

        # 绘图
        plt.figure(figsize=(12, 8))
        plt.title("Top Feature Importances", fontsize=16)
        bars = plt.bar(range(top_n), importances[indices])

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
            )

        plt.xticks(range(top_n), [display_names[i] for i in indices], rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"特征重要性图已保存: {save_path}")

        plt.close()

    except Exception as e:
        logger.warning(f"特征重要性分析失败: {str(e)}")


def main(
    model_path: str = "house_price_pipeline.joblib",
    output_dir: str = "evaluation_results",
    random_state: int = 42,
) -> Dict[str, Any]:
    """完整评估流程"""
    logger.info("=" * 60)
    logger.info("开始模型评估流程")
    logger.info("=" * 60)

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    try:
        # 1. 加载数据
        logger.info("加载数据集...")
        X_train, X_test, y_train, y_test = load_and_split_data(random_state=random_state)
        feature_names = list(X_train.columns)

        # 2. 加载模型
        model = load_model(model_path)

        # 3. 基础评估
        logger.info("执行基础评估...")
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_metrics = calculate_metrics(y_train, y_pred_train)
        test_metrics = calculate_metrics(y_test, y_pred_test)

        logger.info("\n训练集评估结果:")
        for name, value in train_metrics.items():
            logger.info(f"  {name}: {value:.4f}")

        logger.info("\n测试集评估结果:")
        for name, value in test_metrics.items():
            logger.info(f"  {name}: {value:.4f}")

        # 4. 交叉验证
        logger.info("\n执行交叉验证...")
        cv_metrics = cross_validation_evaluation(model, X_train, y_train, cv=5)

        # 5. 可视化分析
        logger.info("\n生成可视化分析...")
        residual_analysis(
            y_test,
            y_pred_test,
            save_path=output_dir / "residual_analysis.png",
        )

        feature_importance_analysis(
            model,
            feature_names,
            save_path=output_dir / "feature_importance.png",
        )

        # 合并所有结果
        results = {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "cv_metrics": cv_metrics,
        }

        # 保存结果
        import json

        with open(output_dir / "evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)

        logger.info("=" * 60)
        logger.info("模型评估完成!")
        logger.info(f"结果已保存到: {output_dir.resolve()}")
        logger.info("=" * 60)

        return results

    except Exception as e:
        logger.error(f"评估流程出现错误: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
