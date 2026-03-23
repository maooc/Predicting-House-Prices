import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)

logger = logging.getLogger(__name__)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
    }

    logger.info(f"{model_name} Evaluation Metrics:")
    logger.info(f"  MSE:  {mse:.4f}")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  MAE:  {mae:.4f}")
    logger.info(f"  R²:   {r2:.4f}")
    logger.info(f"  MAPE: {mape:.4f}")

    return metrics


def evaluate_pipeline(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Pipeline"
) -> dict[str, float]:
    y_pred = pipeline.predict(X_test)
    return evaluate_model(y_test.values, y_pred, model_name)


def format_metrics_report(metrics: dict[str, float]) -> str:
    report = f"""
========================================
Model Evaluation Report
========================================
MSE  (Mean Squared Error):      {metrics['mse']:.4f}
RMSE (Root Mean Squared Error): {metrics['rmse']:.4f}
MAE  (Mean Absolute Error):     {metrics['mae']:.4f}
R²   (R-squared):               {metrics['r2']:.4f}
MAPE (Mean Abs. % Error):       {metrics['mape']:.4f}
========================================
"""
    return report


def compare_models(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(results).T
    df = df[["rmse", "mae", "r2", "mape"]]
    df = df.round(4)
    df = df.sort_values("r2", ascending=False)
    logger.info("\nModel Comparison:\n" + df.to_string())
    return df
