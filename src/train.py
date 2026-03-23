import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from .feature_engineering import (
    DerivedFeaturesTransformer,
    DataFrameSelector,
    create_log_transformer,
    load_california_housing_data,
    prepare_data,
    ORIGINAL_FEATURES,
    LOG_TRANSFORM_FEATURES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, Any] = {
    "random_forest": RandomForestRegressor,
    "xgboost": XGBRegressor,
    "lightgbm": LGBMRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}

DEFAULT_HYPERPARAMS: dict[str, dict[str, Any]] = {
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
    },
    "xgboost": {
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    },
    "lightgbm": {
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    },
    "gradient_boosting": {
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "random_state": 42,
    },
}

DERIVED_FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
    "rooms_per_household", "bedrooms_per_room", "population_per_household",
    "dist_to_sf", "dist_to_la", "dist_to_major_city", "income_per_capita",
]


def create_column_transformer() -> ColumnTransformer:
    log_features = ["Population", "AveOccup"]
    standard_features = [f for f in DERIVED_FEATURES if f not in log_features]

    column_transformer = ColumnTransformer(
        transformers=[
            (
                "log_transform_scale",
                Pipeline([
                    ("log", FunctionTransformer(func=np.log1p, validate=False)),
                    ("scaler", StandardScaler()),
                ]),
                log_features,
            ),
            (
                "standard_scale",
                StandardScaler(),
                standard_features,
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
    )

    return column_transformer


def create_pipeline(model_name: str = "xgboost", hyperparams: dict[str, Any] | None = None) -> Pipeline:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    params = DEFAULT_HYPERPARAMS.get(model_name, {}).copy()
    if hyperparams:
        params.update(hyperparams)

    model_class = MODEL_REGISTRY[model_name]
    model = model_class(**params)

    pipeline = Pipeline([
        ("derived_features", DerivedFeaturesTransformer()),
        ("preprocessor", create_column_transformer()),
        ("model", model),
    ])

    logger.info(f"Created pipeline with {model_name} model")
    logger.info(f"Pipeline steps: {[step[0] for step in pipeline.steps]}")
    logger.info(f"Model hyperparameters: {params}")

    return pipeline


def train_model(
    model_name: str = "xgboost",
    hyperparams: dict[str, Any] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    logger.info("Loading California housing dataset...")
    df = load_california_housing_data()
    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")

    pipeline = create_pipeline(model_name, hyperparams)

    logger.info("Training model...")
    logger.info("Pipeline structure:")
    logger.info("  Step 1: DerivedFeaturesTransformer - 创建衍生特征")
    logger.info("  Step 2: ColumnTransformer - Log变换 + StandardScaler")
    logger.info("  Step 3: Model - XGBoost/RandomForest/LightGBM")

    pipeline.fit(X_train, y_train)

    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    logger.info(f"Train R² score: {train_score:.4f}")
    logger.info(f"Test R² score: {test_score:.4f}")

    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2")
    logger.info(f"Cross-validation R² scores: {cv_scores}")
    logger.info(f"Mean CV R² score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    result = {
        "pipeline": pipeline,
        "train_score": train_score,
        "test_score": test_score,
        "cv_scores": cv_scores,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "model_name": model_name,
        "hyperparams": hyperparams or DEFAULT_HYPERPARAMS.get(model_name, {}),
    }

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, save_path)
        logger.info(f"Pipeline saved to {save_path}")

    return result


def load_pipeline(path: str | Path) -> Pipeline:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline file not found: {path}")

    pipeline = joblib.load(path)
    logger.info(f"Pipeline loaded from {path}")
    return pipeline


def predict(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    predictions = pipeline.predict(X)
    return predictions


if __name__ == "__main__":
    result = train_model(
        model_name="xgboost",
        save_path=Path(__file__).parent.parent / "models" / "house_price_pipeline.pkl"
    )
    print(f"\nTraining completed!")
    print(f"Test R² score: {result['test_score']:.4f}")
    print(f"CV R² score: {result['cv_mean']:.4f} (+/- {result['cv_std'] * 2:.4f})")
