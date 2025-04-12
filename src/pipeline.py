import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

from src.feature_engineering import (
    CombinedFeaturesAdder,
    DistanceFeaturesAdder,
    SpatialClusterAdder,
    LogTransformer,
)


def create_full_pipeline(
    model_params: dict | None = None,
    n_clusters: int = 8,
    random_state: int = 42,
) -> Pipeline:
    """
    创建完整的ML Pipeline，包含特征工程、预处理和模型
    """
    # 默认模型参数
    if model_params is None:
        model_params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_state,
        }

    # 特征工程流水线
    feature_engineering = Pipeline(
        [
            ("log_transform", LogTransformer()),
            ("combined_features", CombinedFeaturesAdder()),
            ("distance_features", DistanceFeaturesAdder()),
            ("spatial_clusters", SpatialClusterAdder(n_clusters=n_clusters, random_state=random_state)),
        ]
    )

    # 数值特征处理（排除经纬度和聚类特征，聚类需要单独编码）
    # 特征索引：0-7(原始), 8-10(combined), 11-15(distance), 16(cluster)
    numeric_features = list(range(16))  # 前16个特征是数值型
    cluster_feature = [16]  # 第17个特征是聚类类别，需要编码

    # 数值特征预处理流水线
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # 分类特征预处理流水线（空间聚类）
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")),
        ]
    )

    # ColumnTransformer：组合不同类型的预处理
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, cluster_feature),
        ],
        remainder="drop",
    )

    # 完整流水线：特征工程 -> 预处理 -> 模型
    full_pipeline = Pipeline(
        [
            ("feature_engineering", feature_engineering),
            ("preprocessor", preprocessor),
            ("model", XGBRegressor(**model_params)),
        ]
    )

    return full_pipeline


def create_baseline_pipeline(
    model_params: dict | None = None,
    random_state: int = 42,
) -> Pipeline:
    """
    创建基础流水线（仅基础预处理，用于对比实验）
    """
    if model_params is None:
        model_params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": random_state,
        }

    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        [("numeric", numeric_transformer, list(range(8)))],
        remainder="drop",
    )

    full_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", XGBRegressor(**model_params)),
        ]
    )

    return full_pipeline
