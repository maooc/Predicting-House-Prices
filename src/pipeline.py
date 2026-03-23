"""ML Pipeline construction for California House Price Prediction."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import CombinedFeaturesTransformer, GeoFeaturesTransformer, LogTransformer

logger = logging.getLogger(__name__)


def build_feature_engineering_pipeline() -> Pipeline:
    """
    构建特征工程Pipeline。
    
    包含以下步骤：
    1. 组合特征生成（rooms_per_household等）
    2. 地理特征生成（到主要城市的距离）
    3. Log变换（对偏态特征）
    
    Returns:
        特征工程Pipeline
    """
    feature_engineering_steps = [
        ("combined_features", CombinedFeaturesTransformer()),
        ("geo_features", GeoFeaturesTransformer()),
        ("log_transform", LogTransformer(
            columns=["MedInc", "Population", "AveOccup"],
            add_indicator=True
        )),
    ]
    
    return Pipeline(steps=feature_engineering_steps)


def get_feature_columns() -> Tuple[List[str], List[str]]:
    """
    获取用于模型训练的特征列列表。
    
    Returns:
        (原始特征列表, 工程特征列表) 的元组
    """
    # 原始特征
    original_features = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms", 
        "Population", "AveOccup", "Latitude", "Longitude"
    ]
    
    # 工程特征
    engineered_features = [
        # 组合特征
        "rooms_per_household",
        "bedrooms_per_room", 
        "population_per_household",
        # 地理特征
        "distance_to_san_francisco",
        "distance_to_los_angeles",
        "distance_to_san_diego",
        "distance_to_sacramento",
        # Log变换特征
        "MedInc_log",
        "Population_log",
        "AveOccup_log",
        # 零值指示器
        "MedInc_is_zero",
        "Population_is_zero",
        "AveOccup_is_zero",
    ]
    
    return original_features, engineered_features


def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    构建数据预处理Pipeline。
    
    对数值特征进行标准化处理。
    
    Returns:
        ColumnTransformer预处理对象
    """
    original_features, engineered_features = get_feature_columns()
    all_features = original_features + engineered_features
    
    # 所有特征都进行标准化
    preprocessor = ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), all_features),
        ],
        remainder="drop"  # 丢弃未指定的列
    )
    
    return preprocessor


def build_model_pipeline(
    model_type: str = "random_forest",
    model_params: Optional[Dict] = None
) -> Pipeline:
    """
    构建完整的模型训练Pipeline。
    
    Args:
        model_type: 模型类型，支持 "random_forest", "xgboost", "lightgbm"
        model_params: 模型超参数字典
        
    Returns:
        完整的ML Pipeline
    """
    logger.info(f"Building model pipeline with model_type={model_type}")
    
    # 特征工程Pipeline
    feature_pipeline = build_feature_engineering_pipeline()
    
    # 预处理Pipeline
    preprocessor = build_preprocessing_pipeline()
    
    # 选择模型
    if model_params is None:
        model_params = {}
    
    if model_type == "random_forest":
        default_params = {
            "n_estimators": 200,
            "max_depth": 20,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1,
        }
        default_params.update(model_params)
        model = RandomForestRegressor(**default_params)
        logger.info(f"Using RandomForest with params: {default_params}")
        
    elif model_type == "xgboost":
        try:
            import xgboost as xgb
            default_params = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42,
                "n_jobs": -1,
            }
            default_params.update(model_params)
            model = xgb.XGBRegressor(**default_params)
            logger.info(f"Using XGBoost with params: {default_params}")
        except ImportError:
            logger.warning("XGBoost not installed, falling back to RandomForest")
            model = RandomForestRegressor(
                n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
            )
            
    elif model_type == "lightgbm":
        try:
            import lightgbm as lgb
            default_params = {
                "n_estimators": 200,
                "max_depth": -1,
                "learning_rate": 0.1,
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }
            default_params.update(model_params)
            model = lgb.LGBMRegressor(**default_params)
            logger.info(f"Using LightGBM with params: {default_params}")
        except ImportError:
            logger.warning("LightGBM not installed, falling back to RandomForest")
            model = RandomForestRegressor(
                n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
            )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    # 构建完整Pipeline
    pipeline = Pipeline(steps=[
        ("features", feature_pipeline),
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    
    return pipeline


def get_feature_importance(
    pipeline: Pipeline, 
    feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    获取特征重要性。
    
    Args:
        pipeline: 训练好的Pipeline
        feature_names: 特征名称列表
        
    Returns:
        特征重要性DataFrame
    """
    model = pipeline.named_steps["model"]
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        logger.warning("Model does not have feature_importances_ attribute")
        return pd.DataFrame()
    
    if feature_names is None:
        original_features, engineered_features = get_feature_columns()
        feature_names = original_features + engineered_features
    
    importance_df = pd.DataFrame({
        "feature": feature_names[:len(importances)],
        "importance": importances,
    }).sort_values("importance", ascending=False)
    
    return importance_df
