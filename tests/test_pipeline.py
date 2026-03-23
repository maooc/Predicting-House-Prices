"""Unit tests for ML pipeline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import fetch_california_housing

from src.pipeline import (
    build_feature_engineering_pipeline,
    build_preprocessing_pipeline,
    build_model_pipeline,
    get_feature_columns,
)


class TestFeatureEngineeringPipeline:
    """测试特征工程Pipeline。"""
    
    @pytest.fixture
    def sample_data(self):
        """创建测试数据。"""
        return pd.DataFrame({
            "MedInc": [3.0, 5.0],
            "HouseAge": [20, 30],
            "AveRooms": [6.0, 5.5],
            "AveBedrms": [1.0, 1.2],
            "Population": [1500, 2000],
            "AveOccup": [3.0, 2.5],
            "Latitude": [37.0, 38.0],
            "Longitude": [-122.0, -121.0],
        })
    
    def test_pipeline_creates_all_features(self, sample_data):
        """测试Pipeline是否创建所有特征。"""
        pipeline = build_feature_engineering_pipeline()
        result = pipeline.fit_transform(sample_data)
        
        # 检查组合特征
        assert "rooms_per_household" in result.columns
        assert "bedrooms_per_room" in result.columns
        
        # 检查地理特征
        assert "distance_to_san_francisco" in result.columns
        
        # 检查Log特征
        assert "MedInc_log" in result.columns


class TestModelPipeline:
    """测试完整模型Pipeline。"""
    
    @pytest.fixture
    def sample_data(self):
        """创建测试数据。"""
        return pd.DataFrame({
            "MedInc": [3.0, 5.0, 4.0],
            "HouseAge": [20, 30, 25],
            "AveRooms": [6.0, 5.5, 5.8],
            "AveBedrms": [1.0, 1.2, 1.1],
            "Population": [1500, 2000, 1800],
            "AveOccup": [3.0, 2.5, 2.8],
            "Latitude": [37.0, 38.0, 37.5],
            "Longitude": [-122.0, -121.0, -121.5],
        })
    
    @pytest.fixture
    def sample_target(self):
        """创建测试目标。"""
        return np.array([2.0, 3.0, 2.5])
    
    def test_random_forest_pipeline(self, sample_data, sample_target):
        """测试Random Forest Pipeline。"""
        pipeline = build_model_pipeline(model_type="random_forest")
        
        # 拟合
        pipeline.fit(sample_data, sample_target)
        
        # 预测
        predictions = pipeline.predict(sample_data)
        
        assert len(predictions) == len(sample_target)
        assert all(np.isfinite(predictions))
    
    def test_pipeline_with_params(self, sample_data, sample_target):
        """测试带参数的Pipeline。"""
        params = {"n_estimators": 50, "max_depth": 5}
        pipeline = build_model_pipeline(
            model_type="random_forest",
            model_params=params
        )
        
        pipeline.fit(sample_data, sample_target)
        model = pipeline.named_steps["model"]
        
        assert model.n_estimators == 50
        assert model.max_depth == 5


class TestFeatureColumns:
    """测试特征列获取函数。"""
    
    def test_returns_tuple_of_lists(self):
        """测试返回类型。"""
        original, engineered = get_feature_columns()
        
        assert isinstance(original, list)
        assert isinstance(engineered, list)
        assert len(original) == 8
        assert len(engineered) > 0
    
    def test_original_features_match_dataset(self):
        """测试原始特征与数据集匹配。"""
        original, _ = get_feature_columns()
        
        housing = fetch_california_housing()
        assert set(original) == set(housing.feature_names)
