"""Integration tests for the complete ML pipeline."""

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from src.pipeline import build_model_pipeline
from src.train import load_data, split_data, train_model


class TestEndToEndPipeline:
    """端到端Pipeline集成测试。"""
    
    @pytest.fixture(scope="class")
    def full_dataset(self):
        """加载完整数据集。"""
        X, y = load_data()
        return X, y
    
    def test_data_loading(self, full_dataset):
        """测试数据加载。"""
        X, y = full_dataset
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert X.shape[0] == y.shape[0]
        assert X.shape[0] == 20640  # California Housing数据集大小
    
    def test_train_test_split(self, full_dataset):
        """测试数据划分。"""
        X, y = full_dataset
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        
        assert len(X_train) == int(20640 * 0.8)
        assert len(X_test) == int(20640 * 0.2)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
    
    def test_full_training_pipeline(self, full_dataset):
        """测试完整训练流程。"""
        X, y = full_dataset
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        
        # 训练模型
        pipeline = train_model(
            X_train, y_train,
            model_type="random_forest",
            model_params={"n_estimators": 50, "max_depth": 10}
        )
        
        # 预测
        predictions = pipeline.predict(X_test)
        
        # 验证预测
        assert len(predictions) == len(y_test)
        assert all(np.isfinite(predictions))
        assert all(predictions >= 0)  # 房价不能为负
    
    def test_model_serialization(self, full_dataset):
        """测试模型序列化。"""
        X, y = full_dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.9, random_state=42  # 使用小数据集加速测试
        )
        
        # 训练模型
        pipeline = train_model(
            X_train, y_train,
            model_type="random_forest",
            model_params={"n_estimators": 10}
        )
        
        # 保存前预测
        predictions_before = pipeline.predict(X_test.iloc[:10])
        
        # 序列化和反序列化
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name
        
        try:
            joblib.dump(pipeline, temp_path)
            loaded_pipeline = joblib.load(temp_path)
            
            # 加载后预测应该一致
            predictions_after = loaded_pipeline.predict(X_test.iloc[:10])
            np.testing.assert_array_almost_equal(predictions_before, predictions_after)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_pipeline_with_single_sample(self):
        """测试Pipeline处理单条样本。"""
        # 单条样本
        single_sample = pd.DataFrame({
            "MedInc": [3.0],
            "HouseAge": [20],
            "AveRooms": [6.0],
            "AveBedrms": [1.0],
            "Population": [1500],
            "AveOccup": [3.0],
            "Latitude": [37.0],
            "Longitude": [-122.0],
        })
        
        # 构建Pipeline
        pipeline = build_model_pipeline(
            model_type="random_forest",
            model_params={"n_estimators": 10}
        )
        
        # 使用训练数据拟合
        X, y = load_data()
        X_train = X.iloc[:100]
        y_train = y.iloc[:100]
        pipeline.fit(X_train, y_train)
        
        # 单条预测
        prediction = pipeline.predict(single_sample)
        
        assert len(prediction) == 1
        assert np.isfinite(prediction[0])


class TestFeatureEngineeringIntegration:
    """特征工程集成测试。"""
    
    def test_feature_engineering_preserves_dataframe(self):
        """测试特征工程保持DataFrame格式。"""
        from src.pipeline import build_feature_engineering_pipeline
        
        X, _ = load_data()
        X_sample = X.iloc[:100]
        
        pipeline = build_feature_engineering_pipeline()
        result = pipeline.fit_transform(X_sample)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == X_sample.shape[0]
        assert result.shape[1] > X_sample.shape[1]  # 应该增加特征
    
    def test_all_original_features_preserved(self):
        """测试所有原始特征都被保留。"""
        from src.pipeline import build_feature_engineering_pipeline
        
        X, _ = load_data()
        original_columns = set(X.columns)
        
        pipeline = build_feature_engineering_pipeline()
        result = pipeline.fit_transform(X.iloc[:100])
        
        # 所有原始列都应该存在
        assert original_columns.issubset(set(result.columns))
