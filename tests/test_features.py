"""Unit tests for feature engineering transformers."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    CombinedFeaturesTransformer,
    GeoFeaturesTransformer,
    LogTransformer,
    FeatureSelector,
)


class TestCombinedFeaturesTransformer:
    """测试组合特征转换器。"""
    
    @pytest.fixture
    def sample_data(self):
        """创建测试数据。"""
        return pd.DataFrame({
            "AveRooms": [6.0, 5.5, 7.0],
            "AveBedrms": [1.0, 1.2, 1.5],
            "Population": [1500, 2000, 1000],
            "AveOccup": [3.0, 2.5, 4.0],
        })
    
    def test_transform_creates_features(self, sample_data):
        """测试是否正确创建组合特征。"""
        transformer = CombinedFeaturesTransformer()
        result = transformer.fit_transform(sample_data)
        
        assert "rooms_per_household" in result.columns
        assert "bedrooms_per_room" in result.columns
        assert "population_per_household" in result.columns
    
    def test_bedrooms_per_room_calculation(self, sample_data):
        """测试卧室占比计算是否正确。"""
        transformer = CombinedFeaturesTransformer()
        result = transformer.fit_transform(sample_data)
        
        expected = sample_data["AveBedrms"] / sample_data["AveRooms"]
        pd.testing.assert_series_equal(
            result["bedrooms_per_room"], 
            expected, 
            check_names=False
        )
    
    def test_handles_zero_rooms(self):
        """测试处理零房间数的情况。"""
        data = pd.DataFrame({
            "AveRooms": [0.0, 5.0],
            "AveBedrms": [1.0, 1.0],
            "Population": [100, 200],
            "AveOccup": [2.0, 3.0],
        })
        
        transformer = CombinedFeaturesTransformer()
        result = transformer.fit_transform(data)
        
        # 零房间时，bedrooms_per_room应为0
        assert result.loc[0, "bedrooms_per_room"] == 0.0


class TestGeoFeaturesTransformer:
    """测试地理特征转换器。"""
    
    @pytest.fixture
    def sample_geo_data(self):
        """创建地理测试数据（旧金山附近）。"""
        return pd.DataFrame({
            "Latitude": [37.7749, 34.0522],
            "Longitude": [-122.4194, -118.2437],
        })
    
    def test_transform_creates_distance_features(self, sample_geo_data):
        """测试是否正确创建距离特征。"""
        transformer = GeoFeaturesTransformer()
        result = transformer.fit_transform(sample_geo_data)
        
        assert "distance_to_san_francisco" in result.columns
        assert "distance_to_los_angeles" in result.columns
        assert "distance_to_san_diego" in result.columns
        assert "distance_to_sacramento" in result.columns
    
    def test_distance_to_sf_is_small(self, sample_geo_data):
        """测试到旧金山的距离应该很小。"""
        transformer = GeoFeaturesTransformer()
        result = transformer.fit_transform(sample_geo_data)
        
        # 第一行是旧金山坐标，距离应该接近0
        assert result.loc[0, "distance_to_san_francisco"] < 1.0
    
    def test_distance_calculation_reasonable(self, sample_geo_data):
        """测试距离计算是否合理。"""
        transformer = GeoFeaturesTransformer()
        result = transformer.fit_transform(sample_geo_data)
        
        # 旧金山到洛杉矶的距离应该约550公里
        sf_to_la = result.loc[0, "distance_to_los_angeles"]
        assert 500 < sf_to_la < 600


class TestLogTransformer:
    """测试Log变换转换器。"""
    
    @pytest.fixture
    def sample_data(self):
        """创建测试数据。"""
        return pd.DataFrame({
            "MedInc": [3.0, 5.0, 0.0, 10.0],
            "Population": [1000, 2000, 0, 5000],
            "Latitude": [37.0, 38.0, 39.0, 40.0],
        })
    
    def test_transform_creates_log_features(self, sample_data):
        """测试是否正确创建Log特征。"""
        transformer = LogTransformer(columns=["MedInc", "Population"])
        result = transformer.fit_transform(sample_data)
        
        assert "MedInc_log" in result.columns
        assert "Population_log" in result.columns
    
    def test_log_calculation(self, sample_data):
        """测试Log计算是否正确。"""
        transformer = LogTransformer(columns=["MedInc"])
        result = transformer.fit_transform(sample_data)
        
        expected = np.log1p(sample_data["MedInc"])
        pd.testing.assert_series_equal(
            result["MedInc_log"],
            expected,
            check_names=False
        )
    
    def test_zero_indicator(self, sample_data):
        """测试零值指示器。"""
        transformer = LogTransformer(columns=["MedInc"], add_indicator=True)
        result = transformer.fit_transform(sample_data)
        
        assert "MedInc_is_zero" in result.columns
        assert result.loc[2, "MedInc_is_zero"] == 1  # 第三行是0
        assert result.loc[0, "MedInc_is_zero"] == 0  # 第一行不是0
    
    def test_auto_detect_columns(self, sample_data):
        """测试自动检测数值列。"""
        transformer = LogTransformer(add_indicator=False)
        result = transformer.fit_transform(sample_data)
        
        # 应该排除经纬度
        assert "Latitude_log" not in result.columns


class TestFeatureSelector:
    """测试特征选择器。"""
    
    @pytest.fixture
    def sample_data(self):
        """创建测试数据。"""
        return pd.DataFrame({
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "C": [7, 8, 9],
        })
    
    def test_selects_correct_columns(self, sample_data):
        """测试是否正确选择列。"""
        selector = FeatureSelector(columns=["A", "C"])
        result = selector.fit_transform(sample_data)
        
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result[:, 0], [1, 2, 3])
        np.testing.assert_array_equal(result[:, 1], [7, 8, 9])
    
    def test_raises_on_missing_columns(self, sample_data):
        """测试缺失列时是否抛出异常。"""
        selector = FeatureSelector(columns=["A", "D"])
        
        with pytest.raises(ValueError, match="Missing columns"):
            selector.fit(sample_data)
