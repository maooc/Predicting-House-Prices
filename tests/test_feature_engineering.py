import numpy as np
import pytest
from sklearn.datasets import fetch_california_housing

from src.feature_engineering import (
    CombinedFeaturesAdder,
    DistanceFeaturesAdder,
    SpatialClusterAdder,
    LogTransformer,
    get_feature_names,
    MEDINC,
    POPULATION,
    AVEROOMS,
    AVEBEDRMS,
    AVEOCCUP,
)


@pytest.fixture
def sample_data():
    """创建测试用的样本数据"""
    housing = fetch_california_housing()
    return housing.data[:100]  # 取前100条作为测试数据


class TestCombinedFeaturesAdder:
    """测试组合特征添加器"""

    def test_fit_returns_self(self, sample_data):
        """测试fit方法返回self"""
        transformer = CombinedFeaturesAdder()
        result = transformer.fit(sample_data)
        assert result is transformer

    def test_transform_adds_3_features(self, sample_data):
        """测试transform添加3个新特征"""
        transformer = CombinedFeaturesAdder()
        result = transformer.transform(sample_data)

        # 原始8个特征 + 3个新特征 = 11个特征
        assert result.shape == (sample_data.shape[0], sample_data.shape[1] + 3)

    def test_combined_features_calculation(self, sample_data):
        """测试组合特征的计算正确性"""
        transformer = CombinedFeaturesAdder()
        result = transformer.transform(sample_data)

        # AVEROOMS是第2列, AVEOCCUP是第5列
        expected_rooms = sample_data[:, 2] / sample_data[:, 5]
        np.testing.assert_array_almost_equal(result[:, 8], expected_rooms)

        # AVEBEDRMS是第3列, AVEROOMS是第2列
        expected_bedrooms = sample_data[:, 3] / sample_data[:, 2]
        np.testing.assert_array_almost_equal(result[:, 9], expected_bedrooms)

        # POPULATION是第4列, AVEOCCUP是第5列
        expected_pop = sample_data[:, 4] / sample_data[:, 5]
        np.testing.assert_array_almost_equal(result[:, 10], expected_pop)

    def test_empty_input(self):
        """测试空输入处理"""
        transformer = CombinedFeaturesAdder()
        empty_data = np.array([]).reshape(0, 8)
        result = transformer.transform(empty_data)
        assert result.shape == (0, 11)


class TestDistanceFeaturesAdder:
    """测试距离特征添加器"""

    def test_fit_returns_self(self, sample_data):
        """测试fit方法返回self"""
        transformer = DistanceFeaturesAdder()
        result = transformer.fit(sample_data)
        assert result is transformer

    def test_transform_adds_5_features(self, sample_data):
        """测试transform添加5个距离特征（4个城市 + 1个最小距离）"""
        transformer = DistanceFeaturesAdder()
        result = transformer.transform(sample_data)

        # 原始8个特征 + 5个距离特征 = 13个特征
        assert result.shape == (sample_data.shape[0], sample_data.shape[1] + 5)

    def test_distance_calculations_positive(self, sample_data):
        """测试距离计算结果应为正数"""
        transformer = DistanceFeaturesAdder()
        result = transformer.transform(sample_data)

        # 所有距离都应该大于0
        distance_features = result[:, 8:13]
        assert np.all(distance_features >= 0)

    def test_min_distance_is_minimum(self, sample_data):
        """测试最小距离确实是四个距离中的最小值"""
        transformer = DistanceFeaturesAdder()
        result = transformer.transform(sample_data)

        four_distances = result[:, 8:12]
        min_distance = result[:, 12]

        expected_min = np.min(four_distances, axis=1)
        np.testing.assert_array_almost_equal(min_distance, expected_min)

    def test_haversine_distance(self):
        """测试Haversine距离计算的正确性"""
        transformer = DistanceFeaturesAdder()

        # 已知两点的距离测试（洛杉矶到圣地亚哥约180公里）
        la_lat, la_lon = 34.0522, -118.2437
        sd_lat, sd_lon = 32.7157, -117.1611

        test_point = np.array([[0, 0, 0, 0, 0, 0, la_lat, la_lon]])
        result = transformer.transform(test_point)

        # 到圣地亚哥的距离应该在170-190公里之间
        distance_to_sd = result[0, 10]  # 第三个城市是SD
        assert 170 < distance_to_sd < 190


class TestSpatialClusterAdder:
    """测试空间聚类添加器"""

    def test_fit_returns_self(self, sample_data):
        """测试fit方法返回self"""
        transformer = SpatialClusterAdder(n_clusters=3)
        result = transformer.fit(sample_data)
        assert result is transformer

    def test_transform_adds_cluster_feature(self, sample_data):
        """测试transform添加聚类特征"""
        transformer = SpatialClusterAdder(n_clusters=3, random_state=42)
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)

        # 原始8个特征 + 1个聚类特征 = 9个特征
        assert result.shape == (sample_data.shape[0], sample_data.shape[1] + 1)

    def test_cluster_values_in_range(self, sample_data):
        """测试聚类标签在正确范围内"""
        n_clusters = 5
        transformer = SpatialClusterAdder(n_clusters=n_clusters, random_state=42)
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)

        cluster_values = result[:, 8]
        # 聚类标签应该是0到n_clusters-1的整数
        assert np.all(cluster_values >= 0)
        assert np.all(cluster_values < n_clusters)
        assert len(np.unique(cluster_values)) <= n_clusters

    def test_deterministic_clustering(self, sample_data):
        """测试相同random_state下聚类结果的确定性"""
        transformer1 = SpatialClusterAdder(n_clusters=3, random_state=42)
        transformer2 = SpatialClusterAdder(n_clusters=3, random_state=42)

        transformer1.fit(sample_data)
        transformer2.fit(sample_data)

        result1 = transformer1.transform(sample_data)
        result2 = transformer2.transform(sample_data)

        np.testing.assert_array_equal(result1[:, 8], result2[:, 8])


class TestLogTransformer:
    """测试Log变换器"""

    def test_fit_returns_self(self, sample_data):
        """测试fit方法返回self"""
        transformer = LogTransformer()
        result = transformer.fit(sample_data)
        assert result is transformer

    def test_transform_preserves_shape(self, sample_data):
        """测试transform保持数据形状"""
        transformer = LogTransformer()
        result = transformer.transform(sample_data)
        assert result.shape == sample_data.shape

    def test_log1p_transformation_correctness(self):
        """测试log1p变换的正确性"""
        transformer = LogTransformer(columns=[0])
        test_data = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])

        result = transformer.transform(test_data)

        # 验证第0列是log1p结果
        expected_col0 = np.log1p(test_data[:, 0])
        np.testing.assert_array_almost_equal(result[:, 0], expected_col0)

        # 验证其他列保持不变
        np.testing.assert_array_equal(result[:, 1:], test_data[:, 1:])

    def test_default_columns(self, sample_data):
        """测试默认转换的列"""
        transformer = LogTransformer()
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)

        # MEDINC(第0列)和POPULATION(第4列)应该被转换
        expected_medinc = np.log1p(sample_data[:, 0])
        expected_pop = np.log1p(sample_data[:, 4])

        np.testing.assert_array_almost_equal(result[:, 0], expected_medinc)
        np.testing.assert_array_almost_equal(result[:, 4], expected_pop)

    def test_handles_zero_values(self):
        """测试处理零值的能力"""
        transformer = LogTransformer(columns=[0])
        test_data = np.array([[0.0], [1.0]])

        # 应该可以处理零值（log1p(0) = 0）
        result = transformer.transform(test_data)
        assert result[0, 0] == 0.0


class TestGetFeatureNames:
    """测试获取特征名称函数"""

    def test_returns_correct_number_of_features(self):
        """测试返回正确数量的特征名"""
        base_features = ["feat1", "feat2", "feat3", "feat4", "feat5", "feat6", "feat7", "feat8"]
        result = get_feature_names(base_features)

        # 8个原始 + 3组合 + 5距离 + 1聚类 = 17个特征
        assert len(result) == 17

    def test_contains_base_features(self):
        """测试结果包含原始特征名"""
        base_features = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
        result = get_feature_names(base_features)

        for feat in base_features:
            assert feat in result

    def test_contains_derived_features(self):
        """测试结果包含衍生特征名"""
        base_features = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
        result = get_feature_names(base_features)

        derived_features = [
            "rooms_per_household",
            "bedrooms_per_room",
            "population_per_household",
            "distance_to_sf",
            "distance_to_la",
            "distance_to_sd",
            "distance_to_sj",
            "min_city_distance",
            "spatial_cluster",
        ]

        for feat in derived_features:
            assert feat in result
