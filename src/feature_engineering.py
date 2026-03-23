import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

# 原始特征列名
MEDINC = "MedInc"
HOUSEAGE = "HouseAge"
AVEROOMS = "AveRooms"
AVEBEDRMS = "AveBedrms"
POPULATION = "Population"
AVEOCCUP = "AveOccup"
LATITUDE = "Latitude"
LONGITUDE = "Longitude"

# 特征列名列表
BASE_FEATURES = [
    MEDINC,
    HOUSEAGE,
    AVEROOMS,
    AVEBEDRMS,
    POPULATION,
    AVEOCCUP,
    LATITUDE,
    LONGITUDE,
]

# 加州主要城市坐标（旧金山、洛杉矶、圣地亚哥、圣何塞）
MAJOR_CITIES = {
    "sf": {"lat": 37.7749, "lon": -122.4194},
    "la": {"lat": 34.0522, "lon": -118.2437},
    "sd": {"lat": 32.7157, "lon": -117.1611},
    "sj": {"lat": 37.3382, "lon": -121.8863},
}


class CombinedFeaturesAdder(BaseEstimator, TransformerMixin):
    """
    自定义Transformer：添加组合业务特征
    - rooms_per_household: 每户房间数
    - bedrooms_per_room: 每房间卧室数
    - population_per_household: 每户人口数
    """

    def fit(self, X: np.ndarray | pd.DataFrame, y=None) -> "CombinedFeaturesAdder":
        return self

    def transform(self, X: np.ndarray | pd.DataFrame, y=None) -> np.ndarray:
        # 转换为numpy数组（如果是DataFrame）
        if isinstance(X, pd.DataFrame):
            # 按正确的列名顺序提取
            X_arr = X[BASE_FEATURES].values
        else:
            X_arr = X

        rooms_per_household = X_arr[:, 2] / X_arr[:, 5]  # AveRooms / AveOccup
        bedrooms_per_room = X_arr[:, 3] / X_arr[:, 2]  # AveBedrms / AveRooms
        population_per_household = X_arr[:, 4] / X_arr[:, 5]  # Population / AveOccup

        return np.c_[
            X_arr,
            rooms_per_household.reshape(-1, 1),
            bedrooms_per_room.reshape(-1, 1),
            population_per_household.reshape(-1, 1),
        ]


class DistanceFeaturesAdder(BaseEstimator, TransformerMixin):
    """
    自定义Transformer：添加到主要城市的距离特征
    使用Haversine公式计算球面距离
    """

    @staticmethod
    def _haversine_distance(
        lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float
    ) -> np.ndarray:
        # 将角度转换为弧度
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

        # Haversine公式
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))

        # 地球平均半径为6371公里
        return c * 6371

    def fit(self, X: np.ndarray | pd.DataFrame, y=None) -> "DistanceFeaturesAdder":
        return self

    def transform(self, X: np.ndarray | pd.DataFrame, y=None) -> np.ndarray:
        # 输入X已经是numpy数组（来自前一个Transformer）
        # 第6列为Latitude，第7列为Longitude
        latitude = X[:, 6]
        longitude = X[:, 7]

        distance_features = []
        for city_coords in MAJOR_CITIES.values():
            dist = self._haversine_distance(
                latitude, longitude, city_coords["lat"], city_coords["lon"]
            )
            distance_features.append(dist.reshape(-1, 1))

        # 添加到最近主要城市的距离
        all_distances = np.hstack(distance_features)
        min_distance = np.min(all_distances, axis=1).reshape(-1, 1)
        distance_features.append(min_distance)

        return np.c_[X, *distance_features]


class SpatialClusterAdder(BaseEstimator, TransformerMixin):
    """
    自定义Transformer：基于经纬度的空间聚类
    """

    def __init__(self, n_clusters: int = 8, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None

    def fit(self, X: np.ndarray | pd.DataFrame, y=None) -> "SpatialClusterAdder":
        # 输入是前一个Transformer的numpy数组输出
        # 第6列为Latitude，第7列为Longitude
        coords = X[:, [6, 7]]
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        self.kmeans.fit(coords)
        return self

    def transform(self, X: np.ndarray | pd.DataFrame, y=None) -> np.ndarray:
        # 处理空输入
        if X.shape[0] == 0:
            return np.hstack([X, np.zeros((0, 1))])
        
        coords = X[:, [6, 7]]
        clusters = self.kmeans.predict(coords)
        return np.c_[X, clusters.reshape(-1, 1)]


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    自定义Transformer：对偏态特征执行Log变换
    """

    def __init__(self, columns: list[int] | list[str] | None = None):
        # 默认转换MedInc(第0列)和Population(第4列)
        self.columns = columns if columns is not None else [0, 4]
        # 存储实际使用的列索引
        self._column_indices = None

    def fit(self, X: np.ndarray | pd.DataFrame, y=None) -> "LogTransformer":
        # 如果输入是DataFrame，确保列顺序正确
        if isinstance(X, pd.DataFrame):
            # 总是通过列名获取索引，确保顺序正确
            self._column_indices = [BASE_FEATURES.index(col) for col in BASE_FEATURES if col in [MEDINC, POPULATION]]
        else:
            self._column_indices = self.columns
        return self

    def transform(self, X: np.ndarray | pd.DataFrame, y=None) -> np.ndarray:
        # 转换为numpy数组，确保列顺序正确
        if isinstance(X, pd.DataFrame):
            X_arr = X[BASE_FEATURES].values
        else:
            X_arr = X.copy()

        # 使用fit时确定的索引
        columns = self._column_indices if self._column_indices is not None else self.columns

        X_transformed = X_arr.copy()
        for col in columns:
            # 添加小值避免log(0)
            X_transformed[:, col] = np.log1p(np.maximum(X_transformed[:, col], 0))
        return X_transformed


def get_feature_names(base_features: list[str]) -> list[str]:
    """
    获取完整的特征名称列表（包括衍生特征）
    """
    extended_features = base_features.copy()

    # CombinedFeaturesAdder添加的特征
    extended_features.extend(
        ["rooms_per_household", "bedrooms_per_room", "population_per_household"]
    )

    # DistanceFeaturesAdder添加的特征
    extended_features.extend(
        ["distance_to_sf", "distance_to_la", "distance_to_sd", "distance_to_sj", "min_city_distance"]
    )

    # SpatialClusterAdder添加的特征
    extended_features.append("spatial_cluster")

    return extended_features
