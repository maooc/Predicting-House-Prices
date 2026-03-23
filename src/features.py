"""Custom feature engineering transformers for California Housing dataset."""

import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class CombinedFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    创建组合特征的业务转换器。
    
    生成以下业务特征：
    - rooms_per_household: 每户房间数
    - bedrooms_per_room: 卧室占比
    - population_per_household: 每户人口数
    """
    
    def __init__(self, 
                 rooms_col: str = "AveRooms",
                 bedrooms_col: str = "AveBedrms", 
                 population_col: str = "Population",
                 households_col: str = "AveOccup") -> None:
        self.rooms_col = rooms_col
        self.bedrooms_col = bedrooms_col
        self.population_col = population_col
        self.households_col = households_col
        
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "CombinedFeaturesTransformer":
        """Fit方法（无状态转换器，直接返回self）。"""
        logger.debug("CombinedFeaturesTransformer.fit() called")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        创建组合特征。
        
        Args:
            X: 输入DataFrame
            
        Returns:
            添加了组合特征的DataFrame
        """
        X_transformed = X.copy()
        
        # 计算每户房间数 (rooms_per_household)
        # AveRooms 已经是每户平均房间数，直接使用
        X_transformed["rooms_per_household"] = X_transformed[self.rooms_col]
        
        # 计算卧室占比 (bedrooms_per_room)
        # 避免除以零
        X_transformed["bedrooms_per_room"] = np.where(
            X_transformed[self.rooms_col] > 0,
            X_transformed[self.bedrooms_col] / X_transformed[self.rooms_col],
            0.0
        )
        
        # 计算每户人口数 (population_per_household)
        # AveOccup 已经是每户平均居住人数，直接使用
        X_transformed["population_per_household"] = X_transformed[self.households_col]
        
        logger.debug(f"Created combined features: {self.get_feature_names_out()}")
        return X_transformed
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """返回新增的特征名称列表。"""
        return ["rooms_per_household", "bedrooms_per_room", "population_per_household"]


class GeoFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    基于经纬度的地理特征转换器。
    
    生成以下地理特征：
    - distance_to_sf: 到旧金山的距离
    - distance_to_la: 到洛杉矶的距离
    - distance_to_san_diego: 到圣地亚哥的距离
    - distance_to_sacramento: 到萨克拉门托的距离
    """
    
    # 加州主要城市坐标
    MAJOR_CITIES = {
        "san_francisco": (37.7749, -122.4194),
        "los_angeles": (34.0522, -118.2437),
        "san_diego": (32.7157, -117.1611),
        "sacramento": (38.5816, -121.4944),
    }
    
    def __init__(self, 
                 lat_col: str = "Latitude",
                 lon_col: str = "Longitude") -> None:
        self.lat_col = lat_col
        self.lon_col = lon_col
        
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "GeoFeaturesTransformer":
        """Fit方法（无状态转换器，直接返回self）。"""
        logger.debug("GeoFeaturesTransformer.fit() called")
        return self
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        计算两点间的Haversine距离（单位：公里）。
        
        Args:
            lat1, lon1: 第一点经纬度
            lat2, lon2: 第二点经纬度
            
        Returns:
            两点间距离（公里）
        """
        R = 6371  # 地球半径（公里）
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_lat / 2) ** 2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        return R * c
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        创建地理特征。
        
        Args:
            X: 输入DataFrame
            
        Returns:
            添加了地理特征的DataFrame
        """
        X_transformed = X.copy()
        
        for city_name, (city_lat, city_lon) in self.MAJOR_CITIES.items():
            feature_name = f"distance_to_{city_name}"
            X_transformed[feature_name] = self._haversine_distance(
                X_transformed[self.lat_col].values,
                X_transformed[self.lon_col].values,
                city_lat,
                city_lon
            )
            logger.debug(f"Created geo feature: {feature_name}")
        
        return X_transformed
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """返回新增的特征名称列表。"""
        return [f"distance_to_{city}" for city in self.MAJOR_CITIES.keys()]


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    对高度偏态特征进行Log变换的转换器。
    
    使用 np.log1p (log(1+x)) 避免对零值取log的问题。
    """
    
    def __init__(self, 
                 columns: Optional[List[str]] = None,
                 add_indicator: bool = True) -> None:
        """
        初始化Log变换器。
        
        Args:
            columns: 需要变换的列名列表，None则自动检测数值列
            add_indicator: 是否添加指示器特征标记原始值是否为0
        """
        self.columns = columns
        self.add_indicator = add_indicator
        self._fitted_columns: Optional[List[str]] = None
        
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "LogTransformer":
        """
        确定需要变换的列。
        
        Args:
            X: 输入DataFrame
            y: 目标变量（未使用）
            
        Returns:
            self
        """
        if self.columns is None:
            # 自动检测所有数值列（排除经纬度）
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            self._fitted_columns = [col for col in numeric_cols 
                                   if col not in ["Latitude", "Longitude"]]
        else:
            self._fitted_columns = self.columns
            
        logger.info(f"LogTransformer will transform columns: {self._fitted_columns}")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        对指定列进行Log变换。
        
        Args:
            X: 输入DataFrame
            
        Returns:
            变换后的DataFrame
        """
        X_transformed = X.copy()
        
        for col in self._fitted_columns:
            if col in X_transformed.columns:
                # 使用log1p避免对0取log的问题
                new_col_name = f"{col}_log"
                X_transformed[new_col_name] = np.log1p(X_transformed[col])
                
                if self.add_indicator:
                    indicator_name = f"{col}_is_zero"
                    X_transformed[indicator_name] = (X_transformed[col] == 0).astype(int)
        
        logger.debug(f"Log-transformed columns: {self._fitted_columns}")
        return X_transformed
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """返回新增的特征名称列表。"""
        if self._fitted_columns is None:
            return []
        
        feature_names = []
        for col in self._fitted_columns:
            feature_names.append(f"{col}_log")
            if self.add_indicator:
                feature_names.append(f"{col}_is_zero")
        return feature_names


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    特征选择器，用于在Pipeline最后选择需要的特征。
    """
    
    def __init__(self, columns: List[str]) -> None:
        """
        初始化特征选择器。
        
        Args:
            columns: 需要保留的列名列表
        """
        self.columns = columns
        
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "FeatureSelector":
        """验证列是否存在。"""
        missing_cols = set(self.columns) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in input data: {missing_cols}")
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        选择指定列并返回numpy数组。
        
        Args:
            X: 输入DataFrame
            
        Returns:
            选择后的特征数组
        """
        return X[self.columns].values
