import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging

logger = logging.getLogger(__name__)

ORIGINAL_FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", 
    "Population", "AveOccup", "Latitude", "Longitude"
]

LOG_TRANSFORM_FEATURES = ["Population", "AveOccup"]

SF_LAT, SF_LON = 37.7749, -122.4194
LA_LAT, LA_LON = 34.0522, -118.2437


class DerivedFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        self._output_features: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "DerivedFeaturesTransformer":
        self._output_features = self._get_output_feature_names(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = self._create_derived_features(X)
        return X

    def _create_derived_features(self, X: pd.DataFrame) -> pd.DataFrame:
        ave_occup_safe = X["AveOccup"].replace(0, 1)
        ave_rooms_safe = X["AveRooms"].replace(0, 1)
        population_safe = X["Population"].replace(0, 1)

        X["rooms_per_household"] = X["AveRooms"]
        X["bedrooms_per_room"] = X["AveBedrms"] / ave_rooms_safe
        X["population_per_household"] = X["Population"] / ave_occup_safe

        X["dist_to_sf"] = np.sqrt(
            (X["Latitude"] - SF_LAT) ** 2 + (X["Longitude"] - SF_LON) ** 2
        )
        X["dist_to_la"] = np.sqrt(
            (X["Latitude"] - LA_LAT) ** 2 + (X["Longitude"] - LA_LON) ** 2
        )
        X["dist_to_major_city"] = np.minimum(X["dist_to_sf"], X["dist_to_la"])

        X["income_per_capita"] = X["MedInc"] / X["population_per_household"].replace(0, 1)

        return X

    def _get_output_feature_names(self, X: pd.DataFrame) -> list[str]:
        return list(ORIGINAL_FEATURES) + [
            "rooms_per_household",
            "bedrooms_per_room", 
            "population_per_household",
            "dist_to_sf",
            "dist_to_la",
            "dist_to_major_city",
            "income_per_capita",
        ]

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        return self._output_features


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str]) -> None:
        super().__init__()
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "DataFrameSelector":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.columns]

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        return self.columns


def create_log_transformer() -> FunctionTransformer:
    return FunctionTransformer(
        func=np.log1p,
        inverse_func=np.expm1,
        validate=False,
        feature_names_out="one-to-one"
    )


def create_preprocessor(derived_features: list[str]) -> ColumnTransformer:
    log_features = [f for f in LOG_TRANSFORM_FEATURES if f in derived_features]
    standard_features = [f for f in derived_features if f not in log_features]

    transformers = []

    if log_features:
        transformers.append(
            ("log_transform", Pipeline([
                ("selector", DataFrameSelector(log_features)),
                ("log", create_log_transformer()),
                ("scaler", StandardScaler()),
            ]), log_features)
        )

    if standard_features:
        transformers.append(
            ("standard_scale", Pipeline([
                ("selector", DataFrameSelector(standard_features)),
                ("scaler", StandardScaler()),
            ]), standard_features)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0,
    )

    return preprocessor


def build_feature_pipeline() -> Pipeline:
    derived_transformer = DerivedFeaturesTransformer()
    
    pipeline = Pipeline([
        ("derived_features", derived_transformer),
    ])
    
    return pipeline


def load_california_housing_data() -> pd.DataFrame:
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    df["MedHouseVal"] = housing.target
    logger.info(f"Loaded California housing data: {df.shape[0]} samples, {df.shape[1]} columns")
    return df


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[ORIGINAL_FEATURES].copy()
    y = df["MedHouseVal"]
    return X, y
