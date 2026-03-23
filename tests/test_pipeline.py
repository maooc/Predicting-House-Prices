import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import fetch_california_housing

from src.feature_engineering import (
    DerivedFeaturesTransformer,
    prepare_data,
    load_california_housing_data,
)
from src.train import create_pipeline, create_column_transformer, MODEL_REGISTRY
from src.evaluate import evaluate_model


@pytest.fixture
def sample_data() -> pd.DataFrame:
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    df["MedHouseVal"] = housing.target
    return df


@pytest.fixture
def sample_input() -> pd.DataFrame:
    return pd.DataFrame({
        "MedInc": [3.9],
        "HouseAge": [30],
        "AveRooms": [5.0],
        "AveBedrms": [1.0],
        "Population": [3000],
        "AveOccup": [3.0],
        "Latitude": [34.0],
        "Longitude": [-118.0],
    })


class TestDerivedFeaturesTransformer:
    def test_derived_features_transformer_fit_transform(self, sample_data: pd.DataFrame) -> None:
        X, y = prepare_data(sample_data)
        transformer = DerivedFeaturesTransformer()
        transformer.fit(X)
        X_transformed = transformer.transform(X)

        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] > X.shape[1]

    def test_derived_features_created(self, sample_input: pd.DataFrame) -> None:
        transformer = DerivedFeaturesTransformer()
        transformer.fit(sample_input)
        X_transformed = transformer.transform(sample_input)

        assert "rooms_per_household" in X_transformed.columns
        assert "bedrooms_per_room" in X_transformed.columns
        assert "dist_to_sf" in X_transformed.columns
        assert "dist_to_la" in X_transformed.columns
        assert "dist_to_major_city" in X_transformed.columns
        assert "population_per_household" in X_transformed.columns
        assert "income_per_capita" in X_transformed.columns

    def test_derived_features_output_shape(self, sample_data: pd.DataFrame) -> None:
        X, _ = prepare_data(sample_data)
        transformer = DerivedFeaturesTransformer()
        transformer.fit(X)
        X_transformed = transformer.transform(X)

        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == 15


class TestColumnTransformer:
    def test_column_transformer_creation(self) -> None:
        preprocessor = create_column_transformer()
        assert preprocessor is not None
        assert len(preprocessor.transformers) == 2

    def test_column_transformer_transform(self, sample_data: pd.DataFrame) -> None:
        X, _ = prepare_data(sample_data.head(100))
        
        derived_transformer = DerivedFeaturesTransformer()
        X_derived = derived_transformer.fit_transform(X)
        
        preprocessor = create_column_transformer()
        X_scaled = preprocessor.fit_transform(X_derived)
        
        assert X_scaled.shape[0] == 100
        assert X_scaled.shape[1] == 15


class TestPipeline:
    def test_create_pipeline_xgboost(self) -> None:
        pipeline = create_pipeline("xgboost")
        assert pipeline is not None
        assert len(pipeline.steps) == 3
        assert pipeline.steps[0][0] == "derived_features"
        assert pipeline.steps[1][0] == "preprocessor"
        assert pipeline.steps[2][0] == "model"

    def test_create_pipeline_random_forest(self) -> None:
        pipeline = create_pipeline("random_forest")
        assert pipeline is not None

    def test_create_pipeline_lightgbm(self) -> None:
        pipeline = create_pipeline("lightgbm")
        assert pipeline is not None

    def test_create_pipeline_invalid_model(self) -> None:
        with pytest.raises(ValueError):
            create_pipeline("invalid_model")

    def test_pipeline_fit_predict(self, sample_data: pd.DataFrame) -> None:
        X, y = prepare_data(sample_data.head(100))
        pipeline = create_pipeline("xgboost")
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)

        assert predictions.shape == y.shape
        assert not np.any(np.isnan(predictions))

    def test_pipeline_custom_hyperparams(self) -> None:
        hyperparams = {"n_estimators": 50, "max_depth": 3}
        pipeline = create_pipeline("xgboost", hyperparams=hyperparams)
        assert pipeline is not None

    def test_pipeline_uses_column_transformer(self) -> None:
        pipeline = create_pipeline("xgboost")
        from sklearn.compose import ColumnTransformer
        assert isinstance(pipeline.named_steps["preprocessor"], ColumnTransformer)


class TestEvaluate:
    def test_evaluate_model(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

        metrics = evaluate_model(y_true, y_pred, "TestModel")

        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "mape" in metrics
        assert metrics["r2"] > 0.9

    def test_evaluate_model_perfect_predictions(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        metrics = evaluate_model(y_true, y_pred)

        assert metrics["mse"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["r2"] == 1.0


class TestDataLoading:
    def test_load_california_housing_data(self) -> None:
        df = load_california_housing_data()

        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 20640
        assert "MedHouseVal" in df.columns

    def test_prepare_data(self, sample_data: pd.DataFrame) -> None:
        X, y = prepare_data(sample_data)

        assert X.shape[1] == 8
        assert len(y) == X.shape[0]
        assert "MedInc" in X.columns
        assert "Latitude" in X.columns
