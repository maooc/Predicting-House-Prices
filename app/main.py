import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.train import load_pipeline, predict
from src.evaluate import evaluate_model

PIPELINE_PATH = Path(__file__).parent.parent / "models" / "house_price_pipeline.pkl"


@st.cache_resource
def load_model() -> Any:
    try:
        pipeline = load_pipeline(PIPELINE_PATH)
        return pipeline
    except FileNotFoundError:
        st.error(
            f"Pipeline not found at {PIPELINE_PATH}. "
            "Please run training first: `python -m src.train`"
        )
        st.stop()


def create_input_dataframe(
    med_inc: float,
    house_age: int,
    ave_rooms: int,
    ave_bedrooms: int,
    population: int,
    ave_occup: float,
    latitude: float,
    longitude: float,
) -> pd.DataFrame:
    return pd.DataFrame({
        "MedInc": [med_inc],
        "HouseAge": [house_age],
        "AveRooms": [ave_rooms],
        "AveBedrms": [ave_bedrooms],
        "Population": [population],
        "AveOccup": [ave_occup],
        "Latitude": [latitude],
        "Longitude": [longitude],
    })


def main() -> None:
    st.set_page_config(
        page_title="California House Price Predictor",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        "<h1 style='text-align: center;'>🏡 California House Price Prediction</h1>",
        unsafe_allow_html=True,
    )
    st.write(
        "Enter the house details below to predict its median value "
        "(in $100,000 units)."
    )

    pipeline = load_model()

    st.sidebar.image(
        "https://cdn.pixabay.com/photo/2017/06/10/07/24/house-2382713_1280.jpg",
        use_container_width=True,
    )
    st.sidebar.title("🔍 Input House Features")

    med_inc = st.sidebar.number_input(
        "💰 Median Income (in $10,000s)",
        min_value=0.0,
        max_value=20.0,
        value=3.9,
        step=0.1,
        help="Median income of the block group (e.g., 3.9 means $39,000)",
    )
    house_age = st.sidebar.slider(
        "🏠 House Age (years)",
        min_value=1,
        max_value=100,
        value=30,
        help="Median house age in the block group",
    )
    ave_rooms = st.sidebar.number_input(
        "🛏️ Avg. Rooms per Household",
        min_value=1.0,
        max_value=50.0,
        value=5.0,
        step=0.1,
        help="Average number of rooms per household",
    )
    ave_bedrooms = st.sidebar.number_input(
        "🚪 Avg. Bedrooms per Household",
        min_value=0.5,
        max_value=20.0,
        value=1.0,
        step=0.1,
        help="Average number of bedrooms per household",
    )
    population = st.sidebar.number_input(
        "👨‍👩‍👧‍👦 Population in Area",
        min_value=1,
        max_value=50000,
        value=3000,
        step=100,
        help="Total population in the block group",
    )
    ave_occup = st.sidebar.number_input(
        "👥 Avg. Occupants per Household",
        min_value=1.0,
        max_value=100.0,
        value=3.0,
        step=0.1,
        help="Average number of household members",
    )
    latitude = st.sidebar.number_input(
        "📍 Latitude",
        min_value=32.0,
        max_value=42.0,
        value=34.0,
        step=0.01,
        help="Block group latitude coordinate",
    )
    longitude = st.sidebar.number_input(
        "📍 Longitude",
        min_value=-125.0,
        max_value=-114.0,
        value=-118.0,
        step=0.01,
        help="Block group longitude coordinate",
    )

    st.sidebar.write(f"📆 House Built in: **{2025 - house_age}**")

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("💰 Predict Price", use_container_width=True):
            input_df = create_input_dataframe(
                med_inc=med_inc,
                house_age=house_age,
                ave_rooms=ave_rooms,
                ave_bedrooms=ave_bedrooms,
                population=population,
                ave_occup=ave_occup,
                latitude=latitude,
                longitude=longitude,
            )

            prediction = predict(pipeline, input_df)[0]
            price_dollars = prediction * 100000

            st.markdown(
                f"<h2 style='text-align: center; color: green;'>"
                f"🏡 Estimated Median House Value</h2>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<h3 style='text-align: center;'>"
                f"${price_dollars:,.0f}</h3>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='text-align: center; color: gray;'>"
                f"(Raw prediction: {prediction:.4f} in $100k units)</p>",
                unsafe_allow_html=True,
            )

            st.markdown("### 📊 Input Summary")
            summary_df = pd.DataFrame({
                "Feature": [
                    "Median Income",
                    "House Age",
                    "Avg Rooms",
                    "Avg Bedrooms",
                    "Population",
                    "Avg Occupants",
                    "Latitude",
                    "Longitude",
                ],
                "Value": [
                    f"${med_inc * 10000:,.0f}",
                    f"{house_age} years",
                    f"{ave_rooms:.1f}",
                    f"{ave_bedrooms:.1f}",
                    f"{population:,}",
                    f"{ave_occup:.1f}",
                    f"{latitude:.2f}",
                    f"{longitude:.2f}",
                ],
            })
            st.table(summary_df)

    st.markdown("---")
    st.markdown(
        "<h5 style='text-align: center;'>Developed with ❤️ using Streamlit</h5>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: gray; font-size: 12px;'>"
        "Powered by XGBoost with feature engineering pipeline</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
