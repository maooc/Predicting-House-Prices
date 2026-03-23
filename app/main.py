"""Streamlit web application for California House Price Prediction."""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# 页面配置
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model(model_path: Path = Path("models/house_price_pipeline.pkl")):
    """
    加载训练好的Pipeline模型（带缓存）。
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        加载的Pipeline
    """
    try:
        logger.info(f"Loading model from {model_path}")
        if not model_path.exists():
            st.error(f"模型文件不存在: {model_path}")
            st.info("请先运行训练脚本: python -m src.train")
            return None
        
        pipeline = joblib.load(model_path)
        logger.info("Model loaded successfully")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        st.error(f"加载模型失败: {e}")
        return None


def create_input_dataframe(
    med_inc: float,
    house_age: int,
    ave_rooms: float,
    ave_bedrooms: float,
    population: int,
    ave_occup: float,
    latitude: float,
    longitude: float,
) -> pd.DataFrame:
    """
    创建输入DataFrame。
    
    注意：所有输入都是原始特征值，Pipeline会自动进行特征工程和标准化。
    
    Args:
        med_inc: 收入中位数
        house_age: 房屋年龄
        ave_rooms: 平均房间数
        ave_bedrooms: 平均卧室数
        population: 人口
        ave_occup: 平均居住人数
        latitude: 纬度
        longitude: 经度
        
    Returns:
        输入特征DataFrame
    """
    input_data = {
        "MedInc": [med_inc],
        "HouseAge": [house_age],
        "AveRooms": [ave_rooms],
        "AveBedrms": [ave_bedrooms],
        "Population": [population],
        "AveOccup": [ave_occup],
        "Latitude": [latitude],
        "Longitude": [longitude],
    }
    return pd.DataFrame(input_data)


def predict_price(pipeline, input_df: pd.DataFrame) -> float:
    """
    使用Pipeline进行预测。
    
    Args:
        pipeline: 训练好的Pipeline
        input_df: 输入特征DataFrame
        
    Returns:
        预测价格
    """
    try:
        prediction = pipeline.predict(input_df)[0]
        return float(prediction)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


def main():
    """主应用函数。"""
    # 标题
    st.markdown(
        "<h1 style='text-align: center;'>🏡 California House Price Prediction</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: gray;'>Enter the house details below to predict its price in California</p>",
        unsafe_allow_html=True,
    )
    
    # 加载模型
    pipeline = load_model()
    
    if pipeline is None:
        st.stop()
    
    # 侧边栏输入
    with st.sidebar:
        st.image(
            "https://cdn.pixabay.com/photo/2017/06/10/07/24/house-2382713_1280.jpg",
            use_container_width=True,
        )
        st.title("🔍 Input House Features")
        
        # 收入中位数
        med_inc = st.slider(
            "💰 Median Income (in $10,000s)",
            min_value=0.5,
            max_value=15.0,
            value=3.0,
            step=0.1,
            help="Median income in block group (in tens of thousands of dollars)",
        )
        
        # 房屋年龄
        house_age = st.slider(
            "🏠 House Age (years)",
            min_value=1,
            max_value=52,
            value=20,
            step=1,
            help="Median house age in block group",
        )
        
        # 平均房间数
        ave_rooms = st.slider(
            "🛏️ Avg. Rooms per Household",
            min_value=1.0,
            max_value=10.0,
            value=5.0,
            step=0.1,
            help="Average number of rooms per household",
        )
        
        # 平均卧室数
        ave_bedrooms = st.slider(
            "🚪 Avg. Bedrooms per Household",
            min_value=0.5,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="Average number of bedrooms per household",
        )
        
        # 人口
        population = st.number_input(
            "👨‍👩‍👧‍👦 Population in Block Group",
            min_value=3,
            max_value=35682,
            value=1500,
            step=100,
            help="Block group population",
        )
        
        # 平均居住人数
        ave_occup = st.slider(
            "👥 Avg. Occupants per Household",
            min_value=0.5,
            max_value=10.0,
            value=3.0,
            step=0.1,
            help="Average number of household members",
        )
        
        # 纬度
        latitude = st.slider(
            "📍 Latitude",
            min_value=32.0,
            max_value=42.0,
            value=36.0,
            step=0.01,
            help="Block group latitude",
        )
        
        # 经度
        longitude = st.slider(
            "📍 Longitude",
            min_value=-125.0,
            max_value=-114.0,
            value=-120.0,
            step=0.01,
            help="Block group longitude",
        )
        
        # 预测按钮
        predict_button = st.button("💰 Predict Price", type="primary", use_container_width=True)
    
    # 主内容区域
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Feature Summary")
        
        # 创建输入摘要
        input_df = create_input_dataframe(
            med_inc, house_age, ave_rooms, ave_bedrooms,
            population, ave_occup, latitude, longitude
        )
        
        # 显示输入数据
        st.dataframe(
            input_df.T.rename(columns={0: "Value"}),
            use_container_width=True,
        )
        
        # 显示地图位置
        st.subheader("🗺️ Location on Map")
        map_data = pd.DataFrame({
            "lat": [latitude],
            "lon": [longitude],
        })
        st.map(map_data, zoom=6)
    
    with col2:
        st.subheader("📈 Prediction Result")
        
        if predict_button:
            with st.spinner("Calculating prediction..."):
                try:
                    prediction = predict_price(pipeline, input_df)
                    
                    # 显示预测结果
                    st.success("Prediction Complete!")
                    st.markdown(
                        f"<h2 style='text-align: center; color: #2ecc71;'>${prediction * 100000:,.0f}</h2>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<p style='text-align: center; color: gray;'>Predicted Median House Value</p>",
                        unsafe_allow_html=True,
                    )
                    
                    # 显示预测区间（假设±10%）
                    lower_bound = prediction * 0.9
                    upper_bound = prediction * 1.1
                    st.markdown(
                        f"<p style='text-align: center; color: gray;'>Estimated Range: ${lower_bound * 100000:,.0f} - ${upper_bound * 100000:,.0f}</p>",
                        unsafe_allow_html=True,
                    )
                    
                    logger.info(f"Prediction made: ${prediction * 100000:,.2f}")
                    
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    logger.error(f"Prediction error: {e}")
        else:
            st.info("Click 'Predict Price' to see the estimated house value.")
    
    # 页脚
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: gray; font-size: 0.8em;'>"
        "California House Price Predictor | Built with Streamlit & Scikit-learn</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
