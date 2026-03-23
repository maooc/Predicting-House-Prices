import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# 页面配置
st.set_page_config(
    page_title="🏡 加州房价预测器",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 常量定义
MODEL_PATH = "house_price_pipeline.joblib"
FEATURE_NAMES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

# 加载模型
@st.cache_resource
def load_model():
    """加载训练好的Pipeline模型"""
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        st.error(f"❌ 模型文件不存在: {model_path.resolve()}\n请先运行 `python -m src.train` 训练模型！")
        st.stop()

    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ 模型加载失败: {str(e)}")
        st.stop()


def create_input_form() -> pd.DataFrame:
    """创建输入表单并返回DataFrame"""
    st.sidebar.title("🔍 输入房屋特征")

    # 输入分组
    with st.sidebar.expander("💵 经济特征", expanded=True):
        med_inc = st.number_input(
            "中位数收入（万美元）",
            min_value=0.1,
            max_value=15.0,
            value=3.9,
            step=0.1,
            help="该街区的收入中位数（单位：万美元）",
        )

    with st.sidebar.expander("🏠 房屋特征", expanded=True):
        house_age = st.slider(
            "房屋平均年限",
            min_value=1,
            max_value=100,
            value=30,
            help="房屋的平均建造年限",
        )
        ave_rooms = st.number_input(
            "平均房间数",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            help="每户的平均房间数量",
        )
        ave_bedrooms = st.number_input(
            "平均卧室数",
            min_value=0.5,
            max_value=10.0,
            value=1.1,
            step=0.1,
            help="每户的平均卧室数量",
        )

    with st.sidebar.expander("👥 人口特征", expanded=True):
        population = st.number_input(
            "街区人口",
            min_value=100,
            max_value=50000,
            value=3000,
            step=100,
            help="该街区的总人口",
        )
        ave_occup = st.number_input(
            "平均居住人数",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.1,
            help="每户的平均居住人数",
        )

    with st.sidebar.expander("📍 地理位置", expanded=True):
        st.markdown("**加州主要城市坐标参考:**")
        st.markdown("- 旧金山: 37.77, -122.42")
        st.markdown("- 洛杉矶: 34.05, -118.24")
        st.markdown("- 圣地亚哥: 32.72, -117.16")
        st.markdown("- 圣何塞: 37.34, -121.89")

        latitude = st.number_input(
            "纬度",
            min_value=32.0,
            max_value=42.0,
            value=34.0,
            step=0.01,
            help="该街区的纬度",
        )
        longitude = st.number_input(
            "经度",
            min_value=-125.0,
            max_value=-114.0,
            value=-118.0,
            step=0.01,
            help="该街区的经度",
        )

    # 构造DataFrame（确保特征顺序与训练一致）
    input_data = pd.DataFrame(
        [
            [
                med_inc,
                house_age,
                ave_rooms,
                ave_bedrooms,
                population,
                ave_occup,
                latitude,
                longitude,
            ]
        ],
        columns=FEATURE_NAMES,
    )

    return input_data


def display_feature_info():
    """显示特征工程信息"""
    with st.expander("📊 模型特征工程说明", expanded=False):
        st.markdown("""
        #### 🔧 自动衍生特征（模型内部处理）

        **组合特征:**
        - `rooms_per_household`: 每户房间数
        - `bedrooms_per_room`: 每房间卧室占比
        - `population_per_household`: 每户人口数

        **空间地理特征:**
        - 到旧金山、洛杉矶、圣地亚哥、圣何塞的球面距离
        - 到最近主要城市的距离
        - 基于经纬度的空间聚类（8类）

        **分布修正:**
        - 对收入、人口等偏态特征执行Log变换
        - 对所有数值特征进行标准化
        """)


def display_prediction(prediction: float, input_data: pd.DataFrame):
    """显示预测结果"""
    # 预测值是10万美元单位，转换为美元
    price_dollars = prediction[0] * 100000

    st.markdown("### 🎯 预测结果")

    # 主要显示
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="🏡 预估房价",
            value=f"${price_dollars:,.0f}",
            delta=f"${prediction[0]:.2f}（10万美元单位）",
        )

    with col2:
        # 价格区间标签
        if price_dollars < 150000:
            price_level = "💰 经济实惠型"
            color = "green"
        elif price_dollars < 300000:
            price_level = "🏠 中等价位型"
            color = "blue"
        elif price_dollars < 500000:
            price_level = "🏘️ 高端舒适型"
            color = "orange"
        else:
            price_level = "🏰 豪华奢侈型"
            color = "red"

        st.markdown(f"<h3 style='color: {color};'>{price_level}</h3>", unsafe_allow_html=True)

    # 价格可视化
    price_ranges = [0, 150000, 300000, 500000, 1000000]
    price_labels = ["经济", "中等", "高端", "豪华"]

    # 计算当前价格在区间中的位置
    progress = min(1.0, max(0.0, (price_dollars - 50000) / 950000))
    st.progress(progress)

    # 显示输入摘要
    with st.expander("📋 输入特征摘要", expanded=True):
        # 添加衍生特征的简单计算用于显示
        display_df = input_data.copy()
        display_df["rooms_per_household"] = (
            display_df["AveRooms"] / display_df["AveOccup"]
        ).round(2)
        display_df["bedrooms_per_room"] = (
            display_df["AveBedrms"] / display_df["AveRooms"]
        ).round(3)

        # 重命名为中文
        display_df = display_df.rename(
            columns={
                "MedInc": "中位数收入",
                "HouseAge": "房屋年限",
                "AveRooms": "平均房间数",
                "AveBedrms": "平均卧室数",
                "Population": "街区人口",
                "AveOccup": "平均居住人数",
                "Latitude": "纬度",
                "Longitude": "经度",
            }
        )

        st.dataframe(display_df.T, use_container_width=True)


def main():
    """主应用函数"""
    # 标题
    st.title("🏡 加州房价预测系统")
    st.markdown("---")

    # 副标题
    st.markdown(
        """
    <div style='text-align: center; padding: 1rem;'>
        <h4>基于XGBoost的工业级房价预测模型</h4>
        <p style='color: #666;'>输入房屋特征，实时预测加州地区的房屋中位数价格</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # 加载模型
    model = load_model()

    # 创建两栏布局
    col1, col2 = st.columns([1, 2])

    with col1:
        # 显示加州地图示意
        st.image(
            "https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=california%20map%20with%20major%20cities%20san%20francisco%20los%20angeles%20simple%20style&image_size=square",
            caption="加州地区示意图",
            use_container_width=True,
        )

    with col2:
        # 创建输入表单
        input_data = create_input_form()

    # 特征工程说明
    display_feature_info()

    # 预测按钮
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        predict_button = st.button(
            "💰 开始预测",
            type="primary",
            use_container_width=True,
            help="点击按钮进行房价预测",
        )

    if predict_button:
        with st.spinner("🔮 模型正在计算中..."):
            try:
                # 使用完整Pipeline预测（自动处理所有特征工程）
                prediction = model.predict(input_data)

                # 显示结果
                display_prediction(prediction, input_data)

                # 成功提示
                st.toast("✅ 预测完成！", icon="🎉")

            except Exception as e:
                st.error(f"❌ 预测过程出错: {str(e)}")
                st.exception(e)

    # 页脚
    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🚀 基于 Streamlit + XGBoost 构建</p>
        <p style='font-size: 0.8rem;'>模型通过完整Pipeline封装，确保训练与推理一致性</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
