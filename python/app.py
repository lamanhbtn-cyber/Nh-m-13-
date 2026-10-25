import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np

# ========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN VÀ TẢI MÔ HÌNH
# ========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Vì app.py nằm trong thư mục python,
# còn hai file .pkl nằm ở thư mục cha (Nh-m-13-):
MODEL_DIR = os.path.dirname(BASE_DIR)

@st.cache_resource
def load_models():
    rf_path = os.path.join(MODEL_DIR, "random_forest.pkl")
    gb_path = os.path.join(MODEL_DIR, "gradient_boosting.pkl")

    # In ra để kiểm tra trên giao diện Streamlit
    st.write("Thư mục app.py:", BASE_DIR)
    st.write("Thư mục chứa mô hình:", MODEL_DIR)
    st.write("Đường dẫn Random Forest:", rf_path)
    st.write("Đường dẫn Gradient Boosting:", gb_path)
    st.write("Random Forest tồn tại:", os.path.isfile(rf_path))
    st.write("Gradient Boosting tồn tại:", os.path.isfile(gb_path))

    models = {}
    if os.path.isfile(rf_path):
        models["Random Forest"] = joblib.load(rf_path)
    if os.path.isfile(gb_path):
        models["Gradient Boosting"] = joblib.load(gb_path)
        
    return models

try:
    models = load_models()

    if not models:
        st.error(
            "Không tìm thấy file mô hình. "
            "Hãy kiểm tra vị trí hai file .pkl."
        )
        st.stop()
    else:
        st.success("Đã tải thành công cả hai mô hình!")

except Exception as e:
    st.error(f"Không tải được mô hình: {e}")
    st.stop()

# ========================================
# 2. NHẬP THÔNG TIN HÀNH KHÁCH
# ========================================
st.subheader("📋 Thông tin hành khách")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "Hạng vé",
        [1, 2, 3],
        format_func=lambda x: {
            1: "Hạng 1",
            2: "Hạng 2",
            3: "Hạng 3"
        }[x]
    )

    sex = st.selectbox(
        "Giới tính",
        ["male", "female"],
        format_func=lambda x: {
            "male": "Nam",
            "female": "Nữ"
        }[x]
    )

    age = st.number_input(
        "Tuổi",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=1.0
    )

with col2:
    fare = st.number_input(
        "Giá vé",
        min_value=0.0,
        max_value=1000.0,
        value=32.0,
        step=1.0
    )

    sibsp = st.number_input(
        "Số anh chị em/vợ chồng đi cùng",
        min_value=0,
        max_value=10,
        value=0
    )

    parch = st.number_input(
        "Số cha mẹ/con đi cùng",
        min_value=0,
        max_value=10,
        value=0
    )

embarked = st.selectbox(
    "Cảng lên tàu",
    ["S", "C", "Q"],
    format_func=lambda x: {
        "S": "Southampton",
        "C": "Cherbourg",
        "Q": "Queenstown"
    }[x]
)

title = st.selectbox(
    "Danh xưng",
    ["Mr", "Mrs", "Miss", "Master"],
    format_func=lambda x: {
        "Mr": "Ông",
        "Mrs": "Bà",
        "Miss": "Cô",
        "Master": "Bé trai"
    }[x]
)

# ========================================
# 3. CHUẨN BỊ DỮ LIỆU ĐẦU VÀO
# ========================================
def prepare_input():
    family_size = sibsp + parch + 1

    row = {
        "PassengerId": 0,
        "Pclass": pclass,
        "Sex": 1 if sex == "female" else 0,
        "Age": age,
        "Fare_log": np.log1p(fare),
        "FamilySize": family_size,
        "IsAlone": 1 if family_size == 1 else 0,
        "Mrs": 1 if title == "Mrs" else 0,
        "Embarked_C": 1 if embarked == "C" else 0,
        "Embarked_Q": 1 if embarked == "Q" else 0,
        "Embarked_S": 1 if embarked == "S" else 0,
        "Title_Master": 1 if title == "Master" else 0,
        "Title_Miss": 1 if title == "Miss" else 0,
        "Title_Mr": 1 if title == "Mr" else 0,
        "Title_Mrs": 1 if title == "Mrs" else 0
    }

    return pd.DataFrame([row])

# ========================================
# 4. CHỌN MÔ HÌNH VÀ DỰ ĐOÁN
# ========================================
st.divider()
st.subheader("🤖 Dự đoán bằng mô hình học máy")

model_name = st.radio(
    "Chọn thuật toán",
    list(models.keys()),
    horizontal=True
)

if st.button("🔍 Dự đoán sống sót", use_container_width=True):
    try:
        model = models[model_name]
        input_df = prepare_input()

        expected_features = getattr(
            model, "feature_names_in_", None
        )

        if expected_features is not None:
            missing = set(expected_features) - set(input_df.columns)

            if missing:
                st.error(
                    "Thiếu đặc trưng khi dự đoán: "
                    + ", ".join(sorted(missing))
                )
                st.stop()

            input_df = input_df[list(expected_features)]

        prediction = int(model.predict(input_df)[0])

        st.divider()
        st.subheader("📊 Kết quả")

        if prediction == 1:
            st.success("Dự đoán: Hành khách có khả năng sống sót.")
        else:
            st.error("Dự đoán: Hành khách có khả năng không sống sót.")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)

            if 1 in classes:
                survival_probability = probabilities[classes.index(1)]

                st.metric(
                    "Xác suất sống sót theo mô hình",
                    f"{survival_probability:.2%}"
                )

                st.progress(float(survival_probability))

        st.caption(
            "Đây là kết quả dự đoán của mô hình, "
            "không phải xác suất chắc chắn trong thực tế."
        )

    except Exception as e:
        st.error(f"Dự đoán thất bại: {e}")

# ========================================
# 5. THÔNG TIN DỰ ÁN
# ========================================
with st.expander("ℹ️ Giới thiệu dự án"):
    st.write("""
    **Đề tài:** Ứng dụng Ensemble Learning vào bài toán
    dự đoán sống sót thảm họa Titanic.

    **Các thuật toán:**
    - Random Forest
    - Gradient Boosting

    Ứng dụng nhận thông tin hành khách và sử dụng mô hình
    đã huấn luyện để đưa ra dự đoán.
    """)