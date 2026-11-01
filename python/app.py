from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# 1. CẤU HÌNH TRANG
# =========================================================
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

FEATURE_COLUMNS = [
    "Pclass",
    "Sex",
    "Age",
    "Fare_log",
    "FamilySize",
    "IsAlone",
    "Embarked_C",
    "Embarked_Q",
    "Embarked_S",
    "Title_Master",
    "Title_Miss",
    "Title_Mr",
    "Title_Mrs",
]

MODEL_FILES = {
    "Random Forest": "random_forest.pkl",
    "Gradient Boosting": "gradient_boosting.pkl",
}


# =========================================================
# 2. GIỚI THIỆU
# =========================================================
st.title("🚢 DỰ ĐOÁN KHẢ NĂNG SỐNG SÓT – TITANIC")
st.caption(
    "Đồ án học máy tích hợp (Ensemble Learning) | "
    "So sánh Random Forest và Gradient Boosting"
)

with st.expander("📘 Giới thiệu đề tài", expanded=False):
    st.write(
        "Ứng dụng sử dụng hai thuật toán học máy Random Forest và "
        "Gradient Boosting để dự đoán khả năng sống sót của hành khách "
        "trên tàu Titanic dựa trên thông tin cá nhân, vé và gia đình."
    )
    st.info(
        "Kết quả là dự đoán của mô hình trên dữ liệu đã huấn luyện, "
        "không phải kết luận chắc chắn về một người."
    )


# =========================================================
# 3. TẢI MÔ HÌNH
# =========================================================
def find_model_file(filename):
    possible_paths = [
        PROJECT_DIR / filename,
        APP_DIR / filename,
    ]

    for path in possible_paths:
        if path.is_file():
            return path

    return None


@st.cache_resource
def load_models():
    loaded = {}
    errors = {}

    for model_name, filename in MODEL_FILES.items():
        model_path = find_model_file(filename)

        if model_path is None:
            errors[model_name] = (
                f"Không tìm thấy {filename}. Hãy đặt file mô hình "
                "trong thư mục gốc dự án hoặc thư mục python."
            )
            continue

        try:
            loaded[model_name] = joblib.load(model_path)
        except Exception as exc:
            errors[model_name] = f"Không thể tải mô hình: {exc}"

    return loaded, errors


models, model_errors = load_models()

with st.sidebar:
    st.header("⚙️ Trạng thái hệ thống")

    for model_name in MODEL_FILES:
        if model_name in models:
            st.success(f"{model_name}: đã tải")
        else:
            st.error(f"{model_name}: chưa sẵn sàng")

    if model_errors:
        with st.expander("Chi tiết lỗi tải mô hình"):
            for model_name, message in model_errors.items():
                st.write(f"**{model_name}:** {message}")

    st.divider()
    st.markdown("**Thông tin đầu vào**")
    st.caption("Ứng dụng sử dụng 13 đặc trưng đã thống nhất với hai mô hình.")


# =========================================================
# 4. CHUẨN BỊ DỮ LIỆU ĐẦU VÀO
# =========================================================
def prepare_input(
    pclass,
    sex,
    age,
    fare,
    sibsp,
    parch,
    embarked,
    title,
):
    family_size = sibsp + parch + 1

    row = {
        "Pclass": int(pclass),
        "Sex": 1 if sex == "male" else 0,
        "Age": float(age),
        "Fare_log": float(np.log1p(fare)),
        "FamilySize": int(family_size),
        "IsAlone": int(family_size == 1),
        "Embarked_C": int(embarked == "C"),
        "Embarked_Q": int(embarked == "Q"),
        "Embarked_S": int(embarked == "S"),
        "Title_Master": int(title == "Master"),
        "Title_Miss": int(title == "Miss"),
        "Title_Mr": int(title == "Mr"),
        "Title_Mrs": int(title == "Mrs"),
    }

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def check_model_features(model, model_name):
    expected = getattr(model, "feature_names_in_", None)

    if expected is None:
        return None

    expected = list(expected)

    if expected != FEATURE_COLUMNS:
        missing = [c for c in FEATURE_COLUMNS if c not in expected]
        extra = [c for c in expected if c not in FEATURE_COLUMNS]

        details = []
        if missing:
            details.append("Thiếu: " + ", ".join(missing))
        if extra:
            details.append("Thừa: " + ", ".join(extra))

        return (
            f"{model_name} không khớp với 13 đặc trưng giao diện. "
            + " | ".join(details)
            + ". Hãy huấn luyện lại mô hình bằng file thuật toán mới."
        )

    return None


def predict_one(model, input_df):
    prediction = int(model.predict(input_df)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)

        if 1 in classes:
            probability = float(probabilities[classes.index(1)])

    return prediction, probability


# =========================================================
# 5. FORM NHẬP THÔNG TIN
# =========================================================
st.subheader("🧾 Nhập thông tin hành khách")
st.write("Điền thông tin rồi nhấn nút dự đoán.")

with st.form("passenger_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Thông tin cá nhân**")

        sex = st.selectbox(
            "Giới tính",
            options=["male", "female"],
            format_func=lambda x: "Nam" if x == "male" else "Nữ",
        )

        age = st.number_input(
            "Tuổi",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=1.0,
        )

        title_options = ["Mr", "Master"] if sex == "male" else ["Mrs", "Miss"]

        title = st.selectbox(
            "Danh xưng",
            options=title_options,
            format_func=lambda x: {
                "Mr": "Mr – Nam trưởng thành",
                "Master": "Master – Bé trai",
                "Mrs": "Mrs – Nữ đã kết hôn",
                "Miss": "Miss – Nữ chưa kết hôn",
            }[x],
        )

    with col2:
        st.markdown("**Thông tin vé**")

        pclass = st.selectbox(
            "Hạng vé",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "Hạng 1",
                2: "Hạng 2",
                3: "Hạng 3",
            }[x],
        )

        fare = st.number_input(
            "Giá vé (Fare)",
            min_value=0.0,
            max_value=1000.0,
            value=32.0,
            step=1.0,
            help="Nhập giá vé gốc; ứng dụng tự tính log1p(Fare).",
        )

        embarked = st.selectbox(
            "Cảng lên tàu",
            options=["S", "C", "Q"],
            format_func=lambda x: {
                "S": "Southampton (S)",
                "C": "Cherbourg (C)",
                "Q": "Queenstown (Q)",
            }[x],
        )

    with col3:
        st.markdown("**Thông tin gia đình**")

        sibsp = st.number_input(
            "Số anh/chị/em hoặc vợ/chồng đi cùng",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
        )

        parch = st.number_input(
            "Số cha/mẹ hoặc con đi cùng",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
        )

        family_size = int(sibsp + parch + 1)
        st.metric("Tổng quy mô gia đình", family_size)
        st.caption(
            "FamilySize = SibSp + Parch + 1. "
            "IsAlone = 1 nếu hành khách đi một mình."
        )

    submitted = st.form_submit_button(
        "🔍 DỰ ĐOÁN KHẢ NĂNG SỐNG SÓT",
        use_container_width=True,
        type="primary",
    )


# =========================================================
# 6. DỰ ĐOÁN VÀ HIỂN THỊ KẾT QUẢ
# =========================================================
if submitted:
    if not models:
        st.error(
            "Chưa tải được mô hình. Hãy kiểm tra hai file "
            "random_forest.pkl và gradient_boosting.pkl."
        )
    else:
        input_df = prepare_input(
            pclass=pclass,
            sex=sex,
            age=age,
            fare=fare,
            sibsp=int(sibsp),
            parch=int(parch),
            embarked=embarked,
            title=title,
        )

        errors = [
            error
            for model_name, model in models.items()
            if (error := check_model_features(model, model_name))
        ]

        if errors:
            st.error("Mô hình chưa tương thích với dữ liệu đầu vào.")
            for error in errors:
                st.warning(error)
        else:
            st.divider()
            st.subheader("📊 Kết quả dự đoán")

            result_columns = st.columns(len(models))

            for column, (model_name, model) in zip(result_columns, models.items()):
                with column:
                    try:
                        prediction, probability = predict_one(model, input_df)

                        st.markdown(f"### {model_name}")

                        if prediction == 1:
                            st.success("DỰ ĐOÁN: CÓ KHẢ NĂNG SỐNG SÓT")
                        else:
                            st.error("DỰ ĐOÁN: KHÔNG CÓ KHẢ NĂNG SỐNG SÓT")

                        if probability is not None:
                            st.metric(
                                "Xác suất dự đoán sống sót",
                                f"{probability * 100:.2f}%",
                            )
                            st.progress(min(max(probability, 0.0), 1.0))
                        else:
                            st.info("Mô hình không hỗ trợ predict_proba().")

                    except Exception as exc:
                        st.error(f"Lỗi khi dự đoán bằng {model_name}: {exc}")

            with st.expander("🔎 Xem dữ liệu sau khi mã hóa"):
                st.dataframe(input_df, use_container_width=True)
                st.caption("13 đặc trưng được đưa vào mô hình; PassengerId đã bị loại bỏ.")


# =========================================================
# 7. CHÂN TRANG
# =========================================================
st.divider()
st.caption(
    "Đồ án học máy tích hợp (Ensemble Learning) – Titanic Survival Prediction. "
    "Kết quả phụ thuộc vào dữ liệu và quá trình huấn luyện mô hình."
)
