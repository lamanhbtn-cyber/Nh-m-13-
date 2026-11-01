
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


# =========================================================
# 1. ĐƯỜNG DẪN VÀ CẤU HÌNH
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent if BASE_DIR.name.lower() == "python" else BASE_DIR

DATA_CANDIDATES = [
    BASE_DIR / "processed_train.csv",
    BASE_DIR / "python" / "processed_train.csv",
    BASE_DIR.parent / "processed_train.csv",
]

MODEL_PATH = PROJECT_DIR / "random_forest.pkl"

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


# =========================================================
# 2. ĐỌC VÀ KIỂM TRA DỮ LIỆU
# =========================================================
def find_data_file():
    for path in DATA_CANDIDATES:
        if path.is_file():
            return path

    raise FileNotFoundError(
        "Không tìm thấy processed_train.csv. "
        "Hãy đặt file này cùng thư mục với random_forest.py "
        "hoặc trong thư mục dự án."
    )


def load_dataset():
    data_path = find_data_file()
    df = pd.read_csv(data_path)

    print("Đường dẫn dữ liệu:", data_path)
    print("Kích thước dữ liệu ban đầu:", df.shape)

    if "Survived" not in df.columns:
        raise ValueError("Dữ liệu không có cột mục tiêu 'Survived'.")

    # Loại bỏ cột định danh, không sử dụng để dự đoán.
    df = df.drop(
        columns=["PassengerId", "Unnamed: 0"],
        errors="ignore",
    )

    missing_columns = [
        column for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    extra_columns = [
        column for column in df.columns
        if column not in FEATURE_COLUMNS + ["Survived"]
    ]

    if missing_columns:
        raise ValueError(
            "Dữ liệu đang thiếu các cột đặc trưng: "
            + ", ".join(missing_columns)
        )

    if extra_columns:
        print("Các cột không sử dụng:", extra_columns)

    # Lấy đúng 13 đặc trưng theo thứ tự của app.py.
    X = df[FEATURE_COLUMNS].copy()
    y = df["Survived"].astype(int).copy()

    if X.isnull().any().any():
        raise ValueError(
            "Dữ liệu X vẫn còn giá trị thiếu. "
            "Hãy kiểm tra lại bước tiền xử lý."
        )

    if y.isnull().any():
        raise ValueError("Cột Survived có giá trị thiếu.")

    print("Số đặc trưng sử dụng:", len(FEATURE_COLUMNS))
    print("Các đặc trưng:", X.columns.tolist())

    print("\nPhân bố nhãn:")
    print(y.value_counts().sort_index())

    return X, y


# =========================================================
# 3. HUẤN LUYỆN VÀ ĐÁNH GIÁ RANDOM FOREST
# =========================================================
def main():
    X, y = load_dataset()

    # Chia dữ liệu: 80% huấn luyện, 20% kiểm tra.
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Khởi tạo mô hình Random Forest.
    model = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        random_state=42,
        class_weight=None,
        n_jobs=-1,
    )

    print("\nĐang huấn luyện Random Forest...")
    model.fit(X_train, y_train)

    # Dự đoán trên tập kiểm tra.
    y_pred = model.predict(X_valid)

    # Tính các chỉ số đánh giá.
    accuracy = accuracy_score(y_valid, y_pred)
    precision = precision_score(y_valid, y_pred, zero_division=0)
    recall = recall_score(y_valid, y_pred, zero_division=0)
    f1 = f1_score(y_valid, y_pred, zero_division=0)

    print("\n========== KẾT QUẢ RANDOM FOREST ==========")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_valid, y_pred, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_valid, y_pred))

    # Lưu mô hình.
    joblib.dump(model, MODEL_PATH)

    print("\nĐã lưu mô hình tại:", MODEL_PATH)


if __name__ == "__main__":
    main()
