
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# =====================================================
# 1. ĐƯỜNG DẪN
# =====================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DATA_PATH = SCRIPT_DIR / "processed_train.csv"
MODEL_PATH = PROJECT_DIR / "gradient_boosting.pkl"


# =====================================================
# 2. DANH SÁCH 13 ĐẶC TRƯNG
# =====================================================
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
    "Title_Mrs"
]


# =====================================================
# 3. ĐỌC DỮ LIỆU
# =====================================================
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy dữ liệu: {DATA_PATH}"
    )

# Cột đầu tiên trong CSV là PassengerId (index),
# không đưa cột này vào huấn luyện.
df = pd.read_csv(DATA_PATH, index_col=0)

print("Kích thước dữ liệu ban đầu:", df.shape)

if "Survived" not in df.columns:
    raise ValueError("Không tìm thấy cột Survived.")

# Tách đặc trưng và nhãn
X = df.drop(columns=["Survived"])
y = df["Survived"]

# Kiểm tra và sắp xếp đúng 13 đặc trưng
missing_columns = [
    col for col in FEATURE_COLUMNS if col not in X.columns
]

if missing_columns:
    raise ValueError(
        f"Dữ liệu thiếu các cột: {missing_columns}"
    )

# Cảnh báo cột thừa (nếu có)
extra_columns = [col for col in X.columns if col not in FEATURE_COLUMNS]
if extra_columns:
    print(f"⚠️ Cột thừa sẽ bị bỏ: {extra_columns}")

X = X[FEATURE_COLUMNS]

print("Số đặc trưng sử dụng:", X.shape[1])
print("Các đặc trưng:", X.columns.tolist())

print("\nPhân bố nhãn:")
print(y.value_counts().sort_index())


# =====================================================
# 4. CHIA DỮ LIỆU TRAIN VÀ TEST
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nKích thước tập huấn luyện:", X_train.shape)
print("Kích thước tập kiểm tra:", X_test.shape)


# =====================================================
# 5. KHỞI TẠO VÀ HUẤN LUYỆN GRADIENT BOOSTING
# =====================================================
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

print("\nĐang huấn luyện Gradient Boosting...")

model.fit(X_train, y_train)


# =====================================================
# 6. DỰ ĐOÁN
# =====================================================
y_pred = model.predict(X_test)


# =====================================================
# 7. ĐÁNH GIÁ MÔ HÌNH
# =====================================================
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

print("\n========== KẾT QUẢ GRADIENT BOOSTING (Train/Test Split) ==========")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# =====================================================
# 8. CROSS-VALIDATION (đánh giá vững hơn)
# =====================================================
print("\n========== CROSS-VALIDATION (5-Fold) ==========")

cv_accuracy = cross_val_score(model, X, y, cv=5, scoring="accuracy")
cv_f1 = cross_val_score(model, X, y, cv=5, scoring="f1")

print(f"CV Accuracy : {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
print(f"CV F1-score : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
print(f"Từng fold (Accuracy): {[round(s, 4) for s in cv_accuracy]}")


# =====================================================
# 9. HUẤN LUYỆN LẠI TRÊN TOÀN BỘ DỮ LIỆU VÀ LƯU
# =====================================================
print("\nHuấn luyện lại trên toàn bộ dữ liệu trước khi lưu...")
model.fit(X, y)

joblib.dump(model, MODEL_PATH)

print(f"Đã lưu mô hình tại: {MODEL_PATH}")
print(f"Mô hình được train trên {X.shape[0]} mẫu (toàn bộ dữ liệu).")

print("\nHoàn thành huấn luyện Gradient Boosting!")
