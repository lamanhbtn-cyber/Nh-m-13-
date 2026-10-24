
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 1. Đọc dữ liệu đã tiền xử lý
df = pd.read_csv("python/processed_train.csv")

print("Kích thước dữ liệu:", df.shape)
print("Các cột:", df.columns.tolist())

# 2. Tách đặc trưng X và nhãn y
X = df.drop(columns=["Survived"])
y = df["Survived"]

# 3. Chia dữ liệu huấn luyện và kiểm thử
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Khởi tạo mô hình Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=None,
    random_state=42,
    class_weight=None
)

# 5. Huấn luyện mô hình
model.fit(X_train, y_train)

print("\nHuấn luyện Random Forest hoàn tất!")

# 6. Dự đoán trên tập kiểm thử
y_pred = model.predict(X_test)

# 7. Đánh giá mô hình
print("\n===== KẾT QUẢ ĐÁNH GIÁ =====")
print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(
    precision_score(y_test, y_pred, zero_division=0), 4
))
print("Recall   :", round(
    recall_score(y_test, y_pred, zero_division=0), 4
))
print("F1-score :", round(
    f1_score(y_test, y_pred, zero_division=0), 4
))

print("\nBáo cáo phân loại:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Ma trận nhầm lẫn:")
print(confusion_matrix(y_test, y_pred))

# 8. Lưu mô hình
joblib.dump(model, "random_forest.pkl")

print("\nĐã lưu mô hình vào random_forest.pkl")
