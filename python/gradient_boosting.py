import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
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
print("\nSố mẫu tập huấn luyện:", X_train.shape[0])
print("Số mẫu tập kiểm thử:", X_test.shape[0])

# 4. Khởi tạo mô hình Gradient Boosting
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
# 5. Huấn luyện mô hình
model.fit(X_train, y_train)
print("\nHuấn luyện Gradient Boosting hoàn tất!")

# 6. Dự đoán trên tập kiểm thử
y_pred = model.predict(X_test)

# 7. Đánh giá mô hình
print("\n===== KẾT QUẢ ĐÁNH GIÁ GRADIENT BOOSTING =====")
print("Accuracy :",round(accuracy_score(y_test, y_pred), 4))
print("Precision:",round(precision_score( y_test, y_pred,zero_division=0),4))
print("Recall   :",round(recall_score(y_test,y_pred,zero_division=0),4))
print( "F1-score :",round(f1_score(y_test,y_pred, zero_division=0), 4))

# 8. Báo cáo phân loại
print("\n===== BÁO CÁO PHÂN LOẠI =====")
print(classification_report(y_test,y_pred,zero_division=0)
    )
# 9. Ma trận nhầm lẫn
print("===== MA TRẬN NHẦM LẪN =====")
print(confusion_matrix(y_test, y_pred))

# 10. Lưu mô hình
joblib.dump(model,"gradient_boosting.pkl")
print("\nĐã lưu mô hình vào gradient_boosting.pkl")