
import numpy as np
import pandas as pd
import joblib


class TitanicPreprocessor:
    MAIN_TITLES = ["Master", "Miss", "Mr", "Mrs"]
    EMBARKED_VALUES = ["C", "Q", "S"]

    def __init__(self, extra_features=False):
      
        self.extra_features = extra_features
        self.is_fitted_ = False

    # ------------------------------------------------------------------ helpers
    @classmethod
    def _extract_title(cls, df):
        title = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
        title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
        # Danh xưng hiếm (Dr, Rev, Col, Countess...) -> Mr / Mrs theo giới tính
        rare = ~title.isin(cls.MAIN_TITLES)
        fallback = df["Sex"].map({"male": "Mr", "female": "Mrs"})
        return title.where(~rare, fallback)

    # ---------------------------------------------------------------------- fit
    def fit(self, df):
        title = self._extract_title(df)

        self.age_by_title_ = df["Age"].groupby(title).median().to_dict()
        self.age_median_ = df["Age"].median()            # dự phòng nếu danh xưng lạ
        self.embarked_mode_ = df["Embarked"].mode()[0]
        self.fare_median_ = df["Fare"].median()

        if self.extra_features:
            # Ngưỡng chia FareBand lấy từ train (tứ phân vị), dùng lại cho test
            self.fare_bins_ = df["Fare"].quantile([0.25, 0.5, 0.75]).to_numpy()

        self.is_fitted_ = True
        self.columns_ = self.transform(df).columns.tolist()
        return self

    # ---------------------------------------------------------------- transform
    def transform(self, df):
        if not self.is_fitted_ and not hasattr(self, "age_by_title_"):
            raise RuntimeError("Cần gọi fit() trước khi transform().")

        df = df.copy()
        if "PassengerId" in df.columns:
            df = df.set_index("PassengerId")

        # 1) Danh xưng
        df["Title"] = self._extract_title(df)

        # 2) Điền giá trị thiếu bằng thống kê của TRAIN
        df["Age"] = df["Age"].fillna(df["Title"].map(self.age_by_title_))
        df["Age"] = df["Age"].fillna(self.age_median_)
        df["Embarked"] = df["Embarked"].fillna(self.embarked_mode_)
        df["Fare"] = df["Fare"].fillna(self.fare_median_)

        # 3) Tạo đặc trưng
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
        df["Sex"] = (df["Sex"] == "male").astype(int)       # 1 = nam, 0 = nữ
        df["Fare_log"] = np.log1p(df["Fare"])

        feature_cols = ["Pclass", "Sex", "Age", "Fare_log", "FamilySize", "IsAlone"]

        if self.extra_features:
            df["Age*Class"] = df["Age"] * df["Pclass"]
            df["Age*Fare"] = df["Age"] * df["Fare"]
            df["AgeBand"] = pd.cut(df["Age"], bins=[0, 12, 20, 40, 60, np.inf],
                                   labels=False, include_lowest=True)
            df["FareBand"] = np.searchsorted(self.fare_bins_, df["Fare"], side="right")
            feature_cols += ["Age*Class", "Age*Fare", "AgeBand", "FareBand"]

        # 4) One-hot với danh sách giá trị CỐ ĐỊNH -> số cột không bao giờ đổi
        df["Embarked"] = pd.Categorical(df["Embarked"], categories=self.EMBARKED_VALUES)
        df["Title"] = pd.Categorical(df["Title"], categories=self.MAIN_TITLES)

        out = pd.get_dummies(df[feature_cols + ["Embarked", "Title"]],
                             columns=["Embarked", "Title"], dtype=int)

        # 5) Giữ đúng thứ tự cột như lúc fit (nếu đã có)
        if hasattr(self, "columns_"):
            out = out.reindex(columns=self.columns_, fill_value=0)
        return out

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    # ---------------------------------------------------------------- lưu / nạp
    def save(self, path="preprocessor.pkl"):
        joblib.dump(self, path)

    @staticmethod
    def load(path="preprocessor.pkl"):
        return joblib.load(path)


# ---------------------------------------------------------------------- chạy file
if __name__ == "__main__":
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    # Tách nhãn ra TRƯỚC khi tiền xử lý
    y = train.set_index("PassengerId")["Survived"]
    train_features = train.drop(columns="Survived")

    prep = TitanicPreprocessor(extra_features=False)
    X = prep.fit_transform(train_features)      # học thống kê từ train
    X_test = prep.transform(test)               # dùng lại thống kê của train

    print("Train:", X.shape, "| Test:", X_test.shape)
    print("Giá trị thiếu còn lại:", int(X.isna().sum().sum()), "/", int(X_test.isna().sum().sum()))
    print("Các cột:", X.columns.tolist())

    X.join(y).to_csv("processed_train.csv")     # có thêm cột Survived để tiện đọc
    X_test.to_csv("processed_test.csv")
    prep.save("preprocessor.pkl")
    print("Đã lưu processed_train.csv, processed_test.csv, preprocessor.pkl")
