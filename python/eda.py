
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# 1. KHAI BÁO ĐƯỜNG DẪN VÀ ĐỌC DỮ LIỆU
# ==================================================

SCRIPT_DIR = Path(__file__).resolve().parent

DATA_PATH = SCRIPT_DIR / "processed_train.csv"
OUTPUT_DIR = SCRIPT_DIR / "eda_results"

OUTPUT_DIR.mkdir(exist_ok=True)

# Đọc dữ liệu
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("PHÂN TÍCH KHÁM PHÁ DỮ LIỆU TITANIC (EDA)")
print("=" * 60)


# ==================================================
# 2. KIỂM TRA TỔNG QUAN DỮ LIỆU
# ==================================================

print("\n1. KÍCH THƯỚC DỮ LIỆU")
print("Số dòng:", df.shape[0])
print("Số cột:", df.shape[1])

print("\n2. THÔNG TIN CÁC CỘT")
print(df.info())

print("\n3. THỐNG KÊ MÔ TẢ")
print(df.describe().round(2))

print("\n4. KIỂM TRA GIÁ TRỊ THIẾU")
print(df.isnull().sum())


# Kiểm tra các cột cần thiết
required_columns = [
    "Survived",
    "Pclass",
    "Sex",
    "Age",
    "Fare_log",
    "FamilySize",
    "IsAlone",
    "Embarked_C",
    "Embarked_Q",
    "Embarked_S"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Thiếu các cột cần thiết trong dữ liệu: {missing_columns}"
    )


# ==================================================
# 3. PHÂN TÍCH BIẾN SURVIVED
# ==================================================

print("\n5. PHÂN BỐ SỐ LƯỢNG HÀNH KHÁCH")

survival_count = df["Survived"].value_counts().sort_index()

survival_rate = (
    df["Survived"].value_counts(normalize=True).sort_index() * 100
)

survival_table = pd.DataFrame({
    "Số lượng": survival_count,
    "Tỷ lệ (%)": survival_rate
})

survival_table.index = ["Không sống sót", "Sống sót"]

print(survival_table.round(2))

plt.figure(figsize=(8, 5))

plt.bar(
    ["Không sống sót", "Sống sót"],
    survival_count.values
)

plt.title("Phân bố số lượng hành khách theo tình trạng sống sót")
plt.xlabel("Tình trạng sống sót")
plt.ylabel("Số lượng hành khách")

for i, value in enumerate(survival_count.values):
    plt.text(i, value + 5, str(value), ha="center")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hinh_2_1_survived.png", dpi=300)
plt.close()


# ==================================================
# 4. PHÂN TÍCH GIỚI TÍNH SEX
# ==================================================

print("\n6. TỶ LỆ SỐNG SÓT THEO GIỚI TÍNH")

sex_table = df.groupby("Sex")["Survived"].agg(
    ["count", "sum", "mean"]
)

sex_table.columns = [
    "Tổng số hành khách",
    "Số người sống sót",
    "Tỷ lệ sống sót"
]

sex_table["Tỷ lệ sống sót"] *= 100

sex_table.index = ["Nữ" if x == 0 else "Nam" for x in sex_table.index]

print(sex_table.round(2))

sex_rate = df.groupby("Sex")["Survived"].mean() * 100

plt.figure(figsize=(8, 5))

labels = [
    "Nữ" if x == 0 else "Nam"
    for x in sex_rate.index
]

plt.bar(labels, sex_rate.values)

plt.title("Tỷ lệ sống sót theo giới tính")
plt.xlabel("Giới tính")
plt.ylabel("Tỷ lệ sống sót (%)")

for i, value in enumerate(sex_rate.values):
    plt.text(i, value + 1, f"{value:.2f}%", ha="center")

plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hinh_2_2_sex.png", dpi=300)
plt.close()


# ==================================================
# 5. PHÂN TÍCH HẠNG VÉ PCLASS
# ==================================================

print("\n7. TỶ LỆ SỐNG SÓT THEO HẠNG VÉ")

pclass_table = df.groupby("Pclass")["Survived"].agg(
    ["count", "sum", "mean"]
)

pclass_table.columns = [
    "Tổng số hành khách",
    "Số người sống sót",
    "Tỷ lệ sống sót"
]

pclass_table["Tỷ lệ sống sót"] *= 100

print(pclass_table.round(2))

pclass_rate = df.groupby("Pclass")["Survived"].mean() * 100

plt.figure(figsize=(8, 5))

plt.bar(
    pclass_rate.index.astype(str),
    pclass_rate.values
)

plt.title("Tỷ lệ sống sót theo hạng vé")
plt.xlabel("Hạng vé")
plt.ylabel("Tỷ lệ sống sót (%)")

for i, value in enumerate(pclass_rate.values):
    plt.text(i, value + 1, f"{value:.2f}%", ha="center")

plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hinh_2_3_pclass.png", dpi=300)
plt.close()


# ==================================================
# 6. PHÂN TÍCH ĐỘ TUỔI AGE
# ==================================================

print("\n8. THỐNG KÊ ĐỘ TUỔI")

print(df["Age"].describe().round(2))

# Chia nhóm độ tuổi
age_bins = [0, 12, 18, 30, 45, 60, 100]

age_labels = [
    "0-12",
    "13-18",
    "19-30",
    "31-45",
    "46-60",
    "Trên 60"
]

df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=age_bins,
    labels=age_labels,
    include_lowest=True
)

age_table = df.groupby(
    "AgeGroup",
    observed=False
)["Survived"].agg(
    ["count", "sum", "mean"]
)

age_table.columns = [
    "Tổng số hành khách",
    "Số người sống sót",
    "Tỷ lệ sống sót"
]

age_table["Tỷ lệ sống sót"] *= 100

print(age_table.round(2))

# Biểu đồ phân bố độ tuổi
plt.figure(figsize=(9, 5))

plt.hist(
    df["Age"],
    bins=20,
    edgecolor="black"
)

plt.title("Phân bố độ tuổi hành khách Titanic")
plt.xlabel("Độ tuổi")
plt.ylabel("Số lượng hành khách")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hinh_2_4_age.png", dpi=300)
plt.close()


# ==================================================
# 7. PHÂN TÍCH QUY MÔ GIA ĐÌNH
# ==================================================

print("\n9. TỶ LỆ SỐNG SÓT THEO QUY MÔ GIA ĐÌNH")

family_table = df.groupby("FamilySize")["Survived"].agg(
    ["count", "sum", "mean"]
)

family_table.columns = [
    "Tổng số hành khách",
    "Số người sống sót",
    "Tỷ lệ sống sót"
]

family_table["Tỷ lệ sống sót"] *= 100

print(family_table.round(2))

# Phân tích đi một mình hay đi cùng gia đình
alone_table = df.groupby("IsAlone")["Survived"].agg(
    ["count", "sum", "mean"]
)

alone_table.columns = [
    "Tổng số hành khách",
    "Số người sống sót",
    "Tỷ lệ sống sót"
]

alone_table["Tỷ lệ sống sót"] *= 100

alone_table.index = [
    "Đi cùng gia đình" if x == 0 else "Đi một mình"
    for x in alone_table.index
]

print("\nTỷ lệ sống sót theo tình trạng đi một mình:")
print(alone_table.round(2))

alone_rate = df.groupby("IsAlone")["Survived"].mean() * 100

plt.figure(figsize=(8, 5))

labels = [
    "Đi cùng gia đình" if x == 0 else "Đi một mình"
    for x in alone_rate.index
]

plt.bar(labels, alone_rate.values)

plt.title("Tỷ lệ sống sót theo tình trạng đi một mình")
plt.xlabel("Tình trạng")
plt.ylabel("Tỷ lệ sống sót (%)")

for i, value in enumerate(alone_rate.values):
    plt.text(i, value + 1, f"{value:.2f}%", ha="center")

plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hinh_2_5_family.png", dpi=300)
plt.close()


# ==================================================
# 8. PHÂN TÍCH GIÁ VÉ FARE_LOG
# ==================================================

print("\n10. THỐNG KÊ GIÁ VÉ SAU BIẾN ĐỔI LOG")

fare_table = df.groupby("Survived")["Fare_log"].agg(
    ["count", "mean", "median", "min", "max"]
)

fare_table.index = ["Không sống sót", "Sống sót"]

print(fare_table.round(2))

plt.figure(figsize=(8, 5))

for survived, label in [(0, "Không sống sót"), (1, "Sống sót")]:
    plt.hist(
        df.loc[df["Survived"] == survived, "Fare_log"],
        bins=20,
        alpha=0.5,
        label=label
    )

plt.title("Phân bố Fare_log theo tình trạng sống sót")
plt.xlabel("Fare_log")
plt.ylabel("Số lượng hành khách")
plt.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hinh_2_6_fare.png", dpi=300)
plt.close()


# ==================================================
# 9. PHÂN TÍCH CẢNG LÊN TÀU EMBARKED
# ==================================================

print("\n11. TỶ LỆ SỐNG SÓT THEO CẢNG LÊN TÀU")

embarked_columns = {
    "C": "Embarked_C",
    "Q": "Embarked_Q",
    "S": "Embarked_S"
}

embarked_results = []

for port, column in embarked_columns.items():

    total = df[column].sum()

    survived = df.loc[
        df[column] == 1, "Survived"
    ].sum()

    rate = survived / total * 100 if total > 0 else 0

    embarked_results.append({
        "Cảng": port,
        "Tổng số hành khách": total,
        "Số người sống sót": survived,
        "Tỷ lệ sống sót (%)": rate
    })

embarked_table = pd.DataFrame(embarked_results)

print(embarked_table.round(2))

plt.figure(figsize=(8, 5))

plt.bar(
    embarked_table["Cảng"],
    embarked_table["Tỷ lệ sống sót (%)"]
)

plt.title("Tỷ lệ sống sót theo cảng lên tàu")
plt.xlabel("Cảng lên tàu")
plt.ylabel("Tỷ lệ sống sót (%)")

for i, value in enumerate(embarked_table["Tỷ lệ sống sót (%)"]):
    plt.text(i, value + 1, f"{value:.2f}%", ha="center")

plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hinh_2_7_embarked.png", dpi=300)
plt.close()


# ==================================================
# 10. LƯU CÁC BẢNG THỐNG KÊ
# ==================================================

survival_table.to_csv(
    OUTPUT_DIR / "thong_ke_survived.csv",
    encoding="utf-8-sig"
)

sex_table.to_csv(
    OUTPUT_DIR / "thong_ke_sex.csv",
    encoding="utf-8-sig"
)

pclass_table.to_csv(
    OUTPUT_DIR / "thong_ke_pclass.csv",
    encoding="utf-8-sig"
)

age_table.to_csv(
    OUTPUT_DIR / "thong_ke_age.csv",
    encoding="utf-8-sig"
)

family_table.to_csv(
    OUTPUT_DIR / "thong_ke_family_size.csv",
    encoding="utf-8-sig"
)

alone_table.to_csv(
    OUTPUT_DIR / "thong_ke_is_alone.csv",
    encoding="utf-8-sig"
)

fare_table.to_csv(
    OUTPUT_DIR / "thong_ke_fare.csv",
    encoding="utf-8-sig"
)

embarked_table.to_csv(
    OUTPUT_DIR / "thong_ke_embarked.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==================================================
# 11. KẾT THÚC
# ==================================================

print("\n" + "=" * 60)
print("ĐÃ HOÀN THÀNH PHÂN TÍCH EDA!")
print("Các biểu đồ và bảng thống kê được lưu tại:")
print(OUTPUT_DIR)
print("=" * 60)
