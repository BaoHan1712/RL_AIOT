import pandas as pd

# ============================
# 1. Load dữ liệu gốc
# ============================
df1 = pd.read_csv(r"uwb_data_log_1s.csv")
df2 = pd.read_csv(r"uwb_data_log_1s.csv")
df3 = pd.read_csv(r"uwb_data_log_1s.csv")

# ============================
# 2. Chuẩn hoá cột thời gian
# (phòng trường hợp tên cột khác nhau)
# ============================

# Tự động tìm tên cột time
def normalize_time_column(df):
    for col in df.columns:
        if col.lower() in ["time", "timestamp", "datetime"]:
            df = df.rename(columns={col: "time"})
            df["time"] = pd.to_datetime(df["time"])
            return df
    raise ValueError("Không tìm thấy cột thời gian trong file!")

df1 = normalize_time_column(df1)
df2 = normalize_time_column(df2)
df3 = normalize_time_column(df3)

# ============================
# 3. Gộp 3 file lại
# ============================
data = pd.concat([df1, df2, df3], ignore_index=True)

# ============================
# 4. Sắp xếp theo time
# ============================
data = data.sort_values("time")

# ============================
# 5. Reset lại index
# ============================
data.reset_index(drop=True, inplace=True)

# ============================
# 6. Lưu file merge (nếu cần)
# ============================
data.to_csv("sensor_merged.csv", index=False)
print("Đã tạo file sensor_merged.csv thành công!")
