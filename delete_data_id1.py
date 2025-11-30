import pandas as pd

# đọc dữ liệu từ file CSV
df = pd.read_csv("uwb_data_adjusted.csv")

# loại bỏ các dòng có id = 1
df_filtered = df[df['id'] != 1].reset_index(drop=True)

# lưu lại file mới nếu muốn
df_filtered.to_csv("sensor_only_id2.csv", index=False)

print("Dữ liệu sau khi lọc:")
print(df_filtered)
