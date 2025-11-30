import pandas as pd

# Đọc file CSV
df = pd.read_csv('uwb_data_log_1s.csv')

# Tạo copy để chỉnh id=1
df_adjusted = df.copy()

# Chỉ cộng thêm cho id=1
mask = df_adjusted['id'] == 1
df_adjusted.loc[mask, 'lux'] = df_adjusted.loc[mask, 'lux'] + 24
df_adjusted.loc[mask, 'current'] = df_adjusted.loc[mask, 'current'] - 0.4

# Lưu file mới
df_adjusted.to_csv('uwb_data_adjusted.csv', index=False)

print(df_adjusted)
