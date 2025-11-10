import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Số bản ghi
n_records = 100_000

# Tạo timestamp liên tục, giả sử 1 giây một lần
start_time = datetime(2025, 10, 16, 17, 31, 12)
timestamps = [start_time + timedelta(seconds=i) for i in range(n_records)]
timestamps = [t.strftime("%Y%m%d_%H%M%S") for t in timestamps]

# Sinh dữ liệu sensor mô phỏng
np.random.seed(42)  # để kết quả reproducible

Altitude = np.random.normal(loc=72.5, scale=0.5, size=n_records)       # Altitude ~72m ±0.5
Humidity = np.clip(np.random.normal(loc=50, scale=5, size=n_records),0,100)   # 0-100%
Lux = np.clip(np.random.normal(loc=50, scale=30, size=n_records),0,1000)      # ánh sáng
Pressure = np.random.normal(loc=100631, scale=50, size=n_records)        # Pa
Temp = np.random.normal(loc=28, scale=1, size=n_records)                 # °C
Current = np.clip(np.random.normal(loc=0.05, scale=0.01, size=n_records),0,1) # Amps

# Tạo DataFrame
df = pd.DataFrame({
    'Timestamp': timestamps,
    'Altitude': Altitude,
    'Humidity': Humidity,
    'Lux': Lux,
    'Pressure': Pressure,
    'Temp': Temp,
    'Current': Current
})

# Lưu CSV
df.to_csv("sensor_data.csv", index=False)
print("Đã tạo file sensor_data.csv với 100000 bản ghi")
