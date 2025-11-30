import pandas as pd
import numpy as np
from stable_baselines3 import PPO

# ======================
# LOAD MODEL + DỮ LIỆU ID2
# ======================
model = PPO.load("ppo_sensor_model.zip")
data = pd.read_csv("sensor_only_id2.csv")  # đọc file CSV

# convert time nếu cần
if not np.issubdtype(data['time'].dtype, np.datetime64):
    data['time'] = pd.to_datetime(data['time'])

data = data.reset_index(drop=True)

# ======================
# MINI ENV – chỉ dùng ID2
# ======================
class MiniEnv:
    def __init__(self, data, drift_limit=30):
        self.data = data
        self.idx = 0
        self.last_send_idx = 0
        self.drift_limit = drift_limit

    def get_state(self):
        if self.idx >= len(self.data):
            return np.zeros(8, dtype=np.float32)

        row = self.data.iloc[self.idx]
        drift = self.idx - self.last_send_idx

        # Giả lập giá trị chuẩn bằng chính ID2
        state = np.array([
            0.0,  # lux
            0.0,  # temp
            0.0,  # hum
            0.0,  # press
            0.0,  # alt
            row["current"] / 100,
            drift / 50,
            1.0 if drift > self.drift_limit else 0.0
        ], dtype=np.float32)

        return state

    def step(self, action):
        if action == 1:
            self.last_send_idx = self.idx
        self.idx += 1
        return action, self.idx - self.last_send_idx

env = MiniEnv(data)

# ======================
# Chạy mô hình và lưu dữ liệu SEND + SKIP
# ======================
rows_all = []

while env.idx < len(data):
    state = env.get_state()
    action, _ = model.predict(state)
    row = data.iloc[env.idx]
    drift = env.idx - env.last_send_idx

    # lưu cả SEND và SKIP, thêm giá trị so với lần gửi trước
    prev_row = data.iloc[env.last_send_idx]
    rows_all.append({
        "time": row['time'],
        "lux": row['lux'],
        "temp": row['temp'],
        "hum": row['hum'],
        "press": row['press'],
        "alt": row['alt'],
        "current": row['current'],
        "drift": drift,
        "decision": "SEND" if action == 1 else "SKIP",
        "lux_change": row['lux'] - prev_row['lux'],
        "temp_change": row['temp'] - prev_row['temp'],
        "hum_change": row['hum'] - prev_row['hum'],
        "press_change": row['press'] - prev_row['press'],
        "alt_change": row['alt'] - prev_row['alt'],
        "current_change": row['current'] - prev_row['current']
    })

    env.step(action)

# tạo DataFrame và xuất CSV
df_all = pd.DataFrame(rows_all)
df_all.to_csv("sensor_decision_full.csv", index=False)
print("✅ CSV 'sensor_decision_full.csv' đã được lưu, gồm cả SEND và SKIP.")
