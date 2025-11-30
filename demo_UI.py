import customtkinter as ctk
import pandas as pd
import numpy as np
from stable_baselines3 import PPO

# ======================
# LOAD MODEL + DỮ LIỆU ID2
# ======================
model = PPO.load("ppo_sensor_model.zip")
data = pd.read_csv("sensor_only_id2.csv")

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
        last = self.data.iloc[self.last_send_idx]
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
# GUI
# ======================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

root = ctk.CTk()
root.title("RL Sensor Agent Simulation")
root.geometry("600x400")

labels = {}
fields = ['Timestamp','Lux','Temp','Hum','Press','Alt','Current','Drift','Decision']
for i, f in enumerate(fields):
    lbl = ctk.CTkLabel(root, text=f"{f}: ", font=("Arial", 14))
    lbl.grid(row=i, column=0, sticky="w", padx=10, pady=3)
    labels[f] = lbl

# ======================
# Lưu kết quả ra CSV
# ======================
results = []

def update_gui():
    if env.idx >= len(data):
        # Khi hết dữ liệu, lưu file CSV
        df_results = pd.DataFrame(results)
        df_results.to_csv("sensor_decisions.csv", index=False)
        print("✅ CSV 'sensor_decisions.csv' đã được lưu!")
        return

    state = env.get_state()
    action, _ = model.predict(state)
    row = data.iloc[env.idx]
    drift = env.idx - env.last_send_idx

    # Cập nhật GUI
    labels['Timestamp'].configure(text=f"Timestamp: {row['time']}")
    labels['Lux'].configure(text=f"Lux: {row['lux']:.2f}")
    labels['Temp'].configure(text=f"Temp: {row['temp']:.2f}")
    labels['Hum'].configure(text=f"Hum: {row['hum']:.2f}")
    labels['Press'].configure(text=f"Press: {row['press']:.2f}")
    labels['Alt'].configure(text=f"Alt: {row['alt']:.2f}")
    labels['Current'].configure(text=f"Current: {row['current']:.2f}")
    labels['Drift'].configure(text=f"Drift: {drift}")
    labels['Decision'].configure(
        text=f"Decision: {'SEND' if action == 1 else 'SKIP'}",
        text_color="green" if action == 1 else "red"
    )

    # Ghi lại kết quả cho CSV
    results.append({
        "time": row['time'],
        "lux": row['lux'],
        "temp": row['temp'],
        "hum": row['hum'],
        "press": row['press'],
        "alt": row['alt'],
        "current": row['current'],
        "drift": drift,
        "decision": "SEND" if action == 1 else "SKIP"
    })

    env.step(action)
    root.after(1000, update_gui)

update_gui()
root.mainloop()
