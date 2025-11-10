import customtkinter as ctk
import pandas as pd
import numpy as np
from stable_baselines3 import PPO

# ====== Load model và dữ liệu ======
model = PPO.load("ppo_sensor_model.zip")
data = pd.read_csv("sensor_data.csv")

# ====== Giả lập môi trường mini cho agent ======
class MiniEnv:
    """Môi trường nhỏ để agent predict action"""
    def __init__(self, data):
        self.data = data
        self.idx = 0
        self.last_send_idx = 0

    def get_state(self):
        if self.idx >= len(self.data):
            return np.zeros(7)
        row = self.data.iloc[self.idx]
        state = np.array([
            row['Altitude']/1000,
            row['Humidity']/100,
            row['Lux']/1000,
            row['Pressure']/120000,
            row['Temp']/100,
            row['Current'],
            (self.idx - self.last_send_idx)/100
        ], dtype=np.float32)
        return state

    def step(self, action):
        if action == 1:
            self.last_send_idx = self.idx
        self.idx += 1

env = MiniEnv(data)

# ====== CustomTkinter setup ======
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

root = ctk.CTk()
root.title("RL Sensor Agent Simulation")
root.geometry("600x400")

labels = {}
sensors = ['Timestamp','Altitude','Humidity','Lux','Pressure','Temp','Current','Decision']
for i, s in enumerate(sensors):
    lbl = ctk.CTkLabel(root, text=f"{s}: ", font=("Arial", 14))
    lbl.grid(row=i, column=0, sticky="w", padx=10, pady=5)
    labels[s] = lbl

# ====== Hàm update GUI mỗi giây ======
def update_gui():
    if env.idx >= len(data):
        return  # hết dữ liệu
    state = env.get_state()
    action, _ = model.predict(state)
    row = data.iloc[env.idx]

    # Cập nhật nhãn
    labels['Timestamp'].configure(text=f"Timestamp: {row['Timestamp']}")
    labels['Altitude'].configure(text=f"Altitude: {row['Altitude']:.2f}")
    labels['Humidity'].configure(text=f"Humidity: {row['Humidity']:.2f}")
    labels['Lux'].configure(text=f"Lux: {row['Lux']:.2f}")
    labels['Pressure'].configure(text=f"Pressure: {row['Pressure']:.2f}")
    labels['Temp'].configure(text=f"Temp: {row['Temp']:.2f}")
    labels['Current'].configure(text=f"Current: {row['Current']:.4f}")
    labels['Decision'].configure(text=f"Decision: {'SEND' if action==1 else 'SKIP'}", 
                                 text_color="green" if action==1 else "red")

    # Agent step
    env.step(action)

    # gọi lại sau 1 giây
    root.after(1000, update_gui)

# ====== Start simulation ======
update_gui()
root.mainloop()
