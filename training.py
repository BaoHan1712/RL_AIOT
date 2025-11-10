import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# ====== Môi trường Sensor cải tiến ======
class SensorEnv(gym.Env):
    def __init__(self, data, lux_thresh=100, hum_thresh=5):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.idx = 0
        self.last_send_idx = 0
        self.lux_thresh = lux_thresh
        self.hum_thresh = hum_thresh

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = 0
        self.last_send_idx = 0
        self.first_step = True  # Bản ghi đầu tiên luôn gửi
        return self._get_state(), {}

    def step(self, action):
        terminated = False
        truncated = False
        reward = 0.0

        # Bản ghi đầu tiên luôn gửi
        if self.first_step:
            action = 1
            self.first_step = False

        state = self._get_state()
        current = float(state[5])
        important = self._data_important()

        if action == 1:
            reward -= current
            self.last_send_idx = self.idx
            if important:
                reward += 1.0  # gửi dữ liệu quan trọng
        else:
            if important:
                reward -= 1.0  # bỏ qua dữ liệu quan trọng
            else:
                reward += 0.1  # tiết kiệm năng lượng

        self.idx += 1
        if self.idx >= len(self.data):
            terminated = True

        next_state = self._get_state()
        return next_state, float(reward), terminated, truncated, {}

    def _get_state(self):
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

    def _data_important(self):
        row = self.data.iloc[self.idx]
        last_row = self.data.iloc[self.last_send_idx]
        return (
            row['Lux'] > self.lux_thresh or
            abs(row['Humidity'] - last_row['Humidity']) > self.hum_thresh
        )

# ====== Load dữ liệu sensor ======
data = pd.read_csv("sensor_data.csv")
env = SensorEnv(data)
check_env(env)

# ====== Train PPO ======
model = PPO('MlpPolicy', env,learning_rate = 0.0005, verbose=1)
model.learn(total_timesteps=10000)

model.save("ppo_sensor_model.zip")
print("✅ Model đã được lưu thành công!")

# ====== Test và ghi hành vi ======
obs, _ = env.reset()
actions_taken = []
rewards = []
energy_used = 0.0

for _ in range(len(data)):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    actions_taken.append(action)
    rewards.append(reward)

    # Nếu gửi thì tính năng lượng tiêu thụ
    if action == 1:
        current = float(env.data.iloc[env.idx - 1]['Current'])
        energy_used += current

    if terminated:
        break

# ====== Thống kê kết quả ======
send_count = sum(1 for a in actions_taken if a == 1)
skip_count = len(actions_taken) - send_count
total_steps = len(actions_taken)
avg_energy = energy_used / max(send_count, 1)

print("\n===== THỐNG KÊ HÀNH VI AGENT =====")
print(f"Tổng số bước: {total_steps}")
print(f"Số lần gửi dữ liệu: {send_count}")
print(f"Số lần bỏ qua: {skip_count}")
print(f"Tổng năng lượng tiêu thụ: {energy_used:.4f}")
print(f"Năng lượng trung bình mỗi lần gửi: {avg_energy:.4f}")
print(f"Tổng reward đạt được: {sum(rewards):.4f}")
print("===================================\n")
