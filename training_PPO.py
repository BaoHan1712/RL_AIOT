import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
import os

# =====================================================
# 1. LOAD & MERGE DATA (NODE 2 vs NODE 1)
# =====================================================
def load_and_merge_data(path_node2, path_node1):
    """
    Load 2 file dữ liệu và đồng bộ hóa theo thời gian.
    - Node 2: Dữ liệu cảm biến (Input cho Agent)
    - Node 1: Dữ liệu chuẩn (Ground Truth để tính RMSE/RAE)
    """
    print(f"📥 Đang tải dữ liệu từ: {path_node2} và {path_node1}")
    
    # Đọc dữ liệu
    df2 = pd.read_csv(path_node2) # Node 2
    df1 = pd.read_csv(path_node1) # Node 1 (Ground Truth)
    
    # Chuẩn hóa tên cột & Thời gian
    for df in [df2, df1]:
        df.columns = df.columns.str.lower().str.strip()
        time_col = 'time' if 'time' in df.columns else df.columns[0]
        df[time_col] = pd.to_datetime(df[time_col])
        df.sort_values(by=time_col, inplace=True)
        df.rename(columns={time_col: 'time'}, inplace=True)

    # Merge asof: Tìm bản ghi Node 1 gần nhất với mỗi bản ghi Node 2
    merged = pd.merge_asof(
        df2, 
        df1, 
        on='time', 
        direction='nearest', 
        suffixes=('_sensor', '_gt'),
        tolerance=pd.Timedelta('10s')
    )
    
    merged = merged.dropna()
    
    # Chọn feature quan trọng (Temp)
    feature_cols = [c for c in merged.columns if 'temp' in c]
    if len(feature_cols) < 2:
        col_sensor = 'temp_sensor' if 'temp_sensor' in merged.columns else merged.columns[1]
        col_gt = 'temp_gt' if 'temp_gt' in merged.columns else merged.columns[-1]
    else:
        col_sensor = 'temp_sensor'
        col_gt = 'temp_gt'

    print(f"✅ Đã merge: {merged.shape[0]} dòng. Sensor: '{col_sensor}', GT: '{col_gt}'")
    
    data_sensor = merged[col_sensor].values
    data_gt = merged[col_gt].values
    
    # Normalize
    min_val = data_sensor.min()
    max_val = data_sensor.max()
    data_sensor_norm = (data_sensor - min_val) / (max_val - min_val + 1e-6)
    
    return data_sensor, data_sensor_norm, data_gt

# =====================================================
# 2. CUSTOM GYM ENVIRONMENT
# =====================================================
class SmartTransmissionEnv(gym.Env):
    def __init__(self, sensor_norm, sensor_raw, gt_raw, energy_cost=0.5, error_penalty=1.0):
        super(SmartTransmissionEnv, self).__init__()
        
        self.sensor_norm = sensor_norm
        self.sensor_raw = sensor_raw
        self.gt_raw = gt_raw
        self.n_steps = len(sensor_norm)
        
        self.energy_cost = energy_cost
        self.alpha = error_penalty
        
        self.action_space = spaces.Discrete(2) # 0: Hold, 1: Send
        # Observation: [Current_Sensor_Norm, Last_Sent_Norm, Diff_Norm]
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        
        self.current_step = 0
        self.last_sent_norm = 0.0
        self.last_sent_raw = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.last_sent_norm = self.sensor_norm[0]
        self.last_sent_raw = self.sensor_raw[0]
        diff = 0.0
        obs = np.array([self.sensor_norm[0], self.last_sent_norm, diff], dtype=np.float32)
        return obs, {}

    def step(self, action):
        current_norm = self.sensor_norm[self.current_step]
        current_raw_sensor = self.sensor_raw[self.current_step]
        current_raw_gt = self.gt_raw[self.current_step]
        
        reward_energy = 0
        if action == 1: # SEND
            self.last_sent_norm = current_norm
            self.last_sent_raw = current_raw_sensor
            reward_energy = -self.energy_cost
        else: # HOLD
            reward_energy = 0
            
        internal_error = abs(current_norm - self.last_sent_norm)
        reward = -(self.alpha * internal_error) + reward_energy
        
        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1
        truncated = False
        
        if not terminated:
            next_norm = self.sensor_norm[self.current_step]
            diff = abs(next_norm - self.last_sent_norm)
            next_obs = np.array([next_norm, self.last_sent_norm, diff], dtype=np.float32)
        else:
            next_obs = np.zeros(3, dtype=np.float32)
            
        info = {
            "server_val": self.last_sent_raw,
            "ground_truth": current_raw_gt,
            "sensor_raw": current_raw_sensor, # Thêm giá trị gốc của sensor để đối chiếu
            "action": action
        }
        
        return next_obs, reward, terminated, truncated, info

# =====================================================
# 3. MAIN EXECUTION
# =====================================================

FILE_NODE_2 = "sensor_only_id2.csv"
FILE_NODE_1 = "sensor_only_id1.csv" 
MODEL_PATH = "ppo_smart_transmission_model" # Tên file model sẽ lưu
REPORT_FILE = "evaluation_report.txt"
LOG_FILE = "simulation_log.csv"

# Tạo dummy nếu thiếu file (để test)
if not os.path.exists(FILE_NODE_1) and os.path.exists(FILE_NODE_2):
    print(f"⚠️ Tạo dummy {FILE_NODE_1}...")
    df = pd.read_csv(FILE_NODE_2)
    df['id'] = 1
    if 'temp' in df.columns: df['temp'] = df['temp'].rolling(5).mean()
    df.to_csv(FILE_NODE_1, index=False)

try:
    # 1. Load Data
    data_sensor, data_norm, data_gt = load_and_merge_data(FILE_NODE_2, FILE_NODE_1)
    
    # 2. Setup Env
    env = SmartTransmissionEnv(data_norm, data_sensor, data_gt, energy_cost=0.1, error_penalty=15.0)
    check_env(env)
    
    # 3. Train
    print("🚀 Bắt đầu training PPO...")
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.001)
    model.learn(total_timesteps=150000)
    print("✅ Training xong.")

    # --- LƯU MODEL ---
    model.save(MODEL_PATH)
    print(f"💾 Đã lưu model tại: {MODEL_PATH}.zip")

    # 4. Evaluation
    print("\n🔍 Đang đánh giá và ghi log...")
    obs, _ = env.reset()
    done = False
    
    # Lists to store logs
    val_server = []
    val_gt = []
    val_sensor = []
    actions = []
    
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        
        val_server.append(info['server_val'])
        val_gt.append(info['ground_truth'])
        val_sensor.append(info['sensor_raw'])
        actions.append(info['action'])
        
    # --- TÍNH TOÁN METRICS ---
    y_pred = np.array(val_server)
    y_true = np.array(val_gt)
    
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(np.abs(y_true - np.mean(y_true)))
    rae = numerator / denominator if denominator != 0 else np.inf
    
    send_count = sum(actions)
    total = len(actions)
    saving = 100 * (1 - send_count/total)
    
    # --- LƯU BÁO CÁO KẾT QUẢ (.txt) ---
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("BÁO CÁO KẾT QUẢ (NODE 2 vs NODE 1 GT)\n")
        f.write("====================================\n")
        f.write(f"Dataset Size:           {total} mẫu\n")
        f.write(f"RMSE (Độ lệch chuẩn):   {rmse:.4f}\n")
        f.write(f"RAE (Sai số tương đối): {rae:.4f}\n")
        f.write(f"Tiết kiệm năng lượng:   {saving:.2f}%\n")
        f.write(f"Số gói tin gửi đi:      {send_count}/{total}\n")
    print(f"📝 Đã lưu báo cáo tại: {REPORT_FILE}")

    # --- LƯU LOG CHI TIẾT (.csv) ---
    log_df = pd.DataFrame({
        "step": range(len(actions)),
        "ground_truth_node1": val_gt,
        "sensor_raw_node2": val_sensor,
        "server_value": val_server,
        "action_send": actions
    })
    log_df.to_csv(LOG_FILE, index=False)
    print(f"📊 Đã lưu log mô phỏng tại: {LOG_FILE}")
    
    # 5. Visualization
    limit = 850
    plt.figure(figsize=(12, 6))
    plt.plot(val_gt[:limit], 'k-', linewidth=2, label='Ground Truth (Node 1)')
    plt.plot(val_server[:limit], 'r--', label='Server Value (Reconstructed)')
    
    send_indices = [i for i, a in enumerate(actions[:limit]) if a == 1]
    send_values = [val_gt[i] for i in send_indices]
    plt.scatter(send_indices, send_values, c='green', marker='^', zorder=5, label='Tx Event')
    
    plt.title(f"Result: RMSE={rmse:.3f}, Saving={saving:.1f}%")
    plt.xlabel("Time Step")
    plt.ylabel("Temperature")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("result_chart.png") # Lưu luôn biểu đồ
    print("🖼️ Đã lưu biểu đồ tại: result_chart.png")
    plt.show()

except Exception as e:
    print(f"❌ Có lỗi xảy ra: {e}")
    import traceback
    traceback.print_exc()