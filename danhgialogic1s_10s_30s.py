import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

# Tạo thư mục lưu nếu chưa có
output_dir = "data_analytic"
os.makedirs(output_dir, exist_ok=True)

# Hàm tính RMSE và RAE, xử lý chia cho 0
def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    denominator = np.sum(np.abs(y_true - np.mean(y_true)))
    if denominator == 0:
        rae = 0.0  # Hoặc np.nan nếu muốn báo không xác định
    else:
        rae = np.sum(np.abs(y_true - y_pred)) / denominator * 100
    return rmse, rae

# Đọc dữ liệu
df = pd.read_csv('uwb_data_adjusted.csv')
total_packets = len(df[df['id'] == 2])  # Node 2 là data gửi
intervals = [1, 10, 30]

for interval in intervals:
    # Lấy dữ liệu Node 2 theo interval (giả lập gửi)
    df_id2 = df[df['id'] == 2].iloc[::interval, :].reset_index(drop=True)
    df_id1 = df[df['id'] == 1].iloc[:len(df_id2), :].reset_index(drop=True)

    # Tính RMSE và RAE trung bình lux + current
    rmse_lux, rae_lux = compute_metrics(df_id1['lux'], df_id2['lux'])
    rmse_current, rae_current = compute_metrics(df_id1['current'], df_id2['current'])
    rmse_avg = (rmse_lux + rmse_current)/2
    rae_avg = (rae_lux + rae_current)/2

    # Tính tiết kiệm năng lượng
    packets_sent = len(df_id2)
    energy_saved = (1 - packets_sent / total_packets) * 100

    # In báo cáo text
    print("📊 BÁO CÁO KẾT QUẢ (NODE 2 vs NODE 1 GT)")
    print("========================================")
    print(f"Gửi mỗi {interval} giây:")
    print(f"📉 RMSE (Độ lệch chuẩn):       {rmse_avg:.4f}")
    print(f"📉 RAE (Sai số tương đối):     {rae_avg:.4f}%")
    print(f"🔋 Tiết kiệm năng lượng:       {energy_saved:.2f}%")
    print(f"📡 Số gói tin gửi đi:          {packets_sent}/{total_packets}")
    print("\n")

    # ===============================
    # VẼ biểu đồ style ví dụ, node1=GT, node2=Sent
    # ===============================
    limit = min(200, len(df_id2))  # chỉ vẽ tối đa 200 điểm
    val_gt = df_id1['lux'][:limit].values
    val_server = df_id2['lux'][:limit].values

    # Tạo action array: 1 nếu gửi, 0 nếu không gửi
    actions = np.zeros(limit)
    send_indices = list(range(0, limit, 1))  # tất cả dòng df_id2 được gửi
    actions[send_indices] = 1

    send_values = [val_gt[i] for i in range(limit) if actions[i]==1]

    plt.figure(figsize=(12,6))
    plt.plot(val_gt, 'k-', linewidth=2, label='Ground Truth (Node 1)')
    plt.plot(val_server, 'r--', label='Node Sent (Node 2)')
    plt.scatter(send_indices, send_values, c='green', marker='^', zorder=5, label='Tx Event')
    plt.title(f"Gửi mỗi {interval}s: RMSE={rmse_avg:.3f}, Saving={energy_saved:.1f}%")
    plt.xlabel("Time Step")
    plt.ylabel("Lux")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Lưu ảnh
    plt.savefig(os.path.join(output_dir, f"rmse_rae_{interval}s.png"), dpi=300)
    plt.close()
