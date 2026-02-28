import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ==============================
# 路径
# ==============================
gain_path = r"G:\RadioMapSeer\gain\IRT2\0_0.png"
dist_path = r"G:\RMdata\distancemap\0_0.npy"

# ==============================
# 读取数据
# ============================== 
gain_img = Image.open(gain_path).convert("F")
gain = np.array(gain_img)
dist = np.load(dist_path)

# ==============================
# 过滤无效点
# ==============================
mask = (dist != 1000) & (gain != 0)
gain_valid = gain[mask]
dist_valid = dist[mask]

# ==============================
# log(distance)
# ==============================
dist_log = np.log10(dist_valid)

# ==============================
# 画散点
# ==============================
plt.figure(figsize=(6,6))
plt.scatter(gain_valid, dist_log, s=1)

# ==============================
# 随便画一条斜率 = -2 的红线
# y = -2x + b
# ==============================
x_line = np.linspace(gain_valid.min(), gain_valid.max(), 100)
b = dist_log.mean() + 2 * gain_valid.mean()   # 随便选个截距
y_line = -2 * x_line + b

plt.plot(x_line, y_line, 'r-', linewidth=2)

plt.xlabel("gain (original)")
plt.ylabel("log10(distance)")
plt.title("log10(Distance) vs Gain (filtered)")
plt.grid(True)
plt.tight_layout()
plt.show()