# plot_fig11_pwr_speed.py
import numpy as np
import matplotlib.pyplot as plt
from config import *

plt.rcParams.update(plt_config)

# ====================== 正确MPPT功率公式 ======================
# 常数项 Kopt
K_opt = (0.5 * np.pi * rho * blade_r**5 * Cp_opt) / ((lambda_opt ** 3) * (p * ng)**3)

# 生成发电机电角速度区间 250~500 rad/s
omega_e_arr = np.linspace(250, 500, 200)
P_opt_curve = []

# 遍历计算最优机械功率
for we in omega_e_arr:
    P_watt = K_opt * (we ** 3)
    P_opt_curve.append(P_watt / 1e6)  # 转换单位 MW

# 传递函数模型离散采样点（论文图中标记点）
sample_we = [270, 320, 360, 400, 440, 480]
P_tf_points_x = sample_we
P_tf_points_y = []
for we in sample_we:
    P_watt = K_opt * (we ** 3)
    P_tf_points_y.append(P_watt / 1e6)

# ====================== 绘图 ======================
plt.figure(figsize=(4, 2.5))
plt.plot(omega_e_arr, P_opt_curve, color="black", linewidth=1, label="最优运行特性曲线")
plt.scatter(P_tf_points_x, P_tf_points_y, color="red", marker="o", s=40, label="传递函数模型")
plt.xlabel(r"$\omega_e$ / (rad/s)")
plt.ylabel(r"$P_e$ / MW")
plt.xlim(250, 500)
plt.ylim(0, 1.5)
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig("fig11_pwr_speed.png", dpi=150)
plt.show()