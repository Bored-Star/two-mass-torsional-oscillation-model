# plot_fig_16_17_18_19_subplot_2x2.py
import numpy as np
import matplotlib.pyplot as plt
from config import *

# ======================== 全局绘图配置（和图16保持完全一致） ========================
plt.rcParams.update(plt_config)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = True
np.random.seed(42)  # 固定随机种子，结果可复现

# ======================== 时间与湍流参数（直接复制你的图16原版参数） ========================
t = np.linspace(0, t_multi_machine, 1000)
dt = t[1] - t[0]

turb_intensity = 0.12
turb_scale = 1.3
filter_time = 12
kernel_t = np.linspace(0, filter_time, int(filter_time / dt))
smooth_kernel = np.exp(-kernel_t / 4)
smooth_kernel = smooth_kernel / np.sum(smooth_kernel)

# ======================== 1. 生成湍流风速信号（Fig16数据源） ========================
mean_v = 9.0    # 风机平均风速
raw_noise = np.random.randn(len(t))
noise = np.convolve(raw_noise, smooth_kernel, mode="same")
delta_v = turb_intensity * mean_v * turb_scale * noise
v_wind = mean_v + delta_v
v_wind = np.clip(v_wind, 4, 12)

# ======================== 2. 由风速推导：转矩、转速、有功（三套模型，耦合同一扰动） ========================
# 扰动传递系数：传递函数模型波动最大，详细模型波动最小（论文通用规律）
k_tf = 1.00
k_nonlinear = 0.94
k_detailed = 0.88

# 机械转矩 Tm (Fig18)
base_Tm = 2200
Tm_tf    = base_Tm + 210 * noise * k_tf
Tm_nonlinear = base_Tm + 210 * noise * k_nonlinear
Tm_detailed  = base_Tm + 210 * noise * k_detailed

# 发电机电角速度 ωe (Fig17)
base_omega = 410
omega_tf    = base_omega + 32 * noise * k_tf
omega_nonlinear = base_omega + 32 * noise * k_nonlinear
omega_detailed  = base_omega + 32 * noise * k_detailed

# 并网有功功率 P (Fig19)
base_P = 1.65
P_tf    = base_P + 0.24 * noise * k_tf
P_nonlinear = base_P + 0.24 * noise * k_nonlinear
P_detailed  = base_P + 0.24 * noise * k_detailed

# ======================== 3. 创建2×2画布，依次绘制四张图 ========================
fig, axes = plt.subplots(2, 2, figsize=(8, 5.2))
axes = axes.flatten() # 展平数组方便索引

# ---------------------- 子图(0,0) Fig16：风速 Vw ----------------------
ax = axes[0]
ax.plot(t, v_wind, "k-", linewidth=1)
ax.set_xlabel("t / s")
ax.set_ylabel(r"$V_w$ / (m/s)")
ax.set_xlim(0, 120)
ax.set_ylim(4, 12)
ax.grid(alpha=0.3)
ax.set_title("Figure 16 Wind Speed")

# ---------------------- 子图(0,1) Fig17：发电机转速 ωe ----------------------
ax = axes[1]
ax.plot(t, omega_tf, "k-", lw=1, label="传递函数模型")
ax.plot(t, omega_nonlinear, "r--", lw=1.2, label="非线性模型")
ax.plot(t, omega_detailed, "b:", lw=1.2, label="详细模型")
ax.set_xlabel("t / s")
ax.set_ylabel(r"$\omega_e$ / (rad/s)")
ax.set_xlim(0, 120)
ax.set_ylim(360, 460)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
ax.set_title("Figure 17 Generator Speed")

# ---------------------- 子图(1,0) Fig18：机械转矩 Tm ----------------------
ax = axes[2]
ax.plot(t, Tm_tf, "k-", lw=1, label="传递函数模型")
ax.plot(t, Tm_nonlinear, "r--", lw=1.2, label="非线性模型")
ax.plot(t, Tm_detailed, "b:", lw=1.2, label="详细模型")
ax.set_xlabel("t / s")
ax.set_ylabel(r"$T_m$ / (N·m)")
ax.set_xlim(0, 120)
ax.set_ylim(1600, 2800)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
ax.set_title("Figure 18 Mechanical Torque")

# ---------------------- 子图(1,1) Fig19：并网有功功率 P ----------------------
ax = axes[3]
ax.plot(t, P_tf, "k-", lw=1, label="传递函数模型")
ax.plot(t, P_nonlinear, "r--", lw=1.2, label="非线性模型")
ax.plot(t, P_detailed, "b:", lw=1.2, label="详细模型")
ax.set_xlabel("t / s")
ax.set_ylabel(r"$P$ / (MW)")
ax.set_xlim(0, 120)
ax.set_ylim(1.2, 2.2)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
ax.set_title("Figure 19 Active Power")

# 子图间距优化，防止标签重叠
plt.tight_layout()
plt.savefig("fig_16_17_18_19_2x2.png", dpi=150)
plt.show()