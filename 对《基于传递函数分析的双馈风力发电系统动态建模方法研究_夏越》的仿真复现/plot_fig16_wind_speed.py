# plot_fig16_wind_speed.py
import numpy as np
import matplotlib.pyplot as plt
from config import *

# 字体修复，消除负号方块
plt.rcParams.update(plt_config)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = True

np.random.seed(42)  # 固定随机序列，每次曲线完全一致
t = np.linspace(0, t_multi_machine, 1000)
dt = t[1] - t[0]
wind_list = []
colors = ["black", "red", "blue", "purple"]
linestyles = ["-", "--", "-.", ":"]

# 湍流参数：调大扰动系数提升幅值，缩短滤波窗口
turb_intensity = 0.12
turb_scale = 1.3    # 放大湍流幅值，解决波动偏小
filter_time = 12    # 缩短滤波窗口，保留更多起伏

# 构造归一化平滑核
kernel_t = np.linspace(0, filter_time, int(filter_time / dt))
smooth_kernel = np.exp(-kernel_t / 4)
smooth_kernel = smooth_kernel / np.sum(smooth_kernel)

for idx, cfg in enumerate(wind_params):
    mean_v = cfg["Vw_mean"]
    raw_noise = np.random.randn(len(t))
    # 使用same卷积，无截断、无边缘填充平顶
    noise = np.convolve(raw_noise, smooth_kernel, mode="same")
    # 放大湍流波动幅度
    delta_v = turb_intensity * mean_v * turb_scale * noise
    v_wind = mean_v + delta_v
    # 限制风速区间4~12，杜绝极端异常尖峰
    v_wind = np.clip(v_wind, 4, 12)
    wind_list.append(v_wind)
    # LaTeX下标修正，消除NameError报错
    plt.plot(t, v_wind, color=colors[idx], linestyle=linestyles[idx], linewidth=1, label=f"$V_{{w{idx+1}}}$")

# 绘图设置
plt.xlabel("t / s")
plt.ylabel(r"$V_w$ / (m/s)")
plt.xlim(0, 120)
plt.ylim(4, 12)
plt.legend(fontsize=7)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig16_wind_speed.png", dpi=150)
plt.show()