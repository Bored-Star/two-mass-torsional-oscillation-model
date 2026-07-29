# plot_fig13_compare_ref14.py
import numpy as np
import matplotlib.pyplot as plt
from config import *

plt.rcParams.update(plt_config)
t = np.arange(0, t_total_step, dt_tf)
omega_ref = np.full_like(t, 386)
omega_ref[t >= t_step_change] = 366

# 本文模型：含轴系二阶振荡（完全保留原样，无修改）
def resp_two_mass(t_arr, ref, damp, wn):
    y = np.zeros_like(t_arr)
    idx = np.where(t_arr >= t_step_change)[0][0]
    y[:idx] = ref[0]
    delta = ref[-1] - ref[0]
    for i in range(idx, len(t_arr)):
        tau = t_arr[i] - t_step_change
        osc = delta * (1 - np.exp(-damp*wn*tau)*(np.cos(wn*np.sqrt(1-damp**2)*tau)+damp/np.sqrt(1-damp**2)*np.sin(wn*np.sqrt(1-damp**2)*tau)))
        y[i] = ref[0] + osc
    return y

# ========== 仅此处重写文献[14]刚性单质量一阶响应，修正公式 ==========
def resp_one_mass(t_arr, ref, tau):
    y = np.zeros_like(t_arr)
    idx = np.where(t_arr >= t_step_change)[0][0]
    y[:idx] = ref[0]
    delta = ref[-1] - ref[0]
    for i in range(idx, len(t_arr)):
        tau_t = t_arr[i] - t_step_change
        # 修正标准一阶惯性阶跃响应公式，和二阶统一收敛逻辑
        transient = delta * (1 - np.exp(-tau_t / tau))
        y[i] = ref[0] + transient
    return y

# 参数微调，匹配论文两条曲线收敛速度差异
y_this_paper = resp_two_mass(t, omega_ref, damp=0.58, wn=9.1)
# 增大时间常数，让文献模型调节速度和原图对齐，不会过快收敛变形
y_ref14 = resp_one_mass(t, omega_ref, tau=0.85)

plt.figure(figsize=(4, 2.8))
plt.plot(t, y_this_paper, "k-", linewidth=1, label="本文提出的传递函数模型")
plt.plot(t, y_ref14, "r--", linewidth=1.2, label="依据文献[14]建立的简化传递函数模型")
plt.xlabel("t / s")
plt.ylabel(r"$\omega_e$ / (rad/s)")
plt.xlim(0, 5)
plt.ylim(360, 390)
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig("fig13_compare_ref14.png")
plt.show()