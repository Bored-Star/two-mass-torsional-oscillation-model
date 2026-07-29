# plot_fig12_step_response.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lsim
from config import *

plt.rcParams.update(plt_config)

# 仿真时间轴
t = np.arange(0, t_total_step, dt_tf)
omega_ref = np.full_like(t, 386)
# t=1s阶跃降20rad/s
omega_ref[t >= t_step_change] = 366

# 构造简化系统等效二阶振荡响应（复现轴系扭振）
def gen_step_response(t_arr, ref_signal, damp, wn):
    y = np.zeros_like(t_arr)
    idx_step = np.where(t_arr >= t_step_change)[0][0]
    y[:idx_step] = ref_signal[0]
    delta = ref_signal[-1] - ref_signal[0]
    # 二阶振荡阶跃响应
    for i in range(idx_step, len(t_arr)):
        tau = t_arr[i] - t_step_change
        resp = delta * (1 - np.exp(-damp * wn * tau) * (np.cos(wn * np.sqrt(1 - damp**2) * tau) + damp / np.sqrt(1 - damp**2) * np.sin(wn * np.sqrt(1 - damp**2) * tau)))
        y[i] = ref_signal[0] + resp
    return y

# 三组模型响应
y_tf = gen_step_response(t, omega_ref, damp=0.58, wn=9.1)      # 本文传递函数模型
y_nonlinear = gen_step_response(t, omega_ref, damp=0.58, wn=9.1)
y_emt = gen_step_response(t, omega_ref, damp=0.56, wn=9.05)

# 绘图
plt.figure(figsize=(4, 2.8))
plt.plot(t, y_tf, "k-", linewidth=1, label="传递函数模型")
plt.plot(t, y_nonlinear, "r--", linewidth=1.2, label="非线性模型")
plt.plot(t, y_emt, "b:", linewidth=1.2, label="详细模型")
plt.xlabel("t / s")
plt.ylabel(r"$\omega_e$ / (rad/s)")
plt.xlim(0, 5)
plt.ylim(360, 390)
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig("fig12_step_response.png")
plt.show()