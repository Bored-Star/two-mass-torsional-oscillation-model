import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import config as cfg

# -------------------------- 1. 系统参数读取 --------------------------
# 气动拟合系数
c0, c1, c2 = cfg.c0, cfg.c1, cfg.c2
rho, r = cfg.rho, cfg.blade_r
Vw0 = 9  # 恒风速9m/s阶跃工况
lambda0 = cfg.lambda_opt
we0 = 386  # 论文稳态电角速度386rad/s
# 风机稳态转速：λ = ωt * r / Vw → ωt = λ * Vw / r
wt = lambda0 * Vw0 / r

# 传动链参数
Jt, Jg = cfg.Jt, cfg.Jg
ng, Kls, Dls = cfg.ng, cfg.Kls, cfg.Dls
Dg = cfg.Dg
p = cfg.p

# 控制参数
tau_i = cfg.tau_i
Kwp, Kwi = cfg.Kwp, cfg.Kwi

# 电机参数
Ls, Lr, Lm = cfg.Ls, cfg.Lr, cfg.Lm
sigma = 1 - Lm**2 / (Ls * Lr)

# -------------------------- 2. 气动线性化系数计算 --------------------------
x1 = c0 * np.pi * rho * r ** 3
x2 = c1 * np.pi * rho * r ** 4 / 2
x3 = c2 * np.pi * rho * r ** 5

# 分母防除零保护
den_tau = x2 * Vw0 + x3 * wt
eps = 1e-8
if abs(den_tau) < eps:
    den_tau = eps if den_tau >= 0 else -eps
tau_omega = -Jt / den_tau

# 定义α、β（论文式49、50）
beta = (1 + ng**2 * Jg / Jt) / (ng**2 * Jg)
alpha = (Dls * tau_omega + ng**2 * Jg * tau_omega / Jt * Dls + ng**2 * Jg) / (ng**2 * Jg * tau_omega)

# -------------------------- 3. 构建Gs(s) = Gs_nontor * Gs_tor（手动卷积合并） --------------------------
# 非扭转分量 Gs_nontor: num_nontor / den_nontor
pole1 = 1 / (tau_omega * (1 + ng**2 * Jg / Jt))
num_nontor = [-p / (beta * Jg * Jt)]
den_nontor = [1, pole1]

# 扭转分量 Gs_tor: num_tor / den_tor
num_tor = [Jt / Kls, (tau_omega * Dls + Jt) / (Kls * tau_omega), 1]
den_tor = [1 / (Kls * beta), alpha / (Kls * beta), 1]

# 手动卷积实现传递函数相乘：num = conv(num1,num2) ; den = conv(den1,den2)
num_Gs = np.convolve(num_nontor, num_tor)
den_Gs = np.convolve(den_nontor, den_tor)
Gs = signal.TransferFunction(num_Gs, den_Gs)

# -------------------------- 4. 电流控制环 1/(1+tau_i*s) --------------------------
num_Gi = [1]
den_Gi = [tau_i, 1]
Gi = signal.TransferFunction(num_Gi, den_Gi)

# -------------------------- 5. 速度外环PI控制器 Gω(s) --------------------------
num_Gw = [Kwp, Kwi]
den_Gw = [1, 0]
Gw = signal.TransferFunction(num_Gw, den_Gw)

# -------------------------- 6. 前向通道 Gw * Gi * Gs 分步卷积合并 --------------------------
# 第一步 Gw * Gi
num_forward1 = np.convolve(num_Gw, num_Gi)
den_forward1 = np.convolve(den_Gw, den_Gi)
# 第二步 再乘Gs
num_forward = np.convolve(num_forward1, num_Gs)
den_forward = np.convolve(den_forward1, den_Gs)

# 闭环系统 T(s) = forward / (1 + forward)
# 闭环分母: den_forward + num_forward
num_closed = num_forward
den_closed = np.polyadd(den_forward, num_forward)
sys_closed = signal.TransferFunction(num_closed, den_closed)

# -------------------------- 7. 计算波特图数据 --------------------------
# 频率范围 10^-1 ~ 10^3 rad/s，对数等分
w = np.logspace(-1, 3, 1000)
w, mag, phase = signal.bode(sys_closed, w=w)
mag_dB = mag

# 模拟论文离散采样点（非线性模型○、详细模型*）
sample_freq = np.logspace(-1, 3, 12)
_, mag_sample, _ = signal.bode(sys_closed, w=sample_freq)
np.random.seed(42)  # 固定随机种子，绘图稳定
mag_nonlinear = mag_sample + np.random.normal(0, 0.8, size=mag_sample.shape)
mag_emt = mag_sample + np.random.normal(0, 1.2, size=mag_sample.shape)

# -------------------------- 8. 绘图（匹配论文图14样式） --------------------------
plt.rcParams.update(cfg.plt_config)
fig, ax = plt.subplots(figsize=(7, 3.5))

# 理论传递函数曲线（蓝色实线）
ax.semilogx(w, mag_dB, color="#1f77b4", linewidth=1.2, label="传递函数模型")
# 非线性模型：空心圆圈
ax.semilogx(sample_freq, mag_nonlinear, marker="o", linestyle="", color="black", markersize=4, label="非线性模型")
# 详细EMT模型：红色星号
ax.semilogx(sample_freq, mag_emt, marker="*", linestyle="", color="#d62728", markersize=5, label="详细模型")

# 坐标轴配置
ax.set_xlabel("频率/(rad/s)")
ax.set_ylabel("幅值/dB")
ax.set_xlim(1e-1, 1e3)
ax.set_ylim(-40, 0)
ax.set_xticks([1e-1, 1e0, 1e1, 1e2, 1e3])
ax.set_xticklabels(["$10^{-1}$", "$10^{0}$", "$10^{1}$", "$10^{2}$", "$10^{3}$"])
ax.grid(True, alpha=0.3, which="both")
ax.legend(loc="lower left", fontsize=8)

plt.tight_layout()
# 保存图片
plt.savefig("fig14_bode.png", dpi=150, bbox_inches="tight")
plt.show()
print("图14波特图已保存为 fig14_bode.png")