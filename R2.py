import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

config = {
    "font.family": 'serif',
    "font.serif": ['SimSun'],  # 设置字体类型
    "axes.unicode_minus": False,  # 解决负号无法显示的问题
    'axes.labelsize': '12',  # 标题字体大小
    'xtick.labelsize': '12',  # x轴数字字体大小
    'ytick.labelsize': '12',  # y轴数字字体大小
    'lines.linewidth': 2.5,  # 线宽
    'legend.fontsize': '15',
    'figure.figsize': '5, 4'  # 图片尺寸
}
rcParams.update(config)
# 读取数据
predicted_data = np.loadtxt('../predicted.txt')
with open('20.txt', 'r') as f:
    lines = [line.rstrip() for line in f if not line.startswith('%')]
data = np.array([line.split() for line in lines], dtype=float)


# 分割数据
xyz = predicted_data[:, :3]
pred_data = predicted_data[:,3:]
true_data = data[:, 3:]  # 确保维度匹配

# 物理量名称和单位
phys_names = ['u', 'v', 'w', 'p']
units = ['mm/s', 'mm/s', 'mm/s', 'Pa']

# 创建2x2子图布局
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs = axs.ravel()

# 颜色设置
true_color = '#2C5F94'  # 蓝色系
pred_color = '#D8262C'  # 红色系

for idx, (ax, name, unit) in enumerate(zip(axs, phys_names, units)):
    # 提取对应物理量数据
    pred = pred_data[:, idx]
    true = true_data[:, idx]

    # 计算统计指标
    mae = np.mean(np.abs(pred - true))
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    r2 = np.corrcoef(pred, true)[0, 1] ** 2

    # 绘制散点密度图
    hb = ax.hexbin(true, pred, gridsize=50, cmap='viridis', bins='log',
                   mincnt=1, alpha=0.7)

    # 添加颜色条
    cb = fig.colorbar(hb, ax=ax)
    # cb.set_label('数据密度')

    # 添加参考线
    lims = [np.min([true.min(), pred.min()]),
            np.max([true.max(), pred.max()])]
    ax.plot(lims, lims, '--', color='gray', linewidth=1, alpha=0.7)

    # 设置文本注释
    stats_text = f'$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            va='top', ha='left', bbox=dict(facecolor='white', alpha=0.8))

    # 设置坐标轴
    ax.set_xlabel(f'真实 {name} ({unit})')
    ax.set_ylabel(f'预测 {name} ({unit})')
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout(pad=2.0)
plt.savefig('pred_true_comparison.png', bbox_inches='tight')
plt.show()
