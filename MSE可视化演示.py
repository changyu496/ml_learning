"""
MSE可视化演示 - 理解最小化均方误差
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 生成示例数据
np.random.seed(42)
X = np.random.rand(20, 1) * 10
y = 2 * X + 1 + np.random.normal(0, 0.5, (20, 1))

print("🎯 MSE可视化演示")
print("="*50)

# 计算不同参数下的MSE
def calculate_mse(X, y, beta_0, beta_1):
    """计算给定参数下的MSE"""
    y_pred = beta_0 + beta_1 * X
    mse = np.mean((y - y_pred) ** 2)
    return mse

# 测试不同的参数组合
beta_0_range = np.linspace(-2, 4, 50)
beta_1_range = np.linspace(0, 4, 50)
mse_values = []

print("计算不同参数组合的MSE...")
for beta_0 in beta_0_range:
    for beta_1 in beta_1_range:
        mse = calculate_mse(X, y, beta_0, beta_1)
        mse_values.append((beta_0, beta_1, mse))

# 找到最佳参数
best_params = min(mse_values, key=lambda x: x[2])
print(f"最佳参数: β₀ = {best_params[0]:.3f}, β₁ = {best_params[1]:.3f}")
print(f"最小MSE: {best_params[2]:.4f}")

# 可视化1：MSE等高线图
print("\n📊 可视化1: MSE等高线图")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 准备网格数据
beta_0_grid, beta_1_grid = np.meshgrid(beta_0_range, beta_1_range)
mse_grid = np.zeros_like(beta_0_grid)

for i in range(len(beta_0_range)):
    for j in range(len(beta_1_range)):
        mse_grid[j, i] = calculate_mse(X, y, beta_0_range[i], beta_1_range[j])

# 等高线图
contour = ax1.contour(beta_0_grid, beta_1_grid, mse_grid, levels=20)
ax1.clabel(contour, inline=True, fontsize=8)
ax1.scatter(best_params[0], best_params[1], color='red', s=100, marker='*', label='最佳参数')
ax1.set_xlabel('β₀ (截距)')
ax1.set_ylabel('β₁ (斜率)')
ax1.set_title('MSE等高线图\n(颜色越深，MSE越小)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 可视化2：数据拟合对比
print("📊 可视化2: 不同参数下的拟合效果")

# 最佳拟合线
y_best = best_params[0] + best_params[1] * X

# 随机选择几个不好的参数
bad_params = [
    (0, 0.5, "参数太差"),
    (3, 0.5, "截距太大"),
    (1, 3, "斜率太大")
]

ax2.scatter(X, y, alpha=0.6, color='blue', label='真实数据')

# 绘制最佳拟合线
ax2.plot(X, y_best, color='red', linewidth=3, label=f'最佳拟合 (MSE={best_params[2]:.3f})')

# 绘制不好的拟合线
colors = ['green', 'orange', 'purple']
for i, (beta_0, beta_1, desc) in enumerate(bad_params):
    y_bad = beta_0 + beta_1 * X
    mse_bad = calculate_mse(X, y, beta_0, beta_1)
    ax2.plot(X, y_bad, color=colors[i], linewidth=2, linestyle='--', 
             label=f'{desc} (MSE={mse_bad:.3f})')

ax2.set_xlabel('特征 X')
ax2.set_ylabel('目标 y')
ax2.set_title('不同参数下的拟合效果对比')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 可视化3：误差分解
print("\n📊 可视化3: 误差分解演示")

plt.figure(figsize=(12, 8))

# 选择几个数据点进行详细分析
sample_indices = [0, 5, 10, 15]
sample_X = X[sample_indices]
sample_y = y[sample_indices]
sample_y_pred = best_params[0] + best_params[1] * sample_X

for i, (x, y_true, y_pred) in enumerate(zip(sample_X, sample_y, sample_y_pred)):
    plt.subplot(2, 2, i+1)
    
    # 绘制所有数据点
    plt.scatter(X, y, alpha=0.3, color='lightblue')
    
    # 高亮当前样本
    plt.scatter(x, y_true, color='red', s=100, zorder=5)
    plt.scatter(x, y_pred, color='green', s=100, zorder=5)
    
    # 绘制拟合线
    plt.plot(X, y_best, color='blue', linewidth=2)
    
    # 绘制误差线
    plt.plot([x, x], [y_true, y_pred], color='red', linewidth=2, linestyle='--')
    
    # 计算误差
    error = y_true - y_pred
    error_squared = error ** 2
    
    plt.title(f'样本 {i+1}: 误差 = {error[0]:.3f}, 平方误差 = {error_squared[0]:.3f}')
    plt.xlabel('特征 X')
    plt.ylabel('目标 y')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 数学解释
print("\n📚 数学解释")
print("="*50)
print("1. MSE = (1/n) × Σ(yᵢ - ŷᵢ)²")
print("2. 目标：找到参数 β₀, β₁，使得 MSE 最小")
print("3. 几何意义：")
print("   - 每个点 (xᵢ, yᵢ) 到拟合线的垂直距离")
print("   - 平方后求和，再求平均")
print("   - 找到使这个总距离最小的直线")

print("\n4. 为什么用平方？")
print("   - 避免正负误差抵消")
print("   - 大误差被更严重惩罚")
print("   - 数学性质好（凸函数）")

print("\n5. 最小化过程：")
print("   - 对 β₀ 求偏导，令其等于0")
print("   - 对 β₁ 求偏导，令其等于0")
print("   - 解方程组得到最佳参数")

# 实际计算演示
print("\n🔢 实际计算演示")
print("="*50)

# 使用sklearn验证
model = LinearRegression()
model.fit(X, y)
sklearn_beta_0 = model.intercept_[0]
sklearn_beta_1 = model.coef_[0][0]
sklearn_mse = calculate_mse(X, y, sklearn_beta_0, sklearn_beta_1)

print(f"我们的最佳参数: β₀ = {best_params[0]:.4f}, β₁ = {best_params[1]:.4f}")
print(f"sklearn的参数:   β₀ = {sklearn_beta_0:.4f}, β₁ = {sklearn_beta_1:.4f}")
print(f"我们的MSE: {best_params[2]:.4f}")
print(f"sklearn的MSE: {sklearn_mse:.4f}")
print(f"差异: {abs(best_params[2] - sklearn_mse):.6f}")

print("\n✅ 结论：最小化MSE就是找到最接近所有数据点的直线！") 