"""
手写线性回归挑战 - 断网练习！🔥
目标：从零实现线性回归算法
时间：40分钟
"""

import numpy as np
import matplotlib.pyplot as plt

print("🔥 手写线性回归挑战 - 断网练习")
print("="*50)
print("⚠️  重要：尽量不查资料，理解每个步骤！")
print("💡 核心思想：找到最佳的直线拟合数据")

# ===== 第1关：理解线性回归 =====
print("\n🎯 第1关：理解线性回归")
print("线性回归公式：y = w*x + b")
print("目标：找到最佳的w（斜率）和b（截距）")

# 创建示例数据
np.random.seed(42)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]) + np.random.randn(10) * 2

print(f"数据：X = {X}")
print(f"数据：y = {y}")

# 任务1.1：可视化数据
print("\n📝 任务1.1：可视化数据")
print("你的代码：")
# 在这里写代码：
# plt.figure(figsize=(10, 6))
# plt.scatter(X, y, alpha=0.6)
# plt.xlabel('X')
# plt.ylabel('y')
# plt.title('原始数据')
# plt.show()

# ===== 第2关：最小二乘法公式 =====
print("\n🎯 第2关：最小二乘法公式")
print("最小二乘法：通过最小化平方误差来找到最佳参数")
print("公式：")
print("w = (Σ(x*y) - n*mean(x)*mean(y)) / (Σ(x²) - n*mean(x)²)")
print("b = mean(y) - w*mean(x)")

# 任务2.1：计算均值
print("\n📝 任务2.1：计算X和y的均值")
print("你的代码：")
# 在这里写代码：
# mean_x = 
# mean_y = 
# print(f"X的均值: {mean_x}")
# print(f"y的均值: {mean_y}")

# 任务2.2：计算需要的统计量
print("\n📝 任务2.2：计算统计量")
print("你的代码：")
# 在这里写代码：
# sum_xy = np.sum(X * y)
# sum_x_squared = np.sum(X ** 2)
# n = len(X)
# print(f"Σ(x*y) = {sum_xy}")
# print(f"Σ(x²) = {sum_x_squared}")
# print(f"样本数量 n = {n}")

# 任务2.3：计算w和b
print("\n📝 任务2.3：使用最小二乘法计算w和b")
print("你的代码：")
# 在这里写代码：
# w = (sum_xy - n * mean_x * mean_y) / (sum_x_squared - n * mean_x**2)
# b = mean_y - w * mean_x
# print(f"斜率 w = {w:.4f}")
# print(f"截距 b = {b:.4f}")

# ===== 第3关：矩阵方式计算 =====
print("\n🎯 第3关：矩阵方式计算")
print("线性回归的矩阵形式：θ = (X^T * X)^(-1) * X^T * y")
print("其中 θ = [b, w]，X = [1, x] (增加偏置列)")

# 任务3.1：构建设计矩阵
print("\n📝 任务3.1：构建设计矩阵")
print("你的代码：")
# 在这里写代码：
# X_matrix = np.column_stack((np.ones(len(X)), X))
# print(f"设计矩阵 X_matrix 形状: {X_matrix.shape}")
# print(f"前5行:\n{X_matrix[:5]}")

# 任务3.2：计算参数
print("\n📝 任务3.2：使用矩阵公式计算参数")
print("你的代码：")
# 在这里写代码：
# theta = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ y
# b_matrix = theta[0]
# w_matrix = theta[1]
# print(f"矩阵方法：b = {b_matrix:.4f}, w = {w_matrix:.4f}")

# 任务3.3：验证结果
print("\n📝 任务3.3：验证两种方法的结果")
print("你的代码：")
# 在这里写代码：
# print(f"最小二乘法：b = {b:.4f}, w = {w:.4f}")
# print(f"矩阵方法：  b = {b_matrix:.4f}, w = {w_matrix:.4f}")
# print(f"结果是否一致: {np.allclose([b, w], [b_matrix, w_matrix])}")

# ===== 第4关：预测和评估 =====
print("\n🎯 第4关：预测和评估")

# 任务4.1：进行预测
print("\n📝 任务4.1：使用我们的模型进行预测")
print("你的代码：")
# 在这里写代码：
# y_pred = w * X + b
# print(f"预测结果: {y_pred}")

# 任务4.2：计算均方误差
print("\n📝 任务4.2：计算均方误差(MSE)")
print("MSE = (1/n) * Σ(y_true - y_pred)²")
print("你的代码：")
# 在这里写代码：
# mse = np.mean((y - y_pred)**2)
# print(f"均方误差: {mse:.4f}")

# 任务4.3：计算R²分数
print("\n📝 任务4.3：计算R²分数")
print("R² = 1 - (SS_res / SS_tot)")
print("SS_res = Σ(y_true - y_pred)²")
print("SS_tot = Σ(y_true - mean(y))²")
print("你的代码：")
# 在这里写代码：
# ss_res = np.sum((y - y_pred)**2)
# ss_tot = np.sum((y - mean_y)**2)
# r2 = 1 - (ss_res / ss_tot)
# print(f"R²分数: {r2:.4f}")

# ===== 第5关：可视化结果 =====
print("\n🎯 第5关：可视化结果")

# 任务5.1：绘制拟合线
print("\n📝 任务5.1：绘制数据和拟合线")
print("你的代码：")
# 在这里写代码：
# plt.figure(figsize=(10, 6))
# plt.scatter(X, y, alpha=0.6, label='真实数据')
# plt.plot(X, y_pred, 'r-', label=f'拟合线: y = {w:.2f}x + {b:.2f}')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.title('手写线性回归结果')
# plt.legend()
# plt.show()

# 任务5.2：绘制残差图
print("\n📝 任务5.2：绘制残差图")
print("你的代码：")
# 在这里写代码：
# residuals = y - y_pred
# plt.figure(figsize=(10, 6))
# plt.scatter(y_pred, residuals, alpha=0.6)
# plt.axhline(y=0, color='r', linestyle='--')
# plt.xlabel('预测值')
# plt.ylabel('残差')
# plt.title('残差图')
# plt.show()

# ===== 第6关：梯度下降实现 =====
print("\n🎯 第6关：梯度下降实现（挑战）")
print("使用梯度下降优化算法来找到最佳参数")

# 任务6.1：实现梯度下降
print("\n📝 任务6.1：实现梯度下降算法")
print("你的代码：")
# 在这里写代码：
# def gradient_descent(X, y, lr=0.01, epochs=1000):
#     # 初始化参数
#     w = 0.0
#     b = 0.0
#     n = len(X)
#     
#     for epoch in range(epochs):
#         # 预测
#         y_pred = w * X + b
#         
#         # 计算损失函数的梯度
#         dw = (-2/n) * np.sum(X * (y - y_pred))
#         db = (-2/n) * np.sum(y - y_pred)
#         
#         # 更新参数
#         w = w - lr * dw
#         b = b - lr * db
#         
#         # 每100轮打印一次损失
#         if epoch % 100 == 0:
#             loss = np.mean((y - y_pred)**2)
#             print(f"Epoch {epoch}, Loss: {loss:.4f}")
#     
#     return w, b

# 任务6.2：运行梯度下降
print("\n📝 任务6.2：运行梯度下降算法")
print("你的代码：")
# 在这里写代码：
# w_gd, b_gd = gradient_descent(X, y)
# print(f"梯度下降结果：w = {w_gd:.4f}, b = {b_gd:.4f}")

# 任务6.3：对比三种方法
print("\n📝 任务6.3：对比三种方法的结果")
print("你的代码：")
# 在这里写代码：
# print("参数对比：")
# print(f"最小二乘法：w = {w:.4f}, b = {b:.4f}")
# print(f"矩阵方法：  w = {w_matrix:.4f}, b = {b_matrix:.4f}")
# print(f"梯度下降：  w = {w_gd:.4f}, b = {b_gd:.4f}")

# ===== 第7关：与sklearn对比 =====
print("\n🎯 第7关：与sklearn对比")

# 任务7.1：使用sklearn验证
print("\n📝 任务7.1：使用sklearn验证我们的结果")
print("你的代码：")
# 在这里写代码：
# from sklearn.linear_model import LinearRegression
# sklearn_model = LinearRegression()
# sklearn_model.fit(X.reshape(-1, 1), y)
# print(f"sklearn结果：w = {sklearn_model.coef_[0]:.4f}, b = {sklearn_model.intercept_:.4f}")

# ===== 自我检查 =====
print("\n" + "="*50)
print("🏁 完成练习后，请检查：")
print("1. 你理解最小二乘法的原理了吗？")
print("2. 你能解释为什么三种方法结果相同吗？")
print("3. 你知道何时使用梯度下降吗？")
print("4. 你能从零实现线性回归了吗？")
print("\n💡 记录下困难的地方，重点复习！")

# ===== 提示区域 =====
print("\n" + "="*50)
print("📖 提示区域 (实在不会时再看)")
print("- 均值：np.mean(array)")
print("- 矩阵乘法：A @ B")
print("- 矩阵转置：A.T")
print("- 矩阵求逆：np.linalg.inv(A)")
print("- 数组形状：array.reshape(-1, 1)")
print("="*50) 