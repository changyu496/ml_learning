"""
基础sklearn操作 - 断网挑战！🔥
目标：不查资料完成sklearn基本操作
时间：20分钟
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

print("🔥 基础sklearn操作挑战 - 断网练习")
print("="*50)
print("⚠️  重要：尽量不查资料，先尝试自己写！")
print("💡 提示：如果卡住了，先思考5分钟再看下面的提示")

# ===== 第1关：创建数据集 =====
print("\n🎯 第1关：创建数据集")
print("任务：创建一个简单的线性关系数据")

# 创建数据
np.random.seed(42)
X = np.random.randn(100, 1) * 10
y = 2 * X.ravel() + 3 + np.random.randn(100) * 5

print(f"数据形状: X={X.shape}, y={y.shape}")
print(f"前5个样本:\nX={X[:5].ravel()}\ny={y[:5]}")

# 任务1.1：数据分割
print("\n📝 任务1.1：将数据分割为训练集和测试集")
print("要求：70%训练，30%测试，随机状态=42")
print("你的代码：")
# 在这里写代码：
# X_train, X_test, y_train, y_test = 

# 任务1.2：检查分割结果
print("\n📝 任务1.2：检查分割结果")
print("你的代码：")
# 在这里写代码：
# print(f"训练集大小: {X_train.shape[0]}")
# print(f"测试集大小: {X_test.shape[0]}")

# ===== 第2关：模型训练 =====
print("\n🎯 第2关：模型训练")

# 任务2.1：创建线性回归模型
print("\n📝 任务2.1：创建线性回归模型")
print("你的代码：")
# 在这里写代码：
# model = 

# 任务2.2：训练模型
print("\n📝 任务2.2：训练模型")
print("你的代码：")
# 在这里写代码：
# model.fit(?, ?)

# 任务2.3：查看模型参数
print("\n📝 任务2.3：查看模型参数")
print("线性回归公式：y = w*x + b")
print("你的代码：")
# 在这里写代码：
# print(f"斜率(w): {model.coef_[0]:.4f}")
# print(f"截距(b): {model.intercept_:.4f}")

# ===== 第3关：模型预测 =====
print("\n🎯 第3关：模型预测")

# 任务3.1：在测试集上预测
print("\n📝 任务3.1：在测试集上预测")
print("你的代码：")
# 在这里写代码：
# y_pred = 

# 任务3.2：显示预测结果
print("\n📝 任务3.2：显示预测结果")
print("你的代码：")
# 在这里写代码：
# print(f"前5个预测结果: {y_pred[:5]}")
# print(f"前5个真实值: {y_test[:5]}")

# ===== 第4关：模型评估 =====
print("\n🎯 第4关：模型评估")

# 任务4.1：计算均方误差(MSE)
print("\n📝 任务4.1：计算均方误差(MSE)")
print("你的代码：")
# 在这里写代码：
# mse = 
# print(f"均方误差: {mse:.4f}")

# 任务4.2：计算R²分数
print("\n📝 任务4.2：计算R²分数")
print("你的代码：")
# 在这里写代码：
# r2 = 
# print(f"R²分数: {r2:.4f}")

# 任务4.3：解释评估结果
print("\n📝 任务4.3：解释评估结果")
print("请回答：")
print("1. MSE越小说明什么？")
print("2. R²接近1说明什么？")
print("3. 这个模型的效果如何？")

# 你的回答：
# 答案1：
# 答案2：
# 答案3：

# ===== 第5关：可视化 =====
print("\n🎯 第5关：可视化")

# 任务5.1：绘制散点图和拟合线
print("\n📝 任务5.1：绘制散点图和拟合线")
print("你的代码：")
# 在这里写代码：
# plt.figure(figsize=(10, 6))
# plt.scatter(X_test, y_test, alpha=0.6, label='真实值')
# plt.plot(X_test, y_pred, 'r-', label='预测值')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.title('线性回归结果')
# plt.legend()
# plt.show()

# 任务5.2：绘制残差图
print("\n📝 任务5.2：绘制残差图")
print("残差 = 真实值 - 预测值")
print("你的代码：")
# 在这里写代码：
# residuals = y_test - y_pred
# plt.figure(figsize=(10, 6))
# plt.scatter(y_pred, residuals, alpha=0.6)
# plt.axhline(y=0, color='r', linestyle='--')
# plt.xlabel('预测值')
# plt.ylabel('残差')
# plt.title('残差图')
# plt.show()

# ===== 第6关：新数据预测 =====
print("\n🎯 第6关：新数据预测")

# 任务6.1：预测新数据
print("\n📝 任务6.1：预测新数据")
print("假设有新的X值：[5, 10, 15]")
print("你的代码：")
# 在这里写代码：
# new_X = np.array([[5], [10], [15]])
# new_pred = 
# print(f"新数据预测结果: {new_pred}")

# 任务6.2：手动验证
print("\n📝 任务6.2：手动验证")
print("使用公式 y = w*x + b 手动计算")
print("你的代码：")
# 在这里写代码：
# w = model.coef_[0]
# b = model.intercept_
# manual_pred = w * new_X.ravel() + b
# print(f"手动计算结果: {manual_pred}")

# ===== 自我检查 =====
print("\n" + "="*50)
print("🏁 完成练习后，请检查：")
print("1. 你能独立完成sklearn的基本操作吗？")
print("2. 你理解模型训练的流程吗？")
print("3. 你知道如何评估模型效果吗？")
print("4. 你能解释各个指标的含义吗？")
print("\n💡 记录下不会的操作，明天重点练习！")

# ===== 提示区域 =====
print("\n" + "="*50)
print("📖 提示区域 (实在不会时再看)")
print("- 数据分割：train_test_split(X, y, test_size=0.3, random_state=42)")
print("- 创建模型：LinearRegression()")
print("- 训练模型：model.fit(X_train, y_train)")
print("- 预测：model.predict(X_test)")
print("- 评估：mean_squared_error(y_true, y_pred), r2_score(y_true, y_pred)")
print("- 模型参数：model.coef_, model.intercept_")
print("="*50) 