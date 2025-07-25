"""
特征值计算挑战 - 断网练习！🔥
目标：手写特征值计算的每个步骤
时间：30分钟
"""

import numpy as np

print("🔥 特征值计算挑战 - 断网练习")
print("="*50)
print("⚠️  重要：尽量不查资料，回忆昨天学的内容！")
print("💡 核心公式：Av = λv")

# ===== 第1关：理解特征值和特征向量 =====
print("\n🎯 第1关：理解特征值和特征向量")
print("回忆：特征值和特征向量的定义是什么？")
print("定义：如果 Av = λv，那么 v 是特征向量，λ 是特征值")

# 给定一个简单的2x2矩阵
A = np.array([[3, 1], 
              [1, 3]])
print(f"\n📊 给定矩阵 A:\n{A}")

# 任务1.1：使用numpy计算特征值和特征向量
print("\n📝 任务1.1：计算特征值和特征向量")
print("你的代码：")
# 在这里写代码：
eigenvalues, eigenvectors = np.linalg.eig(A)

# 任务1.2：显示结果
print("\n📝 任务1.2：显示计算结果")
print("你的代码：")
# 在这里写代码：
print(f"特征值: {eigenvalues}")
print(f"特征向量:\n{eigenvectors}")

# ===== 第2关：手动验证 =====
print("\n🎯 第2关：手动验证 Av = λv")
print("对每个特征值和特征向量，验证 Av = λv")

# 任务2.1：验证第一个特征值和特征向量
print("\n📝 任务2.1：验证第一个特征值和特征向量")
print("步骤：")
print("1. 获取第一个特征值λ₁")
print("2. 获取第一个特征向量v₁")
print("3. 计算 Av₁")
print("4. 计算 λ₁v₁")
print("5. 比较两者是否相等")

print("\n你的代码：")
# 在这里写代码：
λ1 = eigenvalues[0]
v1 = eigenvectors[:,0]
Av1 = A @ v1
λ1v1 = λ1 * v1
print(f"Av₁ = {Av1}")
print(f"λ₁v₁ = {λ1v1}")
print(f"是否相等: {np.abs(Av1-λ1v1) ==  0}")

# 任务2.2：验证第二个特征值和特征向量
print("\n📝 任务2.2：验证第二个特征值和特征向量")
print("你的代码：")
# 在这里写代码：
λ2 = eigenvalues[1]
v2 = eigenvectors[:,1]
Av2 = A @ v2
λ2v2 = λ2 * v2
print(f"Av₂ = {Av2}")
print(f"λ₂v₂ = {λ2v2}")
print(f"是否相等: {np.abs(Av2-λ2v2) ==  0}")

# ===== 第3关：理解几何意义 =====
print("\n🎯 第3关：理解几何意义")
print("特征值告诉我们什么？特征向量告诉我们什么？")

# 任务3.1：分析特征值
print("\n📝 任务3.1：分析特征值的含义")
print("请思考并回答：")
print("1. 哪个特征值更大？")
print("2. 特征值的大小说明什么？")
print("3. 如果特征值是0，说明什么？")

# 你的答案：
# 答案1：
# 答案2：
# 答案3：

# 任务3.2：分析特征向量
print("\n📝 任务3.2：分析特征向量的含义")
print("请思考并回答：")
print("1. 特征向量的方向有什么特殊性？")
print("2. 为什么特征向量在矩阵变换后方向不变？")
print("3. 特征向量的长度重要吗？")

# 你的答案：
# 答案1：
# 答案2：
# 答案3：

# ===== 第4关：实际应用理解 =====
print("\n🎯 第4关：实际应用理解")

# 任务4.1：创建自己的矩阵
print("\n📝 任务4.1：创建一个对角矩阵")
print("对角矩阵的特征值有什么特点？")

# 在这里写代码：
# diagonal_matrix = np.array([[5, 0], 
#                            [0, 2]])
# diag_eigenvalues, diag_eigenvectors = 

# 任务4.2：观察规律
print("\n📝 任务4.2：观察对角矩阵的规律")
print("你的代码：")
# 在这里写代码：
# print(f"对角矩阵:\n{}")
# print(f"特征值: {}")
# print(f"特征向量:\n{}")
# print("规律：对角矩阵的特征值就是___，特征向量是___")

# ===== 第5关：常见错误检查 =====
print("\n🎯 第5关：常见错误检查")
print("检查你是否犯了这些常见错误：")

# 任务5.1：特征向量的获取
print("\n📝 任务5.1：特征向量的正确获取")
print("错误写法：v = eigenvectors[i]")
print("正确写法：v = eigenvectors[:, i]")
print("请解释为什么：")

# 你的解释：
# 解释：eigenvectors是一个矩阵，每___是一个特征向量

# 任务5.2：特征值的顺序
print("\n📝 任务5.2：特征值的顺序")
print("numpy.linalg.eig()返回的特征值有固定顺序吗？")
print("答案：___")
print("如果想要按大小排序，应该怎么做？")

# 你的代码：
# sorted_indices = 
# sorted_eigenvalues = 
# sorted_eigenvectors = 

# ===== 自我检查 =====
print("\n" + "="*50)
print("🏁 完成练习后，请检查：")
print("1. 你能独立写出特征值计算的代码吗？")
print("2. 你理解Av = λv的含义了吗？")
print("3. 你能解释特征值和特征向量的几何意义吗？")
print("4. 你记住了常见的错误写法吗？")
print("\n💡 记录下不清楚的概念，重点复习！")

# ===== 提示区域 =====
print("\n" + "="*50)
print("📖 提示区域 (实在不会时再看)")
print("- 计算特征值：np.linalg.eig(matrix)")
print("- 获取特征向量：eigenvectors[:, i] (第i列)")
print("- 矩阵乘法：A @ v")
print("- 检查相等：np.allclose(a, b)")
print("- 排序：np.argsort()")
print("="*50) 