"""
代码理解挑战 - 断网练习！🔥
目标：理解代码的每一行在做什么
时间：40分钟
"""

import numpy as np

print("🔥 代码理解挑战 - 断网练习")
print("="*50)
print("⚠️  重要：请解释每行代码的作用，不要只是复述！")
print("💡 思考：这行代码为什么要这样写？")

# ===== 挑战1：特征值计算代码理解 =====
print("\n🎯 挑战1：特征值计算代码理解")
print("请解释下面每行代码的作用：")

print("\n代码片段1：")
print("```python")
print("A = np.array([[4, 2], [2, 1]])")
print("eigenvalues, eigenvectors = np.linalg.eig(A)")
print("```")

print("\n请解释：")
print("第1行：")
# 你的解释：

print("第2行：")
# 你的解释：

print("\n代码片段2：")
print("```python")
print("for i in range(len(eigenvalues)):")
print("    v = eigenvectors[:, i]")
print("    λ = eigenvalues[i]")
print("    Av = A @ v")
print("    λv = λ * v")
print("    print(np.allclose(Av, λv))")
print("```")

print("\n请解释：")
print("第1行：")
# 你的解释：

print("第2行：")
# 你的解释：

print("第3行：")
# 你的解释：

print("第4行：")
# 你的解释：

print("第5行：")
# 你的解释：

print("第6行：")
# 你的解释：

# ===== 挑战2：PCA代码理解 =====
print("\n🎯 挑战2：PCA代码理解")
print("请解释下面每行代码的作用：")

print("\n代码片段3：")
print("```python")
print("X_centered = X - np.mean(X, axis=0)")
print("cov_matrix = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1)")
print("eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)")
print("```")

print("\n请解释：")
print("第1行：")
# 你的解释：

print("第2行：")
# 你的解释：

print("第3行：")
# 你的解释：

print("\n代码片段4：")
print("```python")
print("sorted_indices = np.argsort(eigenvalues)[::-1]")
print("sorted_eigenvalues = eigenvalues[sorted_indices]")
print("sorted_eigenvectors = eigenvectors[:, sorted_indices]")
print("```")

print("\n请解释：")
print("第1行：")
# 你的解释：

print("第2行：")
# 你的解释：

print("第3行：")
# 你的解释：

print("\n代码片段5：")
print("```python")
print("W = sorted_eigenvectors[:, :2]")
print("X_pca = X_centered @ W")
print("```")

print("\n请解释：")
print("第1行：")
# 你的解释：

print("第2行：")
# 你的解释：

# ===== 挑战3：数组操作理解 =====
print("\n🎯 挑战3：数组操作理解")
print("请解释下面每行代码的作用：")

print("\n代码片段6：")
print("```python")
print("A = np.array([[1, 2, 3], [4, 5, 6]])")
print("B = A[:, 1]")
print("C = A[0, :]")
print("D = A.T")
print("```")

print("\n请解释：")
print("第1行：")
# 你的解释：

print("第2行：")
# 你的解释：

print("第3行：")
# 你的解释：

print("第4行：")
# 你的解释：

print("\n代码片段7：")
print("```python")
print("result = np.zeros((2, 2))")
print("result[0, 0] = np.sum(A[:, 0])")
print("result[1, 1] = np.mean(A[1, :])")
print("```")

print("\n请解释：")
print("第1行：")
# 你的解释：

print("第2行：")
# 你的解释：

print("第3行：")
# 你的解释：

# ===== 挑战4：常见错误理解 =====
print("\n🎯 挑战4：常见错误理解")
print("请指出下面代码的错误并说明原因：")

print("\n错误代码1：")
print("```python")
print("eigenvalues, eigenvectors = np.linalg.eig(A)")
print("v1 = eigenvectors[0]  # 错误在这里")
print("```")

print("\n错误说明：")
# 你的说明：

print("正确写法：")
# 你的代码：

print("\n错误代码2：")
print("```python")
print("X_centered = X - X.mean()  # 错误在这里")
print("```")

print("\n错误说明：")
# 你的说明：

print("正确写法：")
# 你的代码：

print("\n错误代码3：")
print("```python")
print("cov_matrix = X_centered.T * X_centered  # 错误在这里")
print("```")

print("\n错误说明：")
# 你的说明：

print("正确写法：")
# 你的代码：

# ===== 挑战5：代码优化理解 =====
print("\n🎯 挑战5：代码优化理解")
print("请解释为什么第二种写法更好：")

print("\n写法对比1：")
print("写法A：")
print("```python")
print("result = []")
print("for i in range(len(eigenvalues)):")
print("    result.append(eigenvalues[i] * eigenvectors[:, i])")
print("```")

print("写法B：")
print("```python")
print("result = eigenvalues * eigenvectors.T")
print("```")

print("\n为什么写法B更好？")
# 你的解释：

print("\n写法对比2：")
print("写法A：")
print("```python")
print("X_centered = np.zeros_like(X)")
print("for i in range(X.shape[1]):")
print("    X_centered[:, i] = X[:, i] - np.mean(X[:, i])")
print("```")

print("写法B：")
print("```python")
print("X_centered = X - np.mean(X, axis=0)")
print("```")

print("\n为什么写法B更好？")
# 你的解释：

# ===== 挑战6：理解输出结果 =====
print("\n🎯 挑战6：理解输出结果")
print("请预测下面代码的输出结果：")

print("\n代码：")
print("```python")
print("A = np.array([[2, 0], [0, 3]])")
print("eigenvalues, eigenvectors = np.linalg.eig(A)")
print("print(eigenvalues)")
print("print(eigenvectors)")
print("```")

print("\n预测输出：")
print("eigenvalues = ")
# 你的预测：

print("eigenvectors = ")
# 你的预测：

print("解释为什么：")
# 你的解释：

print("\n代码：")
print("```python")
print("X = np.array([[1, 2], [3, 4]])")
print("X_centered = X - np.mean(X, axis=0)")
print("print(X_centered)")
print("```")

print("\n预测输出：")
print("X_centered = ")
# 你的预测：

print("计算过程：")
# 你的计算：

# ===== 自我检查 =====
print("\n" + "="*50)
print("🏁 完成练习后，请检查：")
print("1. 你能准确解释每行代码的作用吗？")
print("2. 你理解为什么要这样写吗？")
print("3. 你能识别常见的错误吗？")
print("4. 你能预测代码的输出结果吗？")
print("\n💡 记录下不理解的地方，重点学习！")

# ===== 答案提示 =====
print("\n" + "="*50)
print("📖 答案提示 (完成后再看)")
print("- 数组切片：[:, i] 表示所有行的第i列")
print("- 矩阵乘法：@ 用于矩阵乘法，* 用于逐元素乘法")
print("- axis=0 表示沿着行方向（每列），axis=1 表示沿着列方向（每行）")
print("- np.argsort()[::-1] 表示按降序排列的索引")
print("- 对角矩阵的特征值就是对角线元素")
print("="*50) 