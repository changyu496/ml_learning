"""
协方差矩阵详解 - 从零开始理解
"""

import numpy as np
from sklearn.datasets import load_iris

# 加载数据
iris = load_iris()
X = iris.data
print("原始数据形状:", X.shape)
print("前5行数据:")
print(X[:5])

# ===== 步骤1：数据中心化 =====
print("\n" + "="*50)
print("步骤1：数据中心化")
print("为什么要中心化？因为协方差计算需要减去均值")

# 计算每个特征的均值
mean_features = np.mean(X, axis=0)
print("每个特征的均值:", mean_features)

# 中心化数据
X_centered = X - mean_features
print("中心化后的数据均值:", np.mean(X_centered, axis=0))

# ===== 步骤2：手动计算协方差矩阵 =====
print("\n" + "="*50)
print("步骤2：手动计算协方差矩阵")

n_samples = X_centered.shape[0]
n_features = X_centered.shape[1]

print(f"样本数: {n_samples}")
print(f"特征数: {n_features}")

# 方法1：使用矩阵乘法（推荐）
cov_matrix_fast = (X_centered.T @ X_centered) / (n_samples - 1)
print("\n方法1 - 矩阵乘法结果:")
print(cov_matrix_fast)

# 方法2：逐个计算（理解原理）
cov_matrix_manual = np.zeros((n_features, n_features))

for i in range(n_features):
    for j in range(n_features):
        # 计算第i个特征和第j个特征的协方差
        cov_ij = np.sum(X_centered[:, i] * X_centered[:, j]) / (n_samples - 1)
        cov_matrix_manual[i, j] = cov_ij

print("\n方法2 - 手动计算结果:")
print(cov_matrix_manual)

# 验证两种方法结果是否一致
print("\n两种方法结果是否一致:", np.allclose(cov_matrix_fast, cov_matrix_manual))

# ===== 步骤3：理解协方差矩阵的含义 =====
print("\n" + "="*50)
print("步骤3：理解协方差矩阵的含义")

print("协方差矩阵的形状:", cov_matrix_fast.shape)
print("对角线元素（方差）:")
for i in range(n_features):
    print(f"特征{i+1}的方差: {cov_matrix_fast[i, i]:.4f}")

print("\n非对角线元素（协方差）:")
for i in range(n_features):
    for j in range(i+1, n_features):
        print(f"特征{i+1}和特征{j+1}的协方差: {cov_matrix_fast[i, j]:.4f}")

# ===== 步骤4：协方差矩阵的性质 =====
print("\n" + "="*50)
print("步骤4：协方差矩阵的性质")

# 1. 对称性
print("1. 对称性检查:")
print("矩阵是否对称:", np.allclose(cov_matrix_fast, cov_matrix_fast.T))

# 2. 对角线元素非负（方差总是非负的）
print("\n2. 对角线元素（方差）:")
diagonal_elements = np.diag(cov_matrix_fast)
print("所有方差是否非负:", np.all(diagonal_elements >= 0))
print("各特征方差:", diagonal_elements)

# 3. 特征值非负（协方差矩阵是半正定的）
eigenvalues = np.linalg.eigvals(cov_matrix_fast)
print("\n3. 特征值:")
print("所有特征值是否非负:", np.all(eigenvalues >= 0))
print("特征值:", eigenvalues)

# ===== 步骤5：与numpy.cov对比 =====
print("\n" + "="*50)
print("步骤5：与numpy.cov函数对比")

# 使用numpy的cov函数
cov_numpy = np.cov(X.T)
print("numpy.cov结果:")
print(cov_numpy)

print("\n我们的结果与numpy.cov是否一致:", np.allclose(cov_matrix_fast, cov_numpy))

# ===== 步骤6：协方差矩阵的几何意义 =====
print("\n" + "="*50)
print("步骤6：协方差矩阵的几何意义")

print("协方差矩阵描述了数据在各个方向上的分布情况:")
print("- 对角线元素：每个特征方向的方差（数据在该方向的分散程度）")
print("- 非对角线元素：不同特征方向之间的相关性")
print("- 特征值：数据在对应特征向量方向上的方差")
print("- 特征向量：数据变化的主要方向（主成分）")

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix_fast)

print(f"\n特征值（按大小排序）:")
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
print(sorted_eigenvalues)

print(f"\n对应的特征向量（主成分方向）:")
for i, (eigenval, eigenvec) in enumerate(zip(sorted_eigenvalues, eigenvectors.T[sorted_indices])):
    print(f"主成分{i+1} (方差={eigenval:.4f}): {eigenvec}")

print("\n" + "="*50)
print("总结：")
print("1. 协方差矩阵是对称的n×n矩阵")
print("2. 对角线元素是各特征的方差")
print("3. 非对角线元素是特征间的协方差")
print("4. 协方差矩阵的特征值就是PCA中的方差")
print("5. 协方差矩阵的特征向量就是PCA中的主成分方向") 