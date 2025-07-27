"""
PCA手写挑战 - 断网练习！🔥
目标：不看答案实现PCA的核心步骤
时间：30分钟
"""

import numpy as np
from sklearn.datasets import load_iris

print("🔥 PCA手写挑战 - 断网练习")
print("="*50)
print("⚠️  重要：尽量不查资料，回忆PCA的步骤！")
print("💡 核心思想：找到数据变化最大的方向")

# 加载数据
iris = load_iris()
X = iris.data  # 4个特征
print(f"原始数据形状: {X.shape}")
print(f"前5行数据:\n{X[:5]}")

# ===== 第1关：数据中心化 =====
print("\n🎯 第1关：数据中心化")
print("为什么要进行数据中心化？")
print("答案：让数据围绕原点分布，消除均值的影响")

# 任务1.1：计算每个特征的均值
print("\n📝 任务1.1：计算每个特征的均值")
print("你的代码：")
# 在这里写代码：
mean_features = np.mean(X,axis=0)

# 任务1.2：中心化数据
print("\n📝 任务1.2：中心化数据")
print("公式：X_centered = X - mean")
print("你的代码：")
# 在这里写代码：
X_centered = X - mean_features

# 任务1.3：验证中心化结果
print("\n📝 任务1.3：验证中心化结果")
print("中心化后的数据均值应该接近0")
print("你的代码：")
# 在这里写代码：
print(f"中心化后的均值: {np.mean(X_centered,axis=0)}")

# ===== 第2关：计算协方差矩阵 =====
print("\n🎯 第2关：计算协方差矩阵")
print("协方差矩阵反映了什么？")
print("答案：特征之间的关系强度")

# 任务2.1：计算协方差矩阵
print("\n📝 任务2.1：计算协方差矩阵")
print("公式：Cov = (X_centered.T @ X_centered) / (n-1)")
print("你的代码：")
# 在这里写代码：
n = X_centered.shape[0]
cov_matrix = (X_centered.T @ X_centered)/(n-1)

# 任务2.2：验证协方差矩阵
print("\n📝 任务2.2：验证协方差矩阵")
print("协方差矩阵应该是对称的")
print("你的代码：")
# 在这里写代码：
print(f"协方差矩阵形状: {cov_matrix.shape}")
print(f"协方差矩阵:\n{cov_matrix}")

# ===== 第3关：特征值分解 =====
print("\n🎯 第3关：特征值分解")
print("为什么要对协方差矩阵进行特征值分解？")
print("答案：找到数据变化最大的方向（主成分）")

# 任务3.1：计算特征值和特征向量
print("\n📝 任务3.1：计算特征值和特征向量")
print("你的代码：")
# 在这里写代码：
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# 任务3.2：显示结果
print("\n📝 任务3.2：显示特征值和特征向量")
print("你的代码：")
# 在这里写代码：
print(f"特征值: {eigenvalues}")
print(f"特征向量形状: {eigenvectors}")

# ===== 第4关：排序特征值 =====
print("\n🎯 第4关：排序特征值")
print("为什么要对特征值进行排序？")
print("答案：按重要性排序，最大的特征值对应最重要的主成分")

# 任务4.1：按特征值大小排序
print("\n📝 任务4.1：按特征值大小排序")
print("提示：使用 np.argsort()[::-1] 进行倒序排序")
print("你的代码：")
# 在这里写代码：
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:,sorted_indices]

# 任务4.2：显示排序结果
print("\n📝 任务4.2：显示排序结果")
print("你的代码：")
# 在这里写代码：
print(f"排序后的特征值: {sorted_eigenvalues}")
print("特征值从大到小，说明重要性依次递减")

# ===== 第5关：选择主成分 =====
print("\n🎯 第5关：选择主成分")
print("我们要降维到2维，所以选择前2个主成分")

# 任务5.1：选择前2个主成分
print("\n📝 任务5.1：选择前2个主成分")
print("你的代码：")
# 在这里写代码：
n_components = 2
W = sorted_eigenvectors[:, :n_components]
print(f"变换矩阵W的形状: {W.shape}")

# 任务5.2：计算解释方差比例
print("\n📝 任务5.2：计算解释方差比例")
print("前2个主成分能解释多少原始数据的变化？")
print("你的代码：")
# 在这里写代码：
total_variance = np.sum(eigenvalues)
explained_variance = np.sum(sorted_eigenvalues[:n_components])
explained_ratio = explained_variance/total_variance
print(f"前2个主成分解释了 {explained_ratio:.2%} 的方差")

# ===== 第6关：数据投影 =====
print("\n🎯 第6关：数据投影")
print("将原始数据投影到新的坐标系中")

# 任务6.1：投影数据
print("\n📝 任务6.1：投影数据")
print("公式：X_pca = X_centered @ W")
print("你的代码：")
# 在这里写代码：
X_pca = X_centered @ W
print(f"PCA后的数据形状: {X_pca.shape}")

# 任务6.2：验证投影结果
print("\n📝 任务6.2：验证投影结果")
print("你的代码：")
# 在这里写代码：
print(f"原始数据: {X.shape}")
print(f"PCA后数据: {X_pca.shape}")
print(f"前5行PCA结果:\n{X_pca[:5]}")

# ===== 第7关：理解PCA结果 =====
print("\n🎯 第7关：理解PCA结果")

# 任务7.1：分析主成分的含义
print("\n📝 任务7.1：分析主成分的含义")
print("请思考并回答：")
print("1. 第一主成分代表什么？")
print("2. 第二主成分代表什么？")
print("3. 为什么降维后还能保持大部分信息？")

# 你的答案：
# 答案1：第一主成分是数据变化___的方向
# 答案2：第二主成分是在与第一主成分___的方向中，变化最大的方向
# 答案3：因为我们保留了___的主成分

# 任务7.2：与sklearn结果对比
print("\n📝 任务7.2：与sklearn结果对比")
print("你的代码：")
# 在这里写代码：
# from sklearn.decomposition import PCA
# pca_sklearn = PCA(n_components=2)
# X_sklearn = pca_sklearn.fit_transform(X)
# print(f"我们的结果前5行:\n{X_pca[:5]}")
# print(f"sklearn结果前5行:\n{X_sklearn[:5]}")
# print("差异主要来源于符号不同，但绝对值应该相近")

# ===== 自我检查 =====
print("\n" + "="*50)
print("🏁 完成练习后，请检查：")
print("1. 你能独立完成PCA的每个步骤吗？")
print("2. 你理解每个步骤的目的吗？")
print("3. 你知道为什么要这样做吗？")
print("4. 你的结果合理吗？")
print("\n💡 记录下困难的步骤，重点复习！")

# ===== 提示区域 =====
print("\n" + "="*50)
print("📖 提示区域 (实在不会时再看)")
print("- 计算均值：np.mean(X, axis=0)")
print("- 中心化：X - mean")
print("- 协方差矩阵：(X.T @ X) / (n-1)")
print("- 特征值分解：np.linalg.eig()")
print("- 排序：np.argsort()[::-1]")
print("- 矩阵乘法：A @ B")
print("="*50) 