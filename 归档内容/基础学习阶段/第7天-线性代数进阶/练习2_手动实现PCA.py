"""
练习2：手动实现PCA算法
目标：深入理解PCA的五个步骤，手动实现完整算法
"""

import numpy as np
from sklearn.decomposition import PCA

print("📚 练习2：手动实现PCA")
print("="*50)

# 练习任务
print("\n🎯 任务目标：")
print("1. 手动实现PCA的五个步骤")
print("2. 与sklearn的PCA结果进行比较")
print("3. 理解每个步骤的数学原理")

# 生成测试数据
np.random.seed(42)
X = np.random.randn(100, 4)  # 100个样本，4个特征
# 让特征之间有一定的相关性
X[:, 1] = X[:, 0] + 0.5 * np.random.randn(100)
X[:, 2] = X[:, 0] - 0.3 * X[:, 1] + 0.2 * np.random.randn(100)

print(f"\n📊 测试数据形状: {X.shape}")

def manual_pca(X, n_components=2):
    """
    手动实现PCA算法
    
    参数:
    X: 输入数据 (n_samples, n_features)
    n_components: 要保留的主成分数量
    
    返回:
    X_pca: 降维后的数据
    components: 主成分矩阵
    explained_variance_ratio: 方差解释比例
    """
    
    # TODO: 步骤1 - 数据中心化
    print("📝 步骤1：数据中心化")
    print("提示：X_centered = X - np.mean(X, axis=0)")
    
    # 你的代码：
    X_centered = X - np.mean(X,axis=0)
    
    # TODO: 步骤2 - 计算协方差矩阵
    print("\n📝 步骤2：计算协方差矩阵")
    print("提示：cov_matrix = np.cov(X_centered.T)")
    
    # 你的代码：
    cov_matrix = np.cov(X_centered.T)
    
    # TODO: 步骤3 - 计算特征值和特征向量
    print("\n📝 步骤3：特征值分解")
    print("提示：eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)")
    
    # 你的代码：
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # TODO: 步骤4 - 按特征值大小排序
    print("\n📝 步骤4：按重要性排序")
    print("提示：idx = np.argsort(eigenvalues)[::-1]")
    
    # 你的代码：
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[idx]
    eigenvectors_sorted = eigenvectors[:,idx] # 列向量
    
    # TODO: 步骤5 - 选择主成分并投影
    print("\n📝 步骤5：选择主成分并投影数据")
    print("提示：components = eigenvectors_sorted[:, :n_components]")
    print("      X_pca = X_centered @ components")
    
    # 你的代码：
    components = eigenvectors_sorted[:,:n_components]
    X_pca = X_centered @ components
    
    # TODO: 计算方差解释比例
    print("\n📝 额外任务：计算方差解释比例")
    print("提示：explained_variance_ratio = eigenvalues_sorted / np.sum(eigenvalues_sorted)")
    
    # 你的代码：
    explained_variance_ratio = eigenvalues_sorted/np.sum(eigenvalues_sorted)
    
    return X_pca, components, explained_variance_ratio[:n_components]

# TODO: 调用你的函数
print("\n🔄 测试你的实现：")
X_pca_manual, components_manual, explained_ratio_manual = manual_pca(X, n_components=2)

# TODO: 与sklearn对比
print("\n📊 与sklearn PCA对比：")
pca_sklearn = PCA(n_components=2)
X_pca_sklearn = pca_sklearn.fit_transform(X)

# 比较结果（完成上面的代码后取消注释）
print(f"手动PCA结果形状: {X_pca_manual.shape}")
print(f"sklearn PCA结果形状: {X_pca_sklearn.shape}")
print(f"最大差异: {np.max(np.abs(X_pca_manual - X_pca_sklearn)):.10f}")
print(f"手动PCA方差解释比例: {explained_ratio_manual}")
print(f"sklearn PCA方差解释比例: {pca_sklearn.explained_variance_ratio_}")

print("\n✅ 练习2完成！")
print("💡 核心理解：PCA通过特征值分解找到数据的主要变化方向") 