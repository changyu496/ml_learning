#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手写PCA算法实现
作者: ChangYu
日期: 2025-07-28
目标: 通过手写实现加深对PCA的理解
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def my_pca(X, n_components):
    """
    手写PCA算法实现
    
    参数:
    X: 输入数据，形状为 (n_samples, n_features)
    n_components: 要保留的主成分数量
    
    返回:
    X_pca: 降维后的数据
    components: 主成分向量
    explained_variance_ratio: 解释方差比例
    """
    # 1. 数据标准化 (去中心化)
    X_centered = X - X.mean(axis=0)
    print(f"数据标准化完成，形状: {X_centered.shape}")
    
    # 2. 计算协方差矩阵
    cov_matrix = np.cov(X_centered.T)
    print(f"协方差矩阵形状: {cov_matrix.shape}")
    print(f"协方差矩阵是对称的: {np.allclose(cov_matrix, cov_matrix.T)}")
    
    # 3. 特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    print(f"特征值: {eigenvalues}")
    print(f"特征向量形状: {eigenvectors.shape}")
    
    # 4. 按特征值大小排序
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # 5. 选择前n_components个主成分
    selected_components = sorted_eigenvectors[:, :n_components]
    print(f"选择的主成分形状: {selected_components.shape}")
    
    # 6. 投影到主成分空间
    X_pca = X_centered @ selected_components
    
    # 7. 计算解释方差比例
    explained_variance_ratio = sorted_eigenvalues[:n_components] / np.sum(eigenvalues)
    
    return X_pca, selected_components, explained_variance_ratio

def compare_with_sklearn(X, n_components=2):
    """
    与sklearn的PCA结果对比
    """
    print("=" * 50)
    print("手写PCA vs sklearn PCA 对比")
    print("=" * 50)
    
    # 手写PCA
    print("\n1. 手写PCA实现:")
    X_pca_my, components_my, ratio_my = my_pca(X, n_components)
    print(f"降维后数据形状: {X_pca_my.shape}")
    print(f"解释方差比例: {ratio_my}")
    print(f"累计解释方差: {np.sum(ratio_my):.4f}")
    
    # sklearn PCA
    print("\n2. sklearn PCA实现:")
    pca_sklearn = PCA(n_components=n_components)
    X_pca_sklearn = pca_sklearn.fit_transform(X)
    print(f"降维后数据形状: {X_pca_sklearn.shape}")
    print(f"解释方差比例: {pca_sklearn.explained_variance_ratio_}")
    print(f"累计解释方差: {np.sum(pca_sklearn.explained_variance_ratio_):.4f}")
    
    # 比较结果
    print("\n3. 结果对比:")
    print(f"数据差异 (手写 vs sklearn): {np.mean(np.abs(X_pca_my - X_pca_sklearn)):.6f}")
    print(f"解释方差比例差异: {np.mean(np.abs(ratio_my - pca_sklearn.explained_variance_ratio_)):.6f}")
    
    return X_pca_my, X_pca_sklearn, ratio_my, pca_sklearn.explained_variance_ratio_

def visualize_pca_results(X_pca_my, X_pca_sklearn, iris):
    """
    可视化PCA结果
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 手写PCA结果
    scatter1 = axes[0].scatter(X_pca_my[:, 0], X_pca_my[:, 1], 
                               c=iris.target, cmap='viridis', alpha=0.7)
    axes[0].set_title('手写PCA结果')
    axes[0].set_xlabel('第一主成分')
    axes[0].set_ylabel('第二主成分')
    axes[0].grid(True, alpha=0.3)
    
    # sklearn PCA结果
    scatter2 = axes[1].scatter(X_pca_sklearn[:, 0], X_pca_sklearn[:, 1], 
                               c=iris.target, cmap='viridis', alpha=0.7)
    axes[1].set_title('sklearn PCA结果')
    axes[1].set_xlabel('第一主成分')
    axes[1].set_ylabel('第二主成分')
    axes[1].grid(True, alpha=0.3)
    
    # 添加图例
    legend1 = axes[0].legend(*scatter1.legend_elements(), title="鸢尾花品种")
    legend2 = axes[1].legend(*scatter2.legend_elements(), title="鸢尾花品种")
    
    plt.tight_layout()
    plt.show()

def main():
    """
    主函数：演示手写PCA算法
    """
    print("🚀 开始手写PCA算法演示")
    print("=" * 50)
    
    # 1. 加载数据
    print("1. 加载Iris数据集")
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    print(f"数据形状: {X.shape}")
    print(f"特征名称: {iris.feature_names}")
    print(f"目标类别: {iris.target_names}")
    
    # 2. 数据标准化
    print("\n2. 数据标准化")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"标准化后数据形状: {X_scaled.shape}")
    
    # 3. 手写PCA vs sklearn PCA
    print("\n3. 执行PCA降维")
    X_pca_my, X_pca_sklearn, ratio_my, ratio_sklearn = compare_with_sklearn(X_scaled, n_components=2)
    
    # 4. 可视化结果
    print("\n4. 可视化PCA结果")
    visualize_pca_results(X_pca_my, X_pca_sklearn, iris)
    
    # 5. 详细分析
    print("\n5. 详细分析")
    print(f"原始特征数量: {X.shape[1]}")
    print(f"降维后特征数量: {X_pca_my.shape[1]}")
    print(f"降维比例: {X_pca_my.shape[1] / X.shape[1]:.2%}")
    print(f"信息保留比例: {np.sum(ratio_my):.2%}")
    
    print("\n🎉 手写PCA算法演示完成！")

if __name__ == "__main__":
    main() 