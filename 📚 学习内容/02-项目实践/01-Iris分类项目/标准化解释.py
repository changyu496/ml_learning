#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标准化解释 - 帮你理解fit_transform和transform的区别
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# 加载数据
iris = load_iris()
X = iris.data
y = iris.target

# 模拟数据分割
X_train = X[:120]  # 前120个样本作为训练集
X_test = X[120:]    # 后30个样本作为测试集

print("=== 原始数据 ===")
print(f"训练集形状: {X_train.shape}")
print(f"测试集形状: {X_test.shape}")

print("\n=== 训练集前5行 ===")
print(X_train[:5])

print("\n=== 测试集前5行 ===")
print(X_test[:5])

# 创建标准化器
scaler = StandardScaler()

print("\n=== 标准化过程 ===")

# 1. fit_transform (训练集)
print("1. 训练集标准化:")
print("   fit: 学习训练数据的统计信息")
print("   transform: 使用学到的信息转换数据")

# 计算训练集的统计信息
train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)
print(f"   训练集均值: {train_mean}")
print(f"   训练集标准差: {train_std}")

# 标准化训练集
X_train_scaled = scaler.fit_transform(X_train)
print(f"   训练集标准化后: {X_train_scaled[:3]}")

# 2. transform (测试集)
print("\n2. 测试集标准化:")
print("   只用transform，不用fit")
print("   使用训练时学到的统计信息")

# 手动计算测试集标准化（使用训练集的统计信息）
X_test_scaled_manual = (X_test - train_mean) / train_std
X_test_scaled = scaler.transform(X_test)

print(f"   测试集标准化后: {X_test_scaled[:3]}")
print(f"   手动计算结果: {X_test_scaled_manual[:3]}")

print("\n=== 验证结果 ===")
print("手动计算和sklearn结果是否一致:")
print(np.allclose(X_test_scaled, X_test_scaled_manual))

print("\n=== 重要概念 ===")
print("1. fit_transform: 学习+转换（用于训练数据）")
print("2. transform: 只转换（用于测试数据）")
print("3. 原因: 避免数据泄露，确保模型泛化能力")
print("4. 实际应用: 训练时学习统计信息，预测时使用相同的信息") 