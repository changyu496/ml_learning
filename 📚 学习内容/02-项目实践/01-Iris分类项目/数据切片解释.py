#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据切片解释 - 帮你理解X[:, i]的含义
"""

import numpy as np
from sklearn.datasets import load_iris

# 加载数据
iris = load_iris()
X = iris.data
y = iris.target

print("=== 数据形状 ===")
print(f"X的形状: {X.shape}")  # (150, 4)
print(f"y的形状: {y.shape}")  # (150,)

print("\n=== 数据切片解释 ===")
print("X[行, 列] - 第一个数字是行，第二个数字是列")

print("\n1. X[:, 0] - 所有行的第0列（花萼长度）")
print(f"形状: {X[:, 0].shape}")  # (150,)
print(f"前10个值: {X[:10, 0]}")

print("\n2. X[:, 1] - 所有行的第1列（花萼宽度）")
print(f"形状: {X[:, 1].shape}")  # (150,)
print(f"前10个值: {X[:10, 1]}")

print("\n3. X[:, 2] - 所有行的第2列（花瓣长度）")
print(f"形状: {X[:, 2].shape}")  # (150,)
print(f"前10个值: {X[:10, 2]}")

print("\n4. X[:, 3] - 所有行的第3列（花瓣宽度）")
print(f"形状: {X[:, 3].shape}")  # (150,)
print(f"前10个值: {X[:10, 3]}")

print("\n=== 循环中的i值 ===")
for i, feature in enumerate(iris.feature_names):
    print(f"i={i}: {feature}")
    print(f"  数据: X[:, {i}]")
    print(f"  形状: {X[:, i].shape}")
    print(f"  平均值: {X[:, i].mean():.2f}")
    print()

print("=== 子图位置计算 ===")
for i in range(4):
    row, col = i // 2, i % 2
    print(f"i={i}: row={row}, col={col} (位置: 第{row+1}行第{col+1}列)") 