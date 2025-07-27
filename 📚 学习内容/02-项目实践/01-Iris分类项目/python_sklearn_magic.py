#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python/sklearn的"不可思议"之处
展示Python和sklearn的简洁性
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 加载数据
iris = load_iris()
X = iris.data
y = iris.target

print("=== Python/sklearn的'不可思议'之处 ===")

# 1. 数据分割 - 一行代码
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("1. 数据分割: 一行代码完成")

# 2. 特征标准化 - 两行代码
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("2. 特征标准化: 两行代码完成")

# 3. 模型训练 - 两行代码
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)
print("3. 模型训练: 两行代码完成")

# 4. 模型预测 - 一行代码
y_pred = model.predict(X_test_scaled)
print("4. 模型预测: 一行代码完成")

# 5. 准确率计算 - 一行代码
accuracy = (y_pred == y_test).mean()
print(f"5. 准确率计算: 一行代码完成 - {accuracy:.4f}")

# 6. 更高级的评估 - 一行代码
print("\n6. 详细分类报告:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 7. 向量化操作示例
print("\n=== 向量化操作示例 ===")

# 计算每个类别的准确率
for i, class_name in enumerate(iris.target_names):
    mask = y_test == i
    class_accuracy = (y_pred[mask] == y_test[mask]).mean()
    print(f"{class_name}准确率: {class_accuracy:.4f}")

# 8. 链式操作示例
print("\n=== 链式操作示例 ===")

# 数据统计
print("数据统计:")
print(f"  样本数: {len(X)}")
print(f"  特征数: {X.shape[1]}")
print(f"  类别数: {len(np.unique(y))}")

# 特征重要性（对于线性模型）
feature_importance = np.abs(model.coef_[0])
print(f"特征重要性: {feature_importance}")

# 9. 一行代码的复杂操作
print("\n=== 一行代码的复杂操作 ===")

# 计算混淆矩阵
confusion_matrix = np.zeros((3, 3), dtype=int)
for i in range(3):
    for j in range(3):
        confusion_matrix[i, j] = ((y_test == i) & (y_pred == j)).sum()

print("混淆矩阵:")
print(confusion_matrix)

# 10. 数据可视化 - 简洁的pandas操作
print("\n=== 数据探索 - pandas简洁性 ===")

# 创建DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
df['species'] = [iris.target_names[i] for i in y]

# 一行代码查看统计信息
print("数据统计信息:")
print(df.describe())

# 一行代码查看类别分布
print("\n类别分布:")
print(df['species'].value_counts())

print("\n=== 总结 ===")
print("Python/sklearn的简洁性体现在:")
print("1. 统一的API设计")
print("2. 向量化操作")
print("3. 链式操作")
print("4. 一行代码完成复杂任务")
print("5. 高度抽象化")
print("6. 代码可读性强") 