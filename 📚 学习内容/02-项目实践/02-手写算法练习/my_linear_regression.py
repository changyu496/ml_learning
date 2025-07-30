#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手写线性回归算法实现
作者: ChangYu
日期: 2025-07-28
目标: 通过对话方式学习实现线性回归算法
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class MyLinearRegression:
    """
    手写线性回归实现
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        初始化参数
        
        参数:
        learning_rate: 学习率
        n_iterations: 迭代次数
        """
        # TODO: 在这里初始化权重和偏置
        pass
        
    def fit(self, X, y):
        """
        训练线性回归模型
        
        参数:
        X: 特征矩阵，形状为 (n_samples, n_features)
        y: 目标向量，形状为 (n_samples,)
        """
        # TODO: 在这里实现梯度下降算法
        pass
        
    def predict(self, X):
        """
        预测
        
        参数:
        X: 特征矩阵
        
        返回:
        y_pred: 预测结果
        """
        # TODO: 在这里实现预测逻辑
        pass

def main():
    """
    主函数：测试我们的线性回归实现
    """
    print("🚀 开始手写线性回归算法学习")
    print("=" * 50)
    
    # 创建简单的测试数据
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])
    
    print(f"测试数据:")
    print(f"X: {X.flatten()}")
    print(f"y: {y}")
    
    # TODO: 在这里创建和训练我们的模型
    
    print("\n🎉 学习完成！")

if __name__ == "__main__":
    main() 