#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手写线性回归算法实现
作者: ChangYu
日期: 2025-07-28
目标: 通过手写实现加深对线性回归的理解
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

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
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def fit(self, X, y):
        """
        训练线性回归模型
        
        参数:
        X: 特征矩阵，形状为 (n_samples, n_features)
        y: 目标向量，形状为 (n_samples,)
        """
        # 初始化参数
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        print(f"开始训练，样本数: {n_samples}, 特征数: {n_features}")
        print(f"学习率: {self.learning_rate}, 迭代次数: {self.n_iterations}")
        
        # 梯度下降
        for i in range(self.n_iterations):
            # 前向传播
            y_pred = X @ self.weights + self.bias
            
            # 计算损失
            cost = np.mean((y_pred - y) ** 2)
            self.cost_history.append(cost)
            
            # 计算梯度
            dw = (2/n_samples) * X.T @ (y_pred - y)
            db = (2/n_samples) * np.sum(y_pred - y)
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # 每100次迭代打印一次进度
            if (i + 1) % 100 == 0:
                print(f"迭代 {i+1}/{self.n_iterations}, 损失: {cost:.6f}")
        
        print(f"训练完成！最终损失: {self.cost_history[-1]:.6f}")
        
    def predict(self, X):
        """
        预测
        
        参数:
        X: 特征矩阵
        
        返回:
        y_pred: 预测结果
        """
        return X @ self.weights + self.bias
    
    def get_params(self):
        """
        获取模型参数
        """
        return {
            'weights': self.weights,
            'bias': self.bias,
            'cost_history': self.cost_history
        }

def compare_with_sklearn(X_train, X_test, y_train, y_test):
    """
    与sklearn的线性回归结果对比
    """
    print("=" * 50)
    print("手写线性回归 vs sklearn线性回归 对比")
    print("=" * 50)
    
    # 手写线性回归
    print("\n1. 手写线性回归:")
    my_lr = MyLinearRegression(learning_rate=0.01, n_iterations=1000)
    my_lr.fit(X_train, y_train)
    y_pred_my = my_lr.predict(X_test)
    
    # 计算评估指标
    mse_my = mean_squared_error(y_test, y_pred_my)
    r2_my = r2_score(y_test, y_pred_my)
    
    print(f"手写模型 - MSE: {mse_my:.6f}, R²: {r2_my:.6f}")
    print(f"手写模型参数 - 权重: {my_lr.weights}, 偏置: {my_lr.bias:.6f}")
    
    # sklearn线性回归
    print("\n2. sklearn线性回归:")
    sklearn_lr = LinearRegression()
    sklearn_lr.fit(X_train, y_train)
    y_pred_sklearn = sklearn_lr.predict(X_test)
    
    # 计算评估指标
    mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
    r2_sklearn = r2_score(y_test, y_pred_sklearn)
    
    print(f"sklearn模型 - MSE: {mse_sklearn:.6f}, R²: {r2_sklearn:.6f}")
    print(f"sklearn模型参数 - 权重: {sklearn_lr.coef_}, 偏置: {sklearn_lr.intercept_:.6f}")
    
    # 比较结果
    print("\n3. 结果对比:")
    print(f"MSE差异: {abs(mse_my - mse_sklearn):.6f}")
    print(f"R²差异: {abs(r2_my - r2_sklearn):.6f}")
    print(f"权重差异: {np.mean(np.abs(my_lr.weights - sklearn_lr.coef_)):.6f}")
    print(f"偏置差异: {abs(my_lr.bias - sklearn_lr.intercept_):.6f}")
    
    return my_lr, sklearn_lr, y_pred_my, y_pred_sklearn

def visualize_training_process(my_lr):
    """
    可视化训练过程
    """
    plt.figure(figsize=(12, 4))
    
    # 损失函数变化
    plt.subplot(1, 2, 1)
    plt.plot(my_lr.cost_history)
    plt.title('损失函数变化')
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.grid(True, alpha=0.3)
    
    # 损失函数变化（对数尺度）
    plt.subplot(1, 2, 2)
    plt.semilogy(my_lr.cost_history)
    plt.title('损失函数变化（对数尺度）')
    plt.xlabel('迭代次数')
    plt.ylabel('损失值（对数）')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def visualize_predictions(X_test, y_test, y_pred_my, y_pred_sklearn):
    """
    可视化预测结果
    """
    plt.figure(figsize=(12, 4))
    
    # 手写模型预测结果
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_my, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('手写线性回归预测结果')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.grid(True, alpha=0.3)
    
    # sklearn模型预测结果
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_sklearn, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('sklearn线性回归预测结果')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    主函数：演示手写线性回归算法
    """
    print("🚀 开始手写线性回归算法演示")
    print("=" * 50)
    
    # 1. 生成数据
    print("1. 生成回归数据")
    X, y = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)
    print(f"数据形状: X={X.shape}, y={y.shape}")
    
    # 2. 数据分割
    print("\n2. 数据分割")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 3. 数据标准化
    print("\n3. 数据标准化")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("数据标准化完成")
    
    # 4. 手写线性回归 vs sklearn线性回归
    print("\n4. 执行线性回归")
    my_lr, sklearn_lr, y_pred_my, y_pred_sklearn = compare_with_sklearn(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # 5. 可视化训练过程
    print("\n5. 可视化训练过程")
    visualize_training_process(my_lr)
    
    # 6. 可视化预测结果
    print("\n6. 可视化预测结果")
    visualize_predictions(X_test_scaled, y_test, y_pred_my, y_pred_sklearn)
    
    # 7. 详细分析
    print("\n7. 详细分析")
    print(f"手写模型参数数量: {len(my_lr.weights) + 1}")
    print(f"sklearn模型参数数量: {len(sklearn_lr.coef_) + 1}")
    print(f"训练迭代次数: {len(my_lr.cost_history)}")
    print(f"最终损失值: {my_lr.cost_history[-1]:.6f}")
    
    print("\n🎉 手写线性回归算法演示完成！")

if __name__ == "__main__":
    main() 