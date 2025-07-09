#!/usr/bin/env python3
"""
第6天编程练习：微积分基础
时间：15-20分钟
目标：理解导数、偏导数、梯度的概念
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("🧮 第6天编程练习：微积分基础")
print("=" * 35)
print("理解变化的数学语言！")
print()

# ==========================================
# 练习1：导数计算和可视化
# ==========================================
print("📈 练习1：导数计算和可视化")
print("-" * 25)

# 定义函数 f(x) = x³ - 3x + 1
def f(x):
    return x**3 - 3*x + 1

# TODO 1: 计算导数 f'(x) = 3x² - 3
def f_derivative(x):
    # 你的代码：计算 f(x) = x³ - 3x + 1 的导数
    pass

# 测试数据
x_test = 2
print(f"测试点 x = {x_test}")
print(f"f({x_test}) = {f(x_test)}")
# print(f"f'({x_test}) = {f_derivative(x_test)}")  # 取消注释测试

# TODO 2: 可视化函数和导数
# 要求：
# - 创建两个子图
# - 左图显示原函数 f(x)
# - 右图显示导数函数 f'(x)
# - 标记导数为0的点（极值点）

# 你的代码：


print("完成导数练习！")
print()

# ==========================================
# 练习2：偏导数计算
# ==========================================
print("🔍 练习2：偏导数计算")
print("-" * 20)

# 定义二元函数 f(x,y) = x² + y² + 2xy
def f_2d(x, y):
    return x**2 + y**2 + 2*x*y

# TODO 3: 计算偏导数
def partial_x(x, y):
    # 对x的偏导数：∂f/∂x = 2x + 2y
    # 你的代码：
    pass

def partial_y(x, y):
    # 对y的偏导数：∂f/∂y = 2y + 2x
    # 你的代码：
    pass

# 测试偏导数
x_test, y_test = 1, 2
print(f"测试点 ({x_test}, {y_test})")
print(f"f({x_test}, {y_test}) = {f_2d(x_test, y_test)}")
# print(f"∂f/∂x = {partial_x(x_test, y_test)}")  # 取消注释测试
# print(f"∂f/∂y = {partial_y(x_test, y_test)}")  # 取消注释测试

print("完成偏导数练习！")
print()

# ==========================================
# 练习3：梯度计算和可视化
# ==========================================
print("🎯 练习3：梯度计算和可视化")
print("-" * 25)

# TODO 4: 计算梯度
def gradient(x, y):
    # 梯度是所有偏导数组成的向量
    # grad_f = [∂f/∂x, ∂f/∂y]
    # 你的代码：
    pass

# TODO 5: 梯度可视化
# 要求：
# - 创建函数的等高线图
# - 在几个点上绘制梯度向量
# - 观察梯度指向函数增长最快的方向

# 你的代码：


print("完成梯度练习！")
print()

# ==========================================
# 练习4：简单的梯度下降
# ==========================================
print("⚡ 练习4：简单的梯度下降")
print("-" * 25)

# 使用梯度下降找到函数 f(x) = (x-3)² + 1 的最小值
def simple_function(x):
    return (x - 3)**2 + 1

def simple_derivative(x):
    return 2 * (x - 3)

# TODO 6: 实现梯度下降
def gradient_descent():
    # 初始点
    x = 0.0
    learning_rate = 0.1
    steps = 20
    
    print("梯度下降过程：")
    for i in range(steps):
        # 计算当前点的函数值和导数
        fx = simple_function(x)
        grad = simple_derivative(x)
        
        print(f"步骤 {i+1}: x = {x:.3f}, f(x) = {fx:.3f}, f'(x) = {grad:.3f}")
        
        # TODO: 更新 x
        # x = x - learning_rate * grad
        # 你的代码：
        
        # 如果梯度很小，说明接近最小值
        if abs(grad) < 0.001:
            print(f"收敛！最小值点约为 x = {x:.3f}")
            break
    
    return x

# 运行梯度下降
# final_x = gradient_descent()  # 取消注释运行

print("完成梯度下降练习！")
print()

# ==========================================
# 练习总结
# ==========================================
print("🎉 练习完成检查")
print("=" * 20)
print("请检查你是否完成了：")
print("□ 练习1：导数计算和可视化")
print("□ 练习2：偏导数计算")
print("□ 练习3：梯度计算和可视化")
print("□ 练习4：简单的梯度下降")
print()
print("💡 如果遇到困难，可以：")
print("1. 回顾notebook中的理论")
print("2. 查看下面的提示")
print("3. 问我具体问题")

# ==========================================
# 提示区域（卡住了再看）
# ==========================================
print("\n" + "="*50)
print("💡 提示区域（卡住了再看）")
print("="*50)

print("\n📈 练习1提示（导数）:")
print("def f_derivative(x):")
print("    return 3*x**2 - 3")
print()
print("# 可视化")
print("x = np.linspace(-3, 3, 100)")
print("plt.subplot(1, 2, 1)")
print("plt.plot(x, f(x))")
print("plt.title('原函数')")

print("\n🔍 练习2提示（偏导数）:")
print("def partial_x(x, y):")
print("    return 2*x + 2*y")
print()
print("def partial_y(x, y):")
print("    return 2*y + 2*x")

print("\n🎯 练习3提示（梯度）:")
print("def gradient(x, y):")
print("    return [partial_x(x, y), partial_y(x, y)]")
print()
print("# 等高线图")
print("X, Y = np.meshgrid(x_range, y_range)")
print("Z = f_2d(X, Y)")
print("plt.contour(X, Y, Z)")

print("\n⚡ 练习4提示（梯度下降）:")
print("# 梯度下降更新规则")
print("x = x - learning_rate * grad")
print("# learning_rate 是学习率，控制步长")

print("\n🎯 记住：先自己尝试，再看提示！")
print("🚀 完成后你就理解了微积分的核心概念！") 