#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第6天微积分基础 - 强化练习
目标：将理论理解转化为实际操作能力
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

print("🎯 第6天微积分基础强化练习")
print("=" * 50)

# ============================================================================
# 第三部分：编程实践 (20分钟)
# ============================================================================

print("\n💻 第三部分：编程实践")
print("-" * 30)

# ----------------------------------------------------------------------------
# 任务1：基础梯度下降实现 (8分钟)
# ----------------------------------------------------------------------------
print("\n📝 任务1：基础梯度下降实现")

def simple_gradient_descent(start_point, learning_rate, num_iterations):
    """
    实现简单的梯度下降算法
    函数: f(x,y) = (x-3)² + (y-1)²
    目标: 找到最小值点 (3, 1)
    
    参数:
        start_point: 起始点 [x, y]
        learning_rate: 学习率
        num_iterations: 迭代次数
    
    返回:
        最终点的坐标 [x, y]
    """
    # TODO: 你的实现
    # 提示：
    # 1. 定义损失函数 f(x,y) = (x-3)² + (y-1)²
    # 2. 计算梯度 ∇f = [2(x-3), 2(y-1)]
    # 3. 更新参数 new_point = old_point - learning_rate * gradient
    # 4. 重复迭代
    
    current_point = np.array(start_point, dtype=float)
    
    for i in range(num_iterations):
        # 在这里实现梯度下降的一步
        pass
    
    return current_point.tolist()

# 测试你的实现
print("测试基础梯度下降:")
result = simple_gradient_descent([0, 0], 0.1, 20)
print(f"起始点: [0, 0]")
print(f"最终结果: {result}")
print(f"期望结果: [3, 1]")
print(f"误差: {abs(result[0] - 3) + abs(result[1] - 1):.6f}")

# ----------------------------------------------------------------------------
# 任务2：不同学习率对比 (6分钟)
# ----------------------------------------------------------------------------
print("\n📝 任务2：不同学习率对比")

def compare_learning_rates():
    """
    测试不同学习率的效果
    观察收敛速度和稳定性
    """
    learning_rates = [0.01, 0.1, 0.5, 0.9]
    start_point = [0, 0]
    iterations = 50
    
    print("学习率对比实验:")
    print("起始点:", start_point)
    print("迭代次数:", iterations)
    print("-" * 40)
    
    for lr in learning_rates:
        # TODO: 实现对比实验
        # 1. 使用不同学习率运行梯度下降
        # 2. 记录最终结果和收敛情况
        # 3. 分析学习率的影响
        
        result = simple_gradient_descent(start_point, lr, iterations)
        error = abs(result[0] - 3) + abs(result[1] - 1)
        
        print(f"学习率 {lr:4.2f}: 结果 {result}, 误差 {error:.6f}")
    
    print("\n分析:")
    print("- 学习率太小(0.01): 收敛慢")
    print("- 学习率适中(0.1): 收敛快且稳定")
    print("- 学习率较大(0.5): 可能震荡")
    print("- 学习率过大(0.9): 可能不收敛")

# 运行学习率对比
compare_learning_rates()

# ----------------------------------------------------------------------------
# 任务3：不同起始点对比 (6分钟)
# ----------------------------------------------------------------------------
print("\n📝 任务3：不同起始点对比")

def compare_start_points():
    """
    测试不同起始点的收敛结果
    验证梯度下降的鲁棒性
    """
    start_points = [[0, 0], [5, 5], [-2, 3], [1, -1]]
    learning_rate = 0.1
    iterations = 30
    
    print("起始点对比实验:")
    print("学习率:", learning_rate)
    print("迭代次数:", iterations)
    print("-" * 40)
    
    for start in start_points:
        # TODO: 实现对比实验
        # 1. 使用不同起始点运行梯度下降
        # 2. 观察是否都能收敛到同一点
        # 3. 比较收敛速度
        
        result = simple_gradient_descent(start, learning_rate, iterations)
        error = abs(result[0] - 3) + abs(result[1] - 1)
        
        print(f"起始点 {start}: 结果 {result}, 误差 {error:.6f}")
    
    print("\n分析:")
    print("- 对于凸函数，不同起始点都能收敛到全局最优点")
    print("- 距离最优点越近，收敛越快")
    print("- 梯度下降对起始点选择具有鲁棒性")

# 运行起始点对比
compare_start_points()

# ============================================================================
# 第四部分：图形化练习 (5分钟)
# ============================================================================

print("\n📊 第四部分：图形化练习")
print("-" * 30)

def plot_function_and_derivative():
    """
    绘制函数 f(x) = x³ - 3x² + 2x 和其导数
    要求：
    1. 创建1行2列的子图
    2. 左图显示原函数
    3. 右图显示导数函数
    4. 标记导数为0的点
    5. 添加网格和标签
    """
    
    def f(x):
        """原函数 f(x) = x³ - 3x² + 2x"""
        return x**3 - 3*x**2 + 2*x
    
    def df_dx(x):
        """导数函数 f'(x) = 3x² - 6x + 2"""
        return 3*x**2 - 6*x + 2
    
    # 创建数据
    x = np.linspace(-1, 4, 1000)
    y = f(x)
    dy = df_dx(x)
    
    # TODO: 实现图形绘制
    # 1. 创建子图
    # 2. 绘制原函数和导数函数
    # 3. 找到并标记导数为0的点
    # 4. 添加网格、标签、标题
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：原函数
    ax1.plot(x, y, 'b-', linewidth=2, label='f(x) = x³ - 3x² + 2x')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('原函数')
    ax1.legend()
    
    # 右图：导数函数
    ax2.plot(x, dy, 'r-', linewidth=2, label="f'(x) = 3x² - 6x + 2")
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('x')
    ax2.set_ylabel("f'(x)")
    ax2.set_title('导数函数')
    ax2.legend()
    
    # 找到导数为0的点
    # 解方程 3x² - 6x + 2 = 0
    coeffs = [3, -6, 2]  # 3x² - 6x + 2 = 0
    roots = np.roots(coeffs)
    
    # 标记极值点
    for root in roots:
        if -1 <= root <= 4:
            # 在左图标记
            ax1.plot(root, f(root), 'ro', markersize=8)
            ax1.annotate(f'极值点\n({root:.2f}, {f(root):.2f})', 
                        xy=(root, f(root)), xytext=(10, 10),
                        textcoords='offset points', fontsize=9)
            
            # 在右图标记
            ax2.plot(root, 0, 'ro', markersize=8)
            ax2.annotate(f"f'({root:.2f}) = 0", 
                        xy=(root, 0), xytext=(10, 10),
                        textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    print(f"导数为0的点: {roots}")
    print(f"对应的函数值: {[f(root) for root in roots]}")

# 运行图形化练习
print("\n📈 绘制函数和导数对比图:")
plot_function_and_derivative()

# ============================================================================
# 额外练习：梯度下降可视化
# ============================================================================

def visualize_gradient_descent():
    """
    可视化梯度下降过程
    """
    def loss_function(x, y):
        return (x - 3)**2 + (y - 1)**2
    
    def gradient(x, y):
        return np.array([2*(x - 3), 2*(y - 1)])
    
    # 梯度下降过程
    start_point = np.array([0.0, 0.0])
    learning_rate = 0.1
    num_iterations = 20
    
    path = [start_point.copy()]
    current_point = start_point.copy()
    
    for i in range(num_iterations):
        grad = gradient(current_point[0], current_point[1])
        current_point = current_point - learning_rate * grad
        path.append(current_point.copy())
    
    path = np.array(path)
    
    # 绘制等高线和路径
    x = np.linspace(-1, 4, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = loss_function(X, Y)
    
    plt.figure(figsize=(10, 8))
    
    # 绘制等高线
    contour = plt.contour(X, Y, Z, levels=20, alpha=0.6)
    plt.clabel(contour, inline=True, fontsize=8)
    
    # 绘制梯度下降路径
    plt.plot(path[:, 0], path[:, 1], 'ro-', linewidth=2, markersize=4, 
             label='梯度下降路径')
    plt.plot(path[0, 0], path[0, 1], 'go', markersize=10, label='起始点')
    plt.plot(path[-1, 0], path[-1, 1], 'bs', markersize=10, label='终点')
    plt.plot(3, 1, 'r*', markersize=15, label='真实最优点(3,1)')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('梯度下降可视化')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()
    
    print(f"梯度下降路径:")
    for i, point in enumerate(path[:6]):  # 只显示前6步
        print(f"步骤 {i}: ({point[0]:.3f}, {point[1]:.3f})")

print("\n🎨 梯度下降可视化:")
visualize_gradient_descent()

# ============================================================================
# 学习总结
# ============================================================================

print("\n" + "=" * 50)
print("🎓 学习总结")
print("=" * 50)

print("""
今日练习重点：
1. ✅ 实现了基础梯度下降算法
2. ✅ 对比了不同学习率的效果
3. ✅ 测试了不同起始点的收敛性
4. ✅ 绘制了函数和导数的对比图
5. ✅ 可视化了梯度下降过程

关键收获：
- 理解了梯度下降的完整实现流程
- 体验了学习率对收敛的影响
- 观察了梯度下降的几何意义
- 提高了matplotlib绘图技能

下一步：
- 如果掌握良好，可以学习第7天内容
- 如果还有困难，继续练习相关内容
- 重点关注理论与实践的结合
""")

print("\n🎯 恭喜完成第6天强化练习！") 