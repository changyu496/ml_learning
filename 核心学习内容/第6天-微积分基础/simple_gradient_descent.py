import numpy as np
import matplotlib.pyplot as plt
from sympy import true

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def simple_gradient_dscent(start_point,learning_rate,num_interatios):
    """
    实现简单的梯度下降算法
    函数: f(x,y) = (x-3)² + (y-1)²
    目标是找到最小值点(3,1)
    """
    # 第一步：初始化当前位置
    current_x = start_point[0]
    current_y = start_point[1]
    # 第二步：开始迭代
    for i in range(num_interatios):
        # 第三步：计算当前位置的梯度
        # 对x求偏导
        gradient_x = 2 * (current_x-3)
        # 对y求偏导
        gradient_y = 2 * (current_y-1)
        # 第四步：更新位置
        # 新位置 = 旧位置 - 学习率*梯度
        current_x = current_x - learning_rate * gradient_x
        current_y = current_y - learning_rate * gradient_y

        # 打印中间结果：
        print(f"步骤 {i+1}:位置=({current_x:.3f},{current_y:.3f})")

    # 返回最终位置
    return [current_x,current_y]

# simple_gradient_dscent([0,0],0.1,10)

# 不同起点
start_points = [[0,0],[5,5],[-2,3],[1,-1]]
print("不同起点开始")
for start in start_points:
    print(f"起点:{start}")
    simple_gradient_dscent(start,0.01,10)
print("不同起点结束")

# 不同学习率
learning_rates = [0.01,0.1,0.5,0.9]

print("不同学习率开始")
for learning_rate in learning_rates:
    print(f"学习率:{learning_rate}")
    simple_gradient_dscent([0,0],learning_rate,10)
print("不同学习率结束")

# 任务1：函数和导数对比图 
# 绘制函数 f(x) = x³ - 3x² + 2x 和其导数
# 要求：
# 1. 创建1行2列的子图
# 2. 左图显示原函数
# 3. 右图显示导数函数
# 4. 标记导数为0的点
# 5. 添加网格和标签
def f(x):
    return x**3 - 3 * x**2 + 2*x

def f_deriative(x):
    return 3 * x**2 - 6*x

x = np.linspace(-3,3,100)
plt.subplot(1,2,1)
plt.plot(x,f(x))
plt.title("原函数")
plt.grid(true)
plt.subplot(1,2,2)
plt.plot(x,f_deriative(x))
plt.title('导数函数')
plt.grid(true)
plt.show()