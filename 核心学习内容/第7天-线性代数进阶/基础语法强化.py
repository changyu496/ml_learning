"""
基础语法强化练习 - 断网挑战！🔥
目标：不查资料完成numpy基础操作
时间：30分钟
"""

import numpy as np

print("🔥 基础语法强化挑战 - 断网练习")
print("="*50)
print("⚠️  重要：尽量不查资料，先尝试自己写！")
print("💡 提示：如果卡住了，先思考5分钟再看下面的提示")

# ===== 第1关：数组创建 =====
print("\n🎯 第1关：数组创建")
print("任务：创建以下数组")

# 任务1.1：创建一个2x3的全零数组
print("\n📝 任务1.1：创建2x3的全零数组")
print("预期结果：")
print("[[0. 0. 0.]")
print(" [0. 0. 0.]]")
print("\n你的代码：")
# 在这里写代码：
zeros_array = np.zeros((2,3))

# 任务1.2：创建一个3x3的单位矩阵
print("\n📝 任务1.2：创建3x3的单位矩阵")
print("预期结果：对角线为1，其他为0")
print("\n你的代码：")
# 在这里写代码：
identity_matrix = np.eye(3,3)

# 任务1.3：创建一个包含1到10的数组
print("\n📝 任务1.3：创建包含1到10的数组")
print("预期结果：[1 2 3 4 5 6 7 8 9 10]")
print("\n你的代码：")
# 在这里写代码：
range_array = np.arange(1,11)

# ===== 第2关：数组操作 =====
print("\n🎯 第2关：数组操作")

# 给定数组
A = np.array([[1, 2, 3], 
              [4, 5, 6]])
print(f"给定数组 A:\n{A}")

# 任务2.1：获取数组的形状
print("\n📝 任务2.1：获取数组A的形状")
print("预期结果：(2, 3)")
print("\n你的代码：")
# 在这里写代码：
shape_result = A.shape

# 任务2.2：获取第二行的所有元素
print("\n📝 任务2.2：获取第二行的所有元素")
print("预期结果：[4 5 6]")
print("\n你的代码：")
# 在这里写代码：
second_row = A[1]

# 任务2.3：获取第一列的所有元素
print("\n📝 任务2.3：获取第一列的所有元素")
print("预期结果：[1 4]")
print("\n你的代码：")
# 在这里写代码：
first_column = A[:,0]

# ===== 第3关：数组运算 =====
print("\n🎯 第3关：数组运算")

B = np.array([[1, 1, 1], 
              [2, 2, 2]])
print(f"给定数组 B:\n{B}")

# 任务3.1：计算 A + B
print("\n📝 任务3.1：计算 A + B")
print("预期结果：对应位置相加")
print("\n你的代码：")
# 在这里写代码：
add_result = A+B

# 任务3.2：计算 A * B (逐元素相乘)
print("\n📝 任务3.2：计算 A * B (逐元素相乘)")
print("预期结果：对应位置相乘")
print("\n你的代码：")
# 在这里写代码：
multiply_result = A * B

# 任务3.3：计算A的转置
print("\n📝 任务3.3：计算A的转置")
print("预期结果：行变列，列变行")
print("\n你的代码：")
# 在这里写代码：
transpose_result = A.T

# ===== 第4关：矩阵运算 =====
print("\n🎯 第4关：矩阵运算")

C = np.array([[1, 2], 
              [3, 4], 
              [5, 6]])
print(f"给定数组 C:\n{C}")

# 任务4.1：计算 A @ C (矩阵乘法)
print("\n📝 任务4.1：计算 A @ C (矩阵乘法)")
print("预期结果：A(2x3) × C(3x2) = 结果(2x2)")
print("\n你的代码：")
# 在这里写代码：
matmul_result = A @ C

# 任务4.2：计算数组A中所有元素的和
print("\n📝 任务4.2：计算数组A中所有元素的和")
print("预期结果：21")
print("\n你的代码：")
# 在这里写代码：
sum_result = np.sum(A)

# 任务4.3：计算数组A每行的平均值
print("\n📝 任务4.3：计算数组A每行的平均值")
print("预期结果：[2. 5.]")
print("\n你的代码：")
# 在这里写代码：
mean_result = np.mean(A,axis=1)

# ===== 自我检查 =====
print("\n" + "="*50)
print("🏁 完成练习后，请检查：")
print("1. 你能独立完成多少个任务？")
print("2. 哪些操作你不记得了？")
print("3. 需要重点复习哪些内容？")
print("\n💡 记录下不会的操作，明天重点练习！")

# ===== 提示区域 =====
print("\n" + "="*50)
print("📖 提示区域 (实在不会时再看)")
print("- 创建数组：np.zeros(), np.eye(), np.arange()")
print("- 数组形状：array.shape")
print("- 数组切片：array[行索引, 列索引]")
print("- 数组运算：+, -, *, /, @")
print("- 数组函数：np.sum(), np.mean(), array.T")
print("="*50) 