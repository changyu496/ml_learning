"""
练习1：特征值和特征向量的计算
目标：理解和验证特征值、特征向量的计算过程
"""

import numpy as np

print("📚 练习1：特征值和特征向量")
print("="*50)

# 练习任务
print("\n🎯 任务目标：")
print("1. 计算给定矩阵的特征值和特征向量")
print("2. 验证 Av = λv 的关系")
print("3. 理解特征值和特征向量的几何意义")

# 给定矩阵
A = np.array([[4, 2], 
              [2, 1]])

print(f"\n📊 给定矩阵 A:")
print(A)

# 任务1 - 计算特征值和特征向量
print("\n📝 任务1：计算特征值和特征向量")
print("提示：使用 np.linalg.eig() 函数")

# 你的代码：✅ 正确！
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"特征值: {eigenvalues}")
print(f"特征向量矩阵:\n{eigenvectors}")
print("\n💡 解释：")
print("- eigenvalues 是一个数组，包含所有特征值")
print("- eigenvectors 是一个矩阵，每一列是一个特征向量")

# 任务2 - 验证特征向量定义
print("\n📝 任务2：验证 Av = λv")
print("提示：对每个特征值和特征向量进行验证")

print("\n🔍 详细验证过程：")
print("特征值和特征向量的定义：如果 Av = λv，那么 v 是特征向量，λ 是特征值")
print("-" * 60)

# 修正后的代码：
for i in range(len(eigenvalues)):
    # 📌 关键理解：
    # eigenvalues[i] 是第i个特征值（一个数字）
    # eigenvectors[:, i] 是第i个特征向量（一个向量，取第i列）
    
    λ = eigenvalues[i]          # 第i个特征值
    v = eigenvectors[:, i]      # 第i个特征向量（注意是列向量）
    
    # 计算 Av 和 λv
    Av = A @ v                  # 矩阵A乘以特征向量v
    λv = λ * v                  # 特征值λ乘以特征向量v
    
    # 计算差值来验证
    difference = np.abs(Av - λv)
    max_diff = np.max(difference)
    
    print(f"\n🎯 验证第 {i+1} 个特征值和特征向量：")
    print(f"特征值 λ{i+1} = {λ:.6f}")
    print(f"特征向量 v{i+1} = [{v[0]:.6f}, {v[1]:.6f}]")
    print(f"Av{i+1} = [{Av[0]:.6f}, {Av[1]:.6f}]")
    print(f"λv{i+1} = [{λv[0]:.6f}, {λv[1]:.6f}]")
    print(f"差值 = [{difference[0]:.10f}, {difference[1]:.10f}]")
    print(f"最大差值 = {max_diff:.10f}")
    
    if max_diff < 1e-10:
        print("✅ 验证成功！Av = λv")
    else:
        print("⚠️  存在数值误差")

# 任务3 - 分析结果
print("\n📝 任务3：分析结果")
print("请回答以下问题：")
print("1. 这个矩阵有几个特征值？")
print("2. 哪个特征值更大？对应的特征向量有什么几何意义？")
print("3. 验证结果的误差是多少？为什么不是完全的0？")

# 答案分析：
print("\n📋 答案分析：")
print(f"答案1：这个2×2矩阵有 {len(eigenvalues)} 个特征值")

# 找到最大特征值
max_idx = np.argmax(eigenvalues)
max_eigenvalue = eigenvalues[max_idx]
max_eigenvector = eigenvectors[:, max_idx]

print(f"答案2：最大特征值是 {max_eigenvalue:.6f}")
print(f"      对应的特征向量是 [{max_eigenvector[0]:.6f}, {max_eigenvector[1]:.6f}]")
print(f"      几何意义：这个方向是数据变换后拉伸最多的方向")

print(f"答案3：数值误差通常在 1e-15 量级，这是由于计算机浮点数精度限制")

print("\n✅ 练习1完成！")
print("💡 核心理解：特征向量是矩阵变换后方向不变的向量")
print("💡 关键公式：Av = λv （矩阵 × 特征向量 = 特征值 × 特征向量）")