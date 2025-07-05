import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("🎯 特征值可视化：学生成绩例子")

# 生成高度相关的学生成绩数据
np.random.seed(42)
n_students = 100

# 生成沿着y=x线分布的数据（数学好物理也好）
base_ability = np.random.normal(50, 20, n_students)  # 基础学习能力
math_scores = base_ability + np.random.normal(0, 5, n_students)  # 数学成绩
physics_scores = base_ability + np.random.normal(0, 5, n_students)  # 物理成绩

# 确保成绩在0-100范围内
math_scores = np.clip(math_scores, 0, 100)
physics_scores = np.clip(physics_scores, 0, 100)

# 组合数据
student_data = np.column_stack([math_scores, physics_scores])

print(f"学生数据形状: {student_data.shape}")
print(f"数学成绩范围: {math_scores.min():.1f} - {math_scores.max():.1f}")
print(f"物理成绩范围: {physics_scores.min():.1f} - {physics_scores.max():.1f}")

# 计算协方差矩阵和特征值
cov_matrix = np.cov(student_data.T)
eigenvals, eigenvecs = np.linalg.eig(cov_matrix)

# 按特征值大小排序
sorted_indices = np.argsort(eigenvals)[::-1]
eigenvals_sorted = eigenvals[sorted_indices]
eigenvecs_sorted = eigenvecs[:, sorted_indices]

print(f"\n特征值:")
print(f"第1个特征值（大）: {eigenvals_sorted[0]:.1f}")
print(f"第2个特征值（小）: {eigenvals_sorted[1]:.1f}")
print(f"比例: {eigenvals_sorted[0]/eigenvals_sorted[1]:.1f}:1")

# 创建可视化
plt.figure(figsize=(12, 5))

# 左图：原始数据分布
plt.subplot(1, 2, 1)
plt.scatter(math_scores, physics_scores, alpha=0.6, s=30, color='lightblue', 
           edgecolors='blue', linewidth=0.5)

# 计算数据中心
center_x = np.mean(math_scores)
center_y = np.mean(physics_scores)

# 绘制特征向量（主成分方向）
scale = 30  # 箭头长度缩放
for i in range(2):
    # 特征向量方向
    direction = eigenvecs_sorted[:, i]
    # 根据特征值调整箭头长度
    length = scale * np.sqrt(eigenvals_sorted[i] / eigenvals_sorted[0])
    
    # 绘制箭头
    arrow = FancyArrowPatch(
        (center_x - direction[0] * length, center_y - direction[1] * length),
        (center_x + direction[0] * length, center_y + direction[1] * length),
        arrowstyle='->', mutation_scale=20, linewidth=3,
        color='red' if i == 0 else 'orange'
    )
    plt.gca().add_patch(arrow)
    
    # 添加标签
    label_x = center_x + direction[0] * length * 1.3
    label_y = center_y + direction[1] * length * 1.3
    plt.text(label_x, label_y, 
             f'特征值{i+1}={eigenvals_sorted[i]:.1f}\n{"主要方向" if i==0 else "次要方向"}',
             ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", 
                      facecolor='red' if i==0 else 'orange', alpha=0.7))

plt.xlabel('数学成绩')
plt.ylabel('物理成绩')
plt.title('学生成绩分布与主成分方向')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(0, 100)
plt.ylim(0, 100)

# 右图：特征值大小比较
plt.subplot(1, 2, 2)
bars = plt.bar(['特征值1\n(主要方向)', '特征值2\n(次要方向)'], 
               eigenvals_sorted, 
               color=['red', 'orange'], alpha=0.7)
plt.ylabel('特征值大小')
plt.title('特征值比较')
plt.grid(True, alpha=0.3)

# 添加数值标签
for i, bar in enumerate(bars):
    height = bar.get_height()
    percentage = eigenvals_sorted[i] / np.sum(eigenvals_sorted) * 100
    plt.text(bar.get_x() + bar.get_width()/2., height + 10,
             f'{height:.1f}\n({percentage:.1f}%)', 
             ha='center', va='bottom', fontsize=11, weight='bold')

plt.tight_layout()
plt.show()

print(f"\n📊 图解分析:")
print(f"1. 红色箭头：第1主成分方向，特征值={eigenvals_sorted[0]:.1f}")
print(f"   - 这个方向数据变化最大")
print(f"   - 表示学生的'综合学习能力'")
print(f"   - 包含了{eigenvals_sorted[0]/np.sum(eigenvals_sorted)*100:.1f}%的信息")

print(f"\n2. 橙色箭头：第2主成分方向，特征值={eigenvals_sorted[1]:.1f}")
print(f"   - 这个方向数据变化很小")
print(f"   - 主要是噪音和个体差异")
print(f"   - 只包含{eigenvals_sorted[1]/np.sum(eigenvals_sorted)*100:.1f}%的信息")

print(f"\n💡 为什么大特征值重要？")
print(f"- 大特征值方向捕获了数据的主要模式")
print(f"- 如果要降维（2D→1D），保留红色方向就够了")
print(f"- 丢失的信息只有{eigenvals_sorted[1]/np.sum(eigenvals_sorted)*100:.1f}%")
print(f"- 这就是PCA的核心思想！") 