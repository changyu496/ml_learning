#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L2范数与余弦相似度的关系
详细解释向量范数的概念和在相似度计算中的作用
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def explain_l2_norm_relationship():
    print("📐 L2范数与余弦相似度的关系")
    print("=" * 50)
    
    print("🎯 核心发现：L2范数就是余弦相似度公式中的分母！")
    print("-" * 50)
    
    # 示例向量
    A = np.array([3, 4, 5])
    B = np.array([6, 8, 10])
    
    print(f"向量A: {A}")
    print(f"向量B: {B}")
    
    print(f"\n📏 L2范数计算（就是向量长度）:")
    
    # L2范数的多种计算方法
    l2_A_manual = np.sqrt(np.sum(A**2))
    l2_A_numpy = np.linalg.norm(A)
    l2_A_sklearn = np.linalg.norm(A, ord=2)  # 明确指定L2范数
    
    l2_B_manual = np.sqrt(np.sum(B**2))
    l2_B_numpy = np.linalg.norm(B)
    
    print(f"向量A的L2范数:")
    print(f"  手工计算: √(3² + 4² + 5²) = √(9 + 16 + 25) = √50 = {l2_A_manual:.3f}")
    print(f"  NumPy方法: {l2_A_numpy:.3f}")
    print(f"  明确L2: {l2_A_sklearn:.3f}")
    
    print(f"\n向量B的L2范数:")
    print(f"  手工计算: √(6² + 8² + 10²) = √(36 + 64 + 100) = √200 = {l2_B_manual:.3f}")
    print(f"  NumPy方法: {l2_B_numpy:.3f}")
    
    print(f"\n🧮 余弦相似度的完整计算过程:")
    
    # 点积
    dot_product = np.dot(A, B)
    print(f"1. 计算点积: A·B = {A[0]}×{B[0]} + {A[1]}×{B[1]} + {A[2]}×{B[2]} = {dot_product}")
    
    # L2范数（分母）
    print(f"2. 计算L2范数:")
    print(f"   ||A||₂ = {l2_A_numpy:.3f}")
    print(f"   ||B||₂ = {l2_B_numpy:.3f}")
    print(f"   ||A||₂ × ||B||₂ = {l2_A_numpy:.3f} × {l2_B_numpy:.3f} = {l2_A_numpy * l2_B_numpy:.3f}")
    
    # 余弦相似度
    cosine_manual = dot_product / (l2_A_numpy * l2_B_numpy)
    cosine_sklearn = cosine_similarity([A], [B])[0][0]
    
    print(f"3. 计算余弦相似度:")
    print(f"   cos(A,B) = {dot_product} / {l2_A_numpy * l2_B_numpy:.3f} = {cosine_manual:.3f}")
    print(f"   sklearn验证: {cosine_sklearn:.3f}")
    print(f"   计算正确: {abs(cosine_manual - cosine_sklearn) < 1e-10}")
    
    print(f"\n💡 关键理解:")
    print(f"余弦相似度 = 点积 / (L2范数A × L2范数B)")
    print(f"L2范数就是我们之前说的'向量长度'!")

def explain_norm_family():
    """解释范数家族"""
    print(f"\n" + "="*50)
    print(f"📚 范数家族：不只有L2范数")
    print(f"="*50)
    
    vector = np.array([3, -4, 5])
    
    print(f"示例向量: {vector}")
    print(f"\n🔢 常见范数对比:")
    
    # L1范数（曼哈顿距离）
    l1_norm = np.linalg.norm(vector, ord=1)
    l1_manual = np.sum(np.abs(vector))
    
    print(f"\n1️⃣ L1范数（曼哈顿距离）:")
    print(f"   公式: |v₁| + |v₂| + |v₃|")
    print(f"   计算: |3| + |-4| + |5| = 3 + 4 + 5 = {l1_manual}")
    print(f"   NumPy: {l1_norm}")
    print(f"   含义: 各维度距离之和")
    
    # L2范数（欧几里得距离）
    l2_norm = np.linalg.norm(vector, ord=2)
    l2_manual = np.sqrt(np.sum(vector**2))
    
    print(f"\n2️⃣ L2范数（欧几里得距离）:")
    print(f"   公式: √(v₁² + v₂² + v₃²)")
    print(f"   计算: √(3² + (-4)² + 5²) = √(9 + 16 + 25) = √50 = {l2_manual:.3f}")
    print(f"   NumPy: {l2_norm:.3f}")
    print(f"   含义: 直线距离（我们一直在用的）")
    
    # L∞范数（切比雪夫距离）
    linf_norm = np.linalg.norm(vector, ord=np.inf)
    linf_manual = np.max(np.abs(vector))
    
    print(f"\n3️⃣ L∞范数（切比雪夫距离）:")
    print(f"   公式: max(|v₁|, |v₂|, |v₃|)")
    print(f"   计算: max(|3|, |-4|, |5|) = max(3, 4, 5) = {linf_manual}")
    print(f"   NumPy: {linf_norm}")
    print(f"   含义: 最大维度的距离")
    
    print(f"\n🎯 为什么推荐系统主要用L2范数？")
    print(f"1. L2范数考虑所有维度，更全面")
    print(f"2. 平方运算强调大差异，符合偏好强度的概念")
    print(f"3. 数学性质好，可微分，适合优化")
    print(f"4. 几何意义直观（直线距离）")

def demonstrate_normalization_effect():
    """演示标准化效果"""
    print(f"\n" + "="*50)
    print(f"🎯 L2标准化的神奇效果")
    print(f"="*50)
    
    print(f"\n🔍 实验：不同规模的相似向量")
    
    # 创建不同规模但方向相同的向量
    base_direction = np.array([1, 2, 3])  # 基础方向
    
    vectors = {
        '小规模': base_direction * 1,      # [1, 2, 3]
        '中规模': base_direction * 3,      # [3, 6, 9]  
        '大规模': base_direction * 10,     # [10, 20, 30]
        '巨规模': base_direction * 100,    # [100, 200, 300]
    }
    
    print(f"🧮 原始向量（相同方向，不同规模）:")
    for name, vec in vectors.items():
        l2_norm = np.linalg.norm(vec)
        print(f"{name}: {vec}, L2范数: {l2_norm:.1f}")
    
    print(f"\n📐 L2标准化后（除以各自的L2范数）:")
    normalized_vectors = {}
    for name, vec in vectors.items():
        normalized = vec / np.linalg.norm(vec)
        normalized_vectors[name] = normalized
        print(f"{name}: {normalized}, L2范数: {np.linalg.norm(normalized):.3f}")
    
    print(f"\n✨ 神奇发现：标准化后所有向量长度都是1！")
    
    # 计算标准化向量之间的余弦相似度
    print(f"\n🎯 标准化向量之间的余弦相似度:")
    base_normalized = normalized_vectors['小规模']
    
    for name, normalized_vec in normalized_vectors.items():
        if name != '小规模':
            cosine = cosine_similarity([base_normalized], [normalized_vec])[0][0]
            print(f"小规模 vs {name}: {cosine:.6f}")
    
    print(f"\n💡 重要理解:")
    print(f"1. 相同方向的向量，标准化后完全相同")
    print(f"2. 余弦相似度实际上就是在比较标准化后的向量")
    print(f"3. L2标准化消除了规模，只保留方向信息")

def business_applications():
    """商业应用中的L2范数"""
    print(f"\n" + "="*50)
    print(f"🏢 L2范数在商业中的实际应用")
    print(f"="*50)
    
    print(f"\n🛍️ 1. 电商推荐系统:")
    print(f"场景: 用户购买行为向量化")
    
    # 模拟用户数据
    users = {
        '学生用户': np.array([2, 8, 1, 5, 0]),    # [数码, 服装, 奢侈品, 食品, 汽车]
        '白领用户': np.array([5, 12, 3, 8, 0]),   # 消费能力更强
        '富豪用户': np.array([20, 30, 50, 15, 10]) # 高端消费
    }
    
    categories = ['数码', '服装', '奢侈品', '食品', '汽车']
    
    print(f"\n用户购买数据:")
    for name, purchases in users.items():
        l2_norm = np.linalg.norm(purchases)
        total = np.sum(purchases)
        print(f"{name}: {purchases}")
        print(f"  总购买: {total}, L2范数: {l2_norm:.1f}, 集中度: {l2_norm/total:.2f}")
    
    print(f"\n🔍 推荐策略:")
    print(f"1. 基于L2范数大小:")
    print(f"   - 高L2范数用户: 推荐高价值商品")
    print(f"   - 低L2范数用户: 推荐基础商品")
    
    print(f"\n2. 基于余弦相似度:")
    print(f"   - 找到偏好模式相似的用户")
    print(f"   - 推荐相似用户购买过的商品")
    
    # 计算用户间的相似度
    学生 = users['学生用户']
    白领 = users['白领用户'] 
    富豪 = users['富豪用户']
    
    cos_学生白领 = cosine_similarity([学生], [白领])[0][0]
    cos_学生富豪 = cosine_similarity([学生], [富豪])[0][0]
    cos_白领富豪 = cosine_similarity([白领], [富豪])[0][0]
    
    print(f"\n用户相似度分析:")
    print(f"学生 vs 白领: {cos_学生白领:.3f}")
    print(f"学生 vs 富豪: {cos_学生富豪:.3f}")
    print(f"白领 vs 富豪: {cos_白领富豪:.3f}")
    print(f"结论: 白领和富豪最相似，可以互相推荐商品")

def visualization():
    """可视化L2范数和余弦相似度"""
    print(f"\n📊 生成L2范数可视化...")
    
    # 创建示例向量
    vectors = [
        np.array([3, 4]),
        np.array([6, 8]),    # 与第一个方向相同
        np.array([4, 3]),    # 与第一个方向不同
        np.array([-3, -4])   # 与第一个方向相反
    ]
    
    names = ['向量A', '向量B(2倍A)', '向量C', '向量D(-A)']
    colors = ['red', 'blue', 'green', 'orange']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 图1：原始向量
    for i, (vec, name, color) in enumerate(zip(vectors, names, colors)):
        ax1.arrow(0, 0, vec[0], vec[1], head_width=0.3, head_length=0.3,
                 fc=color, ec=color, linewidth=2, label=name)
        ax1.text(vec[0]*1.1, vec[1]*1.1, f"{name}\nL2:{np.linalg.norm(vec):.1f}", 
                fontsize=8, color=color)
    
    ax1.set_xlim(-5, 8)
    ax1.set_ylim(-5, 9)
    ax1.set_title('原始向量（不同长度）')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 图2：L2标准化后的向量
    for i, (vec, name, color) in enumerate(zip(vectors, names, colors)):
        normalized = vec / np.linalg.norm(vec)
        ax2.arrow(0, 0, normalized[0], normalized[1], head_width=0.1, head_length=0.1,
                 fc=color, ec=color, linewidth=2, label=name)
        ax2.text(normalized[0]*1.2, normalized[1]*1.2, f"{name}\nL2:1.0", 
                fontsize=8, color=color)
    
    # 画单位圆
    circle = plt.Circle((0, 0), 1, fill=False, linestyle='--', alpha=0.5)
    ax2.add_patch(circle)
    
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_title('L2标准化后（都在单位圆上）')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 图3：L2范数比较
    l2_norms = [np.linalg.norm(vec) for vec in vectors]
    bars = ax3.bar(names, l2_norms, color=colors, alpha=0.7)
    ax3.set_ylabel('L2范数')
    ax3.set_title('各向量的L2范数')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, norm in zip(bars, l2_norms):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{norm:.1f}', ha='center', va='bottom')
    
    # 图4：余弦相似度（以向量A为基准）
    base_vec = vectors[0]
    similarities = []
    for vec in vectors[1:]:
        sim = cosine_similarity([base_vec], [vec])[0][0]
        similarities.append(sim)
    
    bars = ax4.bar(names[1:], similarities, color=colors[1:], alpha=0.7)
    ax4.set_ylabel('余弦相似度')
    ax4.set_title('与向量A的余弦相似度')
    ax4.set_ylim(-1.1, 1.1)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.tick_params(axis='x', rotation=45)
    
    for bar, sim in zip(bars, similarities):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{sim:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print(f"图表说明:")
    print(f"- 左上：原始向量，长度不同")
    print(f"- 右上：L2标准化后，都在单位圆上") 
    print(f"- 左下：L2范数大小对比")
    print(f"- 右下：余弦相似度（消除长度影响后的方向相似度）")

if __name__ == "__main__":
    explain_l2_norm_relationship()
    explain_norm_family()
    demonstrate_normalization_effect()
    business_applications()
    visualization() 