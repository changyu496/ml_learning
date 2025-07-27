#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量长度（模）的深度理解
解释np.linalg.norm的实际含义和商业应用
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def explain_vector_length():
    print("📏 向量长度（模）的深度理解")
    print("=" * 50)
    
    print("🤔 首先，向量长度到底是什么？")
    print("-" * 30)
    
    # 简单的2D例子
    vector_2d = np.array([3, 4])
    length_2d = np.linalg.norm(vector_2d)
    
    print(f"2D向量例子: {vector_2d}")
    print(f"向量长度: {length_2d}")
    print(f"手工计算: √(3² + 4²) = √(9 + 16) = √25 = 5")
    print(f"几何意义: 从原点(0,0)到点(3,4)的直线距离")
    
    # 3D例子
    vector_3d = np.array([2, 3, 6])
    length_3d = np.linalg.norm(vector_3d)
    
    print(f"\n3D向量例子: {vector_3d}")
    print(f"向量长度: {length_3d:.2f}")
    print(f"手工计算: √(2² + 3² + 6²) = √(4 + 9 + 36) = √49 = 7")
    print(f"几何意义: 从原点(0,0,0)到点(2,3,6)的直线距离")
    
    print(f"\n💡 核心理解:")
    print(f"向量长度 = 从原点到该点的直线距离")
    print(f"np.linalg.norm() = 计算这个距离")
    
    print(f"\n" + "="*50)
    print(f"🛒 实际应用：用户购买行为分析")
    print(f"="*50)
    
    # 用户购买数据示例
    categories = ['3C数码', '服装', '食品', '图书', '运动用品']
    
    # 不同类型的用户
    用户数据 = {
        '轻度用户': np.array([1, 1, 2, 1, 0]),      # 总计5次购买
        '中度用户': np.array([3, 4, 5, 2, 1]),      # 总计15次购买  
        '重度用户': np.array([8, 6, 10, 5, 3]),     # 总计32次购买
        '极端用户': np.array([20, 2, 15, 8, 5]),    # 总计50次购买，但偏科严重
    }
    
    print(f"\n📊 用户购买数据分析:")
    print(f"用户类型\t购买向量\t\t\t向量长度\t总购买次数\t解释")
    print(f"-" * 80)
    
    for user_type, purchases in 用户数据.items():
        length = np.linalg.norm(purchases)
        total = np.sum(purchases)
        
        print(f"{user_type}\t{purchases}\t{length:.2f}\t\t{total}\t\t", end="")
        
        if abs(length - total) < 1:
            print("均衡购买")
        else:
            print("偏科购买" if length < total * 0.8 else "集中购买")
    
    print(f"\n🔍 关键发现:")
    print(f"1. 向量长度 ≠ 总购买次数")
    print(f"2. 向量长度反映的是'购买的集中程度'")
    print(f"3. 同样的总购买量，集中购买的向量长度更大")
    
    # 详细分析两个特殊案例
    print(f"\n📈 详细分析:")
    
    # 案例1：均衡vs集中
    均衡用户 = np.array([4, 4, 4, 4, 4])  # 总共20次，很均衡
    集中用户 = np.array([16, 1, 1, 1, 1])  # 总共20次，很集中
    
    均衡长度 = np.linalg.norm(均衡用户)
    集中长度 = np.linalg.norm(集中用户)
    
    print(f"\n案例对比（总购买次数相同）:")
    print(f"均衡用户: {均衡用户}, 长度={均衡长度:.2f}")
    print(f"集中用户: {集中用户}, 长度={集中长度:.2f}")
    print(f"结论: 集中购买的用户向量长度更大!")
    
    print(f"\n💡 向量长度的实际含义:")
    print(f"- 向量长度大 → 购买行为更集中、偏好更明显")
    print(f"- 向量长度小 → 购买行为更分散、偏好更均衡")
    
    print(f"\n" + "="*50)
    print(f"🎯 向量长度的商业应用场景")
    print(f"="*50)
    
    print(f"\n🛍️ 1. 电商平台用户分析:")
    print(f"向量长度 = 用户购买专注度")
    print(f"- 高长度用户: 专注特定品类，精准推荐")
    print(f"- 低长度用户: 兴趣广泛，多样化推荐")
    
    print(f"\n🎵 2. 音乐平台用户分析:")
    print(f"向量长度 = 音乐偏好专一度")
    print(f"- 高长度用户: 专注某种风格，深度推荐")
    print(f"- 低长度用户: 口味多样，广度推荐")
    
    print(f"\n📱 3. 社交平台用户分析:")
    print(f"向量长度 = 兴趣集中度")
    print(f"- 高长度用户: 垂直领域专家，专业内容")
    print(f"- 低长度用户: 泛娱乐用户，多元内容")
    
    print(f"\n🏥 4. 风险控制:")
    print(f"向量长度 = 行为异常度")
    print(f"- 突然变化的向量长度可能表示账号异常")
    print(f"- 异常高的向量长度可能表示刷单行为")
    
    # 实际计算演示
    print(f"\n" + "="*50)
    print(f"🧮 向量长度计算详解")
    print(f"="*50)
    
    example_vector = np.array([3, 4, 5, 2])
    
    print(f"\n示例向量: {example_vector}")
    print(f"计算过程:")
    print(f"1. 每个元素平方: [3², 4², 5², 2²] = [9, 16, 25, 4]")
    print(f"2. 求和: 9 + 16 + 25 + 4 = 54")
    print(f"3. 开方: √54 = {sqrt(54):.3f}")
    print(f"4. NumPy结果: {np.linalg.norm(example_vector):.3f}")
    
    # 手工验证
    manual_calc = sqrt(sum(x**2 for x in example_vector))
    numpy_calc = np.linalg.norm(example_vector)
    
    print(f"\n验证:")
    print(f"手工计算: {manual_calc:.6f}")
    print(f"NumPy计算: {numpy_calc:.6f}")
    print(f"结果一致: {abs(manual_calc - numpy_calc) < 1e-10}")
    
    print(f"\n🎯 关键总结:")
    print(f"向量长度 = 衡量向量'强度'的指标")
    print(f"- 几何意义: 空间中的距离")
    print(f"- 业务意义: 行为的集中程度/专注度/强度")
    print(f"- 应用价值: 用户分类、推荐策略、异常检测")

def visualize_vector_length():
    """可视化向量长度的概念"""
    print(f"\n📊 生成向量长度可视化...")
    
    # 创建不同长度的向量进行对比
    vectors = {
        '均衡型': np.array([3, 3, 3, 3]),
        '集中型': np.array([6, 1, 1, 1]),
        '两极型': np.array([4, 0, 4, 0]),
        '递减型': np.array([5, 3, 2, 1])
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左图：向量值对比
    categories = ['类别A', '类别B', '类别C', '类别D']
    x = np.arange(len(categories))
    width = 0.2
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (name, vector) in enumerate(vectors.items()):
        ax1.bar(x + i*width, vector, width, label=f"{name} (长度:{np.linalg.norm(vector):.2f})", 
                color=colors[i], alpha=0.7)
    
    ax1.set_xlabel('购买类别')
    ax1.set_ylabel('购买次数')
    ax1.set_title('不同用户的购买模式对比')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 右图：向量长度对比
    names = list(vectors.keys())
    lengths = [np.linalg.norm(vector) for vector in vectors.values()]
    totals = [np.sum(vector) for vector in vectors.values()]
    
    ax2.bar(names, lengths, color='skyblue', alpha=0.7, label='向量长度')
    ax2.bar(names, totals, color='lightcoral', alpha=0.7, label='总购买次数')
    
    ax2.set_ylabel('数值')
    ax2.set_title('向量长度 vs 总和的对比')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, (length, total) in enumerate(zip(lengths, totals)):
        ax2.text(i, length + 0.1, f'{length:.1f}', ha='center', va='bottom')
        ax2.text(i, total + 0.1, f'{total}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n图表说明:")
    print(f"- 左图：不同用户的购买分布模式")
    print(f"- 右图：向量长度与总购买次数的区别")
    print(f"- 关键：总购买次数相近，但向量长度差异很大")
    print(f"- 结论：向量长度反映的是'集中程度'，不是'总量'")

if __name__ == "__main__":
    explain_vector_length()
    visualize_vector_length() 