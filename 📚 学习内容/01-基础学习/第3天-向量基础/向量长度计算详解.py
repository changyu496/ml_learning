#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量长度计算详解：为什么能反映集中程度
用具体例子解释数学原理
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def explain_vector_length_formula():
    print("🧮 向量长度计算公式详解")
    print("=" * 50)
    
    # 用户的例子
    a = np.array([4, 4])
    b = np.array([7, 1])
    
    print(f"📊 用户例子分析:")
    print(f"向量a: {a} (均衡分布)")
    print(f"向量b: {b} (集中分布)")
    print(f"两者元素和相同: a={np.sum(a)}, b={np.sum(b)}")
    
    print(f"\n📏 向量长度计算公式:")
    print(f"||v|| = √(v₁² + v₂² + ... + vₙ²)")
    
    print(f"\n🧮 详细计算过程:")
    
    # a的计算
    print(f"\n向量a = [4, 4]:")
    print(f"1. 各元素平方: [4², 4²] = [16, 16]")
    print(f"2. 求和: 16 + 16 = 32")
    print(f"3. 开平方: √32 = {np.sqrt(32):.3f}")
    print(f"4. NumPy验证: {np.linalg.norm(a):.3f}")
    
    # b的计算
    print(f"\n向量b = [7, 1]:")
    print(f"1. 各元素平方: [7², 1²] = [49, 1]")
    print(f"2. 求和: 49 + 1 = 50")
    print(f"3. 开平方: √50 = {np.sqrt(50):.3f}")
    print(f"4. NumPy验证: {np.linalg.norm(b):.3f}")
    
    print(f"\n🔍 关键发现:")
    print(f"- 向量a长度: {np.linalg.norm(a):.3f}")
    print(f"- 向量b长度: {np.linalg.norm(b):.3f}")
    print(f"- b的长度更大: {np.linalg.norm(b):.3f} > {np.linalg.norm(a):.3f}")
    print(f"- 结论: 集中分布的向量长度更大!")
    
    print(f"\n💡 为什么平方运算能反映集中程度？")
    print(f"="*50)
    
    # 系列例子
    examples = [
        ("极度均衡", [5, 5]),
        ("略微集中", [6, 4]), 
        ("明显集中", [7, 3]),
        ("高度集中", [8, 2]),
        ("极度集中", [9, 1])
    ]
    
    print(f"\n📈 系列对比（元素和都是10）:")
    print(f"分布类型\t向量\t\t平方后\t\t和\t长度\t集中度")
    print(f"-" * 70)
    
    for desc, vec in examples:
        vec_arr = np.array(vec)
        squared = vec_arr ** 2
        sum_squared = np.sum(squared)
        length = np.linalg.norm(vec_arr)
        
        # 集中度：最大值占比
        concentration = max(vec) / sum(vec) * 100
        
        print(f"{desc}\t{vec}\t\t{squared.tolist()}\t\t{sum_squared}\t{length:.2f}\t{concentration:.0f}%")
    
    print(f"\n🎯 关键发现:")
    print(f"1. 元素和相同（都是10）")
    print(f"2. 平方和逐渐增大")
    print(f"3. 向量长度逐渐增大")
    print(f"4. 集中程度逐渐增强")
    
    print(f"\n🔥 数学原理:")
    print(f"平方运算的'放大效应':")
    print(f"- 小数值平方后变得更小（相对）")
    print(f"- 大数值平方后变得更大（相对）")
    print(f"- 集中的分布会产生更大的平方和")
    print(f"- 因此向量长度能反映集中程度!")

def demonstrate_concentration_effect():
    """演示集中效应"""
    print(f"\n" + "="*50)
    print(f"🎯 平方运算的'集中效应'演示")
    print(f"="*50)
    
    # 固定总和为20的不同分布
    distributions = {
        "完全均衡": [5, 5, 5, 5],
        "轻微集中": [7, 5, 4, 4], 
        "中度集中": [10, 4, 3, 3],
        "高度集中": [14, 2, 2, 2],
        "极度集中": [17, 1, 1, 1]
    }
    
    print(f"\n📊 不同分布的向量长度对比（总和都是20）:")
    print(f"分布类型\t向量\t\t\t向量长度\t最大占比")
    print(f"-" * 65)
    
    lengths = []
    names = []
    
    for name, dist in distributions.items():
        vec = np.array(dist)
        length = np.linalg.norm(vec)
        max_ratio = max(dist) / sum(dist) * 100
        
        print(f"{name}\t{dist}\t\t{length:.2f}\t\t{max_ratio:.0f}%")
        
        lengths.append(length)
        names.append(name)
    
    print(f"\n💡 关键观察:")
    print(f"- 总和相同，但向量长度差异巨大")
    print(f"- 最均衡分布: 长度 {min(lengths):.2f}")
    print(f"- 最集中分布: 长度 {max(lengths):.2f}")
    print(f"- 差异倍数: {max(lengths)/min(lengths):.1f}倍")
    
    # 可视化
    plt.figure(figsize=(12, 8))
    
    # 上图：分布对比
    plt.subplot(2, 1, 1)
    categories = ['类别1', '类别2', '类别3', '类别4']
    x = np.arange(len(categories))
    width = 0.15
    
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    for i, (name, dist) in enumerate(distributions.items()):
        plt.bar(x + i*width, dist, width, label=f"{name} (长度:{np.linalg.norm(dist):.1f})", 
                color=colors[i], alpha=0.7)
    
    plt.xlabel('购买类别')
    plt.ylabel('购买次数')
    plt.title('不同集中程度的购买分布（总和相同）')
    plt.xticks(x + width * 2, categories)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 下图：向量长度对比
    plt.subplot(2, 1, 2)
    bars = plt.bar(names, lengths, color=colors, alpha=0.7)
    plt.xlabel('分布类型')
    plt.ylabel('向量长度')
    plt.title('集中程度 vs 向量长度')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, length in zip(bars, lengths):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{length:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def business_application():
    """商业应用场景"""
    print(f"\n" + "="*50)
    print(f"🏢 商业应用：基于向量长度的推荐策略")
    print(f"="*50)
    
    # 三种用户类型
    user_types = {
        "广泛兴趣用户": np.array([3, 3, 3, 3, 3]),     # 向量长度小
        "一般集中用户": np.array([5, 4, 2, 2, 2]),     # 向量长度中等  
        "专业领域用户": np.array([12, 1, 1, 1, 0]),    # 向量长度大
    }
    
    print(f"\n👥 用户类型分析:")
    for user_type, purchases in user_types.items():
        length = np.linalg.norm(purchases)
        total = np.sum(purchases)
        concentration = max(purchases) / total * 100
        
        print(f"\n{user_type}:")
        print(f"  购买向量: {purchases}")
        print(f"  向量长度: {length:.2f}")
        print(f"  总购买量: {total}")
        print(f"  集中程度: {concentration:.0f}%")
        
        # 推荐策略
        if length < 7:
            strategy = "多样化推荐：广撒网，各领域都推"
        elif length < 10:
            strategy = "混合推荐：主推偏好领域，辅助其他"
        else:
            strategy = "精准推荐：深挖专业领域，相关产品"
            
        print(f"  推荐策略: {strategy}")
    
    print(f"\n🎯 向量长度在推荐系统中的价值:")
    print(f"1. 用户分层：自动识别专业用户vs泛用户")
    print(f"2. 策略选择：精准推荐vs多样化推荐")
    print(f"3. 风险控制：识别异常行为（突然的长度变化）")
    print(f"4. 商业价值：专业用户转化率高，泛用户覆盖面广")

if __name__ == "__main__":
    explain_vector_length_formula()
    demonstrate_concentration_effect()
    business_application() 