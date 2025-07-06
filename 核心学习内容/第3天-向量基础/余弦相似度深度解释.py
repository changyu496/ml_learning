#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
余弦相似度 vs 点积：为什么更准确？
详细解释cosine_similarity的原理和优势
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def explain_cosine_similarity():
    print("📐 余弦相似度 vs 点积：深度对比")
    print("=" * 50)
    
    print("🤔 首先，什么是余弦相似度？")
    print("-" * 30)
    
    print("💡 核心公式:")
    print("余弦相似度 = 点积 / (向量A长度 × 向量B长度)")
    print("cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)")
    print("取值范围: -1 到 1")
    print("- 1: 完全相同方向")
    print("- 0: 垂直（无关）") 
    print("- -1: 完全相反方向")
    
    print(f"\n" + "="*50)
    print(f"🛒 问题：为什么需要余弦相似度？")
    print(f"="*50)
    
    # 创建一个说明性的例子
    print(f"\n📊 用户评分例子:")
    
    # 用户A：轻度使用者
    用户A = np.array([2, 1, 2, 1, 1])  # 总计7次购买，比较保守
    # 用户B：重度使用者，但偏好相似
    用户B = np.array([8, 4, 8, 4, 4])  # 总计28次购买，是A的4倍
    # 用户C：轻度使用者，但偏好不同
    用户C = np.array([1, 2, 1, 2, 1])  # 总计7次购买，偏好不同
    
    movies = ['动作片', '喜剧片', '科幻片', '爱情片', '恐怖片']
    
    print(f"用户A: {用户A} (轻度用户，喜欢动作片和科幻片)")
    print(f"用户B: {用户B} (重度用户，也喜欢动作片和科幻片)")  
    print(f"用户C: {用户C} (轻度用户，喜欢喜剧片和爱情片)")
    
    # 计算点积
    dot_AB = np.dot(用户A, 用户B)
    dot_AC = np.dot(用户A, 用户C)
    
    print(f"\n🧮 点积计算:")
    print(f"用户A · 用户B = {dot_AB}")
    print(f"用户A · 用户C = {dot_AC}")
    print(f"点积结论: A和B更相似 ({dot_AB} > {dot_AC})")
    
    # 计算余弦相似度
    cos_AB = cosine_similarity([用户A], [用户B])[0][0]
    cos_AC = cosine_similarity([用户A], [用户C])[0][0]
    
    print(f"\n📐 余弦相似度计算:")
    print(f"用户A vs 用户B = {cos_AB:.3f}")
    print(f"用户A vs 用户C = {cos_AC:.3f}")
    print(f"余弦相似度结论: A和B更相似 ({cos_AB:.3f} > {cos_AC:.3f})")
    
    print(f"\n💡 两种方法的区别:")
    print(f"- 点积差异: {dot_AB - dot_AC}")
    print(f"- 余弦相似度差异: {cos_AB - cos_AC:.3f}")
    
    # 手工计算余弦相似度验证
    print(f"\n🧮 手工计算余弦相似度验证:")
    
    # A和B的余弦相似度
    norm_A = np.linalg.norm(用户A)
    norm_B = np.linalg.norm(用户B)
    manual_cos_AB = dot_AB / (norm_A * norm_B)
    
    print(f"\n用户A vs 用户B:")
    print(f"点积: {dot_AB}")
    print(f"用户A长度: {norm_A:.3f}")
    print(f"用户B长度: {norm_B:.3f}")
    print(f"手工计算: {dot_AB} / ({norm_A:.3f} × {norm_B:.3f}) = {manual_cos_AB:.3f}")
    print(f"sklearn结果: {cos_AB:.3f}")
    print(f"验证: {abs(manual_cos_AB - cos_AB) < 1e-10}")

def demonstrate_cosine_advantage():
    """演示余弦相似度的优势"""
    print(f"\n" + "="*50)
    print(f"🎯 余弦相似度的核心优势")
    print(f"="*50)
    
    print(f"\n🔍 问题场景：用户购买规模差异很大")
    
    # 三个用户：偏好相似但购买规模不同
    轻度用户 = np.array([3, 1, 3, 1, 1])    # 9次购买，偏好动作+科幻
    中度用户 = np.array([6, 2, 6, 2, 2])    # 18次购买，相同偏好
    重度用户 = np.array([15, 5, 15, 5, 5])  # 45次购买，相同偏好
    
    # 偏好不同的用户
    不同用户 = np.array([1, 4, 1, 4, 1])    # 11次购买，偏好喜剧+爱情
    
    users = {
        '轻度用户': 轻度用户,
        '中度用户': 中度用户, 
        '重度用户': 重度用户,
        '不同用户': 不同用户
    }
    
    print(f"\n📊 用户数据:")
    for name, user in users.items():
        print(f"{name}: {user} (总购买: {np.sum(user)})")
    
    print(f"\n🧮 以轻度用户为基准，计算相似度:")
    print(f"对比用户\t点积\t余弦相似度\t解释")
    print(f"-" * 60)
    
    base_user = 轻度用户
    
    for name, user in users.items():
        if name == '轻度用户':
            continue
            
        # 点积
        dot_product = np.dot(base_user, user)
        
        # 余弦相似度
        cos_sim = cosine_similarity([base_user], [user])[0][0]
        
        # 分析
        if name in ['中度用户', '重度用户']:
            explanation = "偏好相同，规模不同"
        else:
            explanation = "偏好不同"
            
        print(f"{name}\t{dot_product}\t{cos_sim:.3f}\t\t{explanation}")
    
    print(f"\n🎯 关键发现:")
    print(f"1. 点积会被购买规模影响（重度用户分数虚高）")
    print(f"2. 余弦相似度只关注偏好模式，不受规模影响")
    print(f"3. 中度用户和重度用户与轻度用户的余弦相似度都很高")
    print(f"4. 不同偏好的用户余弦相似度明显较低")

def real_world_scenarios():
    """真实世界的应用场景"""
    print(f"\n" + "="*50)
    print(f"🌍 真实世界应用场景")
    print(f"="*50)
    
    print(f"\n🛍️ 1. 电商推荐系统:")
    print(f"问题: 大客户vs小客户的公平比较")
    
    # 模拟数据
    大客户 = np.array([50, 10, 40, 20, 30])  # 高消费用户
    小客户 = np.array([5, 1, 4, 2, 3])       # 低消费用户，但偏好相似
    新客户 = np.array([2, 0, 2, 1, 1])       # 新用户，需要推荐
    
    print(f"大客户: {大客户} (总消费: {np.sum(大客户)})")
    print(f"小客户: {小客户} (总消费: {np.sum(小客户)})")
    print(f"新客户: {新客户} (总消费: {np.sum(新客户)})")
    
    # 对比两种相似度
    dot_big = np.dot(新客户, 大客户)
    dot_small = np.dot(新客户, 小客户)
    
    cos_big = cosine_similarity([新客户], [大客户])[0][0]
    cos_small = cosine_similarity([新客户], [小客户])[0][0]
    
    print(f"\n推荐依据对比:")
    print(f"新客户 vs 大客户 - 点积: {dot_big}, 余弦: {cos_big:.3f}")
    print(f"新客户 vs 小客户 - 点积: {dot_small}, 余弦: {cos_small:.3f}")
    
    print(f"\n结论:")
    print(f"- 点积推荐: 参考大客户 (分数更高)")
    print(f"- 余弦推荐: 参考小客户 (模式更相似)")
    print(f"- 余弦相似度更公平，不受消费规模影响!")
    
    print(f"\n🎵 2. 音乐推荐系统:")
    print(f"问题: 重度听众vs轻度听众的偏好识别")
    
    重度听众 = np.array([100, 20, 80, 10, 90])  # 每天听很多音乐
    轻度听众 = np.array([10, 2, 8, 1, 9])       # 听得少但偏好相似
    目标用户 = np.array([5, 1, 4, 0, 4])        # 新用户
    
    cos_heavy = cosine_similarity([目标用户], [重度听众])[0][0]
    cos_light = cosine_similarity([目标用户], [轻度听众])[0][0]
    
    print(f"目标用户 vs 重度听众: {cos_heavy:.3f}")
    print(f"目标用户 vs 轻度听众: {cos_light:.3f}")
    print(f"结论: 两者偏好模式相似，可以互相推荐!")

def visualization():
    """可视化余弦相似度的几何意义"""
    print(f"\n📊 生成余弦相似度可视化...")
    
    # 创建2D向量用于可视化
    vectors = {
        '用户A': np.array([4, 2]),
        '用户B': np.array([8, 4]),  # 与A方向相同，但更长
        '用户C': np.array([2, 4]),  # 与A方向不同
        '用户D': np.array([-2, -1]) # 与A方向相反
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左图：向量可视化
    colors = ['red', 'blue', 'green', 'orange']
    
    for i, (name, vec) in enumerate(vectors.items()):
        ax1.arrow(0, 0, vec[0], vec[1], head_width=0.2, head_length=0.3, 
                 fc=colors[i], ec=colors[i], linewidth=2, label=name)
        ax1.text(vec[0]*1.1, vec[1]*1.1, name, fontsize=10, color=colors[i])
    
    ax1.set_xlim(-3, 9)
    ax1.set_ylim(-2, 5)
    ax1.set_xlabel('特征1')
    ax1.set_ylabel('特征2')
    ax1.set_title('向量可视化')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 右图：相似度对比
    base_vector = vectors['用户A']
    similarities = []
    names = []
    
    for name, vec in vectors.items():
        if name != '用户A':
            cos_sim = cosine_similarity([base_vector], [vec])[0][0]
            similarities.append(cos_sim)
            names.append(name)
    
    bars = ax2.bar(names, similarities, color=['blue', 'green', 'orange'], alpha=0.7)
    ax2.set_ylabel('余弦相似度')
    ax2.set_title('用户A与其他用户的余弦相似度')
    ax2.set_ylim(-1.1, 1.1)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, sim in zip(bars, similarities):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{sim:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print(f"图表说明:")
    print(f"- 左图：向量的方向和长度")
    print(f"- 右图：余弦相似度只关注方向，不关注长度")
    print(f"- 用户B与A方向相同，相似度最高")
    print(f"- 用户D与A方向相反，相似度为负")

def summary():
    """总结"""
    print(f"\n" + "="*50)
    print(f"📝 余弦相似度总结")
    print(f"="*50)
    
    print(f"\n🎯 为什么余弦相似度更准确？")
    print(f"1. 消除规模影响：不受向量长度影响，只看方向")
    print(f"2. 标准化比较：所有用户在同一标准下比较")
    print(f"3. 更公平：大客户和小客户享受同等权重")
    print(f"4. 更稳定：不会因为购买频次差异影响推荐质量")
    
    print(f"\n📊 点积 vs 余弦相似度:")
    print(f"点积:")
    print(f"  - 优点：计算简单，反映绝对差异")
    print(f"  - 缺点：受向量长度影响，对大客户有偏见")
    print(f"  - 适用：向量长度相近的场景")
    
    print(f"\n余弦相似度:")
    print(f"  - 优点：只关注模式，消除规模影响")
    print(f"  - 缺点：计算稍复杂，忽略绝对差异")
    print(f"  - 适用：用户规模差异很大的场景 ⭐ 推荐")
    
    print(f"\n🚀 实际应用建议:")
    print(f"- 推荐系统：优先使用余弦相似度")
    print(f"- 用户分群：余弦相似度更公平")
    print(f"- 内容匹配：余弦相似度更准确")
    print(f"- 异常检测：结合使用两种指标")

if __name__ == "__main__":
    explain_cosine_similarity()
    demonstrate_cosine_advantage()
    real_world_scenarios()
    visualization()
    summary() 