#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量相似度计算演示 - 简化版推荐系统
作者：大模型学习者
日期：第3天学习内容
目标：理解向量在推荐系统中的应用
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("🎬 简化版电影推荐系统演示")
    print("=" * 50)
    
    # 模拟用户评分数据
    # 每行代表一个用户对5部电影的评分（1-5分）
    users_ratings = np.array([
        [5, 3, 4, 2, 1],  # 张三：喜欢动作片、科幻片
        [4, 3, 5, 2, 2],  # 李四：喜欢动作片、科幻片
        [1, 2, 1, 4, 5],  # 王五：喜欢爱情片、恐怖片
        [5, 4, 4, 1, 1],  # 赵六：喜欢动作片、喜剧片、科幻片
        [2, 1, 2, 5, 4],  # 钱七：喜欢爱情片、恐怖片
        [3, 5, 3, 3, 2],  # 孙八：喜欢喜剧片
    ])
    
    user_names = ['张三', '李四', '王五', '赵六', '钱七', '孙八']
    movies = ['动作片', '喜剧片', '科幻片', '爱情片', '恐怖片']
    
    print("\n📊 用户评分数据:")
    print("用户\t", end="")
    for movie in movies:
        print(f"{movie}\t", end="")
    print()
    print("-" * 50)
    
    for i, name in enumerate(user_names):
        print(f"{name}\t", end="")
        for rating in users_ratings[i]:
            print(f"{rating}\t", end="")
        print()
    
    # 选择目标用户
    target_user = 0  # 张三
    print(f"\n🎯 为 {user_names[target_user]} 寻找相似用户并推荐电影")
    print("-" * 50)
    
    # 计算相似度
    similarities = []
    for i in range(len(users_ratings)):
        if i != target_user:
            # 使用余弦相似度
            sim = cosine_similarity([users_ratings[target_user]], [users_ratings[i]])[0][0]
            similarities.append((i, user_names[i], sim))
            print(f"{user_names[target_user]} vs {user_names[i]}: 相似度 = {sim:.3f}")
    
    # 找到最相似的用户
    most_similar = max(similarities, key=lambda x: x[2])
    most_similar_idx, most_similar_name, most_similar_score = most_similar
    
    print(f"\n🏆 最相似的用户: {most_similar_name} (相似度: {most_similar_score:.3f})")
    
    # 推荐逻辑
    print(f"\n💡 推荐分析:")
    print(f"张三的评分: {users_ratings[target_user]}")
    print(f"{most_similar_name}的评分: {users_ratings[most_similar_idx]}")
    
    # 找到推荐的电影
    recommendations = []
    for i, movie in enumerate(movies):
        target_rating = users_ratings[target_user][i]
        similar_rating = users_ratings[most_similar_idx][i]
        
        # 如果相似用户喜欢(4+分)但目标用户评分不高(<4分)，则推荐
        if similar_rating >= 4 and target_rating < 4:
            recommendations.append((movie, similar_rating, target_rating))
    
    print(f"\n🎬 推荐结果:")
    if recommendations:
        for movie, similar_rating, target_rating in recommendations:
            print(f"推荐《{movie}》: {most_similar_name}评分{similar_rating}分，但你只给了{target_rating}分")
            print(f"  → 因为{most_similar_name}和你品味相似，你可能也会喜欢这部电影！")
    else:
        print("暂无推荐，继续收集更多评分数据...")
    
    # 可视化相似度
    visualize_similarity(users_ratings, user_names, target_user)
    
    print(f"\n🎯 核心原理:")
    print(f"1. 每个用户的评分 = 一个向量")
    print(f"2. 向量相似度 = 用户品味相似度")
    print(f"3. 相似用户喜欢的内容 = 推荐候选")
    print(f"4. 这就是推荐系统的数学基础！")
    
    print(f"\n🌟 现在你理解了淘宝、网易云、Netflix的推荐原理！")

def visualize_similarity(users_ratings, user_names, target_user):
    """可视化用户相似度"""
    print(f"\n📊 生成相似度可视化图表...")
    
    # 计算目标用户与所有其他用户的相似度
    similarities = []
    names = []
    
    for i in range(len(users_ratings)):
        if i != target_user:
            sim = cosine_similarity([users_ratings[target_user]], [users_ratings[i]])[0][0]
            similarities.append(sim)
            names.append(user_names[i])
    
    # 创建柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, similarities, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
    
    # 添加数值标签
    for bar, sim in zip(bars, similarities):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{sim:.3f}', ha='center', va='bottom')
    
    plt.title(f'用户相似度分析 - 以{user_names[target_user]}为基准')
    plt.xlabel('其他用户')
    plt.ylabel('相似度分数')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    
    # 标记最相似的用户
    max_idx = np.argmax(similarities)
    bars[max_idx].set_color('red')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 