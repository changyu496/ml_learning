#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第3天：向量基础 - 今日编码练习（需要完成）
包含多个需要你自己完成的任务和挑战
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def challenge_1_create_users():
    """挑战1：创建用户数据"""
    print("🎯 挑战1：创建用户数据")
    print("="*50)
    
    # TODO: 在这里创建5个用户的电影评分向量
    # 电影类别：动作、喜剧、科幻、恐怖、爱情
    # 每个用户对5个类别进行1-5分的评分
    # 要求：每个用户要有不同的偏好特点
    
    users = {
        # TODO: 在这里添加5个用户的评分向量
        # 例如：
        # '张三': np.array([5, 3, 4, 2, 1]),  # 喜欢动作，不喜欢爱情
        # '李四': np.array([4, 5, 3, 1, 4]),  # 喜欢喜剧和爱情
        # 继续添加3个用户...
    }
    
    categories = ['动作', '喜剧', '科幻', '恐怖', '爱情']
    
    print("📊 你创建的用户评分数据:")
    for name, ratings in users.items():
        print(f"{name}: {ratings}")
    
    return users, categories

def challenge_2_vector_operations():
    """挑战2：向量运算"""
    print("\n🎯 挑战2：向量运算")
    print("="*50)
    
    users, categories = challenge_1_create_users()
    
    # TODO: 选择两个用户进行运算
    user_A_name = "用户A"  # 替换为实际的用户名
    user_B_name = "用户B"  # 替换为实际的用户名
    
    user_A = users[user_A_name]
    user_B = users[user_B_name]
    
    print(f"用户A（{user_A_name}）: {user_A}")
    print(f"用户B（{user_B_name}）: {user_B}")
    
    # TODO: 计算向量加法（平均偏好）
    average_preference = None  # 在这里计算
    print(f"\n📈 平均偏好: {average_preference}")
    
    # TODO: 计算向量减法（偏好差异）
    preference_diff = None  # 在这里计算
    print(f"📊 偏好差异: {preference_diff}")
    
    # TODO: 计算点积
    dot_product = None  # 在这里计算
    print(f"🎯 点积: {dot_product}")
    
    # TODO: 计算向量长度
    length_A = None  # 在这里计算
    length_B = None  # 在这里计算
    print(f"📏 用户A向量长度: {length_A:.3f}")
    print(f"📏 用户B向量长度: {length_B:.3f}")
    
    # TODO: 计算余弦相似度
    cosine_sim = None  # 在这里计算
    print(f"🎯 余弦相似度: {cosine_sim:.3f}")
    
    return users, categories

def challenge_3_recommendation_system():
    """挑战3：推荐系统"""
    print("\n🎯 挑战3：推荐系统")
    print("="*50)
    
    users, categories = challenge_2_vector_operations()
    
    # TODO: 实现找最相似用户的函数
    def find_most_similar_user(target_user, all_users):
        """
        找到与目标用户最相似的用户
        
        参数:
        target_user: 目标用户名
        all_users: 所有用户字典
        
        返回:
        (最相似用户名, 相似度分数)
        """
        # TODO: 在这里实现函数逻辑
        # 1. 遍历所有用户
        # 2. 计算与目标用户的余弦相似度
        # 3. 找到相似度最高的用户
        # 4. 返回(用户名, 相似度)
        
        return None, None  # 返回最相似的用户和相似度
    
    # 测试函数
    user_names = list(users.keys())
    print("🤝 为每个用户找最相似的朋友:")
    for user in user_names:
        most_similar, score = find_most_similar_user(user, users)
        print(f"{user} 最相似的朋友: {most_similar} (相似度: {score:.3f})")
    
    return users, categories

def challenge_4_similarity_matrix():
    """挑战4：相似度矩阵"""
    print("\n🎯 挑战4：相似度矩阵")
    print("="*50)
    
    users, categories = challenge_3_recommendation_system()
    
    user_names = list(users.keys())
    n_users = len(user_names)
    
    # TODO: 创建相似度矩阵
    similarity_matrix = np.zeros((n_users, n_users))
    
    # TODO: 填充相似度矩阵
    for i, name1 in enumerate(user_names):
        for j, name2 in enumerate(user_names):
            if i == j:
                similarity_matrix[i][j] = 1.0
            else:
                # TODO: 在这里计算两个用户的余弦相似度
                sim = None  # 在这里计算
                similarity_matrix[i][j] = sim
    
    # 显示矩阵
    print("📊 用户相似度矩阵:")
    print("      ", end="")
    for name in user_names:
        print(f"{name:>6}", end="")
    print()
    
    for i, name1 in enumerate(user_names):
        print(f"{name1:>6}", end="")
        for j in range(n_users):
            print(f"{similarity_matrix[i][j]:>6.3f}", end="")
        print()
    
    return users, categories, similarity_matrix

def challenge_5_visualization():
    """挑战5：数据可视化"""
    print("\n🎯 挑战5：数据可视化")
    print("="*50)
    
    users, categories, similarity_matrix = challenge_4_similarity_matrix()
    
    # TODO: 创建可视化图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 图1：用户偏好对比
    x = np.arange(len(categories))
    width = 0.15
    
    # TODO: 在这里绘制柱状图
    for i, (name, ratings) in enumerate(users.items()):
        # TODO: 在这里添加绘图代码
        pass
    
    ax1.set_xlabel('电影类别')
    ax1.set_ylabel('评分')
    ax1.set_title('用户电影偏好对比')
    ax1.set_xticks(x + width * 2)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2：相似度热力图
    # TODO: 在这里绘制热力图
    im = ax2.imshow(similarity_matrix, cmap='YlOrRd', vmin=0, vmax=1)
    ax2.set_xticks(range(len(users)))
    ax2.set_yticks(range(len(users)))
    ax2.set_xticklabels(list(users.keys()))
    ax2.set_yticklabels(list(users.keys()))
    ax2.set_title('用户相似度热力图')
    
    # TODO: 添加数值标签
    for i in range(len(users)):
        for j in range(len(users)):
            # TODO: 在这里添加数值标签
            pass
    
    plt.colorbar(im, ax=ax2)
    
    # 图3：向量长度对比
    # TODO: 计算并绘制向量长度
    lengths = []  # 在这里计算每个用户的向量长度
    user_names = list(users.keys())
    
    # TODO: 在这里绘制向量长度对比图
    bars = ax3.bar(user_names, lengths, color='skyblue', alpha=0.7)
    ax3.set_ylabel('L2范数')
    ax3.set_title('用户向量长度对比')
    
    # TODO: 添加数值标签
    for bar, length in zip(bars, lengths):
        # TODO: 在这里添加数值标签
        pass
    
    # 图4：推荐强度分析
    # TODO: 以第一个用户为基准，计算与其他用户的推荐强度
    base_user = list(users.keys())[0]
    base_vector = users[base_user]
    recommendation_strengths = []
    other_users = []
    
    # TODO: 在这里计算推荐强度
    for name, user_vector in users.items():
        if name != base_user:
            # TODO: 在这里计算相似度作为推荐强度
            strength = None  # 在这里计算
            recommendation_strengths.append(strength)
            other_users.append(name)
    
    # TODO: 在这里绘制推荐强度图
    bars = ax4.bar(other_users, recommendation_strengths, color='lightcoral', alpha=0.7)
    ax4.set_ylabel('推荐强度')
    ax4.set_title(f'为{base_user}的推荐强度')
    ax4.set_ylim(0, 1)
    
    # TODO: 添加数值标签
    for bar, strength in zip(bars, recommendation_strengths):
        # TODO: 在这里添加数值标签
        pass
    
    plt.tight_layout()
    plt.show()
    
    print("📈 图表说明:")
    print("- 左上：用户偏好对比")
    print("- 右上：用户相似度热力图")
    print("- 左下：向量长度对比")
    print("- 右下：推荐强度分析")

def challenge_6_business_application():
    """挑战6：商业应用"""
    print("\n🏢 挑战6：商业应用")
    print("="*50)
    
    # TODO: 创建电商用户购买行为数据
    # 要求：至少包含5种不同类型的用户
    # 购买类别：数码、服装、奢侈品、食品、汽车
    ecommerce_users = {
        # TODO: 在这里添加电商用户数据
        # 例如：
        # '学生用户': np.array([2, 8, 1, 5, 0]),    # 数码, 服装, 奢侈品, 食品, 汽车
        # '白领用户': np.array([5, 12, 3, 8, 0]),   # 消费能力更强
        # 继续添加更多用户...
    }
    
    categories = ['数码', '服装', '奢侈品', '食品', '汽车']
    
    print("🛍️ 电商用户购买行为分析:")
    for name, purchases in ecommerce_users.items():
        # TODO: 在这里计算总购买、L2范数、集中度
        total = None  # 在这里计算
        l2_norm = None  # 在这里计算
        concentration = None  # 在这里计算
        print(f"{name}: {purchases}")
        print(f"  总购买: {total}, L2范数: {l2_norm:.1f}, 集中度: {concentration:.2f}")
        print()
    
    # TODO: 分析用户相似度
    print("🔍 用户相似度分析:")
    user_names = list(ecommerce_users.keys())
    for i, name1 in enumerate(user_names):
        for j, name2 in enumerate(user_names):
            if i < j:
                # TODO: 在这里计算相似度
                sim = None  # 在这里计算
                print(f"{name1} vs {name2}: {sim:.3f}")
    
    print("\n💡 商业洞察:")
    print("- 高集中度用户: 推荐该类别的高端商品")
    print("- 相似用户: 可以互相推荐商品")
    print("- 不同用户: 推荐差异化的商品")

def challenge_7_advanced_function():
    """挑战7：高级函数"""
    print("\n🚀 挑战7：高级函数")
    print("="*50)
    
    # TODO: 实现一个完整的推荐函数
    def create_recommendation_system(users, categories):
        """
        创建推荐系统
        
        参数:
        users: 用户字典
        categories: 类别列表
        
        返回:
        推荐结果字典
        """
        recommendations = {}
        
        # TODO: 在这里实现推荐逻辑
        # 1. 为每个用户找到最相似的其他用户
        # 2. 分析相似用户喜欢但目标用户不太喜欢的类别
        # 3. 生成推荐结果
        
        return recommendations
    
    # 测试数据
    test_users = {
        '用户A': np.array([5, 3, 4, 2, 1]),
        '用户B': np.array([4, 5, 3, 1, 4]),
        '用户C': np.array([2, 4, 5, 3, 2])
    }
    
    test_categories = ['类别1', '类别2', '类别3', '类别4', '类别5']
    
    # TODO: 调用推荐函数
    recommendations = create_recommendation_system(test_users, test_categories)
    
    print("推荐结果:")
    for user, recommendation in recommendations.items():
        print(f"{user}: {recommendation}")

def main():
    """主函数：运行所有挑战"""
    print("🎯 第3天：向量基础 - 今日编码练习（需要完成）")
    print("="*60)
    print("包含7个挑战，需要你自己完成所有TODO部分")
    print("="*60)
    
    # 运行所有挑战
    challenge_1_create_users()
    challenge_2_vector_operations()
    challenge_3_recommendation_system()
    challenge_4_similarity_matrix()
    challenge_5_visualization()
    challenge_6_business_application()
    challenge_7_advanced_function()
    
    print("\n🎉 恭喜完成所有挑战！")
    print("="*60)
    print("今日收获:")
    print("✅ 掌握了向量基础操作")
    print("✅ 实现了推荐系统")
    print("✅ 理解了商业应用")
    print("✅ 学会了数据可视化")
    print("✅ 体验了Python的优雅")
    print("\n🚀 继续保持这个学习节奏！")

if __name__ == "__main__":
    main() 