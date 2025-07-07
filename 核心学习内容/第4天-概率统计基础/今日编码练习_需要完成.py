#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第3天和第4天综合编程练习
包含向量基础 + 概率统计应用

作者：大模型转型学习
日期：第3-4天
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("🎯 第3-4天综合编程练习")
print("="*50)
print("包含：向量基础 + 概率统计应用")
print("="*50)

# ============================================================================
# 第3天：向量基础练习
# ============================================================================

print("\n📚 第3天：向量基础练习")
print("-" * 30)

# 练习1：基础向量操作
print("\n🔢 练习1：基础向量操作")
print("任务：完成以下向量运算")

def exercise_1_vector_operations():
    """
    练习1：基础向量操作
    完成TODO标记的任务
    """
    print("开始练习1：基础向量操作")
    
    # 创建两个向量
    vector_a = np.array([1, 2, 3, 4, 5])
    vector_b = np.array([2, 4, 6, 8, 10])
    
    print(f"向量A: {vector_a}")
    print(f"向量B: {vector_b}")
    
    # TODO 1.1: 计算向量A的L2范数
    # 提示：使用 np.linalg.norm() 或 np.sqrt(np.sum(vector_a**2))
    l2_norm_a = np.linalg.norm(vector_a)  # 请完成这个计算
    print(f"向量A的L2范数: {l2_norm_a}")
    
    # TODO 1.2: 计算向量A和向量B的点积
    # 提示：使用 np.dot() 或 np.sum(vector_a * vector_b)
    dot_product = np.dot(vector_a, vector_b)  # 请完成这个计算
    print(f"向量A和B的点积: {dot_product}")
    
    # TODO 1.3: 计算向量A和向量B的余弦相似度
    # 提示：cos_sim = dot_product / (norm_a * norm_b)
    cosine_similarity = dot_product / (l2_norm_a * np.linalg.norm(vector_b))  # 请完成这个计算
    print(f"余弦相似度: {cosine_similarity}")
    
    # TODO 1.4: 计算向量A和向量B的欧几里得距离
    # 提示：使用 np.linalg.norm(vector_a - vector_b)
    euclidean_distance = np.linalg.norm(vector_a - vector_b)  # 请完成这个计算
    print(f"欧几里得距离: {euclidean_distance}")
    
    return {
        'l2_norm_a': l2_norm_a,
        'dot_product': dot_product,
        'cosine_similarity': cosine_similarity,
        'euclidean_distance': euclidean_distance
    }

# 练习2：推荐系统实现
print("\n🎯 练习2：推荐系统实现")
print("任务：实现基于余弦相似度的推荐系统")

def exercise_2_recommendation_system():
    """
    练习2：推荐系统实现
    完成TODO标记的任务
    """
    print("开始练习2：推荐系统实现")
    
    # 用户-物品评分矩阵 (用户数=5, 物品数=4)
    # 0表示未评分
    ratings_matrix = np.array([
        [5, 3, 0, 1],  # 用户1
        [4, 0, 0, 1],  # 用户2
        [1, 1, 0, 5],  # 用户3
        [1, 0, 0, 4],  # 用户4
        [0, 1, 5, 4]   # 用户5
    ])
    
    print("用户-物品评分矩阵:")
    print(ratings_matrix)
    
    # TODO 2.1: 计算用户1和用户2的余弦相似度
    # 提示：只考虑两个用户都评过分的物品
    user1_ratings = ratings_matrix[0]  # 用户1的评分
    user2_ratings = ratings_matrix[1]  # 用户2的评分
    
    # 找到两个用户都评过分的物品索引
    common_items = None  # 请完成这个计算
    print(f"共同评分的物品索引: {common_items}")
    
    # 提取共同评分的向量
    user1_common = None  # 请完成这个计算
    user2_common = None  # 请完成这个计算
    
    # 计算余弦相似度
    similarity_1_2 = None  # 请完成这个计算
    print(f"用户1和用户2的余弦相似度: {similarity_1_2}")
    
    # TODO 2.2: 为用户1推荐物品
    # 找到用户1未评分的物品
    user1_unrated = None  # 请完成这个计算
    print(f"用户1未评分的物品: {user1_unrated}")
    
    # 计算用户1与其他用户的相似度
    similarities = []
    for i in range(1, len(ratings_matrix)):  # 跳过用户1自己
        # 计算用户1与用户i的相似度
        similarity = None  # 请完成这个计算
        similarities.append(similarity)
    
    print(f"用户1与其他用户的相似度: {similarities}")
    
    # 基于相似度预测评分
    # 对于用户1未评分的每个物品，计算预测评分
    predictions = {}
    for item_idx in user1_unrated:
        # 计算预测评分
        # 公式：pred = sum(similarity * rating) / sum(similarity)
        predicted_rating = None  # 请完成这个计算
        predictions[item_idx] = predicted_rating
    
    print(f"用户1的预测评分: {predictions}")
    
    return {
        'similarity_1_2': similarity_1_2,
        'user1_unrated': user1_unrated,
        'similarities': similarities,
        'predictions': predictions
    }

# 练习3：向量可视化
print("\n📊 练习3：向量可视化")
print("任务：创建向量可视化图表")

def exercise_3_vector_visualization():
    """
    练习3：向量可视化
    完成TODO标记的任务
    """
    print("开始练习3：向量可视化")
    
    # 创建多个向量
    vectors = np.array([
        [1, 2],   # 向量1
        [3, 1],   # 向量2
        [2, 3],   # 向量3
        [-1, 2],  # 向量4
        [0, 3]    # 向量5
    ])
    
    print("向量数据:")
    print(vectors)
    
    # TODO 3.1: 创建向量散点图
    # 在2D平面上绘制这些向量
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：向量散点图
    # 请完成散点图的绘制
    # 提示：使用 ax1.scatter() 绘制点，使用 ax1.arrow() 绘制箭头
    
    ax1.set_xlim(-2, 4)
    ax1.set_ylim(-1, 4)
    ax1.set_xlabel('X坐标')
    ax1.set_ylabel('Y坐标')
    ax1.set_title('向量散点图')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # TODO 3.2: 创建向量相似度热力图
    # 计算所有向量两两之间的余弦相似度
    n_vectors = len(vectors)
    similarity_matrix = np.zeros((n_vectors, n_vectors))
    
    # 请完成相似度矩阵的计算
    # 提示：使用双重循环计算每对向量的余弦相似度
    
    # 绘制热力图
    im = ax2.imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax2.set_xticks(range(n_vectors))
    ax2.set_yticks(range(n_vectors))
    ax2.set_xticklabels([f'向量{i+1}' for i in range(n_vectors)])
    ax2.set_yticklabels([f'向量{i+1}' for i in range(n_vectors)])
    ax2.set_title('向量相似度热力图')
    
    # 添加数值标签
    for i in range(n_vectors):
        for j in range(n_vectors):
            text = ax2.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black")
    
    plt.colorbar(im, ax=ax2)
    plt.tight_layout()
    plt.show()
    
    return {
        'vectors': vectors,
        'similarity_matrix': similarity_matrix
    }

# ============================================================================
# 第4天：概率统计练习
# ============================================================================

print("\n📚 第4天：概率统计练习")
print("-" * 30)

# 练习4：数据探索性分析
print("\n📊 练习4：数据探索性分析")
print("任务：分析模拟电商数据")

def exercise_4_exploratory_analysis():
    """
    练习4：数据探索性分析
    完成TODO标记的任务
    """
    print("开始练习4：数据探索性分析")
    
    # 生成模拟电商数据
    np.random.seed(42)
    n_users = 1000
    
    # 用户年龄（正态分布）
    ages = np.random.normal(35, 10, n_users)
    ages = np.clip(ages, 18, 70)
    
    # 消费金额（对数正态分布）
    spending = np.random.lognormal(4, 0.8, n_users)
    
    # 购买频次（泊松分布）
    purchase_frequency = np.random.poisson(5, n_users)
    
    # 用户满意度（1-5分）
    satisfaction = np.random.choice([1,2,3,4,5], n_users, p=[0.05, 0.1, 0.2, 0.4, 0.25])
    
    # 创建DataFrame
    df = pd.DataFrame({
        'age': ages,
        'spending': spending,
        'purchase_frequency': purchase_frequency,
        'satisfaction': satisfaction
    })
    
    print("数据概览:")
    print(df.head())
    print(f"\n数据形状: {df.shape}")
    
    # TODO 4.1: 计算基本统计量
    # 计算每个变量的均值、中位数、标准差
    stats_summary = {}
    
    # 请完成统计量的计算
    # 提示：使用 df.describe() 或分别计算每个变量
    
    print("\n基本统计量:")
    print(stats_summary)
    
    # TODO 4.2: 检测异常值
    # 使用3倍标准差方法检测异常值
    outliers = {}
    
    # 请完成异常值检测
    # 提示：对于每个数值变量，找出超出均值±3倍标准差范围的值
    
    print("\n异常值检测:")
    for var, outlier_count in outliers.items():
        print(f"{var}: {outlier_count} 个异常值")
    
    # TODO 4.3: 计算变量间相关性
    # 计算数值变量间的相关系数
    correlation_matrix = None  # 请完成相关性计算
    
    print("\n相关性矩阵:")
    print(correlation_matrix)
    
    # 可视化分析
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 年龄分布
    ax1.hist(df['age'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('年龄')
    ax1.set_ylabel('频次')
    ax1.set_title('用户年龄分布')
    ax1.grid(True, alpha=0.3)
    
    # 消费金额分布
    ax2.hist(df['spending'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_xlabel('消费金额')
    ax2.set_ylabel('频次')
    ax2.set_title('用户消费金额分布')
    ax2.grid(True, alpha=0.3)
    
    # 购买频次分布
    ax3.hist(df['purchase_frequency'], bins=range(0, 15), alpha=0.7, color='orange', edgecolor='black')
    ax3.set_xlabel('购买频次')
    ax3.set_ylabel('频次')
    ax3.set_title('用户购买频次分布')
    ax3.grid(True, alpha=0.3)
    
    # 满意度分布
    satisfaction_counts = df['satisfaction'].value_counts().sort_index()
    ax4.bar(satisfaction_counts.index, satisfaction_counts.values, alpha=0.7, color='purple')
    ax4.set_xlabel('满意度评分')
    ax4.set_ylabel('用户数')
    ax4.set_title('用户满意度分布')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'df': df,
        'stats_summary': stats_summary,
        'outliers': outliers,
        'correlation_matrix': correlation_matrix
    }

# 练习5：假设检验
print("\n🔬 练习5：假设检验")
print("任务：进行统计假设检验")

def exercise_5_hypothesis_testing():
    """
    练习5：假设检验
    完成TODO标记的任务
    """
    print("开始练习5：假设检验")
    
    # 使用上一练习的数据
    df = exercise_4_exploratory_analysis()['df']
    
    # 创建用户群体
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 100], labels=['青年', '中年', '老年'])
    df['spending_group'] = pd.cut(df['spending'], bins=[0, 50, 100, 1000], labels=['低消费', '中消费', '高消费'])
    
    print("用户群体分布:")
    print(df['age_group'].value_counts())
    print(f"\n消费群体分布:")
    print(df['spending_group'].value_counts())
    
    # TODO 5.1: 不同年龄群体的消费金额差异检验
    # 使用单因素方差分析(ANOVA)
    age_groups = df['age_group'].unique()
    group_data = [df[df['age_group'] == group]['spending'].values for group in age_groups]
    
    # 请完成ANOVA检验
    # 提示：使用 stats.f_oneway(*group_data)
    f_stat = None  # 请完成计算
    p_value = None  # 请完成计算
    
    print(f"\n年龄群体消费差异检验:")
    print(f"F统计量: {f_stat:.4f}")
    print(f"p值: {p_value:.4f}")
    print(f"结论: {'存在显著差异' if p_value < 0.05 else '无显著差异'}")
    
    # TODO 5.2: 消费金额与满意度的关系检验
    # 使用t检验比较不同满意度群体的消费金额
    low_satisfaction = df[df['satisfaction'] <= 3]['spending']
    high_satisfaction = df[df['satisfaction'] >= 4]['spending']
    
    # 请完成t检验
    # 提示：使用 stats.ttest_ind(low_satisfaction, high_satisfaction)
    t_stat = None  # 请完成计算
    t_pvalue = None  # 请完成计算
    
    print(f"\n满意度消费差异检验:")
    print(f"t统计量: {t_stat:.4f}")
    print(f"p值: {t_pvalue:.4f}")
    print(f"结论: {'存在显著差异' if t_pvalue < 0.05 else '无显著差异'}")
    
    # TODO 5.3: 年龄群体与消费群体的关联性检验
    # 使用卡方检验
    contingency_table = pd.crosstab(df['age_group'], df['spending_group'])
    
    # 请完成卡方检验
    # 提示：使用 stats.chi2_contingency(contingency_table)
    chi2_stat = None  # 请完成计算
    chi2_pvalue = None  # 请完成计算
    dof = None  # 请完成计算
    
    print(f"\n年龄消费关联检验:")
    print(f"卡方统计量: {chi2_stat:.4f}")
    print(f"p值: {chi2_pvalue:.4f}")
    print(f"自由度: {dof}")
    print(f"结论: {'存在显著关联' if chi2_pvalue < 0.05 else '无显著关联'}")
    
    return {
        'f_stat': f_stat,
        'p_value': p_value,
        't_stat': t_stat,
        't_pvalue': t_pvalue,
        'chi2_stat': chi2_stat,
        'chi2_pvalue': chi2_pvalue,
        'dof': dof
    }

# 练习6：A/B测试模拟
print("\n🏢 练习6：A/B测试模拟")
print("任务：模拟推荐系统A/B测试")

def exercise_6_ab_testing():
    """
    练习6：A/B测试模拟
    完成TODO标记的任务
    """
    print("开始练习6：A/B测试模拟")
    
    # 模拟A/B测试数据
    np.random.seed(42)
    n_users_per_group = 5000
    
    # 对照组：传统推荐算法
    control_conversion = np.random.binomial(1, 0.12, n_users_per_group)
    control_revenue = np.random.exponential(50, n_users_per_group) * control_conversion
    
    # 实验组：新推荐算法
    treatment_conversion = np.random.binomial(1, 0.15, n_users_per_group)
    treatment_revenue = np.random.exponential(55, n_users_per_group) * treatment_conversion
    
    print("A/B测试数据概览:")
    print(f"对照组用户数: {n_users_per_group}")
    print(f"实验组用户数: {n_users_per_group}")
    
    # TODO 6.1: 计算关键指标
    # 计算转化率和平均收入的提升
    control_conv_rate = None  # 请完成计算
    treatment_conv_rate = None  # 请完成计算
    conv_lift = None  # 请完成计算
    
    control_avg_revenue = None  # 请完成计算
    treatment_avg_revenue = None  # 请完成计算
    revenue_lift = None  # 请完成计算
    
    print(f"\n关键指标对比:")
    print(f"转化率: 对照组 {control_conv_rate:.3f} vs 实验组 {treatment_conv_rate:.3f} (提升 {conv_lift:.1f}%)")
    print(f"平均收入: 对照组 {control_avg_revenue:.2f} vs 实验组 {treatment_avg_revenue:.2f} (提升 {revenue_lift:.1f}%)")
    
    # TODO 6.2: 统计显著性检验
    # 转化率差异检验（比例检验）
    from scipy.stats import proportions_ztest
    conv_counts = [np.sum(treatment_conversion), np.sum(control_conversion)]
    conv_nobs = [n_users_per_group, n_users_per_group]
    
    # 请完成比例检验
    # 提示：使用 proportions_ztest(conv_counts, conv_nobs)
    conv_z_stat = None  # 请完成计算
    conv_p_value = None  # 请完成计算
    
    # 收入差异检验（t检验）
    # 请完成t检验
    # 提示：使用 stats.ttest_ind(treatment_revenue, control_revenue)
    revenue_t_stat = None  # 请完成计算
    revenue_p_value = None  # 请完成计算
    
    print(f"\n统计显著性检验:")
    print(f"转化率差异检验: z统计量={conv_z_stat:.4f}, p值={conv_p_value:.4f}")
    print(f"转化率显著性: {'显著' if conv_p_value < 0.05 else '不显著'}")
    print(f"收入差异检验: t统计量={revenue_t_stat:.4f}, p值={revenue_p_value:.4f}")
    print(f"收入显著性: {'显著' if revenue_p_value < 0.05 else '不显著'}")
    
    # 业务决策建议
    print(f"\n💼 业务决策建议:")
    if conv_p_value < 0.05 and revenue_p_value < 0.05:
        print(f"✅ 建议: 采用新推荐算法")
        print(f"理由: 转化率和收入都有显著提升")
    elif conv_p_value < 0.05:
        print(f"⚠️ 建议: 谨慎采用新算法")
        print(f"理由: 转化率有提升，但收入提升不显著")
    else:
        print(f"❌ 建议: 不采用新算法")
        print(f"理由: 关键指标提升不显著")
    
    return {
        'control_conv_rate': control_conv_rate,
        'treatment_conv_rate': treatment_conv_rate,
        'conv_lift': conv_lift,
        'control_avg_revenue': control_avg_revenue,
        'treatment_avg_revenue': treatment_avg_revenue,
        'revenue_lift': revenue_lift,
        'conv_z_stat': conv_z_stat,
        'conv_p_value': conv_p_value,
        'revenue_t_stat': revenue_t_stat,
        'revenue_p_value': revenue_p_value
    }

# ============================================================================
# 主函数：运行所有练习
# ============================================================================

def main():
    """
    主函数：运行所有练习
    """
    print("🚀 开始运行所有练习...")
    
    # 第3天练习
    print("\n" + "="*50)
    print("第3天：向量基础练习")
    print("="*50)
    
    # 练习1：基础向量操作
    print("\n🔢 运行练习1：基础向量操作")
    result_1 = exercise_1_vector_operations()
    
    # 练习2：推荐系统实现
    print("\n🎯 运行练习2：推荐系统实现")
    result_2 = exercise_2_recommendation_system()
    
    # 练习3：向量可视化
    print("\n📊 运行练习3：向量可视化")
    result_3 = exercise_3_vector_visualization()
    
    # 第4天练习
    print("\n" + "="*50)
    print("第4天：概率统计练习")
    print("="*50)
    
    # 练习4：数据探索性分析
    print("\n📊 运行练习4：数据探索性分析")
    result_4 = exercise_4_exploratory_analysis()
    
    # 练习5：假设检验
    print("\n🔬 运行练习5：假设检验")
    result_5 = exercise_5_hypothesis_testing()
    
    # 练习6：A/B测试模拟
    print("\n🏢 运行练习6：A/B测试模拟")
    result_6 = exercise_6_ab_testing()
    
    print("\n" + "="*50)
    print("🎉 所有练习运行完成！")
    print("="*50)
    
    # 总结
    print("\n📋 练习总结:")
    print("✅ 第3天：向量基础 - 向量运算、推荐系统、可视化")
    print("✅ 第4天：概率统计 - 数据分析、假设检验、A/B测试")
    print("\n💡 关键学习点:")
    print("1. 向量运算是机器学习的基础")
    print("2. 概率统计是数据分析的核心")
    print("3. 实际应用需要结合业务理解")
    print("4. 可视化是理解数据的重要工具")
    
    return {
        'day3_results': [result_1, result_2, result_3],
        'day4_results': [result_4, result_5, result_6]
    }

if __name__ == "__main__":
    # 运行所有练习
    results = main()
    
    print("\n🎯 练习完成！")
    print("请检查TODO标记的任务是否已完成")
    print("如有疑问，请参考提示和文档") 