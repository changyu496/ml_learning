"""
练习5：PCA实际应用案例
目标：在真实场景中应用PCA，理解其实际价值
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print("📚 练习5：PCA实际应用案例")
print("="*50)

print("\n🎯 任务目标：")
print("1. 模拟真实的客户数据分析场景")
print("2. 应用数据标准化和PCA降维")
print("3. 分析主成分的含义和贡献")
print("4. 给出实际的降维建议")

def pca_customer_analysis():
    """PCA在客户数据分析中的应用"""
    
    print("\n📊 场景设定：电商客户数据分析")
    print("你是一名数据分析师，需要分析客户行为数据")
    print("原始数据包含6个特征，需要降维以便后续分析")
    
    # TODO: 任务1 - 生成模拟客户数据
    print("\n📝 任务1：生成客户特征数据")
    print("提示：使用np.random生成6个特征的数据")
    
    np.random.seed(42)
    n_customers = 1000
    
    # 你的代码：
    # 生成6个特征的数据
    # age = np.random.normal(35, 10, n_customers)  # 年龄
    # income = np.random.normal(50000, 15000, n_customers)  # 收入
    # purchase_freq = np.random.poisson(12, n_customers)  # 购买频率
    # avg_order = np.random.normal(150, 50, n_customers)  # 平均订单金额
    # time_on_site = np.random.exponential(20, n_customers)  # 网站停留时间
    # rating = np.random.normal(4.2, 0.8, n_customers)  # 评分
    
    # 添加特征间的相关性
    # purchase_freq = purchase_freq + 0.3 * (income / 10000) + np.random.normal(0, 2, n_customers)
    # avg_order = avg_order + 0.5 * (income / 1000) + np.random.normal(0, 20, n_customers)
    
    # 组合所有特征
    # customer_features = np.column_stack([age, income, purchase_freq, avg_order, time_on_site, rating])
    feature_names = ['年龄', '收入', '购买频率', '平均订单金额', '网站停留时间', '评分']
    
    # print(f"客户数据形状: {customer_features.shape}")
    # print(f"特征名称: {feature_names}")
    
    # TODO: 任务2 - 数据标准化
    print("\n📝 任务2：数据标准化")
    print("提示：不同特征的量纲差异很大，需要标准化")
    print("使用 StandardScaler 进行标准化")
    
    # 你的代码：
    # scaler = StandardScaler()
    # customer_features_scaled = scaler.fit_transform(customer_features)
    
    # TODO: 任务3 - 应用PCA
    print("\n📝 任务3：应用PCA分析")
    print("提示：先不限制主成分数量，分析所有主成分")
    
    # 你的代码：
    # pca = PCA()
    # customer_pca = pca.fit_transform(customer_features_scaled)
    
    # TODO: 任务4 - 分析每个主成分的方差解释比例
    print("\n📝 任务4：分析主成分贡献")
    
    # 你的代码：
    # print("各主成分的方差解释比例:")
    # for i, ratio in enumerate(pca.explained_variance_ratio_):
    #     print(f"PC{i+1}: {ratio:.2%}")
    
    # 计算累积方差解释比例
    # cumulative_ratio = np.cumsum(pca.explained_variance_ratio_)
    # print("\n累积方差解释比例:")
    # for i, ratio in enumerate(cumulative_ratio):
    #     print(f"前{i+1}个主成分: {ratio:.2%}")
    
    # TODO: 任务5 - 创建综合可视化
    print("\n📝 任务5：创建四合一可视化图表")
    print("左上：各主成分方差解释比例")
    print("右上：累积方差解释比例曲线") 
    print("左下：前两个主成分的客户分布")
    print("右下：主成分载荷图（特征贡献热力图）")
    
    # 你的代码：
    # fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 左上：各主成分方差解释比例
    # axes[0, 0].bar(range(1, len(pca.explained_variance_ratio_) + 1), 
    #                pca.explained_variance_ratio_)
    # axes[0, 0].set_title('各主成分方差解释比例')
    # axes[0, 0].set_xlabel('主成分')
    # axes[0, 0].set_ylabel('方差解释比例')
    
    # 右上：累积方差解释比例
    # axes[0, 1].plot(range(1, len(cumulative_ratio) + 1), cumulative_ratio, 'o-')
    # axes[0, 1].axhline(y=0.8, color='r', linestyle='--', label='80%阈值')
    # axes[0, 1].axhline(y=0.9, color='g', linestyle='--', label='90%阈值') 
    # axes[0, 1].set_title('累积方差解释比例')
    # axes[0, 1].legend()
    
    # 左下：前两个主成分的数据分布
    # axes[1, 0].scatter(customer_pca[:, 0], customer_pca[:, 1], alpha=0.6)
    # axes[1, 0].set_title('客户在主成分空间的分布')
    # axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    # axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    
    # 右下：主成分载荷图
    # components_matrix = pca.components_[:2, :]  # 前两个主成分
    # im = axes[1, 1].imshow(components_matrix, cmap='RdBu', aspect='auto')
    # axes[1, 1].set_title('主成分载荷图')
    # axes[1, 1].set_xticks(range(len(feature_names)))
    # axes[1, 1].set_xticklabels(feature_names, rotation=45)
    # axes[1, 1].set_yticks([0, 1])
    # axes[1, 1].set_yticklabels(['PC1', 'PC2'])
    
    # plt.tight_layout()
    # plt.show()
    
    # TODO: 任务6 - 给出降维建议
    print("\n📝 任务6：制定降维策略")
    
    # 你的代码：
    # 计算达到不同阈值所需的主成分数量
    # n_components_80 = np.argmax(cumulative_ratio >= 0.8) + 1
    # n_components_90 = np.argmax(cumulative_ratio >= 0.9) + 1
    
    # print(f"\n📊 降维建议:")
    # print(f"保留80%方差: 使用 {n_components_80} 个主成分")
    # print(f"保留90%方差: 使用 {n_components_90} 个主成分")
    # print(f"降维效果: 从 {len(feature_names)} 维降到 {n_components_80} 维")
    # print(f"数据压缩比: {n_components_80/len(feature_names):.1%}")
    
    print("\n❓ 业务分析问题：")
    print("1. 第一主成分主要由哪些客户特征组成？这代表什么业务含义？")
    print("2. 第二主成分的特征组合有什么特点？")
    print("3. 如果要保留90%的信息，需要几个维度？这对后续分析有什么影响？")
    print("4. 从载荷图中，你能发现哪些特征之间的关系？")
    print("5. 在实际业务中，你会建议使用几个主成分？为什么？")

def advanced_pca_interpretation():
    """高级任务：主成分的业务解释"""
    
    print("\n🔥 高级挑战：主成分的业务解释")
    print("-" * 40)
    
    print("假设你已经完成了上述PCA分析，现在需要向业务团队解释结果")
    print("\n📝 请思考并回答：")
    print("1. 如何用商业语言解释'主成分'这个概念？")
    print("2. PC1如果主要由收入和订单金额组成，应该命名为什么？")
    print("3. PC2如果主要由年龄和评分组成，代表什么客户特征？")
    print("4. 降维后的数据可以用于哪些具体的业务分析？")
    print("5. 向非技术人员展示时，你会重点强调PCA的哪些优势？")

# TODO: 开始练习
print("\n🔄 开始客户数据分析练习：")
# pca_customer_analysis()

print("\n🔄 尝试高级挑战：")
# advanced_pca_interpretation()

print("\n✅ 练习5完成！")
print("💡 核心理解：PCA不仅是数学工具，更是理解数据内在结构的方法")
print("🎯 实际价值：帮助我们从复杂数据中提取关键信息，支持业务决策") 