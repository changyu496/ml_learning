"""
练习4：不同主成分数量的效果分析
目标：学会如何选择合适的主成分数量
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification

print("📚 练习4：不同主成分数量的效果分析")
print("="*50)

print("\n🎯 任务目标：")
print("1. 生成高维数据并分析不同主成分数量的效果")
print("2. 绘制方差解释比例曲线")
print("3. 确定达到指定方差解释比例所需的主成分数量")
print("4. 理解降维的效果和代价")

def analyze_pca_components():
    """分析不同主成分数量的降维效果"""
    
    # TODO: 任务1 - 生成高维测试数据
    print("\n📝 任务1：生成高维数据")
    print("提示：使用 make_classification 生成10维数据")
    print("参数：n_samples=200, n_features=10, n_redundant=5, n_informative=5")
    
    # 你的代码：
    # X, y = make_classification(?)
    
    # TODO: 任务2 - 定义要测试的主成分数量
    print("\n📝 任务2：定义测试的主成分数量")
    n_components_list = [1, 2, 3, 5, 7, 10]
    
    print(f"要测试的主成分数量: {n_components_list}")
    
    # TODO: 任务3 - 计算不同主成分数量的方差解释比例
    print("\n📝 任务3：计算各主成分数量的方差解释比例")
    print("提示：对每个主成分数量，创建PCA对象并fit数据")
    
    explained_ratios = []
    
    print("主成分数量 | 方差解释比例 | 累积方差解释比例")
    print("-" * 50)
    
    # 你的代码：
    # for n_comp in n_components_list:
    #     pca = PCA(n_components=n_comp)
    #     pca.fit(X)
    #     total_explained = pca.explained_variance_ratio_.sum()
    #     explained_ratios.append(total_explained)
    #     print(f"    {n_comp:2d}       |     {total_explained:.2%}      |        {total_explained:.2%}")
    
    # TODO: 任务4 - 可视化方差解释比例
    print("\n📝 任务4：绘制方差解释比例曲线")
    print("提示：创建线图，添加90%和95%的阈值线")
    
    # 你的代码：
    # plt.figure(figsize=(10, 6))
    # plt.plot(n_components_list, explained_ratios, 'o-', linewidth=2, markersize=8)
    # plt.axhline(y=0.9, color='r', linestyle='--', label='90%阈值')
    # plt.axhline(y=0.95, color='g', linestyle='--', label='95%阈值')
    # plt.xlabel('主成分数量')
    # plt.ylabel('累积方差解释比例') 
    # plt.title('不同主成分数量的方差解释效果')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.show()
    
    # TODO: 任务5 - 分析结果
    print("\n📝 任务5：分析达到特定阈值所需的主成分数量")
    
    # 你的代码：
    # 找到达到90%方差解释所需的主成分数量
    # for i, ratio in enumerate(explained_ratios):
    #     if ratio >= 0.9:
    #         print(f"🎯 达到90%方差解释需要 {n_components_list[i]} 个主成分")
    #         break
    
    # 找到达到95%方差解释所需的主成分数量
    # for i, ratio in enumerate(explained_ratios):
    #     if ratio >= 0.95:
    #         print(f"🎯 达到95%方差解释需要 {n_components_list[i]} 个主成分")
    #         break
    
    print("\n❓ 思考问题：")
    print("1. 观察曲线形状，在哪个点之后方差解释比例增长变慢？")
    print("2. 如果你要在信息保留和计算效率之间平衡，会选择多少个主成分？")
    print("3. 为什么前几个主成分的方差解释比例通常比较大？")
    print("4. 在实际项目中，如何确定合适的主成分数量？")
    
    # return explained_ratios

# TODO: 额外挑战 - 详细的主成分分析
def detailed_pca_analysis():
    """详细分析每个主成分的贡献"""
    
    print("\n🔥 额外挑战：详细主成分分析")
    print("-" * 40)
    
    # TODO: 创建数据并应用完整PCA
    # X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    # pca = PCA()  # 不限制主成分数量
    # pca.fit(X)
    
    # TODO: 创建两个子图
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # TODO: 左图：各主成分的方差解释比例
    # ax1.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
    #         pca.explained_variance_ratio_)
    # ax1.set_title('各主成分方差解释比例')
    # ax1.set_xlabel('主成分')
    # ax1.set_ylabel('方差解释比例')
    
    # TODO: 右图：累积方差解释比例
    # cumsum_ratio = np.cumsum(pca.explained_variance_ratio_)
    # ax2.plot(range(1, len(cumsum_ratio) + 1), cumsum_ratio, 'o-')
    # ax2.axhline(y=0.8, color='orange', linestyle='--', label='80%')
    # ax2.axhline(y=0.9, color='red', linestyle='--', label='90%')
    # ax2.axhline(y=0.95, color='green', linestyle='--', label='95%')
    # ax2.set_title('累积方差解释比例')
    # ax2.set_xlabel('主成分数量')
    # ax2.set_ylabel('累积方差解释比例')
    # ax2.legend()
    
    # plt.tight_layout()
    # plt.show()
    
    pass

# TODO: 开始练习
print("\n🔄 开始练习：")
# analyze_pca_components()

print("\n🔄 尝试额外挑战：")
# detailed_pca_analysis()

print("\n✅ 练习4完成！")
print("💡 核心理解：选择合适的主成分数量需要平衡信息保留和计算效率") 