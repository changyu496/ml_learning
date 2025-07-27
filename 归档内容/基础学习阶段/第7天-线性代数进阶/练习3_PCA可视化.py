"""
练习3：PCA降维效果可视化
目标：通过可视化理解PCA降维的效果和意义
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification

# 智能字体设置
import matplotlib.font_manager as fm

def setup_chinese_font():
    chinese_fonts = ['PingFang SC', 'Helvetica', 'STHeiti', 'Arial Unicode MS', 'DejaVu Sans']
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            return font
    return 'DejaVu Sans'

current_font = setup_chinese_font()
use_chinese_labels = current_font not in ['DejaVu Sans']

print("📚 练习3：PCA降维效果可视化")
print("="*50)

print("\n🎯 任务目标：")
print("1. 生成具有明显结构的4维数据")
print("2. 使用PCA降维到2维")
print("3. 可视化对比原始数据和降维后数据")
print("4. 分析方差解释比例")

def visualize_pca_effect():
    """可视化PCA降维效果"""
    
    # TODO: 任务1 - 生成测试数据
    print("\n📝 任务1：生成具有明显结构的数据")
    print("提示：使用 make_classification 生成分类数据")
    print("参数：n_samples=300, n_features=4, n_redundant=0, n_informative=2")
    
    # 你的代码：
    # X, y = make_classification(?)
    
    # TODO: 任务2 - 应用PCA降维
    print("\n📝 任务2：应用PCA降维")
    print("提示：pca = PCA(n_components=2)")
    print("     X_pca = pca.fit_transform(X)")
    
    # 你的代码：
    # pca = ?
    # X_pca = ?
    
    # TODO: 任务3 - 创建可视化图形
    print("\n📝 任务3：创建2x2子图布局")
    print("提示：fig, axes = plt.subplots(2, 2, figsize=(12, 10))")
    
    # 你的代码：
    # fig, axes = ?
    
    # TODO: 任务4 - 绘制原始数据的前两个特征
    print("\n📝 任务4：绘制原始数据对比")
    print("左上图：原始数据的特征1 vs 特征2")
    print("右上图：原始数据的特征3 vs 特征4")
    
    # 你的代码：
    # 左上图
    # axes[0, 0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
    # axes[0, 0].set_title(?)
    # axes[0, 0].set_xlabel(?)
    # axes[0, 0].set_ylabel(?)
    
    # 右上图
    # axes[0, 1].scatter(?)
    # 设置标题和标签
    
    # TODO: 任务5 - 绘制PCA降维后的数据
    print("\n📝 任务5：绘制PCA降维结果")
    print("左下图：PCA降维后的数据分布")
    
    # 你的代码：
    # axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    # 添加标题，包含方差解释比例信息
    
    # TODO: 任务6 - 绘制方差解释比例条形图
    print("\n📝 任务6：绘制方差解释比例")
    print("右下图：各主成分的方差解释比例")
    
    # 你的代码：
    # axes[1, 1].bar(['PC1', 'PC2'], pca.explained_variance_ratio_)
    # 设置标题和标签
    
    # TODO: 任务7 - 显示图形并分析结果
    print("\n📝 任务7：显示图形并分析")
    
    # 你的代码：
    # plt.tight_layout()
    # plt.show()
    
    # 分析结果
    # print(f"原始数据维度: {X.shape[1]}")
    # print(f"PCA后维度: {X_pca.shape[1]}")  
    # print(f"总方差解释比例: {pca.explained_variance_ratio_.sum():.2%}")
    
    print("\n❓ 思考问题：")
    print("1. 观察四个子图，哪个图中的数据分离效果最好？")
    print("2. PCA降维后是否保持了原数据的主要结构？")
    print("3. 两个主成分分别解释了多少比例的方差？")
    print("4. 如果总方差解释比例较低，说明什么问题？")

# TODO: 调用函数开始练习
print("\n🔄 开始练习：")
# visualize_pca_effect()

print("\n✅ 练习3完成！")
print("💡 核心理解：PCA能在保持主要信息的同时实现有效降维") 