"""
第7天练习 - 线性代数进阶
目标：通过编程练习巩固特征值、特征向量和PCA的理解
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

# 智能中文字体设置
import matplotlib.font_manager as fm

def setup_chinese_font():
    """设置中文字体，自动检测系统可用字体"""
    chinese_fonts = [
        'PingFang SC',      # macOS 苹方
        'Helvetica',        # macOS 通用
        'STHeiti',          # macOS 华文黑体
        'Arial Unicode MS', # macOS/Windows
        'SimHei',          # Windows 黑体
        'Microsoft YaHei', # Windows 微软雅黑
        'DejaVu Sans'      # 备用
    ]
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✅ 使用字体: {font}")
            return font
    
    print("⚠️  未找到中文字体，将使用英文标签")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return 'DejaVu Sans'

current_font = setup_chinese_font()
use_chinese_labels = current_font not in ['DejaVu Sans']

print("🎯 第7天练习 - 线性代数进阶")
print("="*50)

# 练习1：特征值和特征向量的计算
print("\n📚 练习1：特征值和特征向量")
print("-" * 30)

def practice_eigenvalues():
    """练习计算特征值和特征向量"""
    # 创建一个对称矩阵（更容易理解）
    A = np.array([[4, 2], 
                  [2, 1]])
    
    print("矩阵 A:")
    print(A)
    
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    print(f"\n特征值: {eigenvalues}")
    print(f"特征向量:\n{eigenvectors}")
    
    # 验证 Av = λv
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        λ = eigenvalues[i]
        
        print(f"\n验证特征向量 {i+1}:")
        print(f"Av = {A @ v}")
        print(f"λv = {λ * v}")
        print(f"误差: {np.linalg.norm(A @ v - λ * v):.10f}")
    
    return eigenvalues, eigenvectors

eigenvalues, eigenvectors = practice_eigenvalues()


# 练习2：手动实现PCA
print("\n📚 练习2：手动实现PCA")
print("-" * 30)

def manual_pca(X, n_components=2):
    """手动实现PCA算法"""
    # 步骤1：数据中心化
    X_centered = X - np.mean(X, axis=0)
    
    # 步骤2：计算协方差矩阵
    cov_matrix = np.cov(X_centered.T)
    
    # 步骤3：计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # 步骤4：按特征值大小排序
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[idx]
    eigenvectors_sorted = eigenvectors[:, idx]
    
    # 步骤5：选择前n_components个主成分
    components = eigenvectors_sorted[:, :n_components]
    
    # 步骤6：投影数据
    X_pca = X_centered @ components
    
    # 计算方差解释比例
    explained_variance_ratio = eigenvalues_sorted / np.sum(eigenvalues_sorted)
    
    return X_pca, components, explained_variance_ratio[:n_components]

# 生成测试数据
np.random.seed(42)
X_test = np.random.randn(100, 4)  # 100个样本，4个特征
# 让特征之间有一定的相关性
X_test[:, 1] = X_test[:, 0] + 0.5 * np.random.randn(100)
X_test[:, 2] = X_test[:, 0] - 0.3 * X_test[:, 1] + 0.2 * np.random.randn(100)

print(f"测试数据形状: {X_test.shape}")

# 使用手动实现的PCA
X_pca_manual, components_manual, explained_ratio_manual = manual_pca(X_test, n_components=2)

# 使用sklearn的PCA对比
pca_sklearn = PCA(n_components=2)
X_pca_sklearn = pca_sklearn.fit_transform(X_test)

print(f"\n手动PCA结果形状: {X_pca_manual.shape}")
print(f"sklearn PCA结果形状: {X_pca_sklearn.shape}")
print(f"最大差异: {np.max(np.abs(X_pca_manual - X_pca_sklearn)):.10f}")
print(f"手动PCA方差解释比例: {explained_ratio_manual}")
print(f"sklearn PCA方差解释比例: {pca_sklearn.explained_variance_ratio_}")


# 练习3：PCA降维效果可视化
print("\n📚 练习3：PCA降维效果可视化")
print("-" * 30)

def visualize_pca_effect():
    """可视化PCA降维效果"""
    # 生成具有明显结构的数据
    X, y = make_classification(n_samples=300, n_features=4, n_redundant=0, 
                              n_informative=2, n_clusters_per_class=1, 
                              random_state=42)
    
    # 应用PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 原始数据的前两个特征
    scatter1 = axes[0, 0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
    if use_chinese_labels:
        axes[0, 0].set_title('原始数据 (特征1 vs 特征2)')
        axes[0, 0].set_xlabel('特征1')
        axes[0, 0].set_ylabel('特征2')
    else:
        axes[0, 0].set_title('Original Data (Feature 1 vs 2)')
        axes[0, 0].set_xlabel('Feature 1')
        axes[0, 0].set_ylabel('Feature 2')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 原始数据的后两个特征
    scatter2 = axes[0, 1].scatter(X[:, 2], X[:, 3], c=y, cmap='viridis', alpha=0.7)
    if use_chinese_labels:
        axes[0, 1].set_title('原始数据 (特征3 vs 特征4)')
        axes[0, 1].set_xlabel('特征3')
        axes[0, 1].set_ylabel('特征4')
    else:
        axes[0, 1].set_title('Original Data (Feature 3 vs 4)')
        axes[0, 1].set_xlabel('Feature 3')
        axes[0, 1].set_ylabel('Feature 4')
    axes[0, 1].grid(True, alpha=0.3)
    
    # PCA降维后的数据
    scatter3 = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    axes[1, 0].set_title(f'PCA降维后\n(解释方差: {pca.explained_variance_ratio_.sum():.2%})')
    axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 方差解释比例
    axes[1, 1].bar(['PC1', 'PC2'], pca.explained_variance_ratio_)
    axes[1, 1].set_title('各主成分方差解释比例')
    axes[1, 1].set_ylabel('方差解释比例')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"原始数据维度: {X.shape[1]}")
    print(f"PCA后维度: {X_pca.shape[1]}")
    print(f"总方差解释比例: {pca.explained_variance_ratio_.sum():.2%}")
    
    return X, X_pca, pca

X_original, X_pca_viz, pca_viz = visualize_pca_effect()


# 练习4：不同主成分数量的效果
print("\n📚 练习4：不同主成分数量的效果")
print("-" * 30)

def compare_pca_components():
    """比较不同主成分数量的降维效果"""
    # 生成高维数据
    X, y = make_classification(n_samples=200, n_features=10, n_redundant=5, 
                              n_informative=5, random_state=42)
    
    # 测试不同的主成分数量
    n_components_list = [1, 2, 3, 5, 7, 10]
    
    print("主成分数量 | 方差解释比例 | 累积方差解释比例")
    print("-" * 50)
    
    explained_ratios = []
    cumulative_ratios = []
    
    for n_comp in n_components_list:
        pca = PCA(n_components=n_comp)
        pca.fit(X)
        
        total_explained = pca.explained_variance_ratio_.sum()
        explained_ratios.append(total_explained)
        
        print(f"    {n_comp:2d}       |     {total_explained:.2%}      |        {total_explained:.2%}")
    
    # 可视化
    plt.figure(figsize=(10, 6))
    plt.plot(n_components_list, explained_ratios, 'o-', linewidth=2, markersize=8)
    plt.axhline(y=0.9, color='r', linestyle='--', label='90%阈值')
    plt.axhline(y=0.95, color='g', linestyle='--', label='95%阈值')
    plt.xlabel('主成分数量')
    plt.ylabel('累积方差解释比例')
    plt.title('不同主成分数量的方差解释效果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 找到达到90%方差解释所需的主成分数量
    for i, ratio in enumerate(explained_ratios):
        if ratio >= 0.9:
            print(f"\n🎯 达到90%方差解释需要 {n_components_list[i]} 个主成分")
            break
    
    return explained_ratios

explained_ratios = compare_pca_components()


# 练习5：PCA的实际应用案例
print("\n📚 练习5：PCA实际应用案例")
print("-" * 30)

def pca_application_case():
    """PCA在实际问题中的应用案例"""
    # 模拟一个客户数据集
    np.random.seed(42)
    n_customers = 1000
    
    # 生成客户特征（模拟电商数据）
    # 年龄、收入、购买频率、平均订单金额、网站停留时间、评分等
    age = np.random.normal(35, 10, n_customers)
    income = np.random.normal(50000, 15000, n_customers)
    purchase_freq = np.random.poisson(12, n_customers)
    avg_order = np.random.normal(150, 50, n_customers)
    time_on_site = np.random.exponential(20, n_customers)
    rating = np.random.normal(4.2, 0.8, n_customers)
    
    # 添加一些相关性
    purchase_freq = purchase_freq + 0.3 * (income / 10000) + np.random.normal(0, 2, n_customers)
    avg_order = avg_order + 0.5 * (income / 1000) + np.random.normal(0, 20, n_customers)
    
    # 组合特征
    customer_features = np.column_stack([age, income, purchase_freq, avg_order, time_on_site, rating])
    feature_names = ['年龄', '收入', '购买频率', '平均订单金额', '网站停留时间', '评分']
    
    print("客户数据集:")
    print(f"样本数: {customer_features.shape[0]}")
    print(f"特征数: {customer_features.shape[1]}")
    print(f"特征名称: {feature_names}")
    
    # 标准化数据
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    customer_features_scaled = scaler.fit_transform(customer_features)
    
    # 应用PCA
    pca = PCA()
    customer_pca = pca.fit_transform(customer_features_scaled)
    
    # 分析结果
    print(f"\n各主成分的方差解释比例:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {ratio:.2%}")
    
    # 累积方差解释比例
    cumulative_ratio = np.cumsum(pca.explained_variance_ratio_)
    print(f"\n累积方差解释比例:")
    for i, ratio in enumerate(cumulative_ratio):
        print(f"前{i+1}个主成分: {ratio:.2%}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 各主成分方差解释比例
    axes[0, 0].bar(range(1, len(pca.explained_variance_ratio_) + 1), 
                   pca.explained_variance_ratio_)
    axes[0, 0].set_title('各主成分方差解释比例')
    axes[0, 0].set_xlabel('主成分')
    axes[0, 0].set_ylabel('方差解释比例')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 累积方差解释比例
    axes[0, 1].plot(range(1, len(cumulative_ratio) + 1), cumulative_ratio, 'o-')
    axes[0, 1].axhline(y=0.8, color='r', linestyle='--', label='80%阈值')
    axes[0, 1].axhline(y=0.9, color='g', linestyle='--', label='90%阈值')
    axes[0, 1].set_title('累积方差解释比例')
    axes[0, 1].set_xlabel('主成分数量')
    axes[0, 1].set_ylabel('累积方差解释比例')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 前两个主成分的数据分布
    axes[1, 0].scatter(customer_pca[:, 0], customer_pca[:, 1], alpha=0.6)
    axes[1, 0].set_title('前两个主成分的数据分布')
    axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 主成分载荷图（特征贡献）
    components_df = pca.components_[:2, :]  # 前两个主成分
    axes[1, 1].imshow(components_df, cmap='RdBu', aspect='auto')
    axes[1, 1].set_title('主成分载荷图')
    axes[1, 1].set_xlabel('原始特征')
    axes[1, 1].set_ylabel('主成分')
    axes[1, 1].set_xticks(range(len(feature_names)))
    axes[1, 1].set_xticklabels(feature_names, rotation=45)
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_yticklabels(['PC1', 'PC2'])
    
    plt.tight_layout()
    plt.show()
    
    # 降维建议
    n_components_80 = np.argmax(cumulative_ratio >= 0.8) + 1
    n_components_90 = np.argmax(cumulative_ratio >= 0.9) + 1
    
    print(f"\n📊 降维建议:")
    print(f"保留80%方差: 使用 {n_components_80} 个主成分")
    print(f"保留90%方差: 使用 {n_components_90} 个主成分")
    print(f"降维效果: 从 {len(feature_names)} 维降到 {n_components_80} 维")
    
    return customer_pca, pca

customer_pca, pca_customer = pca_application_case()

print("\n🎯 练习完成！")
print("="*50)
print("✅ 你已经完成了所有线性代数进阶练习")
print("✅ 理解了特征值、特征向量的计算和应用")
print("✅ 掌握了PCA的原理和实现")
print("✅ 学会了分析和选择合适的主成分数量")
print("✅ 了解了PCA在实际问题中的应用")
print("\n🚀 现在你可以开始更高级的机器学习算法学习了！") 