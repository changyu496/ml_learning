"""
PCA可视化演示
目标：通过直观的可视化帮助理解PCA的工作原理
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs, load_iris
import matplotlib.patches as patches

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

print("🎨 PCA可视化演示")
print("="*50)

def demo_pca_concept():
    """演示PCA的基本概念"""
    print("\n📚 演示1：PCA的基本概念")
    print("-" * 30)
    
    # 生成二维数据
    np.random.seed(42)
    X = np.random.randn(100, 2)
    # 给数据一个相关性
    X[:, 1] = X[:, 0] + 0.5 * X[:, 1]
    
    # 应用PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始数据
    axes[0].scatter(X[:, 0], X[:, 1], alpha=0.7, c='blue')
    if use_chinese_labels:
        axes[0].set_title('原始数据')
        axes[0].set_xlabel('特征1')
        axes[0].set_ylabel('特征2')
    else:
        axes[0].set_title('Original Data')
        axes[0].set_xlabel('Feature 1')
        axes[0].set_ylabel('Feature 2')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')
    
    # 显示主成分方向
    axes[1].scatter(X[:, 0], X[:, 1], alpha=0.7, c='blue')
    
    # 计算数据中心
    center = np.mean(X, axis=0)
    
    # 绘制主成分方向
    for i, component in enumerate(pca.components_):
        # 缩放向量以便显示
        scale = 2 * np.sqrt(pca.explained_variance_[i])
        axes[1].arrow(center[0], center[1], 
                     component[0] * scale, component[1] * scale,
                     head_width=0.1, head_length=0.1, 
                     fc=f'C{i}', ec=f'C{i}', linewidth=2,
                     label=f'PC{i+1} ({pca.explained_variance_ratio_[i]:.2%})')
    
    axes[1].set_title('主成分方向')
    axes[1].set_xlabel('特征1')
    axes[1].set_ylabel('特征2')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_aspect('equal')
    
    # 变换后的数据
    axes[2].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, c='blue')
    axes[2].set_title('PCA变换后的数据')
    axes[2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    axes[2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    print(f"方差解释比例: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")
    print(f"总方差解释: {pca.explained_variance_ratio_.sum():.2%}")


def demo_dimensionality_reduction():
    """演示降维效果"""
    print("\n📚 演示2：降维效果对比")
    print("-" * 30)
    
    # 生成3D数据
    np.random.seed(42)
    X = np.random.randn(200, 3)
    # 创建相关性
    X[:, 1] = X[:, 0] + 0.3 * X[:, 1]
    X[:, 2] = X[:, 0] - 0.5 * X[:, 1] + 0.2 * X[:, 2]
    
    # 应用PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # 3D可视化
    fig = plt.figure(figsize=(15, 5))
    
    # 原始3D数据
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.6, c='blue')
    ax1.set_title('原始3D数据')
    ax1.set_xlabel('特征1')
    ax1.set_ylabel('特征2')
    ax1.set_zlabel('特征3')
    
    # 2D降维结果
    ax2 = fig.add_subplot(132)
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, c='blue')
    ax2.set_title('PCA降维到2D')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    ax2.grid(True, alpha=0.3)
    
    # 方差解释比例
    ax3 = fig.add_subplot(133)
    components = ['PC1', 'PC2', '剩余']
    ratios = [pca.explained_variance_ratio_[0], 
              pca.explained_variance_ratio_[1], 
              1 - pca.explained_variance_ratio_.sum()]
    colors = ['red', 'green', 'gray']
    
    ax3.pie(ratios, labels=components, colors=colors, autopct='%1.1f%%')
    ax3.set_title('方差解释比例')
    
    plt.tight_layout()
    plt.show()
    
    print(f"信息保留: {pca.explained_variance_ratio_.sum():.2%}")
    print(f"信息损失: {1 - pca.explained_variance_ratio_.sum():.2%}")


def demo_iris_pca():
    """使用鸢尾花数据集演示PCA"""
    print("\n📚 演示3：鸢尾花数据集PCA分析")
    print("-" * 30)
    
    # 加载鸢尾花数据
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    print(f"原始数据形状: {X.shape}")
    print(f"特征名称: {iris.feature_names}")
    
    # 应用PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 原始特征的散点图
    feature_pairs = [(0, 1), (0, 2), (0, 3)]
    colors = ['red', 'green', 'blue']
    target_names = iris.target_names
    
    for i, (f1, f2) in enumerate(feature_pairs):
        for j, color, target_name in zip(range(3), colors, target_names):
            axes[0, i].scatter(X[y == j, f1], X[y == j, f2], 
                             c=color, label=target_name, alpha=0.7)
        
        axes[0, i].set_xlabel(iris.feature_names[f1])
        axes[0, i].set_ylabel(iris.feature_names[f2])
        axes[0, i].set_title(f'{iris.feature_names[f1]} vs {iris.feature_names[f2]}')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
    
    # PCA结果
    for i, color, target_name in zip(range(3), colors, target_names):
        axes[1, 0].scatter(X_pca[y == i, 0], X_pca[y == i, 1], 
                         c=color, label=target_name, alpha=0.7)
    
    axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    axes[1, 0].set_title('PCA降维结果')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 方差解释比例
    all_pca = PCA()
    all_pca.fit(X)
    
    axes[1, 1].bar(range(1, len(all_pca.explained_variance_ratio_) + 1), 
                   all_pca.explained_variance_ratio_)
    axes[1, 1].set_title('各主成分方差解释比例')
    axes[1, 1].set_xlabel('主成分')
    axes[1, 1].set_ylabel('方差解释比例')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 载荷图
    components_df = pca.components_
    im = axes[1, 2].imshow(components_df, cmap='RdBu', aspect='auto')
    axes[1, 2].set_title('主成分载荷图')
    axes[1, 2].set_xlabel('原始特征')
    axes[1, 2].set_ylabel('主成分')
    axes[1, 2].set_xticks(range(len(iris.feature_names)))
    axes[1, 2].set_xticklabels([name[:8] for name in iris.feature_names], rotation=45)
    axes[1, 2].set_yticks([0, 1])
    axes[1, 2].set_yticklabels(['PC1', 'PC2'])
    
    # 添加颜色条
    plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.show()
    
    print(f"前2个主成分解释了 {pca.explained_variance_ratio_.sum():.2%} 的方差")
    print(f"各主成分方差解释比例: {pca.explained_variance_ratio_}")


def demo_pca_noise_reduction():
    """演示PCA的去噪效果"""
    print("\n📚 演示4：PCA去噪效果")
    print("-" * 30)
    
    # 生成有噪声的数据
    np.random.seed(42)
    n_samples = 1000
    
    # 生成主要信号
    t = np.linspace(0, 4*np.pi, n_samples)
    signal1 = np.sin(t) + 0.5 * np.cos(2*t)
    signal2 = np.cos(t) - 0.3 * np.sin(3*t)
    
    # 添加噪声
    noise_level = 0.5
    X_clean = np.column_stack([signal1, signal2])
    X_noisy = X_clean + noise_level * np.random.randn(n_samples, 2)
    
    # 应用PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_noisy)
    
    # 重构数据（只使用第一个主成分）
    pca_denoised = PCA(n_components=1)
    X_denoised_pca = pca_denoised.fit_transform(X_noisy)
    X_denoised = pca_denoised.inverse_transform(X_denoised_pca)
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 原始干净数据
    axes[0, 0].scatter(X_clean[:, 0], X_clean[:, 1], alpha=0.6, c='blue', s=10)
    axes[0, 0].set_title('原始干净数据')
    axes[0, 0].set_xlabel('特征1')
    axes[0, 0].set_ylabel('特征2')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 带噪声的数据
    axes[0, 1].scatter(X_noisy[:, 0], X_noisy[:, 1], alpha=0.6, c='red', s=10)
    axes[0, 1].set_title('带噪声的数据')
    axes[0, 1].set_xlabel('特征1')
    axes[0, 1].set_ylabel('特征2')
    axes[0, 1].grid(True, alpha=0.3)
    
    # PCA去噪后的数据
    axes[1, 0].scatter(X_denoised[:, 0], X_denoised[:, 1], alpha=0.6, c='green', s=10)
    axes[1, 0].set_title('PCA去噪后的数据')
    axes[1, 0].set_xlabel('特征1')
    axes[1, 0].set_ylabel('特征2')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 方差解释比例
    axes[1, 1].bar(['PC1', 'PC2'], pca.explained_variance_ratio_)
    axes[1, 1].set_title('方差解释比例')
    axes[1, 1].set_ylabel('方差解释比例')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 计算去噪效果
    mse_original = np.mean((X_noisy - X_clean)**2)
    mse_denoised = np.mean((X_denoised - X_clean)**2)
    
    print(f"原始噪声MSE: {mse_original:.4f}")
    print(f"PCA去噪后MSE: {mse_denoised:.4f}")
    print(f"去噪效果: {(mse_original - mse_denoised) / mse_original:.2%} 的误差减少")


def demo_pca_data_compression():
    """演示PCA的数据压缩效果"""
    print("\n📚 演示5：PCA数据压缩效果")
    print("-" * 30)
    
    # 生成高维数据
    np.random.seed(42)
    n_samples = 500
    n_features = 50
    
    # 生成有结构的数据
    X = np.random.randn(n_samples, n_features)
    # 让前几个特征有相关性
    for i in range(1, 10):
        X[:, i] = X[:, 0] + 0.3 * X[:, i]
    
    print(f"原始数据形状: {X.shape}")
    print(f"原始数据存储大小: {X.nbytes} 字节")
    
    # 测试不同的压缩比
    compression_ratios = [0.1, 0.2, 0.3, 0.5, 0.8]
    
    results = []
    for ratio in compression_ratios:
        n_components = int(n_features * ratio)
        if n_components < 1:
            n_components = 1
        
        # 应用PCA
        pca = PCA(n_components=n_components)
        X_compressed = pca.fit_transform(X)
        X_reconstructed = pca.inverse_transform(X_compressed)
        
        # 计算重构误差
        mse = np.mean((X - X_reconstructed)**2)
        
        # 计算压缩大小
        compressed_size = X_compressed.nbytes + pca.components_.nbytes + pca.mean_.nbytes
        
        results.append({
            'ratio': ratio,
            'n_components': n_components,
            'explained_variance': pca.explained_variance_ratio_.sum(),
            'mse': mse,
            'compressed_size': compressed_size,
            'compression_ratio': compressed_size / X.nbytes
        })
    
    # 可视化结果
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 方差解释比例
    ratios = [r['ratio'] for r in results]
    explained_vars = [r['explained_variance'] for r in results]
    axes[0, 0].plot(ratios, explained_vars, 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('保留的主成分比例')
    axes[0, 0].set_ylabel('方差解释比例')
    axes[0, 0].set_title('方差解释 vs 主成分比例')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 重构误差
    mses = [r['mse'] for r in results]
    axes[0, 1].plot(ratios, mses, 'o-', linewidth=2, markersize=8, color='red')
    axes[0, 1].set_xlabel('保留的主成分比例')
    axes[0, 1].set_ylabel('均方误差')
    axes[0, 1].set_title('重构误差 vs 主成分比例')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 压缩比例
    comp_ratios = [r['compression_ratio'] for r in results]
    axes[1, 0].plot(ratios, comp_ratios, 'o-', linewidth=2, markersize=8, color='green')
    axes[1, 0].set_xlabel('保留的主成分比例')
    axes[1, 0].set_ylabel('实际压缩比例')
    axes[1, 0].set_title('压缩比例 vs 主成分比例')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 综合效果
    axes[1, 1].scatter(comp_ratios, explained_vars, s=100, alpha=0.7)
    for i, (x, y) in enumerate(zip(comp_ratios, explained_vars)):
        axes[1, 1].annotate(f'{ratios[i]:.1f}', (x, y), xytext=(5, 5), 
                          textcoords='offset points')
    axes[1, 1].set_xlabel('压缩比例')
    axes[1, 1].set_ylabel('方差解释比例')
    axes[1, 1].set_title('压缩效果综合分析')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 打印结果表格
    print("\n压缩效果分析:")
    print("-" * 80)
    print(f"{'主成分比例':<12} {'主成分数':<8} {'方差解释':<12} {'重构误差':<12} {'压缩比':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['ratio']:<12.1f} {r['n_components']:<8} {r['explained_variance']:<12.2%} "
              f"{r['mse']:<12.4f} {r['compression_ratio']:<12.2%}")


# 运行所有演示
if __name__ == "__main__":
    demo_pca_concept()
    demo_dimensionality_reduction()
    demo_iris_pca()
    demo_pca_noise_reduction()
    demo_pca_data_compression()
    
    print("\n🎯 PCA可视化演示完成！")
    print("="*50)
    print("✅ 你已经通过可视化理解了PCA的核心概念")
    print("✅ 了解了PCA在降维、去噪、压缩中的应用")
    print("✅ 掌握了如何选择合适的主成分数量")
    print("✅ 学会了分析PCA的效果和性能")
    print("\n💡 现在你可以在自己的项目中应用PCA了！") 