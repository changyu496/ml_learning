#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第5天：线性代数基础 - 编程练习

这个文件包含第5天的所有编程练习，涵盖：
1. 向量空间基础操作
2. 线性变换实现
3. 矩阵分解应用
4. PCA降维实战

完成这些练习将帮助你：
- 掌握线性代数的核心概念
- 理解矩阵分解的原理
- 实现PCA降维算法
- 应用线性代数解决实际问题
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris, make_blobs
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("🧮 第5天：线性代数基础 - 编程练习")
print("="*60)

# ==========================================
# 练习1：向量空间基础操作
# ==========================================

def exercise_1_vector_space():
    """
    练习1：向量空间基础操作
    
    任务：
    1. 实现向量的线性组合
    2. 检验向量的线性无关性
    3. 寻找向量空间的基
    4. 计算向量在不同基下的坐标
    """
    print("\n📐 练习1：向量空间基础操作")
    print("-" * 40)
    
    # TODO: 定义三个2D向量
    v1 = None  # 例如: np.array([1, 2])
    v2 = None  # 例如: np.array([3, 1])
    v3 = None  # 例如: np.array([2, 3])
    
    # TODO: 计算v3是否可以表示为v1和v2的线性组合
    # 提示：求解方程 a*v1 + b*v2 = v3
    # 使用 np.linalg.solve() 或 np.linalg.lstsq()
    
    # TODO: 检验v1和v2是否线性无关
    # 提示：计算由v1, v2组成的矩阵的行列式
    # 使用 np.linalg.det()
    
    # TODO: 将向量v3在基{v1, v2}下表示
    # 如果v1, v2线性无关，它们构成2D空间的一组基
    
    # TODO: 可视化结果
    # 绘制v1, v2, v3以及线性组合的几何关系
    
    print("练习1完成！")
    return v1, v2, v3

# ==========================================
# 练习2：线性变换实现
# ==========================================

def exercise_2_linear_transformation():
    """
    练习2：线性变换实现
    
    任务：
    1. 实现旋转变换矩阵
    2. 实现缩放变换矩阵
    3. 实现反射变换矩阵
    4. 可视化变换效果
    """
    print("\n🔄 练习2：线性变换实现")
    print("-" * 40)
    
    # TODO: 创建一组2D点作为测试数据
    # 例如：正方形的四个顶点
    points = None  # 形状应该是 (2, n)，其中n是点的数量
    
    # TODO: 实现旋转变换矩阵
    def rotation_matrix(angle):
        """
        创建2D旋转矩阵
        
        参数:
        angle: 旋转角度（弧度）
        
        返回:
        2x2的旋转矩阵
        """
        # 提示：旋转矩阵的公式
        # [[cos(θ), -sin(θ)]
        #  [sin(θ),  cos(θ)]]
        pass
    
    # TODO: 实现缩放变换矩阵
    def scaling_matrix(sx, sy):
        """
        创建2D缩放矩阵
        
        参数:
        sx: x方向缩放因子
        sy: y方向缩放因子
        
        返回:
        2x2的缩放矩阵
        """
        # 提示：缩放矩阵的公式
        # [[sx,  0]
        #  [ 0, sy]]
        pass
    
    # TODO: 实现反射变换矩阵
    def reflection_matrix(axis='x'):
        """
        创建2D反射矩阵
        
        参数:
        axis: 反射轴 ('x' 或 'y')
        
        返回:
        2x2的反射矩阵
        """
        # 提示：关于x轴反射 [[1, 0], [0, -1]]
        #       关于y轴反射 [[-1, 0], [0, 1]]
        pass
    
    # TODO: 应用变换并可视化结果
    # 1. 旋转45度
    # 2. 缩放(2, 0.5)
    # 3. 关于x轴反射
    # 4. 复合变换：先旋转再缩放
    
    print("练习2完成！")
    return points

# ==========================================
# 练习3：矩阵分解应用
# ==========================================

def exercise_3_matrix_decomposition():
    """
    练习3：矩阵分解应用
    
    任务：
    1. 实现特征值分解（EVD）
    2. 实现奇异值分解（SVD）
    3. 使用SVD进行图像压缩
    4. 比较不同分解方法的特点
    """
    print("\n🔧 练习3：矩阵分解应用")
    print("-" * 40)
    
    # TODO: 创建一个对称矩阵进行特征值分解
    # 对称矩阵的特征值都是实数，特征向量正交
    A = None  # 例如: np.array([[4, 2], [2, 3]])
    
    # TODO: 计算特征值和特征向量
    # 使用 np.linalg.eig()
    eigenvalues = None
    eigenvectors = None
    
    # TODO: 验证特征值分解
    # A = Q * Λ * Q^T，其中Λ是对角矩阵，Q是正交矩阵
    
    # TODO: 创建一个矩阵进行SVD分解
    B = None  # 例如: np.random.randn(4, 3)
    
    # TODO: 计算SVD分解
    # 使用 np.linalg.svd()
    U = None
    S = None  
    Vt = None
    
    # TODO: 验证SVD分解
    # B = U * Σ * V^T
    
    # TODO: 使用SVD进行低秩近似（模拟图像压缩）
    # 只保留最大的k个奇异值
    def low_rank_approximation(matrix, k):
        """
        使用SVD进行低秩近似
        
        参数:
        matrix: 输入矩阵
        k: 保留的奇异值个数
        
        返回:
        近似矩阵
        """
        # 提示：
        # 1. 对矩阵进行SVD分解
        # 2. 只保留前k个奇异值
        # 3. 重构矩阵
        pass
    
    # TODO: 可视化不同压缩比的效果
    
    print("练习3完成！")
    return A, eigenvalues, eigenvectors

# ==========================================
# 练习4：PCA降维实战
# ==========================================

def exercise_4_pca_implementation():
    """
    练习4：PCA降维实战
    
    任务：
    1. 从零实现PCA算法
    2. 使用sklearn的PCA进行对比
    3. 在真实数据集上应用PCA
    4. 分析主成分的含义
    """
    print("\n📊 练习4：PCA降维实战")
    print("-" * 40)
    
    # TODO: 从零实现PCA
    def my_pca(X, n_components):
        """
        从零实现PCA算法
        
        参数:
        X: 数据矩阵 (n_samples, n_features)
        n_components: 主成分个数
        
        返回:
        X_transformed: 降维后的数据
        components: 主成分
        explained_variance_ratio: 解释方差比例
        """
        # 步骤1: 数据中心化
        # TODO: 计算每个特征的均值
        mean = None
        
        # TODO: 中心化数据
        X_centered = None
        
        # 步骤2: 计算协方差矩阵
        # TODO: 计算协方差矩阵
        # 提示: cov = X_centered.T @ X_centered / (n_samples - 1)
        cov_matrix = None
        
        # 步骤3: 特征值分解
        # TODO: 计算协方差矩阵的特征值和特征向量
        eigenvalues = None
        eigenvectors = None
        
        # 步骤4: 选择主成分
        # TODO: 按特征值大小排序
        # 提示: 使用 np.argsort()[::-1] 进行降序排序
        idx = None
        eigenvalues = None
        eigenvectors = None
        
        # TODO: 选择前n_components个主成分
        components = None
        
        # 步骤5: 数据变换
        # TODO: 将数据投影到主成分空间
        X_transformed = None
        
        # 计算解释方差比例
        explained_variance_ratio = None
        
        return X_transformed, components, explained_variance_ratio
    
    # TODO: 加载鸢尾花数据集进行测试
    # 使用 load_iris() 函数
    iris = None
    X = None
    y = None
    
    # TODO: 使用自实现的PCA
    X_pca_mine, components_mine, var_ratio_mine = my_pca(X, n_components=2)
    
    # TODO: 使用sklearn的PCA进行对比
    pca_sklearn = None
    X_pca_sklearn = None
    
    # TODO: 比较结果
    print("自实现PCA的解释方差比例:", var_ratio_mine)
    print("sklearn PCA的解释方差比例:", pca_sklearn.explained_variance_ratio_)
    
    # TODO: 可视化降维结果
    # 绘制原始4D数据在前2个主成分上的投影
    
    # TODO: 分析主成分的含义
    # 查看每个主成分中各个原始特征的权重
    
    print("练习4完成！")
    return X_pca_mine, X_pca_sklearn

# ==========================================
# 练习5：综合应用 - 人脸识别中的PCA
# ==========================================

def exercise_5_pca_face_recognition():
    """
    练习5：综合应用 - 人脸识别中的PCA（特征脸）
    
    任务：
    1. 生成模拟人脸数据
    2. 使用PCA提取特征脸
    3. 实现简单的人脸识别
    4. 分析不同主成分数量的影响
    """
    print("\n👤 练习5：PCA在人脸识别中的应用")
    print("-" * 40)
    
    # TODO: 生成模拟人脸数据
    # 创建一些简单的"人脸"图像（例如不同的几何图案）
    def generate_face_data(n_faces=50, img_size=(20, 20)):
        """
        生成模拟人脸数据
        
        参数:
        n_faces: 人脸数量
        img_size: 图像大小
        
        返回:
        faces: 人脸数据矩阵 (n_faces, height*width)
        labels: 人脸标签
        """
        # TODO: 创建不同类型的"人脸"
        # 可以使用几何图形、噪声等创建不同的模式
        pass
    
    # TODO: 生成数据
    faces, labels = generate_face_data()
    
    # TODO: 应用PCA提取特征脸
    # 特征脸 = PCA的主成分，代表人脸的主要变化方向
    
    # TODO: 可视化特征脸
    # 将主成分重塑为图像形状并显示
    
    # TODO: 实现简单的人脸识别
    def face_recognition(train_faces, train_labels, test_face, n_components=10):
        """
        基于PCA的简单人脸识别
        
        参数:
        train_faces: 训练人脸数据
        train_labels: 训练标签
        test_face: 测试人脸
        n_components: 主成分数量
        
        返回:
        predicted_label: 预测标签
        """
        # TODO: 
        # 1. 在训练数据上训练PCA
        # 2. 将训练和测试数据投影到PCA空间
        # 3. 使用最近邻分类器进行识别
        pass
    
    # TODO: 测试人脸识别效果
    
    # TODO: 分析不同主成分数量对识别效果的影响
    
    print("练习5完成！")

# ==========================================
# 主函数 - 运行所有练习
# ==========================================

def main():
    """运行所有练习"""
    print("开始第5天的线性代数基础练习...")
    
    # 练习1：向量空间基础操作
    exercise_1_vector_space()
    
    # 练习2：线性变换实现
    exercise_2_linear_transformation()
    
    # 练习3：矩阵分解应用
    exercise_3_matrix_decomposition()
    
    # 练习4：PCA降维实战
    exercise_4_pca_implementation()
    
    # 练习5：综合应用
    exercise_5_pca_face_recognition()
    
    print("\n🎉 恭喜！你已经完成了第5天的所有练习！")
    print("通过这些练习，你应该已经掌握了：")
    print("✅ 向量空间的基本概念和操作")
    print("✅ 线性变换的实现和应用")
    print("✅ 矩阵分解的原理和用途")
    print("✅ PCA算法的完整实现")
    print("✅ 线性代数在机器学习中的应用")

if __name__ == "__main__":
    main() 