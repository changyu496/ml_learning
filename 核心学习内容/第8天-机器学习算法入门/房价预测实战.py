"""
房价预测实战项目 - 第一个完整的机器学习项目！🏠
目标：使用线性回归预测房价
时间：30分钟
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

print("🏠 房价预测实战项目")
print("="*50)
print("🎯 目标：建立一个完整的机器学习项目")
print("📊 数据：房屋特征 → 房价")
print("💡 这是你的第一个端到端机器学习项目！")

# ===== 第1关：数据准备 =====
print("\n🎯 第1关：数据准备")

# 创建模拟的房价数据
np.random.seed(42)
n_samples = 1000

# 生成特征数据
house_size = np.random.normal(150, 50, n_samples)  # 房屋面积 (平方米)
bedrooms = np.random.poisson(2.5, n_samples)       # 卧室数量
bathrooms = np.random.poisson(1.8, n_samples)      # 浴室数量
age = np.random.uniform(0, 50, n_samples)          # 房龄 (年)
location_score = np.random.uniform(1, 10, n_samples)  # 地段评分

# 生成房价 (基于真实关系 + 噪声)
price = (house_size * 50 +           # 面积影响
         bedrooms * 20000 +          # 卧室数量影响
         bathrooms * 15000 +         # 浴室数量影响
         -age * 1000 +               # 房龄影响 (负相关)
         location_score * 30000 +    # 地段影响
         np.random.normal(0, 50000, n_samples))  # 噪声

# 创建DataFrame
data = pd.DataFrame({
    'house_size': house_size,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'age': age,
    'location_score': location_score,
    'price': price
})

print(f"📊 数据集创建完成！")
print(f"数据形状: {data.shape}")
print(f"前5行数据:")
print(data.head())

# 任务1.1：数据探索
print("\n📝 任务1.1：探索数据基本信息")
print("你的代码：")
# 在这里写代码：
# print("数据描述:")
# print(data.describe())
# print("\n数据类型:")
# print(data.dtypes)
# print("\n是否有缺失值:")
# print(data.isnull().sum())

# ===== 第2关：数据可视化 =====
print("\n🎯 第2关：数据可视化")

# 任务2.1：房价分布
print("\n📝 任务2.1：查看房价分布")
print("你的代码：")
# 在这里写代码：
# plt.figure(figsize=(15, 10))
# 
# plt.subplot(2, 3, 1)
# plt.hist(data['price'], bins=30, alpha=0.7)
# plt.title('房价分布')
# plt.xlabel('价格')
# plt.ylabel('频数')

# 任务2.2：特征相关性
print("\n📝 任务2.2：特征与房价的关系")
print("你的代码：")
# 在这里写代码：
# plt.subplot(2, 3, 2)
# plt.scatter(data['house_size'], data['price'], alpha=0.6)
# plt.title('面积 vs 房价')
# plt.xlabel('面积')
# plt.ylabel('价格')
# 
# plt.subplot(2, 3, 3)
# plt.scatter(data['location_score'], data['price'], alpha=0.6)
# plt.title('地段评分 vs 房价')
# plt.xlabel('地段评分')
# plt.ylabel('价格')
# 
# plt.subplot(2, 3, 4)
# plt.scatter(data['age'], data['price'], alpha=0.6)
# plt.title('房龄 vs 房价')
# plt.xlabel('房龄')
# plt.ylabel('价格')
# 
# plt.subplot(2, 3, 5)
# plt.scatter(data['bedrooms'], data['price'], alpha=0.6)
# plt.title('卧室数 vs 房价')
# plt.xlabel('卧室数')
# plt.ylabel('价格')
# 
# plt.subplot(2, 3, 6)
# plt.scatter(data['bathrooms'], data['price'], alpha=0.6)
# plt.title('浴室数 vs 房价')
# plt.xlabel('浴室数')
# plt.ylabel('价格')
# 
# plt.tight_layout()
# plt.show()

# 任务2.3：相关性热力图
print("\n📝 任务2.3：相关性热力图")
print("你的代码：")
# 在这里写代码：
# plt.figure(figsize=(10, 8))
# correlation_matrix = data.corr()
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
# plt.title('特征相关性热力图')
# plt.show()

# ===== 第3关：数据预处理 =====
print("\n🎯 第3关：数据预处理")

# 任务3.1：分离特征和目标
print("\n📝 任务3.1：分离特征和目标变量")
print("你的代码：")
# 在这里写代码：
# X = data.drop('price', axis=1)
# y = data['price']
# 
# print(f"特征矩阵 X 形状: {X.shape}")
# print(f"目标向量 y 形状: {y.shape}")

# 任务3.2：数据分割
print("\n📝 任务3.2：分割训练集和测试集")
print("你的代码：")
# 在这里写代码：
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
# 
# print(f"训练集大小: {X_train.shape[0]}")
# print(f"测试集大小: {X_test.shape[0]}")

# ===== 第4关：模型训练 =====
print("\n🎯 第4关：模型训练")

# 任务4.1：创建和训练模型
print("\n📝 任务4.1：创建和训练线性回归模型")
print("你的代码：")
# 在这里写代码：
# model = LinearRegression()
# model.fit(X_train, y_train)
# print("模型训练完成！")

# 任务4.2：查看模型参数
print("\n📝 任务4.2：查看模型参数")
print("你的代码：")
# 在这里写代码：
# print("模型参数:")
# feature_names = X.columns
# for i, feature in enumerate(feature_names):
#     print(f"{feature}: {model.coef_[i]:.2f}")
# print(f"截距: {model.intercept_:.2f}")

# ===== 第5关：模型预测 =====
print("\n🎯 第5关：模型预测")

# 任务5.1：预测
print("\n📝 任务5.1：使用模型进行预测")
print("你的代码：")
# 在这里写代码：
# y_train_pred = model.predict(X_train)
# y_test_pred = model.predict(X_test)
# 
# print(f"训练集预测完成，形状: {y_train_pred.shape}")
# print(f"测试集预测完成，形状: {y_test_pred.shape}")

# ===== 第6关：模型评估 =====
print("\n🎯 第6关：模型评估")

# 任务6.1：计算评估指标
print("\n📝 任务6.1：计算评估指标")
print("你的代码：")
# 在这里写代码：
# # 训练集评估
# train_mse = mean_squared_error(y_train, y_train_pred)
# train_r2 = r2_score(y_train, y_train_pred)
# 
# # 测试集评估
# test_mse = mean_squared_error(y_test, y_test_pred)
# test_r2 = r2_score(y_test, y_test_pred)
# 
# print("模型评估结果:")
# print(f"训练集 MSE: {train_mse:.2f}")
# print(f"训练集 R²: {train_r2:.4f}")
# print(f"测试集 MSE: {test_mse:.2f}")
# print(f"测试集 R²: {test_r2:.4f}")

# 任务6.2：预测结果可视化
print("\n📝 任务6.2：预测结果可视化")
print("你的代码：")
# 在这里写代码：
# plt.figure(figsize=(15, 5))
# 
# plt.subplot(1, 3, 1)
# plt.scatter(y_test, y_test_pred, alpha=0.6)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
# plt.xlabel('真实价格')
# plt.ylabel('预测价格')
# plt.title('真实价格 vs 预测价格')
# 
# plt.subplot(1, 3, 2)
# residuals = y_test - y_test_pred
# plt.scatter(y_test_pred, residuals, alpha=0.6)
# plt.axhline(y=0, color='r', linestyle='--')
# plt.xlabel('预测价格')
# plt.ylabel('残差')
# plt.title('残差图')
# 
# plt.subplot(1, 3, 3)
# plt.hist(residuals, bins=30, alpha=0.7)
# plt.xlabel('残差')
# plt.ylabel('频数')
# plt.title('残差分布')
# 
# plt.tight_layout()
# plt.show()

# ===== 第7关：实际应用 =====
print("\n🎯 第7关：实际应用")

# 任务7.1：预测新房价
print("\n📝 任务7.1：预测新房价")
print("假设有一套新房子：")
print("- 面积: 120平方米")
print("- 卧室: 3个")
print("- 浴室: 2个")
print("- 房龄: 5年")
print("- 地段评分: 8分")
print("你的代码：")
# 在这里写代码：
# new_house = pd.DataFrame({
#     'house_size': [120],
#     'bedrooms': [3],
#     'bathrooms': [2],
#     'age': [5],
#     'location_score': [8]
# })
# 
# predicted_price = model.predict(new_house)[0]
# print(f"预测价格: {predicted_price:.2f} 元")

# 任务7.2：特征重要性分析
print("\n📝 任务7.2：特征重要性分析")
print("你的代码：")
# 在这里写代码：
# feature_importance = abs(model.coef_)
# feature_names = X.columns
# 
# plt.figure(figsize=(10, 6))
# plt.barh(feature_names, feature_importance)
# plt.xlabel('特征重要性 (绝对值)')
# plt.title('特征重要性排序')
# plt.show()
# 
# # 输出重要性排序
# importance_df = pd.DataFrame({
#     'feature': feature_names,
#     'importance': feature_importance
# }).sort_values('importance', ascending=False)
# 
# print("特征重要性排序:")
# print(importance_df)

# ===== 第8关：项目总结 =====
print("\n🎯 第8关：项目总结")

# 任务8.1：项目总结
print("\n📝 任务8.1：项目总结")
print("请总结：")
print("1. 这个项目解决了什么问题？")
print("2. 模型的效果如何？")
print("3. 哪个特征最重要？")
print("4. 还有哪些改进空间？")

# 你的总结：
# 总结1：这个项目解决了___问题
# 总结2：模型的R²分数为___，说明___
# 总结3：最重要的特征是___，因为___
# 总结4：改进空间：___

print("\n🎉 恭喜！你完成了第一个完整的机器学习项目！")
print("📊 项目完成清单:")
print("- [x] 数据准备和探索")
print("- [x] 数据可视化分析")
print("- [x] 特征工程")
print("- [x] 模型训练")
print("- [x] 模型评估")
print("- [x] 实际应用")
print("- [x] 项目总结")

print("\n💡 这个项目展示了机器学习的完整流程！")
print("🚀 你现在已经具备了基本的机器学习项目能力！")

# ===== 自我检查 =====
print("\n" + "="*50)
print("🏁 完成项目后，请检查：")
print("1. 你理解机器学习项目的完整流程了吗？")
print("2. 你能独立完成类似的项目吗？")
print("3. 你知道如何评估模型效果吗？")
print("4. 你能解释模型的预测结果吗？")
print("\n💡 这是你的第一个完整项目，值得骄傲！")

print("\n" + "="*50)
print("🎯 下一步学习建议：")
print("1. 尝试其他回归算法（多项式回归、岭回归等）")
print("2. 学习分类算法（逻辑回归、决策树等）")
print("3. 掌握更多特征工程技术")
print("4. 学习交叉验证等高级技术")
print("="*50) 