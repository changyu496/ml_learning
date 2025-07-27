from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

print(f'数据形状:{X.shape}')
print(f'特征名称:{iris.feature_names}')
print(f'目标类别:{iris.target_names}')
print(f'前5行的数据:{X[:5]}')

print(f'目标标签的分布，每个品种的有多少朵花:{np.bincount(y)}')
print(f'特征数据的平均值:{X.mean(axis=0)}')
print(f'特征数据的标准差:{X.std(axis=0)}')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 当前图表是由2*2个组成
fig,axes = plt.subplots(2,2,figsize=(12,10))
fig.suptitle("Iris数据集特征分布",fontsize=16)

for i,feature in enumerate(iris.feature_names):
    row,col = i//2,i%2
    axes[row,col].hist(X[:,i],bins=20,alpha=0.3)
    axes[row,col].set_title(f'{feature}分布')
    axes[row,col].set_xlabel(feature)
    axes[row,col].set_ylabel('频次')

plt.tight_layout()
plt.show()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

print(f"训练集大小:{X_train.shape}")
print(f"测试集大小:{X_test.shape}")
print(f"训练集标准化后:{X_train_scaler[:5]}")

# 定义模型
models = {
    '逻辑回归':LogisticRegression(random_state = 42),
    '决策树':DecisionTreeClassifier(random_state = 42),
    '随机森林':RandomForestClassifier(random_state = 42,n_estimators=100)
}

# 训练模型
for name,model in models.items():
    print(f"训练模型:{name}")
    model.fit(X_train_scaler,y_train)
    y_pred = model.predict(X_test_scaler)
    accuracy = (y_pred == y_test).mean()
    print(f"准确率:{accuracy}")
