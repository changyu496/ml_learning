# 第6天微积分基础 - 复习强化计划

## 🎯 学习目标
**将理论理解转化为实际操作能力**

---

## ⏰ 时间安排 (50分钟)

### 📚 第一部分：快速复习 (10分钟)
**目标**: 巩固昨天的理论理解

#### 任务清单
- [ ] **导数公式默写** (3分钟)
  - 写出 x^n, e^x, ln(x), sin(x), cos(x) 的导数
  - 写出四个运算法则：加减、常数倍、乘积、链式
  
- [ ] **梯度下降步骤回顾** (3分钟)
  - 步骤1: 在当前点计算梯度
  - 步骤2: 乘以学习率得到更新步长
  - 步骤3: 更新参数位置
  - 步骤4: 重复直到收敛
  
- [ ] **关键概念确认** (4分钟)
  - 梯度向量的含义
  - 学习率的作用
  - 随机初始化的原因

---

### ✏️ 第二部分：手工计算强化 (15分钟)
**目标**: 提高计算速度和准确性

#### 练习1：基本导数计算 (4分钟)
计算下列函数的导数：
1. f(x) = 2x³ - 5x² + 3x - 1
2. g(x) = e^x + ln(x) + sin(x)
3. h(x) = (x² + 1)³
4. k(x) = x·e^x
5. m(x) = sin(2x + 1)

#### 练习2：偏导数计算 (5分钟)
计算下列函数的偏导数：
1. f(x,y) = x²y + xy² - 2x + 3y
   - 求 ∂f/∂x 和 ∂f/∂y
2. g(x,y) = e^(x+y) + x²y³
   - 求 ∂g/∂x 和 ∂g/∂y
3. h(x,y) = sin(xy) + x²
   - 求 ∂h/∂x 和 ∂h/∂y

#### 练习3：梯度向量计算 (3分钟)
对于函数 L(w₁, w₂) = (w₁-2)² + (w₂+1)²
在以下点计算梯度向量：
1. 点 (0, 0)
2. 点 (1, -0.5)
3. 点 (2, -1)

#### 练习4：手工梯度下降 (3分钟)
使用函数 f(x,y) = x² + y² - 2x - 4y + 5
- 起始点: (0, 0)
- 学习率: 0.1
- 手工计算前3步的梯度下降过程

---

### 💻 第三部分：编程实践 (20分钟)
**目标**: 独立实现梯度下降算法

#### 任务1：基础梯度下降实现 (8分钟)
```python
# 不看答案，独立实现以下函数
def simple_gradient_descent(start_point, learning_rate, num_iterations):
    """
    实现简单的梯度下降算法
    函数: f(x,y) = (x-3)² + (y-1)²
    """
    # TODO: 你的实现
    pass
    

# 测试你的实现
result = simple_gradient_descent([0, 0], 0.1, 20)
print(f"最终结果: {result}")
print(f"期望结果: [3, 1]")
```

#### 任务2：不同学习率对比 (6分钟)
```python
# 测试学习率 [0.01, 0.1, 0.5, 0.9] 的效果
# 观察收敛速度和稳定性
learning_rates = [0.01, 0.1, 0.5, 0.9]
for lr in learning_rates:
    # TODO: 实现对比实验
    pass
```

#### 任务3：不同起始点对比 (6分钟)
```python
# 测试不同起始点的收敛结果
start_points = [[0,0], [5,5], [-2,3], [1,-1]]
for start in start_points:
    # TODO: 实现对比实验
    pass
```

---

### 📊 第四部分：图形化练习 (5分钟)
**目标**: 熟练使用matplotlib绘图

#### 任务1：函数和导数对比图 (5分钟)
```python
# 绘制函数 f(x) = x³ - 3x² + 2x 和其导数
# 要求：
# 1. 创建1行2列的子图
# 2. 左图显示原函数
# 3. 右图显示导数函数
# 4. 标记导数为0的点
# 5. 添加网格和标签
```

---

## 🎯 成功标准

### 💪 能力检验
完成所有练习后，你应该能够：

#### 手工计算能力
- [ ] 5分钟内计算出5个基本导数，准确率90%以上
- [ ] 快速计算偏导数，无需查阅公式
- [ ] 手工执行3步梯度下降过程

#### 编程实现能力
- [ ] 独立写出梯度下降的核心代码
- [ ] 理解不同学习率的影响
- [ ] 能够调试和优化算法

#### 图形化能力
- [ ] 创建包含子图的完整图形
- [ ] 标记特殊点和添加注释
- [ ] 美化图表（网格、标签、标题）

#### 理解深度
- [ ] 能够解释每一步的数学原理
- [ ] 理解梯度下降的几何意义
- [ ] 知道如何调整超参数

---

## 🔍 自我检验方法

### 计算检验
```
练习1答案：
1. f'(x) = 6x² - 10x + 3
2. g'(x) = e^x + 1/x + cos(x)
3. h'(x) = 3(x²+1)²·2x = 6x(x²+1)²
4. k'(x) = e^x + x·e^x = e^x(1+x)
5. m'(x) = 2cos(2x+1)
```

### 编程检验
```python
# 测试你的梯度下降实现
def test_gradient_descent():
    result = your_gradient_descent([0, 0], 0.1, 50)
    expected = [3, 1]
    error = abs(result[0] - expected[0]) + abs(result[1] - expected[1])
    print(f"误差: {error}")
    return error < 0.01  # 误差小于0.01认为成功
```

---

## 📝 学习记录表

### 完成情况记录
- [ ] 第一部分：快速复习 ___/10分钟
- [ ] 第二部分：手工计算 ___/15分钟
- [ ] 第三部分：编程实践 ___/20分钟
- [ ] 第四部分：图形化练习 ___/5分钟

### 困难记录
**遇到的问题**：
1. _________________________________
2. _________________________________
3. _________________________________

**解决方案**：
1. _________________________________
2. _________________________________
3. _________________________________

### 收获记录
**新的理解**：
1. _________________________________
2. _________________________________
3. _________________________________

---

## 🚀 完成后的下一步

### 如果全部完成且理解良好
- 可以开始学习第7天内容：线性代数进阶
- 重点关注特征值、特征向量、PCA等概念

### 如果还有困难
- 重复今天的练习，特别是困难的部分
- 寻找更多相关练习题
- 重新阅读理论材料

### 长期目标
- 为后续的神经网络学习打下坚实基础
- 理解反向传播算法的数学原理
- 掌握各种优化算法的核心思想

---

**今日座右铭**: "理论要理解，实践出真知！" 