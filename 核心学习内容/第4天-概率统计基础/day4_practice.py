#!/usr/bin/env python3
"""
第4天编程练习：统计学基础（练习模板）
时间：15-25分钟
目标：练习今天学的3个概念

请在标记的地方填写代码！
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("📝 第4天编程练习")
print("=" * 30)
print("请在TODO标记处填写你的代码！")
print()

# ==========================================
# 练习1：计算基本统计量
# ==========================================
print("🔥 练习1：计算基本统计量")
print("-" * 20)

# 数据：某公司员工月薪（单位：千元）
salaries = [8, 12, 15, 18, 22, 25, 28, 35, 45, 80]

print("员工月薪数据:", salaries)
print()

# TODO 1.1: 计算平均值
# 提示：使用 np.mean(salaries)
mean_salary = 0  # 替换这行代码

# TODO 1.2: 计算中位数
# 提示：使用 np.median(salaries)
median_salary = 0  # 替换这行代码

# TODO 1.3: 计算标准差
# 提示：使用 np.std(salaries)
std_salary = 0  # 替换这行代码

print(f"平均薪资: {mean_salary:.1f}千元")
print(f"中位数薪资: {median_salary:.1f}千元")
print(f"薪资标准差: {std_salary:.1f}千元")

# 思考题：为什么平均值比中位数大？
print("\n🤔 思考：为什么平均值比中位数大？")
print("你的答案：")  
# TODO 1.4: 在这里写出你的理解

print()

# ==========================================
# 练习2：比较两组数据
# ==========================================
print("🔥 练习2：比较两组数据")
print("-" * 20)

# 两个部门的工作满意度评分（1-10分）
dept_A = [7, 8, 7, 9, 8, 7, 8, 9, 8, 7]  # 稳定部门
dept_B = [5, 9, 6, 10, 4, 8, 3, 9, 7, 9]  # 波动部门

print("部门A满意度:", dept_A)
print("部门B满意度:", dept_B)

# TODO 2.1: 计算部门A的平均值和标准差
mean_A = 0  # 替换这行代码
std_A = 0   # 替换这行代码

# TODO 2.2: 计算部门B的平均值和标准差
mean_B = 0  # 替换这行代码
std_B = 0   # 替换这行代码

print(f"\n部门A: 平均{mean_A:.1f}分, 标准差{std_A:.1f}")
print(f"部门B: 平均{mean_B:.1f}分, 标准差{std_B:.1f}")

# TODO 2.3: 分析哪个部门更稳定
print("\n你的分析：")
# 在这里写出你的分析

print()

# ==========================================
# 练习3：正态分布应用
# ==========================================
print("🔥 练习3：正态分布应用")
print("-" * 20)

# 模拟考试成绩：平均75分，标准差12分
np.random.seed(42)  # 固定随机种子，确保结果一致
exam_scores = np.random.normal(75, 12, 100)

print("模拟100个学生的考试成绩")
mean_score = np.mean(exam_scores)
std_score = np.std(exam_scores)
print(f"平均分: {mean_score:.1f}")
print(f"标准差: {std_score:.1f}")

# TODO 3.1: 计算1个标准差范围内的学生比例
# 提示：
# 1. 先计算条件：(exam_scores >= mean_score - std_score) & (exam_scores <= mean_score + std_score)
# 2. 用 np.sum() 计算满足条件的个数
# 3. 除以总数 len(exam_scores) 得到比例

within_1_std = 0  # 替换这行代码

# TODO 3.2: 计算2个标准差范围内的学生比例
# 提示：类似上面，但是用 2*std_score

within_2_std = 0  # 替换这行代码

print(f"\n验证68-95-99.7法则:")
print(f"1个标准差内: {within_1_std:.1%} (理论值68%)")
print(f"2个标准差内: {within_2_std:.1%} (理论值95%)")

# TODO 3.3: 找出"异常"成绩（超过2个标准差的成绩）
# 提示：使用条件 (exam_scores < mean_score - 2*std_score) | (exam_scores > mean_score + 2*std_score)

abnormal_scores = []  # 替换这行代码

print(f"\n'异常'成绩个数: {len(abnormal_scores)}")
if len(abnormal_scores) > 0:
    print(f"具体异常分数: {abnormal_scores.round(1)}")

print()

# ==========================================
# 练习完成检查
# ==========================================
print("🎉 练习完成检查")
print("=" * 30)
print("请检查你是否完成了：")
print("□ 练习1：计算了平均值、中位数、标准差")
print("□ 练习2：比较了两组数据的稳定性")
print("□ 练习3：应用了正态分布找异常值")
print()
print("💡 如果遇到困难，可以：")
print("1. 回顾notebook中的例子")
print("2. 查看下面的提示")
print("3. 问我具体的问题")

# ==========================================
# 提示区域（如果卡住了可以参考）
# ==========================================
print("\n" + "="*50)
print("💡 提示区域（卡住了再看）")
print("="*50)

print("\n📝 练习1提示:")
print("mean_salary = np.mean(salaries)")
print("median_salary = np.median(salaries)")
print("std_salary = np.std(salaries)")

print("\n📝 练习2提示:")
print("mean_A = np.mean(dept_A)")
print("std_A = np.std(dept_A)")
print("# 标准差小的部门更稳定")

print("\n📝 练习3提示:")
print("# 1个标准差内的比例:")
print("condition = (exam_scores >= mean_score - std_score) & (exam_scores <= mean_score + std_score)")
print("within_1_std = np.sum(condition) / len(exam_scores)")
print()
print("# 异常值:")
print("abnormal_scores = exam_scores[(exam_scores < mean_score - 2*std_score) | (exam_scores > mean_score + 2*std_score)]")

print("\n🎯 记住：先自己尝试，再看提示！")
print("🚀 完成后你就掌握了统计学的核心技能！") 