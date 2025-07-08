#!/usr/bin/env python3
"""
第4天编程练习：统计学基础（简化版）
时间：15-25分钟
目标：练习今天学的3个概念
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("📝 第4天编程练习")
print("=" * 30)
print("只有3个简单练习，轻松完成！")
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

# TODO: 计算平均值、中位数、标准差
# 提示：使用 np.mean(), np.median(), np.std()

# 你的代码：
mean_salary = np.mean(salaries)
median_salary = np.median(salaries)
std_salary = np.std(salaries)

print(f"平均薪资: {mean_salary:.1f}千元")
print(f"中位数薪资: {median_salary:.1f}千元")
print(f"薪资标准差: {std_salary:.1f}千元")

# 思考题：为什么平均值比中位数大？
print("\n🤔 思考：为什么平均值比中位数大？")
print("答案：因为有高薪员工（80千元），拉高了平均值，但中位数不受影响")

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

# TODO: 计算两个部门的平均值和标准差
# 然后分析哪个部门更稳定

# 你的代码：
mean_A = np.mean(dept_A)
mean_B = np.mean(dept_B)
std_A = np.std(dept_A)
std_B = np.std(dept_B)

print(f"\n部门A: 平均{mean_A:.1f}分, 标准差{std_A:.1f}")
print(f"部门B: 平均{mean_B:.1f}分, 标准差{std_B:.1f}")

# 分析结果
if std_A < std_B:
    print("结论: 部门A更稳定（标准差更小）")
else:
    print("结论: 部门B更稳定（标准差更小）")

print()

# ==========================================
# 练习3：正态分布应用
# ==========================================
print("🔥 练习3：正态分布应用")
print("-" * 20)

# 模拟考试成绩：平均75分，标准差12分
np.random.seed(42)
exam_scores = np.random.normal(75, 12, 100)

print("模拟100个学生的考试成绩")
print(f"平均分: {np.mean(exam_scores):.1f}")
print(f"标准差: {np.std(exam_scores):.1f}")

# TODO: 根据68-95-99.7法则，计算各个分数段的学生比例
mean_score = np.mean(exam_scores)
std_score = np.std(exam_scores)

# 你的代码：
# 计算在不同标准差范围内的学生比例
within_1_std = np.sum((exam_scores >= mean_score - std_score) & 
                      (exam_scores <= mean_score + std_score)) / len(exam_scores)
within_2_std = np.sum((exam_scores >= mean_score - 2*std_score) & 
                      (exam_scores <= mean_score + 2*std_score)) / len(exam_scores)

print(f"\n实际验证68-95-99.7法则:")
print(f"1个标准差内({mean_score-std_score:.0f}-{mean_score+std_score:.0f}分): {within_1_std:.1%}")
print(f"2个标准差内({mean_score-2*std_score:.0f}-{mean_score+2*std_score:.0f}分): {within_2_std:.1%}")

# 找出"异常"成绩（超过2个标准差）
abnormal_scores = exam_scores[(exam_scores < mean_score - 2*std_score) | 
                              (exam_scores > mean_score + 2*std_score)]
print(f"\n'异常'成绩（超过2个标准差）: {len(abnormal_scores)}个")
if len(abnormal_scores) > 0:
    print(f"具体分数: {abnormal_scores.round(1)}")

# 简单可视化
plt.figure(figsize=(10, 6))
plt.hist(exam_scores, bins=20, alpha=0.7, color='lightblue', density=True)
plt.axvline(mean_score, color='red', linestyle='-', linewidth=2, label=f'平均分: {mean_score:.1f}')
plt.axvline(mean_score - std_score, color='orange', linestyle='--', label='±1σ')
plt.axvline(mean_score + std_score, color='orange', linestyle='--')
plt.axvline(mean_score - 2*std_score, color='green', linestyle='--', label='±2σ')
plt.axvline(mean_score + 2*std_score, color='green', linestyle='--')

plt.xlabel('考试成绩')
plt.ylabel('概率密度')
plt.title('考试成绩分布')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print()

# ==========================================
# 练习总结
# ==========================================
print("🎉 练习完成！")
print("=" * 30)
print("今天你学会了：")
print("1. 计算平均值、中位数、标准差")
print("2. 比较不同数据组的特征")
print("3. 应用正态分布判断异常值")
print()
print("💡 关键收获：")
print("• 标准差帮我们理解数据的稳定性")
print("• 正态分布的68-95-99.7法则很实用")
print("• 数据分析就是用数字讲故事")
print()
print("🚀 准备好学习第5天的内容了吗？")

# ==========================================
# 可选挑战（时间充裕的话）
# ==========================================
print("\n🌟 可选挑战（时间充裕的话）")
print("-" * 20)
print("尝试分析自己的数据：")
print("1. 记录一周的睡眠时间，计算平均值和标准差")
print("2. 记录每天的步数，看看是否符合正态分布")
print("3. 分析手机使用时间的变化趋势")
print()
print("记住：统计学就在我们身边！") 