#!/usr/bin/env python3
"""
第5天编程练习：数据可视化基础
时间：15-25分钟
目标：练习4种基本图表的绘制
"""

from statistics import correlation
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("📊 第5天编程练习：数据可视化")
print("=" * 35)
print("请完成以下4个图表练习！")
print()

# ==========================================
# 练习1：折线图 - 网站访问量趋势
# ==========================================
print("📈 练习1：折线图 - 网站访问量趋势")
print("-" * 30)

# 数据：一周的网站访问量
days = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
visits = [1200, 1350, 1180, 1420, 1650, 2100, 1800]

print("一周网站访问量数据:", visits)

# TODO 1: 创建折线图
# 要求：
# - 使用 plt.plot() 创建折线图
# - 添加标题"网站一周访问量趋势"
# - 设置x轴标签为"日期"，y轴标签为"访问量"
# - 添加网格线

# 你的代码：
plt.figure(figsize=(10, 6))
plt.plot(days, visits, 'o-', linewidth=2)
plt.title('网站一周访问量趋势')
plt.xlabel('日期')
plt.ylabel('访问量')
plt.grid(True)
plt.show()


print("完成折线图练习！")
print()

# ==========================================
# 练习2：柱状图 - 产品销量对比
# ==========================================
print("📊 练习2：柱状图 - 产品销量对比")
print("-" * 30)

# 数据：不同产品的销量
products = ['产品A', '产品B', '产品C', '产品D', '产品E']
sales = [450, 320, 580, 290, 410]

print("产品销量数据:", sales)

# TODO 2: 创建柱状图
# 要求：
# - 使用 plt.bar() 创建柱状图
# - 添加标题"产品销量对比"
# - 设置x轴标签为"产品"，y轴标签为"销量"
# - 为每个柱子添加数值标签

# 你的代码：
plt.figure(figsize=(10, 6))
bars = plt.bar(products, sales)
plt.title('产品销量对比')
plt.xlabel('产品')
plt.ylabel('销量')
for bar,value in zip(bars,sales):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, f'{value}', ha='center')
plt.grid(True)
plt.show()

print("完成柱状图练习！")
print()

# ==========================================
# 练习3：散点图 - 广告投入与销售额关系
# ==========================================
print("📈 练习3：散点图 - 广告投入与销售额关系")
print("-" * 35)

# 数据：10个月的广告投入和销售额
ad_spend = [10, 15, 8, 20, 12, 18, 25, 14, 22, 16]  # 广告投入（万元）
revenue = [120, 180, 100, 240, 150, 210, 300, 160, 280, 190]  # 销售额（万元）

print("广告投入数据:", ad_spend)
print("销售额数据:", revenue)

# TODO 3: 创建散点图
# 要求：
# - 使用 plt.scatter() 创建散点图
# - 添加标题"广告投入与销售额关系"
# - 设置x轴标签为"广告投入（万元）"，y轴标签为"销售额（万元）"
# - 计算并输出相关系数

# 你的代码：
plt.figure(figsize=(10, 6))
plt.scatter(ad_spend, revenue, s=100)
plt.title('广告投入与销售额关系')
plt.xlabel('广告投入（万元）')
plt.ylabel('销售额（万元）')
plt.grid(True)
correlation = np.corrcoef(ad_spend,revenue)[0][1]
print(f"广告投入与销售额相关系数: {correlation:.3f}")
plt.show()


print("完成散点图练习！")
print()

# ==========================================
# 练习4：直方图 - 员工工作年限分布
# ==========================================
print("📊 练习4：直方图 - 员工工作年限分布")
print("-" * 30)

# 数据：50个员工的工作年限
np.random.seed(42)
work_years = np.random.exponential(5, 50)  # 指数分布，平均5年
work_years = np.clip(work_years, 0, 20)  # 限制在0-20年

print(f"员工工作年限数据（前10个）: {work_years[:10].round(1)}")
print(f"平均工作年限: {np.mean(work_years):.1f}年")

# TODO 4: 创建直方图
# 要求：
# - 使用 plt.hist() 创建直方图
# - 添加标题"员工工作年限分布"
# - 设置x轴标签为"工作年限（年）"，y轴标签为"人数"
# - 设置bins=10
# - 分析并输出不同年限段的员工数量

# 你的代码：
plt.figure(figsize=(10, 6))
plt.hist(work_years, bins=10, edgecolor='black')
plt.title('员工工作年限分布')
plt.xlabel('工作年限（年）')
plt.ylabel('人数')
plt.grid(True)
plt.show()

new_employees = np.sum(work_years < 2)
experienced = np.sum(work_years >= 10)
print(f"新员工（<2年）: {new_employees}人")
print(f"资深员工（≥10年）: {experienced}人")


print("完成直方图练习！")
print()

# ==========================================
# 练习总结
# ==========================================
print("🎉 练习完成检查")
print("=" * 20)
print("请检查你是否完成了：")
print("□ 练习1：折线图展示趋势")
print("□ 练习2：柱状图比较数据")
print("□ 练习3：散点图分析关系")
print("□ 练习4：直方图观察分布")
print()
print("💡 如果遇到困难，可以：")
print("1. 回顾notebook中的例子")
print("2. 查看下面的提示")
print("3. 问我具体问题")

# ==========================================
# 提示区域（卡住了再看）
# ==========================================
print("\n" + "="*50)
print("💡 提示区域（卡住了再看）")
print("="*50)

print("\n📈 练习1提示（折线图）:")
print("plt.figure(figsize=(10, 6))")
print("plt.plot(days, visits, 'o-', linewidth=2)")
print("plt.title('网站一周访问量趋势')")
print("plt.xlabel('日期')")
print("plt.ylabel('访问量')")
print("plt.grid(True)")
print("plt.show()")

print("\n📊 练习2提示（柱状图）:")
print("plt.figure(figsize=(10, 6))")
print("bars = plt.bar(products, sales)")
print("plt.title('产品销量对比')")
print("# 添加数值标签")
print("for bar, value in zip(bars, sales):")
print("    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,")
print("             f'{value}', ha='center')")

print("\n📈 练习3提示（散点图）:")
print("plt.figure(figsize=(10, 6))")
print("plt.scatter(ad_spend, revenue, s=100)")
print("plt.title('广告投入与销售额关系')")
print("correlation = np.corrcoef(ad_spend, revenue)[0, 1]")
print("print(f'相关系数: {correlation:.3f}')")

print("\n📊 练习4提示（直方图）:")
print("plt.figure(figsize=(10, 6))")
print("plt.hist(work_years, bins=10, edgecolor='black')")
print("plt.title('员工工作年限分布')")
print("# 分析年限段")
print("new_emp = np.sum(work_years < 2)")
print("exp_emp = np.sum(work_years >= 10)")

print("\n🎯 记住：先自己尝试，再看提示！")
print("🚀 完成后你就掌握了数据可视化的基本技能！") 