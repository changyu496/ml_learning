#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量加法和减法的实际应用
重点：平均画像和用户特征分析
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def vector_operations_real_world():
    print("🎯 向量加法和减法的实际应用")
    print("=" * 50)
    
    # 用户购买数据：[3C数码, 服装, 食品, 图书, 运动用品]
    categories = ['3C数码', '服装', '食品', '图书', '运动用品']
    
    # 多个用户的购买次数
    用户数据 = np.array([
        [10, 2, 5, 8, 3],   # 用户1：IT男，爱买数码产品和书
        [12, 1, 4, 9, 2],   # 用户2：程序员，类似用户1
        [3, 15, 8, 2, 6],   # 用户3：时尚女性，爱买衣服
        [2, 18, 7, 1, 4],   # 用户4：时尚女性，类似用户3
        [5, 8, 12, 5, 15],  # 用户5：运动达人，爱健身和美食
        [4, 6, 14, 3, 18],  # 用户6：运动达人，类似用户5
    ])
    
    用户名 = ['用户1', '用户2', '用户3', '用户4', '用户5', '用户6']
    
    print("📊 用户购买数据（每月购买次数）:")
    print("用户\t", end="")
    for cat in categories:
        print(f"{cat}\t", end="")
    print()
    print("-" * 70)
    
    for i, name in enumerate(用户名):
        print(f"{name}\t", end="")
        for purchase in 用户数据[i]:
            print(f"{purchase}\t\t", end="")
        print()
    
    print(f"\n" + "="*50)
    print(f"🧮 向量加法：计算平均画像")
    print(f"="*50)
    
    # 1. 向量加法：计算平均用户画像
    总和向量 = np.sum(用户数据, axis=0)
    平均画像 = 总和向量 / len(用户数据)
    
    print(f"\n📈 计算过程:")
    print(f"所有用户总和: {总和向量}")
    print(f"用户数量: {len(用户数据)}")
    print(f"平均用户画像: {平均画像}")
    print(f"平均用户画像（四舍五入）: {np.round(平均画像, 1)}")
    
    print(f"\n💡 平均画像的含义:")
    for i, cat in enumerate(categories):
        print(f"- {cat}: 平均每月购买 {平均画像[i]:.1f} 次")
    
    print(f"\n🎯 平均画像的实际应用:")
    print(f"1. 新用户推荐：按平均画像推荐商品")
    print(f"2. 异常检测：偏离平均画像太远的用户")
    print(f"3. 用户细分：根据与平均画像的差异分类")
    print(f"4. 库存管理：按平均需求准备库存")
    
    print(f"\n" + "="*50)
    print(f"➖ 向量减法：与平均画像比较")
    print(f"="*50)
    
    # 2. 向量减法：分析每个用户与平均画像的差异
    print(f"\n📊 每个用户与平均画像的差异:")
    print(f"用户\t差异向量\t\t\t\t用户特征")
    print(f"-" * 80)
    
    用户特征 = []
    for i, name in enumerate(用户名):
        差异向量 = 用户数据[i] - 平均画像
        
        print(f"{name}\t{np.round(差异向量, 1)}\t", end="")
        
        # 分析用户特征
        max_diff_idx = np.argmax(np.abs(差异向量))
        if 差异向量[max_diff_idx] > 2:
            特征 = f"偏爱{categories[max_diff_idx]}"
        elif 差异向量[max_diff_idx] < -2:
            特征 = f"不喜欢{categories[max_diff_idx]}"
        else:
            特征 = "接近平均用户"
        
        用户特征.append(特征)
        print(f"\t{特征}")
    
    print(f"\n🔍 差异分析的实际应用:")
    
    # 具体案例分析
    print(f"\n📈 具体分析:")
    
    # 分析用户1
    用户1差异 = 用户数据[0] - 平均画像
    print(f"\n👤 用户1差异分析:")
    print(f"原始数据: {用户数据[0]}")
    print(f"平均画像: {np.round(平均画像, 1)}")
    print(f"差异向量: {np.round(用户1差异, 1)}")
    
    for i, cat in enumerate(categories):
        diff = 用户1差异[i]
        if diff > 2:
            print(f"- {cat}: +{diff:.1f} → 比平均用户更爱买")
        elif diff < -2:
            print(f"- {cat}: {diff:.1f} → 比平均用户更少买")
    
    print(f"\n🎯 基于差异的推荐策略:")
    print(f"1. 正差异大的类别 → 推荐更多此类商品")
    print(f"2. 负差异大的类别 → 避免推荐此类商品")
    print(f"3. 接近0的类别 → 按平均推荐策略")
    
    # 实际业务应用
    print(f"\n" + "="*50)
    print(f"🚀 实际业务应用场景")
    print(f"="*50)
    
    print(f"\n📦 1. 电商平台应用:")
    print(f"向量加法: 计算商品品类的平均购买量")
    print(f"向量减法: 识别用户偏好与一般用户的差异")
    print(f"应用: 个性化推荐、库存管理、用户画像")
    
    print(f"\n🎵 2. 音乐平台应用:")
    print(f"向量加法: 计算各音乐类型的平均收听时长")
    print(f"向量减法: 识别用户音乐偏好的独特性")
    print(f"应用: 个性化歌单、新人推荐、音乐发现")
    
    print(f"\n🎬 3. 视频平台应用:")
    print(f"向量加法: 计算各类视频的平均观看时长")
    print(f"向量减法: 识别用户观看习惯的特殊性")
    print(f"应用: 个性化推荐、内容制作指导")
    
    print(f"\n💡 关键理解:")
    print(f"- 向量加法 = 求平均、找基准、建立标准画像")
    print(f"- 向量减法 = 找差异、识别特征、个性化分析")
    print(f"- 平均画像 = 业务基准线，用于比较和推荐")

def visualize_user_analysis():
    """可视化用户画像分析"""
    print(f"\n📊 生成用户画像可视化...")
    
    # 示例数据
    categories = ['3C数码', '服装', '食品', '图书', '运动用品']
    用户数据 = np.array([
        [10, 2, 5, 8, 3],   # 用户1
        [3, 15, 8, 2, 6],   # 用户3
        [5, 8, 12, 5, 15],  # 用户5
    ])
    
    用户名 = ['用户1(IT男)', '用户3(时尚女)', '用户5(运动达人)']
    平均画像 = np.mean(用户数据, axis=0)
    
    # 创建雷达图
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # 闭合图形
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    # 绘制平均画像
    平均数据 = np.concatenate((平均画像, [平均画像[0]]))
    ax.plot(angles, 平均数据, 'o-', linewidth=2, label='平均画像', color='gray')
    ax.fill(angles, 平均数据, alpha=0.25, color='gray')
    
    # 绘制各用户
    colors = ['red', 'blue', 'green']
    for i, (name, data) in enumerate(zip(用户名, 用户数据)):
        用户数据_闭合 = np.concatenate((data, [data[0]]))
        ax.plot(angles, 用户数据_闭合, 'o-', linewidth=2, label=name, color=colors[i])
        ax.fill(angles, 用户数据_闭合, alpha=0.25, color=colors[i])
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 20)
    ax.set_title('用户购买偏好雷达图', size=16, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.show()
    
    print(f"图表说明:")
    print(f"- 灰色区域：平均用户画像")
    print(f"- 彩色区域：具体用户偏好")
    print(f"- 偏离灰色区域越远 = 个性化程度越高")

if __name__ == "__main__":
    vector_operations_real_world()
    visualize_user_analysis() 