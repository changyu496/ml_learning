#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点积 → 相似度的直觉理解
解释为什么点积能表示相似度
"""

import numpy as np

def explain_dot_product_similarity():
    print("🎯 点积 → 相似度的直觉理解")
    print("=" * 50)
    
    # 电影评分例子
    movies = ['动作片', '喜剧片', '科幻片', '爱情片', '恐怖片']
    
    # 三个用户的评分
    张三 = np.array([5, 3, 4, 2, 1])  # 喜欢动作片、科幻片
    李四 = np.array([4, 3, 5, 2, 2])  # 也喜欢动作片、科幻片  
    王五 = np.array([1, 2, 1, 4, 5])  # 喜欢爱情片、恐怖片
    
    print("📊 用户评分数据:")
    print(f"张三: {张三} (喜欢动作片、科幻片)")
    print(f"李四: {李四} (喜欢动作片、科幻片)")  
    print(f"王五: {王五} (喜欢爱情片、恐怖片)")
    
    print(f"\n🧮 计算点积:")
    
    # 计算张三与李四的点积
    dot_张三_李四 = np.dot(张三, 李四)
    print(f"\n张三 · 李四:")
    print(f"= {张三[0]}×{李四[0]} + {张三[1]}×{李四[1]} + {张三[2]}×{李四[2]} + {张三[3]}×{李四[3]} + {张三[4]}×{李四[4]}")
    print(f"= {张三[0]*李四[0]} + {张三[1]*李四[1]} + {张三[2]*李四[2]} + {张三[3]*李四[3]} + {张三[4]*李四[4]}")
    print(f"= {dot_张三_李四}")
    
    # 计算张三与王五的点积
    dot_张三_王五 = np.dot(张三, 王五)
    print(f"\n张三 · 王五:")
    print(f"= {张三[0]}×{王五[0]} + {张三[1]}×{王五[1]} + {张三[2]}×{王五[2]} + {张三[3]}×{王五[3]} + {张三[4]}×{王五[4]}")
    print(f"= {张三[0]*王五[0]} + {张三[1]*王五[1]} + {张三[2]*王五[2]} + {张三[3]*王五[3]} + {张三[4]*王五[4]}")
    print(f"= {dot_张三_王五}")
    
    print(f"\n💡 关键发现:")
    print(f"张三 · 李四 = {dot_张三_李四} (较大)")
    print(f"张三 · 王五 = {dot_张三_王五} (较小)")
    print(f"结论: 张三和李四更相似！")
    
    print(f"\n🤔 为什么点积大 = 相似度高？")
    
    # 详细分析每一项的贡献
    print(f"\n📈 逐项分析:")
    print(f"电影类型\t张三×李四\t张三×王五\t解释")
    print(f"-" * 60)
    
    explanations = [
        "都喜欢动作片：高分×高分=大贡献",
        "都一般喜欢喜剧片：中分×中分=中贡献", 
        "都喜欢科幻片：高分×高分=大贡献",
        "都不太喜欢爱情片：低分×低分=小贡献",
        "都不喜欢恐怖片：低分×低分=小贡献"
    ]
    
    explanations_different = [
        "张三喜欢但王五不喜欢：高分×低分=小贡献",
        "都一般：中分×低分=小贡献",
        "张三喜欢但王五不喜欢：高分×低分=小贡献", 
        "张三不喜欢但王五喜欢：低分×高分=小贡献",
        "张三不喜欢但王五很喜欢：低分×高分=小贡献"
    ]
    
    for i, movie in enumerate(movies):
        contrib_similar = 张三[i] * 李四[i]
        contrib_different = 张三[i] * 王五[i]
        
        print(f"{movie}\t\t{contrib_similar}\t\t{contrib_different}\t\t", end="")
        
        if i < len(explanations):
            if contrib_similar > contrib_different:
                print("✅ 相似偏好")
            else:
                print("❌ 不同偏好")
    
    print(f"\n🎯 核心原理:")
    print(f"1. 相同偏好: 高分×高分 = 大贡献")
    print(f"2. 相同偏好: 低分×低分 = 小但正向贡献") 
    print(f"3. 不同偏好: 高分×低分 = 很小贡献")
    print(f"4. 总和越大 = 共同偏好越多 = 越相似")
    
    # 极端例子
    print(f"\n🔥 极端例子验证:")
    
    # 完全相同的用户
    用户A = np.array([5, 4, 3, 2, 1])
    用户B = np.array([5, 4, 3, 2, 1])  # 完全相同
    用户C = np.array([1, 2, 3, 4, 5])  # 完全相反
    
    dot_AB = np.dot(用户A, 用户B)
    dot_AC = np.dot(用户A, 用户C)
    
    print(f"用户A: {用户A}")
    print(f"用户B: {用户B} (完全相同)")
    print(f"用户C: {用户C} (完全相反)")
    print(f"")
    print(f"A·B = {dot_AB} (完全相同 → 点积最大)")
    print(f"A·C = {dot_AC} (完全相反 → 点积较小)")
    
    print(f"\n✨ 总结:")
    print(f"点积 = 共同偏好的累计得分")
    print(f"点积越大 = 共同偏好越多 = 用户越相似")
    print(f"这就是推荐系统的数学基础！")

if __name__ == "__main__":
    explain_dot_product_similarity() 