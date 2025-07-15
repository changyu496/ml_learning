#!/usr/bin/env python3
"""
概念深度解析 - HTML转换脚本
将Markdown文件转换为HTML格式，便于在手机/平板上阅读
"""

import os
import markdown2
import re
from pathlib import Path

def create_html_css():
    """创建HTML样式"""
    return """
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }
        
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-top: 0;
        }
        
        h2 {
            color: #34495e;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 5px;
            margin-top: 30px;
        }
        
        h3 {
            color: #2c3e50;
            margin-top: 25px;
        }
        
        blockquote {
            border-left: 4px solid #3498db;
            background: #f8f9fa;
            margin: 20px 0;
            padding: 15px 20px;
            font-style: italic;
            color: #2c3e50;
        }
        
        code {
            background: #f1f2f6;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', monospace;
            color: #e74c3c;
        }
        
        pre {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            overflow-x: auto;
            font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', monospace;
            font-size: 14px;
            line-height: 1.4;
        }
        
        pre code {
            background: none;
            color: inherit;
            padding: 0;
        }
        
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        
        th {
            background-color: #3498db;
            color: white;
        }
        
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        
        .toc {
            background: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }
        
        .toc h3 {
            margin-top: 0;
            color: #2c3e50;
        }
        
        .toc ul {
            list-style-type: none;
            padding-left: 0;
        }
        
        .toc li {
            margin: 5px 0;
        }
        
        .toc a {
            color: #3498db;
            text-decoration: none;
        }
        
        .toc a:hover {
            text-decoration: underline;
        }
        
        .nav-buttons {
            display: flex;
            justify-content: space-between;
            margin: 30px 0;
            flex-wrap: wrap;
        }
        
        .nav-button {
            background: #3498db;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            text-decoration: none;
            display: inline-block;
            margin: 5px;
            transition: background 0.3s;
        }
        
        .nav-button:hover {
            background: #2980b9;
        }
        
        .concept-meta {
            background: #e8f5e8;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            border-left: 4px solid #27ae60;
        }
        
        .difficulty {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 12px;
            color: white;
            margin: 0 5px;
        }
        
        .easy { background: #27ae60; }
        .medium { background: #f39c12; }
        .hard { background: #e74c3c; }
        .expert { background: #8e44ad; }
        
        .importance {
            color: #f39c12;
            font-size: 16px;
        }
        
        .print-button {
            position: fixed;
            top: 20px;
            right: 20px;
            background: #27ae60;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            z-index: 1000;
        }
        
        .print-button:hover {
            background: #219a52;
        }
        
        @media (max-width: 600px) {
            body {
                padding: 10px;
            }
            
            .container {
                padding: 20px;
            }
            
            pre {
                font-size: 12px;
                padding: 15px;
            }
            
            table {
                font-size: 14px;
            }
            
            .nav-buttons {
                flex-direction: column;
            }
            
            .nav-button {
                width: 100%;
                text-align: center;
            }
        }
        
        @media print {
            body {
                background: white;
                color: black;
            }
            
            .container {
                box-shadow: none;
                padding: 0;
            }
            
            .print-button {
                display: none;
            }
            
            .nav-buttons {
                display: none;
            }
            
            pre {
                background: #f5f5f5;
                color: black;
                border: 1px solid #ddd;
            }
        }
    </style>
    """

def extract_metadata(content):
    """提取文件元数据"""
    metadata = {
        'title': '概念深度解析',
        'difficulty': 'medium',
        'time': '30分钟',
        'importance': 5
    }
    
    # 提取标题
    title_match = re.search(r'^# (.+)', content, re.MULTILINE)
    if title_match:
        metadata['title'] = title_match.group(1).strip()
    
    # 根据内容长度估计阅读时间
    word_count = len(content.split())
    reading_time = max(20, word_count // 200 * 5)  # 200词/分钟，转换为5分钟单位
    metadata['time'] = f"{reading_time}分钟"
    
    # 根据文件名判断难度
    if any(keyword in content.lower() for keyword in ['numpy', '向量化', '广播']):
        metadata['difficulty'] = 'easy'
    elif any(keyword in content.lower() for keyword in ['特征值', '梯度下降', 'pca']):
        metadata['difficulty'] = 'hard'
    elif any(keyword in content.lower() for keyword in ['深度学习', '神经网络']):
        metadata['difficulty'] = 'expert'
    
    return metadata

def create_navigation_buttons(current_file, all_files):
    """创建导航按钮"""
    navigation = '<div class="nav-buttons">\n'
    
    # 返回索引
    navigation += '<a href="index.html" class="nav-button">📚 返回索引</a>\n'
    
    # 找到当前文件的位置
    try:
        current_index = all_files.index(current_file)
        
        # 上一个文件
        if current_index > 0:
            prev_file = all_files[current_index - 1]
            prev_name = prev_file.replace('_', ' ').replace('.md', '')
            navigation += f'<a href="{prev_file.replace(".md", ".html")}" class="nav-button">⬅️ {prev_name}</a>\n'
        
        # 下一个文件
        if current_index < len(all_files) - 1:
            next_file = all_files[current_index + 1]
            next_name = next_file.replace('_', ' ').replace('.md', '')
            navigation += f'<a href="{next_file.replace(".md", ".html")}" class="nav-button">➡️ {next_name}</a>\n'
    except ValueError:
        pass
    
    navigation += '</div>\n'
    return navigation

def create_toc(content):
    """创建目录"""
    toc = '<div class="toc">\n<h3>📋 目录</h3>\n<ul>\n'
    
    # 提取标题
    headings = re.findall(r'^(#{1,3})\s+(.+)', content, re.MULTILINE)
    
    for level, title in headings:
        if len(level) == 1:  # h1
            continue
        
        # 创建锚点
        anchor = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '-').lower()
        
        # 添加到目录
        indent = '  ' * (len(level) - 2)
        toc += f'{indent}<li><a href="#{anchor}">{title}</a></li>\n'
    
    toc += '</ul>\n</div>\n'
    return toc

def add_anchors_to_headings(html_content):
    """为标题添加锚点"""
    def replace_heading(match):
        level = len(match.group(1))
        title = match.group(2)
        
        if level == 1:  # h1 不需要锚点
            return match.group(0)
        
        # 创建锚点
        anchor = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '-').lower()
        
        return f'<h{level} id="{anchor}">{title}</h{level}>'
    
    return re.sub(r'<h(\d+)>(.+?)</h\d+>', replace_heading, html_content)

def convert_md_to_html(md_file, output_dir, all_files):
    """将Markdown文件转换为HTML"""
    
    # 读取Markdown文件
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # 提取元数据
    metadata = extract_metadata(md_content)
    
    # 转换为HTML
    html_content = markdown2.markdown(md_content, extras=['fenced-code-blocks', 'tables', 'break-on-newline'])
    
    # 添加锚点
    html_content = add_anchors_to_headings(html_content)
    
    # 创建目录
    toc = create_toc(md_content)
    
    # 创建导航
    current_file = os.path.basename(md_file)
    navigation = create_navigation_buttons(current_file, all_files)
    
    # 创建元数据显示
    difficulty_class = metadata['difficulty']
    importance_stars = '⭐' * metadata['importance']
    
    meta_html = f"""
    <div class="concept-meta">
        <h3>📊 概念信息</h3>
        <p>
            <strong>难度:</strong> <span class="difficulty {difficulty_class}">{metadata['difficulty'].upper()}</span>
            <strong>预计阅读时间:</strong> {metadata['time']}
            <strong>重要性:</strong> <span class="importance">{importance_stars}</span>
        </p>
    </div>
    """
    
    # 组合完整HTML
    full_html = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{metadata['title']} - 概念深度解析</title>
        {create_html_css()}
    </head>
    <body>
        <button class="print-button" onclick="window.print()">🖨️ 打印/保存PDF</button>
        
        <div class="container">
            {navigation}
            {meta_html}
            {toc}
            {html_content}
            {navigation}
        </div>
        
        <script>
            // 平滑滚动到锚点
            document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
                anchor.addEventListener('click', function (e) {{
                    e.preventDefault();
                    document.querySelector(this.getAttribute('href')).scrollIntoView({{
                        behavior: 'smooth'
                    }});
                }});
            }});
        </script>
    </body>
    </html>
    """
    
    # 写入HTML文件
    output_file = os.path.join(output_dir, os.path.basename(md_file).replace('.md', '.html'))
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    print(f"✅ 转换完成: {os.path.basename(md_file)} -> {os.path.basename(output_file)}")

def create_index_html(output_dir, all_files):
    """创建索引页面"""
    
    # 概念文件信息
    concepts = [
        {
            'file': '01_NumPy数组与向量化运算深度解析.md',
            'title': 'NumPy数组与向量化运算深度解析',
            'description': '理解向量化运算的本质，掌握NumPy高性能计算的核心技巧',
            'difficulty': 'easy',
            'time': '25分钟',
            'importance': 5,
            'category': '基础概念'
        },
        {
            'file': '02_广播机制深度解析.md',
            'title': '广播机制深度解析',
            'description': '掌握NumPy广播规则，实现不同形状数组的高效运算',
            'difficulty': 'medium',
            'time': '30分钟',
            'importance': 5,
            'category': '基础概念'
        },
        {
            'file': '03_矩阵乘法深度解析.md',
            'title': '矩阵乘法深度解析',
            'description': '理解矩阵乘法的几何意义，掌握机器学习中的核心运算',
            'difficulty': 'medium',
            'time': '35分钟',
            'importance': 5,
            'category': '基础概念'
        },
        {
            'file': '04_特征值与特征向量深度解析.md',
            'title': '特征值与特征向量深度解析',
            'description': '深入理解特征值分解，为PCA和SVD打下坚实基础',
            'difficulty': 'hard',
            'time': '40分钟',
            'importance': 5,
            'category': '进阶概念'
        },
        {
            'file': '05_梯度下降算法详解.md',
            'title': '梯度下降算法详解',
            'description': '掌握机器学习优化算法的核心，理解参数学习的本质',
            'difficulty': 'hard',
            'time': '45分钟',
            'importance': 5,
            'category': '进阶概念'
        },
        {
            'file': '06_PCA主成分分析详解.md',
            'title': 'PCA主成分分析详解',
            'description': '理解数据降维的经典方法，掌握特征提取的艺术',
            'difficulty': 'hard',
            'time': '35分钟',
            'importance': 5,
            'category': '应用概念'
        },
        {
            'file': '07_向量相似度深度解析.md',
            'title': '向量相似度深度解析',
            'description': '掌握各种相似度度量方法，理解推荐系统的数学基础',
            'difficulty': 'medium',
            'time': '30分钟',
            'importance': 4,
            'category': '应用概念'
        }
    ]
    
    # 按类别分组
    categories = {}
    for concept in concepts:
        category = concept['category']
        if category not in categories:
            categories[category] = []
        categories[category].append(concept)
    
    # 创建索引HTML
    index_html = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>概念深度解析 - 通勤学习专用</title>
        {create_html_css()}
    </head>
    <body>
        <div class="container">
            <h1>📚 概念深度解析</h1>
            
            <div class="concept-meta">
                <h3>🎯 使用指南</h3>
                <p>欢迎使用概念深度解析系列！每个概念都包含完整的理论、代码示例和实际应用。</p>
                <ul>
                    <li><strong>🔰 基础概念</strong>：NumPy和向量化运算的核心技巧</li>
                    <li><strong>🔬 进阶概念</strong>：线性代数和优化算法的深入理解</li>
                    <li><strong>🎯 应用概念</strong>：机器学习中的实际应用案例</li>
                </ul>
            </div>
            
            <div class="toc">
                <h3>📈 学习统计</h3>
                <p>
                    <strong>总概念数：</strong> {len(concepts)} 个 &nbsp;
                    <strong>总学习时间：</strong> ~4-5小时 &nbsp;
                    <strong>难度分布：</strong> 基础(3) 进阶(2) 应用(2)
                </p>
            </div>
    """
    
    # 为每个类别创建内容
    for category, items in categories.items():
        category_icon = {
            '基础概念': '🔰',
            '进阶概念': '🔬',
            '应用概念': '🎯'
        }.get(category, '📝')
        
        index_html += f"""
            <h2>{category_icon} {category}</h2>
            <div class="nav-buttons">
        """
        
        for concept in items:
            difficulty_class = concept['difficulty']
            importance_stars = '⭐' * concept['importance']
            
            html_file = concept['file'].replace('.md', '.html')
            
            index_html += f"""
                <div style="flex: 1; margin: 10px; padding: 20px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #3498db;">
                    <h3 style="margin-top: 0;">
                        <a href="{html_file}" style="text-decoration: none; color: #2c3e50;">
                            {concept['title']}
                        </a>
                    </h3>
                    <p style="color: #7f8c8d; margin: 10px 0;">{concept['description']}</p>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 15px;">
                        <div>
                            <span class="difficulty {difficulty_class}">{concept['difficulty'].upper()}</span>
                            <span style="margin-left: 10px;">⏱️ {concept['time']}</span>
                        </div>
                        <div>
                            <span class="importance">{importance_stars}</span>
                        </div>
                    </div>
                    <div style="margin-top: 10px;">
                        <a href="{html_file}" class="nav-button" style="margin: 0;">开始学习 →</a>
                    </div>
                </div>
            """
        
        index_html += "</div>\n"
    
    # 添加学习路径建议
    index_html += f"""
            <h2>🗺️ 学习路径建议</h2>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0;">
                <div style="background: #e8f5e8; padding: 20px; border-radius: 8px; border-left: 4px solid #27ae60;">
                    <h3 style="margin-top: 0; color: #27ae60;">📚 基础路径</h3>
                    <p>适合初学者，循序渐进</p>
                    <ol style="margin: 0;">
                        <li>NumPy数组与向量化运算</li>
                        <li>广播机制深度解析</li>
                        <li>矩阵乘法深度解析</li>
                        <li>向量相似度深度解析</li>
                    </ol>
                </div>
                
                <div style="background: #fef9e7; padding: 20px; border-radius: 8px; border-left: 4px solid #f39c12;">
                    <h3 style="margin-top: 0; color: #f39c12;">🎯 机器学习路径</h3>
                    <p>专注机器学习算法</p>
                    <ol style="margin: 0;">
                        <li>NumPy数组与向量化运算</li>
                        <li>矩阵乘法深度解析</li>
                        <li>梯度下降算法详解</li>
                        <li>PCA主成分分析详解</li>
                    </ol>
                </div>
                
                <div style="background: #f4f3ff; padding: 20px; border-radius: 8px; border-left: 4px solid #8e44ad;">
                    <h3 style="margin-top: 0; color: #8e44ad;">🔬 数据科学路径</h3>
                    <p>深入数据分析技术</p>
                    <ol style="margin: 0;">
                        <li>矩阵乘法深度解析</li>
                        <li>特征值与特征向量深度解析</li>
                        <li>PCA主成分分析详解</li>
                        <li>向量相似度深度解析</li>
                    </ol>
                </div>
            </div>
            
            <div class="toc">
                <h3>💡 使用建议</h3>
                <ul>
                    <li><strong>通勤学习</strong>：每个概念都可以在20-45分钟内完成</li>
                    <li><strong>移动友好</strong>：针对手机和平板优化的界面</li>
                    <li><strong>离线使用</strong>：下载HTML文件后可离线阅读</li>
                    <li><strong>PDF导出</strong>：点击"打印/保存PDF"按钮导出PDF</li>
                    <li><strong>实践优先</strong>：理论结合代码示例，边学边练</li>
                </ul>
            </div>
            
            <div style="text-align: center; margin: 30px 0; padding: 20px; background: #e3f2fd; border-radius: 8px;">
                <h3 style="color: #1976d2; margin-bottom: 10px;">🚀 开始你的学习之旅</h3>
                <p style="color: #424242;">选择一个概念开始深入学习，每一步都将带你更接近机器学习的本质！</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # 写入索引文件
    with open(os.path.join(output_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(index_html)
    
    print("✅ 索引页面创建完成: index.html")

def main():
    """主函数"""
    print("📚 概念深度解析 - HTML转换工具")
    print("=" * 50)
    
    # 设置目录
    current_dir = Path(__file__).parent
    output_dir = current_dir / "HTMLs"
    
    # 创建输出目录
    output_dir.mkdir(exist_ok=True)
    
    # 查找所有Markdown文件
    md_files = []
    for file in current_dir.glob("*.md"):
        if file.name != "README.md":
            md_files.append(file.name)
    
    # 排序文件
    md_files.sort()
    
    print(f"📁 找到 {len(md_files)} 个概念文件")
    
    # 转换每个文件
    for md_file in md_files:
        convert_md_to_html(current_dir / md_file, output_dir, md_files)
    
    # 创建索引页面
    create_index_html(output_dir, md_files)
    
    print("\n🎉 转换完成！")
    print(f"📂 HTML文件保存在: {output_dir}")
    print(f"🌐 在浏览器中打开: {output_dir / 'index.html'}")
    print("\n💡 使用提示:")
    print("• 在手机浏览器中打开index.html开始学习")
    print("• 点击每页的'打印/保存PDF'按钮可导出PDF")
    print("• 所有文件都支持离线阅读")

if __name__ == "__main__":
    main() 