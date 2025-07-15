#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import markdown2
from pathlib import Path

def convert_md_to_html():
    """将Markdown文件转换为HTML，然后可以在浏览器中保存为PDF"""
    
    print("📚 开始转换通勤学习材料为HTML格式...")
    print("=" * 50)
    
    # 创建HTML输出目录
    html_dir = Path("HTMLs")
    html_dir.mkdir(exist_ok=True)
    
    # 需要转换的文件列表
    files = [
        "01_数学基础速查手册.md",
        "02_机器学习概念入门.md", 
        "03_学习进度和重点回顾.md",
        "04_常见问题和答疑集锦.md",
        "05_学习清单和复习计划.md",
        "README.md",
        "使用说明.md"
    ]
    
    # HTML模板
    html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'PingFang SC', 'Microsoft YaHei', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        h1 {{
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            border-bottom: 1px solid #ecf0f1;
            padding-bottom: 5px;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.9em;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            border-left: 4px solid #3498db;
        }}
        pre code {{
            background-color: transparent;
            padding: 0;
        }}
        blockquote {{
            border-left: 4px solid #3498db;
            margin: 0;
            padding-left: 20px;
            color: #555;
            font-style: italic;
            background-color: #f8f9fa;
            padding: 10px 20px;
            border-radius: 0 5px 5px 0;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        ul, ol {{
            padding-left: 20px;
        }}
        li {{
            margin: 5px 0;
        }}
        .emoji {{
            font-size: 1.2em;
        }}
        .print-button {{
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 20px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }}
        .print-button:hover {{
            background-color: #2980b9;
        }}
        @media print {{
            .print-button {{
                display: none;
            }}
            body {{
                margin: 0;
                padding: 15px;
            }}
        }}
    </style>
</head>
<body>
    <button class="print-button" onclick="window.print()">保存为PDF</button>
    {content}
</body>
</html>"""
    
    success_count = 0
    generated_files = []
    
    for file_name in files:
        file_path = Path(file_name)
        
        if file_path.exists():
            print(f"🔄 正在转换: {file_name}")
            
            try:
                # 读取markdown文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    md_content = f.read()
                
                # 转换为HTML
                html_content = markdown2.markdown(md_content, extras=['tables', 'code-friendly', 'fenced-code-blocks'])
                
                # 获取标题
                title = file_path.stem
                
                # 生成完整HTML
                full_html = html_template.format(title=title, content=html_content)
                
                # 输出HTML文件路径
                html_path = html_dir / f"{title}.html"
                
                # 保存HTML文件
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(full_html)
                
                print(f"✅ 转换成功: {html_path}")
                generated_files.append(html_path)
                success_count += 1
                
            except Exception as e:
                print(f"❌ 转换失败: {file_name} - {str(e)}")
                
        else:
            print(f"⚠️  文件不存在: {file_name}")
    
    print("=" * 50)
    print(f"📱 HTML转换完成！成功转换 {success_count} 个文件")
    print(f"📁 输出目录: {html_dir}")
    
    # 显示生成的文件
    if generated_files:
        print("📖 生成的HTML文件:")
        for html_file in generated_files:
            print(f"  - {html_file.name}")
    
    print("\n💡 转换为PDF的方法:")
    print("1. 在浏览器中打开HTML文件")
    print("2. 点击页面右上角的'保存为PDF'按钮")
    print("3. 或者使用浏览器的打印功能 (Cmd+P)，选择'另存为PDF'")
    print("\n🔧 快速打开方法:")
    print("- 双击HTML文件在默认浏览器中打开")
    print("- 或者右键选择'用浏览器打开'")
    
    print("\n📱 使用建议:")
    print("1. 转换为PDF后同步到手机或iPad")
    print("2. 推荐使用支持标注的PDF阅读器")
    print("3. 根据通勤时间长度选择合适的材料")
    print("4. 定期更新学习进度")
    print("\n🚀 开始你的通勤学习之旅吧！")
    
    # 创建一个索引文件
    create_index_file(html_dir, generated_files)

def create_index_file(html_dir, files):
    """创建一个索引HTML文件"""
    index_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>通勤学习材料索引</title>
    <style>
        body {
            font-family: 'PingFang SC', 'Microsoft YaHei', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .file-list {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .file-item {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            background-color: #f8f9fa;
            transition: box-shadow 0.3s;
        }
        .file-item:hover {
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .file-item h3 {
            margin-top: 0;
            color: #2c3e50;
        }
        .file-item a {
            color: #3498db;
            text-decoration: none;
            font-weight: bold;
        }
        .file-item a:hover {
            text-decoration: underline;
        }
        .instructions {
            background-color: #e8f6f3;
            border-left: 4px solid #27ae60;
            padding: 15px;
            margin: 20px 0;
            border-radius: 0 5px 5px 0;
        }
    </style>
</head>
<body>
    <h1>📚 通勤学习材料索引</h1>
    
    <div class="instructions">
        <h3>📱 使用方法:</h3>
        <ol>
            <li>点击下方链接打开对应的学习材料</li>
            <li>在打开的页面中点击"保存为PDF"按钮</li>
            <li>或使用浏览器打印功能 (Cmd+P) 保存为PDF</li>
            <li>将PDF文件同步到手机或iPad即可通勤时阅读</li>
        </ol>
    </div>
    
    <div class="file-list">
"""
    
    # 文件描述
    descriptions = {
        "01_数学基础速查手册": "第1-7天数学基础的核心总结，适合快速复习",
        "02_机器学习概念入门": "ML基础概念和理论预习，为实践做准备",
        "03_学习进度和重点回顾": "个人学习报告和进度分析",
        "04_常见问题和答疑集锦": "40个常见问题的详细解答",
        "05_学习清单和复习计划": "分时间段的具体学习任务和计划",
        "README": "总索引和完整使用指南",
        "使用说明": "快速上手指南"
    }
    
    for file_path in files:
        file_name = file_path.stem
        description = descriptions.get(file_name, "学习材料")
        
        index_content += f"""
        <div class="file-item">
            <h3><a href="{file_path.name}">{file_name}</a></h3>
            <p>{description}</p>
        </div>
        """
    
    index_content += """
    </div>
    
    <div class="instructions">
        <h3>⏰ 通勤时间建议:</h3>
        <ul>
            <li><strong>5-10分钟</strong>: 快速复习清单</li>
            <li><strong>10-20分钟</strong>: 常见问题解答</li>
            <li><strong>20-30分钟</strong>: 概念入门学习</li>
            <li><strong>30分钟以上</strong>: 系统复习</li>
        </ul>
    </div>
</body>
</html>
"""
    
    index_path = html_dir / "index.html"
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_content)
    
    print(f"📄 已创建索引文件: {index_path}")

if __name__ == "__main__":
    try:
        convert_md_to_html()
    except Exception as e:
        print(f"❌ 转换过程中出现错误: {str(e)}")
        print("�� 请检查文件是否存在或联系技术支持") 