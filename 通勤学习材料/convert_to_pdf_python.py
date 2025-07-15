#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import markdown2
import pdfkit
from pathlib import Path

def convert_md_to_pdf():
    """将Markdown文件转换为PDF"""
    
    print("📚 开始使用Python转换通勤学习材料为PDF格式...")
    print("=" * 50)
    
    # 创建PDF输出目录
    pdf_dir = Path("PDFs")
    pdf_dir.mkdir(exist_ok=True)
    
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
    html_template = """
    <!DOCTYPE html>
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
            code {{
                background-color: #f4f4f4;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Monaco', 'Menlo', monospace;
            }}
            pre {{
                background-color: #f4f4f4;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
            }}
            blockquote {{
                border-left: 4px solid #3498db;
                margin: 0;
                padding-left: 20px;
                color: #555;
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
            }}
            .emoji {{
                font-size: 1.2em;
            }}
        </style>
    </head>
    <body>
        {content}
    </body>
    </html>
    """
    
    # 转换选项
    options = {
        'page-size': 'A4',
        'margin-top': '0.75in',
        'margin-right': '0.75in',
        'margin-bottom': '0.75in',
        'margin-left': '0.75in',
        'encoding': "UTF-8",
        'no-outline': None,
        'enable-local-file-access': None
    }
    
    success_count = 0
    
    for file_name in files:
        file_path = Path(file_name)
        
        if file_path.exists():
            print(f"🔄 正在转换: {file_name}")
            
            try:
                # 读取markdown文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    md_content = f.read()
                
                # 转换为HTML
                html_content = markdown2.markdown(md_content, extras=['tables', 'code-friendly'])
                
                # 获取标题
                title = file_path.stem
                
                # 生成完整HTML
                full_html = html_template.format(title=title, content=html_content)
                
                # 输出PDF文件路径
                pdf_path = pdf_dir / f"{title}.pdf"
                
                # 转换为PDF
                pdfkit.from_string(full_html, str(pdf_path), options=options)
                
                print(f"✅ 转换成功: {pdf_path}")
                success_count += 1
                
            except Exception as e:
                print(f"❌ 转换失败: {file_name} - {str(e)}")
                
        else:
            print(f"⚠️  文件不存在: {file_name}")
    
    print("=" * 50)
    print(f"📱 PDF转换完成！成功转换 {success_count} 个文件")
    print(f"📁 输出目录: {pdf_dir}")
    
    # 显示生成的文件
    if pdf_dir.exists():
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if pdf_files:
            print("📖 生成的PDF文件:")
            for pdf_file in pdf_files:
                print(f"  - {pdf_file.name}")
    
    print("\n💡 使用建议:")
    print("1. 将PDF文件同步到手机或iPad")
    print("2. 推荐使用支持标注的PDF阅读器")
    print("3. 根据通勤时间长度选择合适的材料")
    print("4. 定期更新学习进度")
    print("\n🚀 开始你的通勤学习之旅吧！")

if __name__ == "__main__":
    try:
        convert_md_to_pdf()
    except ImportError as e:
        print("❌ 缺少依赖包，请先安装:")
        print("pip install markdown2 pdfkit")
        print("注意：pdfkit还需要安装wkhtmltopdf")
    except Exception as e:
        print(f"❌ 转换过程中出现错误: {str(e)}")
        print("💡 建议使用在线转换工具或其他方法") 