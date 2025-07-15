#!/bin/bash

# 通勤学习材料批量转换PDF脚本
# 使用方法: ./convert_to_pdf.sh

echo "📚 开始批量转换通勤学习材料为PDF格式..."
echo "=========================================="

# 检查是否安装了pandoc
if ! command -v pandoc &> /dev/null; then
    echo "❌ 未检测到pandoc，请先安装："
    echo "   macOS: brew install pandoc"
    echo "   Ubuntu: sudo apt-get install pandoc"
    echo "   Windows: 从https://pandoc.org/installing.html下载"
    exit 1
fi

# 创建PDF输出目录
mkdir -p "PDFs"

# 需要转换的文件列表
files=(
    "01_数学基础速查手册.md"
    "02_机器学习概念入门.md"
    "03_学习进度和重点回顾.md"
    "04_常见问题和答疑集锦.md"
    "05_学习清单和复习计划.md"
    "README.md"
)

# 转换每个文件
for file in "${files[@]}"; do
    if [[ -f "$file" ]]; then
        echo "🔄 正在转换: $file"
        
        # 获取文件名（不含扩展名）
        filename=$(basename "$file" .md)
        
        # 转换为PDF
        pandoc "$file" -o "PDFs/${filename}.pdf" \
            --pdf-engine=xelatex \
            --variable mainfont="PingFang SC" \
            --variable sansfont="PingFang SC" \
            --variable monofont="Monaco" \
            --variable geometry:margin=1in \
            --variable fontsize=12pt \
            --variable colorlinks=true \
            --variable linkcolor=blue \
            --variable urlcolor=blue \
            --variable toccolor=gray \
            --toc \
            --toc-depth=2 \
            --highlight-style=tango \
            --standalone
        
        if [[ $? -eq 0 ]]; then
            echo "✅ 转换成功: PDFs/${filename}.pdf"
        else
            echo "❌ 转换失败: $file"
        fi
    else
        echo "⚠️  文件不存在: $file"
    fi
done

echo "=========================================="
echo "📱 PDF转换完成！"
echo ""
echo "📁 输出目录: PDFs/"
echo "📖 包含文件:"
ls -la PDFs/

echo ""
echo "💡 使用建议:"
echo "1. 将PDF文件同步到手机或iPad"
echo "2. 推荐使用支持标注的PDF阅读器"
echo "3. 根据通勤时间长度选择合适的材料"
echo "4. 定期更新学习进度"
echo ""
echo "🚀 开始你的通勤学习之旅吧！" 