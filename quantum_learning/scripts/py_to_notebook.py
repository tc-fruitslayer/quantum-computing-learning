#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python练习文件到Jupyter Notebook的转换工具

这个脚本将量子计算学习项目中的Python练习文件转换为交互式Jupyter Notebook格式。
它会智能地识别文档字符串、注释和代码块，并将它们转换为适当的Notebook单元格。

用法:
    python py_to_notebook.py <input_python_file> [<output_notebook_file>]

示例:
    python py_to_notebook.py ../exercises/ex01_basics.py ../notebooks/ex01_basics.ipynb
"""

import sys
import os
import re
import json
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import argparse

def convert_docstring_to_markdown(docstring):
    """将Python文档字符串转换为Markdown格式"""
    # 删除开头的引号和空白
    docstring = re.sub(r'^"""|\s*"""$', '', docstring.strip())
    # 将小标题行(带有破折号的行)转换为Markdown标题
    docstring = re.sub(r'^([^\n]+)\n-+$', r'## \1', docstring, flags=re.MULTILINE)
    return docstring

def convert_comments_to_markdown(comments):
    """将Python注释转换为Markdown格式"""
    # 删除每行开头的#并保留内容
    markdown = '\n'.join(line[1:].strip() if line.strip() else '' 
                          for line in comments.split('\n'))
    return markdown

def extract_exercise_info(docstring):
    """从练习的文档字符串中提取标题和描述"""
    lines = docstring.strip().split('\n')
    title = lines[0] if lines else "练习"
    description = '\n'.join(lines[1:]) if len(lines) > 1 else ""
    return title, description

def parse_python_file(file_path):
    """解析Python文件，将其分割为Markdown和代码块"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 分割文件内容
    cells = []
    
    # 处理文件顶部的模块文档字符串
    module_docstring_match = re.search(r'^"""(.*?)"""', content, re.DOTALL)
    if module_docstring_match:
        module_docstring = module_docstring_match.group(1)
        # 添加标题和介绍
        cells.append(new_markdown_cell("# " + os.path.basename(file_path).replace('.py', '') + "\n\n" + 
                                      convert_docstring_to_markdown(module_docstring)))
        # 从内容中移除已处理的文档字符串
        content = content[module_docstring_match.end():]
    
    # 查找所有三引号文档字符串
    docstring_pattern = re.compile(r'(""".*?""")', re.DOTALL)
    
    # 查找所有连续的注释块
    comment_pattern = re.compile(r'((?:^#.*\n)+)', re.MULTILINE)
    
    # 查找import语句块
    import_pattern = re.compile(r'((?:^(?:import|from).*\n)+)', re.MULTILINE)
    
    # 标记所有需要分割的位置
    splits = []
    
    # 添加文档字符串位置
    for match in docstring_pattern.finditer(content):
        splits.append((match.start(), match.end(), 'docstring', match.group(1)))
    
    # 添加注释块位置
    for match in comment_pattern.finditer(content):
        # 忽略井号开头的shebang和编码声明
        if not match.group(1).startswith('#!') and not 'coding' in match.group(1):
            splits.append((match.start(), match.end(), 'comment', match.group(1)))
    
    # 添加import块位置
    for match in import_pattern.finditer(content):
        splits.append((match.start(), match.end(), 'import', match.group(1)))
    
    # 按位置排序
    splits.sort(key=lambda x: x[0])
    
    # 处理每个分割点
    last_end = 0
    for start, end, type_, text in splits:
        # 添加上一个分割点到当前分割点之间的代码(如果不是纯空白)
        code_between = content[last_end:start].strip()
        if code_between and not code_between.isspace():
            cells.append(new_code_cell(code_between))
        
        # 根据类型处理当前文本
        if type_ == 'docstring':
            markdown_text = convert_docstring_to_markdown(text)
            
            # 将"练习X: 标题"格式的文档字符串转换为Markdown标题
            if "练习" in markdown_text.split('\n')[0]:
                title, description = extract_exercise_info(markdown_text)
                cells.append(new_markdown_cell("## " + title + "\n\n" + description))
            else:
                cells.append(new_markdown_cell(markdown_text))
        
        elif type_ == 'comment':
            cells.append(new_markdown_cell(convert_comments_to_markdown(text)))
        
        elif type_ == 'import':
            # 将import语句包装成代码单元格
            cells.append(new_code_cell(text))
        
        last_end = end
    
    # 添加文件末尾的代码(如果有)
    final_code = content[last_end:].strip()
    if final_code and not final_code.isspace():
        cells.append(new_code_cell(final_code))
    
    return cells

def enhance_notebook_interactivity(cells):
    """增强笔记本的交互性，添加可视化和互动元素"""
    enhanced_cells = []
    
    for cell in cells:
        enhanced_cells.append(cell)
        
        # 为特定代码模式添加交互元素
        if cell['cell_type'] == 'code':
            code = cell['source']
            
            # 在代码单元后添加可视化输出单元(对包含plt的单元格)
            if 'plt.' in code and not 'plt.close' in code and not 'plt.savefig' in code:
                vis_cell = new_markdown_cell("**可视化输出:**\n\n"
                                           "运行上面的代码可以查看图形输出。调整参数以观察结果如何变化。")
                enhanced_cells.append(vis_cell)
            
            # 对于包含待完成代码的单元格，添加提示
            if '# 您的代码:' in code or '...' in code:
                hint_cell = new_markdown_cell("> **练习提示:**\n> 在上面的代码单元格中完成实现。"
                                            "可以使用`?`查看相关函数的文档，例如`qml.Hadamard?`。")
                enhanced_cells.append(hint_cell)
    
    # 在末尾添加挑战任务
    challenge_cell = new_markdown_cell("## 💡 挑战任务\n\n"
                                     "尝试扩展上面的练习，探索以下内容：\n"
                                     "1. 更改电路参数并观察结果的变化\n"
                                     "2. 尝试实现不同的量子态或算法\n"
                                     "3. 可视化更多量子测量的结果")
    enhanced_cells.append(challenge_cell)
    
    return enhanced_cells

def convert_py_to_notebook(input_file, output_file=None):
    """将Python文件转换为Jupyter Notebook格式并保存"""
    if not output_file:
        # 如果未指定输出文件，则使用相同的名称但扩展名为.ipynb
        output_file = os.path.splitext(input_file)[0] + '.ipynb'
    
    # 解析Python文件
    cells = parse_python_file(input_file)
    
    # 增强交互性
    cells = enhance_notebook_interactivity(cells)
    
    # 创建新笔记本
    notebook = new_notebook(cells=cells, metadata={
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'codemirror_mode': {
                'name': 'ipython',
                'version': 3
            },
            'file_extension': '.py',
            'mimetype': 'text/x-python',
            'name': 'python',
            'nbconvert_exporter': 'python',
            'pygments_lexer': 'ipython3',
            'version': '3.9.13'
        }
    })
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # 保存笔记本
    with open(output_file, 'w', encoding='utf-8') as f:
        nbformat.write(notebook, f)
    
    print(f"已成功将 {input_file} 转换为 {output_file}")
    return output_file

def convert_all_exercises(exercises_dir, notebooks_dir):
    """转换指定目录下的所有练习文件"""
    converted_files = []
    
    # 确保输出目录存在
    os.makedirs(notebooks_dir, exist_ok=True)
    
    # 查找所有Python文件
    for file in os.listdir(exercises_dir):
        if file.endswith('.py'):
            input_path = os.path.join(exercises_dir, file)
            output_path = os.path.join(notebooks_dir, file.replace('.py', '.ipynb'))
            
            try:
                convert_py_to_notebook(input_path, output_path)
                converted_files.append(output_path)
            except Exception as e:
                print(f"转换 {file} 时出错: {str(e)}")
    
    return converted_files

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='将Python练习文件转换为Jupyter Notebook')
    parser.add_argument('input', nargs='?', help='输入的Python文件或目录')
    parser.add_argument('output', nargs='?', help='输出的Jupyter Notebook文件或目录')
    parser.add_argument('--all', action='store_true', help='转换exercises目录下的所有练习文件')
    
    args = parser.parse_args()
    
    if args.all:
        # 获取脚本所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 确定exercises和notebooks目录的路径
        exercises_dir = os.path.abspath(os.path.join(script_dir, '..', 'exercises'))
        notebooks_dir = os.path.abspath(os.path.join(script_dir, '..', 'notebooks'))
        
        print(f"转换目录 {exercises_dir} 中的所有练习文件到 {notebooks_dir}")
        converted_files = convert_all_exercises(exercises_dir, notebooks_dir)
        
        if converted_files:
            print(f"成功转换了 {len(converted_files)} 个文件:")
            for file in converted_files:
                print(f"  - {file}")
        else:
            print("没有找到练习文件或转换过程中出现错误。")
    
    elif args.input:
        if os.path.isdir(args.input):
            # 如果输入是目录
            exercises_dir = args.input
            notebooks_dir = args.output if args.output else os.path.join(os.path.dirname(exercises_dir), 'notebooks')
            
            print(f"转换目录 {exercises_dir} 中的所有练习文件到 {notebooks_dir}")
            convert_all_exercises(exercises_dir, notebooks_dir)
        else:
            # 如果输入是单个文件
            convert_py_to_notebook(args.input, args.output)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 