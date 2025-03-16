#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python文件到Jupyter Notebook的批量转换工具

这个脚本将量子计算学习项目中的所有Python文件转换为交互式Jupyter Notebook格式。
它会递归遍历指定目录，找到所有.py文件并转换为.ipynb文件。
转换完成后，可以选择删除原始的Python文件。

用法:
    python convert_all_python_files.py [--delete-originals]

选项:
    --delete-originals: 转换完成后删除原始的Python文件
"""

import os
import sys
import argparse
import shutil
import importlib.util

def find_all_python_files(root_dir):
    """递归查找指定目录下的所有Python文件"""
    python_files = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 跳过scripts目录和__pycache__目录
        if 'scripts' in dirpath.split(os.sep) or '__pycache__' in dirpath.split(os.sep):
            continue
            
        for filename in filenames:
            if filename.endswith('.py'):
                full_path = os.path.join(dirpath, filename)
                python_files.append(full_path)
    
    return python_files

def ensure_converter_script():
    """确保转换脚本存在，如果不存在则复制一份"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    converter_path = os.path.join(script_dir, 'py_to_notebook.py')
    
    if not os.path.exists(converter_path):
        # 尝试从pennylane目录复制
        pennylane_converter = os.path.join(script_dir, '..', 'pennylane', 'scripts', 'py_to_notebook.py')
        if os.path.exists(pennylane_converter):
            shutil.copy2(pennylane_converter, converter_path)
            print(f"已将转换脚本复制到 {converter_path}")
        else:
            print("错误：找不到转换脚本 py_to_notebook.py")
            sys.exit(1)
    
    return converter_path

def import_converter_module(converter_path):
    """导入转换器模块"""
    spec = importlib.util.spec_from_file_location("py_to_notebook", converter_path)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    return converter

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='将所有Python文件转换为Jupyter Notebook')
    parser.add_argument('--delete-originals', action='store_true', help='转换完成后删除原始Python文件')
    
    args = parser.parse_args()
    
    # 确保转换脚本存在
    converter_path = ensure_converter_script()
    
    # 导入转换器模块
    converter = import_converter_module(converter_path)
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)  # 项目根目录
    
    # 查找所有Python文件
    python_files = find_all_python_files(root_dir)
    print(f"找到 {len(python_files)} 个Python文件")
    
    # 转换所有Python文件
    converted_files = []
    for py_file in python_files:
        # 确定输出文件路径
        ipynb_file = py_file.replace('.py', '.ipynb')
        
        try:
            # 使用导入的转换器模块进行转换
            converter.convert_py_to_notebook(py_file, ipynb_file)
            converted_files.append((py_file, ipynb_file))
        except Exception as e:
            print(f"转换 {py_file} 时出错: {str(e)}")
    
    # 打印转换结果
    if converted_files:
        print(f"\n成功转换了 {len(converted_files)} 个文件:")
        for py_file, ipynb_file in converted_files:
            print(f"  - {py_file} -> {ipynb_file}")
        
        # 如果指定了删除原始文件选项
        if args.delete_originals:
            print("\n正在删除原始Python文件...")
            for py_file, _ in converted_files:
                try:
                    os.remove(py_file)
                    print(f"  - 已删除 {py_file}")
                except Exception as e:
                    print(f"  - 删除 {py_file} 时出错: {str(e)}")
    else:
        print("没有成功转换任何文件。")

if __name__ == "__main__":
    main() 