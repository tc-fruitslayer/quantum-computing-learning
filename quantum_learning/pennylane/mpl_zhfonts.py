#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
matplotlib中文字体支持模块
用于解决matplotlib绘图时中文显示为方块或问号的问题
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os

def set_chinese_font():
    """
    设置matplotlib使用中文字体
    根据不同操作系统选择合适的字体
    """
    system = platform.system()
    
    if system == 'Darwin':  # macOS
        # macOS常见中文字体
        font_list = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']
    elif system == 'Windows':
        # Windows常见中文字体
        font_list = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
    else:  # Linux或其他系统
        # Linux常见中文字体
        font_list = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Droid Sans Fallback']
    
    # 添加一些通用字体作为后备
    font_list.extend(['DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif'])
    
    # 尝试找到可用的中文字体
    chinese_font = None
    for font in font_list:
        # 检查该字体是否存在于系统中
        matches = [f for f in fm.findSystemFonts() if font.lower() in os.path.basename(f).lower()]
        if matches:
            chinese_font = font
            break
    
    if chinese_font:
        plt.rcParams['font.family'] = [chinese_font, 'sans-serif']
        print(f"已设置matplotlib使用中文字体: {chinese_font}")
    else:
        # 如果没有找到任何中文字体，尝试使用通用设置
        plt.rcParams['font.family'] = ['sans-serif']
        # 一种不太理想但可能有效的替代方法，避免使用DejaVu Sans
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Bitstream Vera Sans', 'sans-serif']
        print("警告: 未找到合适的中文字体，使用通用字体设置")
    
    # 修复负号显示问题
    plt.rcParams['axes.unicode_minus'] = False 