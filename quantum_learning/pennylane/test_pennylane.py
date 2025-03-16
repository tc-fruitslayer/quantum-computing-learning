#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PennyLane测试文件 - 确认基本功能
"""

import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# 导入中文字体支持
try:
    from mpl_zhfonts import set_chinese_font
    set_chinese_font()
    print("已启用中文字体支持")
except ImportError:
    print("警告: 未找到中文字体支持模块，图表中的中文可能无法正确显示")

# 打印PennyLane版本
print(f"PennyLane版本: {qml.version()}")

# 创建量子设备
dev = qml.device('default.qubit', wires=2)

# 定义一个简单的量子电路
@qml.qnode(dev)
def quantum_circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

# 运行电路
params = np.array([0.5, 0.3])
result = quantum_circuit(params)
print(f"量子电路输出: {result}")

# 计算电路梯度
grad = qml.grad(quantum_circuit)(params)
print(f"电路梯度: {grad}")

# 输出消息
print("PennyLane功能测试成功!") 