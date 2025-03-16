#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Xanadu PennyLane框架 - 基础练习

本文件包含关于PennyLane基础知识的练习。
完成这些练习将帮助您理解PennyLane的核心概念和用法。
"""

import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

print("===== PennyLane基础练习 =====")

"""
练习1: 创建量子设备和简单电路
------------------------------
任务:
1. 创建一个拥有2个量子比特的'default.qubit'设备
2. 定义一个量子函数，对第一个量子比特应用Hadamard门，对第二个量子比特应用X门
3. 将该函数转换为QNode
4. 执行电路并打印结果 (<Z0>和<Z1>的期望值)
"""

print("\n练习1: 创建量子设备和简单电路")

# 您的代码:
# dev = ...

# def my_circuit():
#     ...
#     return ...

# qnode = ...
# result = ...

"""
练习2: 量子态制备和测量
------------------------------
任务:
1. 创建一个拥有3个量子比特的设备
2. 编写一个量子函数，准备GHZ态 (|000⟩ + |111⟩)/√2
3. 返回每个量子比特的PauliZ期望值、PauliX期望值，以及所有量子比特的PauliZ张量积
4. 分析结果，解释它们与GHZ态的理论预期如何对应
"""

print("\n练习2: 量子态制备和测量")

# 您的代码:
# dev = ...

# def ghz_state():
#     ...
#     return ...

# qnode = ...
# results = ...

"""
练习3: 参数化电路与自动微分
------------------------------
任务:
1. 创建一个拥有1个量子比特的设备
2. 创建一个包含RX、RY和RZ旋转的参数化量子电路
3. 返回PauliZ的期望值
4. 对不同参数值执行电路，并计算梯度
5. 绘制电路输出相对于第一个参数的变化图
"""

print("\n练习3: 参数化电路与自动微分")

# 您的代码:
# dev = ...

# def param_circuit(params):
#     ...
#     return ...

# qnode = ...

# 计算不同参数值的电路输出
# x_values = ...
# y_values = ...
# gradient_values = ...

# 绘图(可选)
# plt.figure()
# plt.plot(...)
# plt.xlabel(...)
# plt.ylabel(...)
# plt.title(...)
# plt.savefig('param_circuit_output.png')
# plt.close()

"""
练习4: 使用PennyLane模板
------------------------------
任务:
1. 创建一个拥有4个量子比特的设备
2. 使用AngleEmbedding模板将4个随机特征嵌入量子电路
3. 使用StronglyEntanglingLayers应用2层强纠缠层
4. 返回所有量子比特的PauliZ期望值
5. 绘制电路图并解释电路的结构
"""

print("\n练习4: 使用PennyLane模板")

# 您的代码:
# dev = ...

# def template_circuit(features, weights):
#     ...
#     return ...

# qnode = ...

# features = ...
# weights = ...
# results = ...

"""
练习5: 贝尔不等式测试
------------------------------
任务:
1. 创建一个拥有2个量子比特的设备
2. 准备一个最大纠缠态 (Bell态)
3. 计算CHSH不等式的期望值: ⟨A₁B₁⟩ + ⟨A₁B₂⟩ + ⟨A₂B₁⟩ - ⟨A₂B₂⟩
   其中A₁, A₂为量子比特0上的不同测量，B₁, B₂为量子比特1上的不同测量
4. 验证结果是否超过经典极限2
"""

print("\n练习5: 贝尔不等式测试")

# 您的代码:
# dev = ...

# def bell_test():
#     # 准备Bell态
#     ...
    
#     # 计算相关性
#     # A₁ = σz⊗I, B₁ = I⊗σz
#     # A₂ = σx⊗I, B₂ = I⊗σx
#     ...
    
#     # 计算CHSH值
#     # CHSH = ⟨A₁B₁⟩ + ⟨A₁B₂⟩ + ⟨A₂B₁⟩ - ⟨A₂B₂⟩
#     ...
    
#     return ...

# qnode = ...
# chsh_value = ...

"""
练习6: 量子隐形传态
------------------------------
任务:
1. 创建一个拥有3个量子比特的设备
2. 实现量子隐形传态协议:
   - 在量子比特0准备任意状态 |ψ⟩ = cosθ|0⟩ + sinθ|1⟩
   - 在量子比特1和2之间创建Bell对
   - 对量子比特0和1执行Bell测量
   - 根据测量结果在量子比特2上应用适当的门
3. 验证量子比特2最终状态与初始状态|ψ⟩相同
"""

print("\n练习6: 量子隐形传态")

# 您的代码:
# dev = ...

# def teleport(theta):
#     # 准备初始状态
#     ...
    
#     # 创建Bell对
#     ...
    
#     # Bell测量
#     ...
    
#     # 根据测量结果应用校正
#     ...
    
#     # 返回量子比特2的状态
#     return ...

# qnode = ...
# initial_theta = np.pi/4
# teleported_state = ...

"""
练习7: 量子相位估计
------------------------------
任务:
1. 创建一个拥有足够量子比特的设备
2. 实现简化版的量子相位估计算法:
   - 使用3个量子比特来估计相位
   - 使用具有已知特征值e^(2πiφ)的单一算子
   - 估计相位φ
3. 比较估计值与真实值
"""

print("\n练习7: 量子相位估计")

# 您的代码:
# dev = ...

# def phase_estimation(phase):
#     # 初始化寄存器
#     ...
    
#     # 应用Hadamard门到估计量子比特
#     ...
    
#     # 应用受控旋转
#     ...
    
#     # 应用逆量子傅里叶变换
#     ...
    
#     # 测量相位
#     ...
    
#     return ...

# qnode = ...
# true_phase = 0.25  # 例如φ=1/4
# estimated_phase = ...

print("\n完成所有练习后，请查看解决方案文件以比较您的实现。")
print("下一步: 学习变分量子电路和量子机器学习技术。") 