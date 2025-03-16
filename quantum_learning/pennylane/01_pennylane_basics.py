#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PennyLane基础教程
================

本教程介绍PennyLane的基础概念和用法，包括：
- 量子设备创建
- 量子节点定义
- 基本量子门操作
- 量子测量
- 参数化量子电路
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

print("===== PennyLane基础教程 =====")
print(f"PennyLane版本: {qml.version()}")

# ===== 第1部分：创建量子设备 =====
print("\n===== 第1部分：创建量子设备 =====")

# 默认模拟器，2个量子比特
dev = qml.device('default.qubit', wires=2)
print(f"创建设备: default.qubit")
print(f"量子比特数量: 2")  # 硬编码wire数量

# ===== 第2部分：创建量子电路(QNode) =====
print("\n===== 第2部分：创建量子电路(QNode) =====")

# 使用装饰器定义量子电路
@qml.qnode(dev)
def my_circuit():
    # 将第0个量子比特置于叠加态
    qml.Hadamard(wires=0)
    # 将第1个量子比特置于|1⟩态
    qml.PauliX(wires=1)
    # 添加CNOT门，控制比特为0，目标比特为1
    qml.CNOT(wires=[0, 1])
    # 返回两个量子比特的计算基测量结果
    return qml.probs(wires=[0, 1])

# 执行电路并打印结果
result = my_circuit()
print("电路执行结果（态概率）：")
for i, prob in enumerate(result):
    state = format(i, '02b')  # 将索引转换为二进制表示
    print(f"|{state}⟩: {prob:.6f}")

# 打印电路图
print("\n电路图：")
print(qml.draw(my_circuit)())

# ===== 第3部分：参数化量子电路 =====
print("\n===== 第3部分：参数化量子电路 =====")

@qml.qnode(dev)
def rotation_circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

# 使用不同参数执行电路
params1 = np.array([0.0, 0.0])
params2 = np.array([np.pi/2, np.pi/4])
params3 = np.array([np.pi, np.pi])

result1 = rotation_circuit(params1)
result2 = rotation_circuit(params2)
result3 = rotation_circuit(params3)

print(f"参数 [0.0, 0.0]: {result1}")
print(f"参数 [π/2, π/4]: {result2}")
print(f"参数 [π, π]: {result3}")

# ===== 第4部分：观测值期望值 =====
print("\n===== 第4部分：观测值期望值 =====")

@qml.qnode(dev)
def observable_circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    # 返回不同观测量的期望值
    return [
        qml.expval(qml.PauliX(0)),  # ⟨X₀⟩
        qml.expval(qml.PauliY(1)),  # ⟨Y₁⟩
        qml.expval(qml.PauliZ(0)),  # ⟨Z₀⟩
        qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))  # ⟨Z₀ ⊗ Z₁⟩
    ]

params = np.array([np.pi/4, np.pi/4])
exp_vals = observable_circuit(params)

print(f"⟨X₀⟩ = {exp_vals[0]:.6f}")
print(f"⟨Y₁⟩ = {exp_vals[1]:.6f}")
print(f"⟨Z₀⟩ = {exp_vals[2]:.6f}")
print(f"⟨Z₀ ⊗ Z₁⟩ = {exp_vals[3]:.6f}")

# ===== 第5部分：旋转扫描 =====
print("\n===== 第5部分：旋转扫描 =====")

@qml.qnode(dev)
def rotation_scan_circuit(phi):
    qml.RX(phi, wires=0)
    return qml.expval(qml.PauliZ(0))

# 扫描参数从0到2π
phi_values = np.linspace(0, 2*np.pi, 50)
expectation_values = [rotation_scan_circuit(phi) for phi in phi_values]

# 绘制期望值随参数变化的曲线
plt.figure(figsize=(8, 6))
plt.plot(phi_values, expectation_values, 'b-')
plt.grid(True)
plt.xlabel('φ (弧度)')
plt.ylabel('⟨Z⟩')
plt.title('RX旋转角度φ与Z测量期望值的关系')
plt.savefig('rotation_scan.png')
print("旋转扫描图已保存为 rotation_scan.png")

# ===== 第6部分：电路梯度 =====
print("\n===== 第6部分：电路梯度 =====")

@qml.qnode(dev)
def circuit_with_gradient(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

# 计算梯度
params = np.array([0.5, 0.5])
grad_fn = qml.grad(circuit_with_gradient)
gradients = grad_fn(params)

print(f"参数: {params}")
print(f"函数值: {circuit_with_gradient(params):.6f}")
print(f"梯度: [∂f/∂θ₁, ∂f/∂θ₂] = {gradients}")

print("\n===== PennyLane基础教程完成 =====") 