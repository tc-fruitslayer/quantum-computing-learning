#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Google Cirq框架学习 1：Cirq基础和特性
本文件介绍Cirq的基本概念、数据结构和操作方式
"""

import cirq
import numpy as np
import matplotlib.pyplot as plt

print("===== Cirq基础和特性 =====")

# 1. 量子比特表示
print("\n1. 量子比特表示")
print("   Cirq中有多种不同类型的量子比特表示方式")

# 1.1 线性量子比特（LineQubit）- 最常用
print("\n1.1 线性量子比特（LineQubit）")
q0 = cirq.LineQubit(0)  # 单个线性量子比特，索引为0
q1 = cirq.LineQubit(1)  # 单个线性量子比特，索引为1

# 创建一系列连续的线性量子比特
line_qubits = cirq.LineQubit.range(5)  # 索引为0,1,2,3,4的5个量子比特
print(f"单个线性量子比特: {q0}, {q1}")
print(f"连续的线性量子比特: {line_qubits}")

# 1.2 网格量子比特（GridQubit）- 二维网格上的量子比特
print("\n1.2 网格量子比特（GridQubit）")
grid_q00 = cirq.GridQubit(0, 0)  # 位于(0,0)的网格量子比特
grid_q01 = cirq.GridQubit(0, 1)  # 位于(0,1)的网格量子比特

# 创建一个2x3的网格量子比特阵列
grid = cirq.GridQubit.rect(2, 3)  # 2行3列的网格
print(f"单个网格量子比特: {grid_q00}, {grid_q01}")
print(f"网格量子比特阵列: {grid}")

# 1.3 命名量子比特（NamedQubit）- 使用字符串标识的量子比特
print("\n1.3 命名量子比特（NamedQubit）")
alice = cirq.NamedQubit("Alice")
bob = cirq.NamedQubit("Bob")
print(f"命名量子比特: {alice}, {bob}")

# 2. 量子门
print("\n2. Cirq中的基本量子门")

# 2.1 常见单量子比特门
print("\n2.1 常见单量子比特门")
print(f"X门 (NOT门): {cirq.X}")
print(f"Y门: {cirq.Y}")
print(f"Z门: {cirq.Z}")
print(f"H门 (Hadamard): {cirq.H}")
print(f"S门 (相位门): {cirq.S}")
print(f"T门: {cirq.T}")

# 2.2 旋转门
print("\n2.2 旋转门")
theta = np.pi/4
rx = cirq.rx(theta)
ry = cirq.ry(theta)
rz = cirq.rz(theta)
print(f"Rx(π/4): {rx}")
print(f"Ry(π/4): {ry}")
print(f"Rz(π/4): {rz}")

# 2.3 多量子比特门
print("\n2.3 多量子比特门")
print(f"CNOT门: {cirq.CNOT}")
print(f"CZ门: {cirq.CZ}")
print(f"SWAP门: {cirq.SWAP}")

# 3. 创建量子电路
print("\n3. 创建量子电路")

# 3.1 创建一个空电路
circuit = cirq.Circuit()
print("\n3.1 空电路:")
print(circuit)

# 3.2 添加操作
print("\n3.2 添加操作到电路")
# 创建两个量子比特
q0, q1 = cirq.LineQubit.range(2)

# 方法1：使用append方法
circuit.append(cirq.H(q0))
circuit.append(cirq.CNOT(q0, q1))
print("使用append方法添加门后的电路:")
print(circuit)

# 方法2：直接从操作列表创建
ops = [
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.measure(q0, q1, key='result')
]
bell_circuit = cirq.Circuit(ops)
print("\n从操作列表直接创建的Bell态电路:")
print(bell_circuit)

# 3.3 创建和应用矩量（Moment）- 同时执行的操作集合
print("\n3.3 创建和应用矩量（Moment）")
q0, q1, q2 = cirq.LineQubit.range(3)
moment1 = cirq.Moment([cirq.H(q0), cirq.H(q1), cirq.H(q2)])  # 3个Hadamard门并行
moment2 = cirq.Moment([cirq.CNOT(q0, q1), cirq.X(q2)])       # CNOT和X门并行
moment_circuit = cirq.Circuit([moment1, moment2])
print("使用矩量创建的电路:")
print(moment_circuit)

# 4. 电路可视化
print("\n4. 电路可视化")
# 创建一个稍复杂的电路进行可视化
q0, q1, q2 = cirq.LineQubit.range(3)
complex_circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.CNOT(q1, q2),
    cirq.H(q2),
    cirq.X(q0),
    cirq.measure(q0, q1, q2, key='result')
)
print("复杂电路的文本表示:")
print(complex_circuit)

print("\n复杂电路的ASCII图表示:")
print(complex_circuit.to_text_diagram(transpose=True))

# 5. 模拟执行电路
print("\n5. 模拟执行电路")
# 创建一个Bell态电路
q0, q1 = cirq.LineQubit.range(2)
bell_circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.measure(q0, key='m0'),
    cirq.measure(q1, key='m1')
)

# 使用Cirq的模拟器执行电路
simulator = cirq.Simulator()
repetitions = 1000
result = simulator.run(bell_circuit, repetitions=repetitions)

# 分析结果
print(f"Bell态电路的运行结果 ({repetitions} 次重复):")
print(result.histogram(key='m0'))  # 测量q0的结果直方图
print(result.histogram(key='m1'))  # 测量q1的结果直方图

# 计算m0和m1的相关性
m0_results = result.measurements['m0'].flatten()
m1_results = result.measurements['m1'].flatten()
print("\n量子比特测量结果的相关性:")
matches = sum(m0_results == m1_results)
print(f"相同结果的比例: {matches / repetitions:.2f} (理论值应为1.0)")

# 6. Cirq的特殊功能：参数化电路
print("\n6. 参数化电路")
# 定义参数
theta = cirq.Parameter('θ')
phi = cirq.Parameter('φ')

# 创建带参数的电路
q0 = cirq.LineQubit(0)
param_circuit = cirq.Circuit(
    cirq.rx(theta)(q0),
    cirq.rz(phi)(q0),
    cirq.measure(q0, key='result')
)

print("参数化电路:")
print(param_circuit)

# 绑定参数
resolver = cirq.ParamResolver({theta: np.pi/4, phi: np.pi/2})
resolved_circuit = cirq.resolve_parameters(param_circuit, resolver)

print("\n绑定参数后的电路 (θ=π/4, φ=π/2):")
print(resolved_circuit)

# 7. 比较Cirq和其他框架的主要区别
print("\n7. Cirq的独特特性")
print("   1. 设备特定的拓扑结构：针对特定量子硬件的限制")
print("   2. 矩量（Moment）：明确控制并行操作")
print("   3. 强大的参数化支持")
print("   4. 与Google的量子硬件和TensorFlow Quantum的紧密集成")
print("   5. 集中于NISQ时代的应用")

print("\n总结:")
print("1. Cirq提供了丰富的量子比特表示方式")
print("2. 支持标准量子门和自定义门操作")
print("3. 电路创建和修改非常灵活")
print("4. 提供多种模拟器和可视化工具")
print("5. 特别适合针对特定硬件拓扑的电路设计")
print("6. 与Google量子生态系统紧密集成") 