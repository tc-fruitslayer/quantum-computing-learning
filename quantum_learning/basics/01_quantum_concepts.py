#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
量子计算基础 1：量子力学基本概念
本文件介绍量子计算的核心概念，包括叠加、纠缠、测量和不确定性原理
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_bloch_multivector, plot_histogram

print("===== 量子力学基本概念 =====")
print("\n1. 量子叠加原理")
print("   经典比特只能处于0或1状态")
print("   量子比特可以同时处于0和1的线性组合（叠加）状态")
print("   一般表示为：|ψ⟩ = α|0⟩ + β|1⟩，其中|α|² + |β|² = 1")

# 创建一个量子电路，展示叠加态
print("\n叠加态示例：")
superposition_circuit = QuantumCircuit(1, 1)
superposition_circuit.h(0)  # 应用Hadamard门创建叠加态 (|0⟩ + |1⟩)/√2
superposition_circuit.measure(0, 0)

print("量子电路（创建叠加态）：")
print(superposition_circuit.draw())

# 执行电路
simulator = Aer.get_backend('statevector_simulator')
result = simulator.run(superposition_circuit).result()
statevector = result.get_statevector()
print(f"状态向量: {statevector}")

# 绘制量子态在布洛赫球上的表示
print("\n布洛赫球表示：")
print("叠加态在布洛赫球上位于赤道，表示|0⟩和|1⟩的均等叠加")

# 执行多次测量
print("\n多次测量结果：")
qasm_simulator = Aer.get_backend('qasm_simulator')
compiled_circuit = transpile(superposition_circuit, qasm_simulator)
job = qasm_simulator.run(compiled_circuit, shots=1000)
result = job.result()
counts = result.get_counts(compiled_circuit)
print(f"测量结果: {counts}")
print("注意：每次测量都会得到确定的0或1，概率各为50%")

print("\n2. 量子纠缠")
print("   两个或多个粒子的量子态无法独立描述")
print("   测量一个粒子会立即影响另一个粒子，无论相距多远")

# 创建Bell态（最简单的纠缠态）
print("\nBell态示例（纠缠态）：")
bell_circuit = QuantumCircuit(2, 2)
bell_circuit.h(0)
bell_circuit.cx(0, 1)  # CNOT门，将两个量子比特纠缠
bell_circuit.measure([0, 1], [0, 1])

print("量子电路（创建Bell态）：")
print(bell_circuit.draw())

# 执行电路
simulator = Aer.get_backend('qasm_simulator')
compiled_circuit = transpile(bell_circuit, simulator)
job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
counts = result.get_counts(compiled_circuit)
print(f"测量结果: {counts}")
print("注意：测量结果只会得到00或11，表明两个比特总是完全相关的")

print("\n3. 量子测量和状态坍缩")
print("   测量会导致量子态坍缩为测量到的基态")
print("   测量前的状态决定了坍缩到各基态的概率")

# 演示不同测量基的影响
print("\n不同量子态的测量概率：")
measure_circuit = QuantumCircuit(1, 1)
# 准备一个偏向|1⟩的状态 (1/√5)|0⟩ + (2/√5)|1⟩
measure_circuit.initialize([1/np.sqrt(5), 2/np.sqrt(5)], 0)
measure_circuit.measure(0, 0)

print("量子电路（创建偏向的叠加态）：")
print(measure_circuit.draw())

# 执行电路
simulator = Aer.get_backend('qasm_simulator')
compiled_circuit = transpile(measure_circuit, simulator)
job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
counts = result.get_counts(compiled_circuit)
print(f"测量结果: {counts}")
print("理论预期：|0⟩的概率为20%，|1⟩的概率为80%")

print("\n4. 不确定性原理")
print("   在量子力学中，无法同时精确测量某些成对的物理量")
print("   例如：位置和动量、不同方向的自旋等")
print("   在量子计算中表现为：无法同时精确知道X、Y、Z方向的量子态")

print("\n总结：")
print("1. 量子叠加让量子比特可以同时处于多个状态")
print("2. 量子纠缠使得量子比特之间产生超越经典物理的关联")
print("3. 量子测量会导致状态坍缩，结果具有概率性")
print("4. 量子不确定性原理限制了我们对量子系统的认知")

print("\n这些概念是量子计算的理论基础，也是量子计算能够超越经典计算的根本原因。") 