#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
量子计算基础 2：量子比特和量子门
本文件介绍量子比特的表示方法，以及常见的量子门操作及其效果
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_bloch_multivector, plot_histogram, array_to_latex

print("===== 量子比特和量子门 =====")

# 1. 量子比特表示
print("\n1. 量子比特表示")
print("   量子比特的状态可以用二维复向量表示：|ψ⟩ = α|0⟩ + β|1⟩")
print("   其中|0⟩和|1⟩是计算基向量，在代数表示中为:")
print("   |0⟩ = [1, 0]^T")
print("   |1⟩ = [0, 1]^T")

# 创建一些典型的量子比特状态
print("\n典型的量子比特状态：")
# |0⟩状态
state_0 = np.array([1, 0], dtype=complex)
print(f"  |0⟩ = {state_0}")

# |1⟩状态
state_1 = np.array([0, 1], dtype=complex)
print(f"  |1⟩ = {state_1}")

# |+⟩状态 = (|0⟩ + |1⟩)/sqrt(2)
state_plus = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
print(f"  |+⟩ = {state_plus}")

# |−⟩状态 = (|0⟩ - |1⟩)/sqrt(2)
state_minus = np.array([1/np.sqrt(2), -1/np.sqrt(2)], dtype=complex)
print(f"  |−⟩ = {state_minus}")

# 2. Bloch球表示
print("\n2. Bloch球表示")
print("   单量子比特的纯态可以表示为Bloch球上的一个点")
print("   |0⟩对应北极，|1⟩对应南极")
print("   赤道上的点对应|0⟩和|1⟩的均等叠加，但相位不同")

# 使用Qiskit创建不同的量子态并可视化（仅显示文本说明）
print("\n通过量子电路准备不同的量子态：")

# |0⟩状态
print("\n|0⟩态（初始态）:")
qc_0 = QuantumCircuit(1)
# 不需要操作，初始态已经是|0⟩

# |1⟩状态
print("\n|1⟩态（应用X门）:")
qc_1 = QuantumCircuit(1)
qc_1.x(0)  # X门将|0⟩变为|1⟩

# |+⟩状态
print("\n|+⟩态（应用H门）:")
qc_plus = QuantumCircuit(1)
qc_plus.h(0)  # H门将|0⟩变为|+⟩

# |−⟩状态
print("\n|−⟩态（应用X门后应用H门）:")
qc_minus = QuantumCircuit(1)
qc_minus.x(0)
qc_minus.h(0)  # H门将|1⟩变为|−⟩

# 3. 单量子比特门
print("\n3. 单量子比特门")

# Pauli-X门（量子NOT门）
print("\nPauli-X门 (NOT门)：")
print("矩阵表示：")
X_matrix = np.array([[0, 1], [1, 0]])
print(X_matrix)
print("效果：将|0⟩变为|1⟩，将|1⟩变为|0⟩")
print("在Bloch球上：绕X轴旋转π角度")

# Pauli-Y门
print("\nPauli-Y门：")
print("矩阵表示：")
Y_matrix = np.array([[0, -1j], [1j, 0]])
print(Y_matrix)
print("效果：将|0⟩变为i|1⟩，将|1⟩变为-i|0⟩")
print("在Bloch球上：绕Y轴旋转π角度")

# Pauli-Z门
print("\nPauli-Z门：")
print("矩阵表示：")
Z_matrix = np.array([[1, 0], [0, -1]])
print(Z_matrix)
print("效果：将|0⟩保持不变，将|1⟩变为-|1⟩（相位翻转）")
print("在Bloch球上：绕Z轴旋转π角度")

# Hadamard门
print("\nHadamard门 (H门)：")
print("矩阵表示：")
H_matrix = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]])
print(H_matrix)
print("效果：")
print("  H|0⟩ = |+⟩ = (|0⟩ + |1⟩)/√2")
print("  H|1⟩ = |−⟩ = (|0⟩ - |1⟩)/√2")
print("在Bloch球上：先绕Y轴旋转π/2角度，再绕X轴旋转π角度")

# S门（相位门）
print("\nS门（相位门）：")
print("矩阵表示：")
S_matrix = np.array([[1, 0], [0, 1j]])
print(S_matrix)
print("效果：将|0⟩保持不变，将|1⟩变为i|1⟩（π/2相位旋转）")
print("在Bloch球上：绕Z轴旋转π/2角度")

# T门
print("\nT门：")
print("矩阵表示：")
T_matrix = np.array([[1, 0], [0, np.exp(1j * np.pi/4)]])
print(T_matrix)
print("效果：将|0⟩保持不变，将|1⟩变为e^(iπ/4)|1⟩（π/4相位旋转）")
print("在Bloch球上：绕Z轴旋转π/4角度")

# 4. 多量子比特门
print("\n4. 多量子比特门")

# CNOT门（受控非门）
print("\nCNOT门（受控非门）：")
print("矩阵表示（在计算基础|00⟩,|01⟩,|10⟩,|11⟩下）：")
CNOT_matrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
])
print(CNOT_matrix)
print("效果：当控制位为|1⟩时，翻转目标位；当控制位为|0⟩时，保持目标位不变")
print("例如：")
print("  CNOT|00⟩ = |00⟩")
print("  CNOT|01⟩ = |01⟩")
print("  CNOT|10⟩ = |11⟩")
print("  CNOT|11⟩ = |10⟩")

# 创建CNOT演示电路
print("\nCNOT电路示例：")
cnot_circuit = QuantumCircuit(2)
cnot_circuit.h(0)    # 将第一个量子比特置于叠加态
cnot_circuit.cx(0, 1)  # 控制位为0，目标位为1

print("电路图：")
print(cnot_circuit.draw())
print("效果：创建Bell态 (|00⟩ + |11⟩)/√2")

# SWAP门
print("\nSWAP门：")
print("效果：交换两个量子比特的状态")
print("例如：")
print("  SWAP|01⟩ = |10⟩")
print("  SWAP|10⟩ = |01⟩")

# 创建SWAP演示电路
print("\nSWAP电路示例：")
swap_circuit = QuantumCircuit(2)
swap_circuit.x(0)    # 将第一个量子比特置于|1⟩态
swap_circuit.swap(0, 1)  # 交换两个量子比特

print("电路图：")
print(swap_circuit.draw())
print("效果：从|10⟩变为|01⟩")

# 5. 量子门电路示例
print("\n5. 量子门电路示例")

# 创建一个更复杂的量子电路
complex_circuit = QuantumCircuit(3, 3)
complex_circuit.h(0)    # 将第一个量子比特置于叠加态
complex_circuit.cx(0, 1)  # 将第一和第二个量子比特纠缠
complex_circuit.x(2)    # 翻转第三个量子比特
complex_circuit.cx(1, 2)  # 将第二和第三个量子比特纠缠
complex_circuit.h(0)    # 再次应用H门到第一个量子比特
complex_circuit.measure([0, 1, 2], [0, 1, 2])  # 测量所有量子比特

print("复杂电路图：")
print(complex_circuit.draw())

# 执行电路
simulator = Aer.get_backend('qasm_simulator')
compiled_circuit = transpile(complex_circuit, simulator)
job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
counts = result.get_counts(compiled_circuit)
print(f"测量结果: {counts}")

print("\n总结：")
print("1. 量子比特是量子计算的基本单位，可以处于叠加态")
print("2. 量子门对量子比特进行操作，实现各种量子变换")
print("3. 单量子比特门包括Pauli门(X,Y,Z)、Hadamard门(H)、相位门(S,T)等")
print("4. 多量子比特门如CNOT、SWAP使得量子比特之间可以相互作用，产生纠缠")
print("5. 结合不同的量子门可以构建复杂的量子电路，实现各种量子算法") 