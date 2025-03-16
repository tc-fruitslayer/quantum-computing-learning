#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Google Cirq框架学习 2：量子门和电路
本文件详细介绍Cirq中的量子门类型、使用方式和电路构建
"""

import cirq
import numpy as np
import matplotlib.pyplot as plt
import sympy

print("===== Cirq中的量子门和电路 =====")

# 1. 单量子比特门的详细介绍
print("\n1. 单量子比特门")

# 创建一个量子比特用于演示
q = cirq.LineQubit(0)

# 1.1 Pauli门
print("\n1.1 Pauli门")
print(f"Pauli-X门 (量子NOT门): {cirq.X}")
# X门的矩阵表示
print("X门矩阵表示:")
print(cirq.unitary(cirq.X))

print(f"\nPauli-Y门: {cirq.Y}")
# Y门的矩阵表示
print("Y门矩阵表示:")
print(cirq.unitary(cirq.Y))

print(f"\nPauli-Z门: {cirq.Z}")
# Z门的矩阵表示
print("Z门矩阵表示:")
print(cirq.unitary(cirq.Z))

# 1.2 Hadamard门
print("\n1.2 Hadamard门 (H)")
print(f"Hadamard门: {cirq.H}")
# H门的矩阵表示
print("H门矩阵表示:")
print(cirq.unitary(cirq.H))

# 1.3 相位门
print("\n1.3 相位门")
print(f"S门 (π/2相位门): {cirq.S}")
print(f"S门共轭转置: {cirq.S**-1}")

print(f"\nT门 (π/4相位门): {cirq.T}")
print(f"T门共轭转置: {cirq.T**-1}")

# 1.4 旋转门
print("\n1.4 旋转门")
theta = np.pi/4  # 示例角度

# 绕X轴旋转
rx_gate = cirq.rx(theta)
rx_op = rx_gate.on(q)
print(f"Rx(π/4): {rx_op}")
print("Rx(π/4)矩阵表示:")
print(cirq.unitary(rx_op))

# 绕Y轴旋转
ry_gate = cirq.ry(theta)
ry_op = ry_gate.on(q)
print(f"\nRy(π/4): {ry_op}")
print("Ry(π/4)矩阵表示:")
print(cirq.unitary(ry_op))

# 绕Z轴旋转
rz_gate = cirq.rz(theta)
rz_op = rz_gate.on(q)
print(f"\nRz(π/4): {rz_op}")
print("Rz(π/4)矩阵表示:")
print(cirq.unitary(rz_op))

# 1.5 任意单量子比特门
print("\n1.5 任意单量子比特门")
# 使用欧拉分解创建一个任意单量子比特门（ZYZ分解）
alpha, beta, gamma = 0.1, 0.2, 0.3  # 示例角度
arbitrary_gate = cirq.PhasedXZGate(x_exponent=beta/np.pi, z_exponent_before=alpha/np.pi, z_exponent_after=gamma/np.pi)
arbitrary_op = arbitrary_gate.on(q)
print(f"任意单量子比特门 (ZYZ分解): {arbitrary_op}")
print("矩阵表示:")
print(cirq.unitary(arbitrary_op))

# 2. 多量子比特门
print("\n2. 多量子比特门")
# 创建两个量子比特用于演示
q0, q1 = cirq.LineQubit.range(2)

# 2.1 CNOT门 (Controlled-X)
print("\n2.1 CNOT门 (Controlled-X)")
cnot = cirq.CNOT(q0, q1)  # q0为控制比特，q1为目标比特
print(f"CNOT门: {cnot}")
print("CNOT门矩阵表示:")
print(cirq.unitary(cnot))

# 2.2 CZ门 (Controlled-Z)
print("\n2.2 CZ门 (Controlled-Z)")
cz = cirq.CZ(q0, q1)
print(f"CZ门: {cz}")
print("CZ门矩阵表示:")
print(cirq.unitary(cz))

# 2.3 SWAP门
print("\n2.3 SWAP门")
swap = cirq.SWAP(q0, q1)
print(f"SWAP门: {swap}")
print("SWAP门矩阵表示:")
print(cirq.unitary(swap))

# 2.4 受控门的其他变体
print("\n2.4 受控门变体")
# 创建受控Y门
cy = cirq.ControlledGate(sub_gate=cirq.Y)
cy_op = cy.on(q0, q1)  # q0控制q1
print(f"受控-Y门: {cy_op}")

# 创建多控制门（Toffoli门或CCNOT）
q0, q1, q2 = cirq.LineQubit.range(3)
ccnot = cirq.TOFFOLI(q0, q1, q2)  # 等价于CCNOT
print(f"Toffoli门 (CCNOT): {ccnot}")

# 2.5 参数化的多量子比特门
print("\n2.5 参数化的多量子比特门")
# 创建iSWAP门，在交换时添加相位
iswap = cirq.ISWAP(q0, q1)
print(f"iSWAP门: {iswap}")

# 创建可参数化的iSWAP^t门
t = 0.5  # 部分应用
iswap_pow = cirq.ISWAP**t
iswap_pow_op = iswap_pow.on(q0, q1)
print(f"iSWAP^0.5门: {iswap_pow_op}")

# 3. 量子电路结构与操作
print("\n3. 量子电路结构与操作")

# 3.1 创建电路的不同方式
print("\n3.1 创建电路的不同方式")
# 方式1：空电路然后添加操作
q0, q1 = cirq.LineQubit.range(2)
circuit1 = cirq.Circuit()
circuit1.append(cirq.H(q0))
circuit1.append(cirq.CNOT(q0, q1))
print("方式1 - 逐步添加操作:")
print(circuit1)

# 方式2：从操作列表创建
ops = [cirq.H(q0), cirq.CNOT(q0, q1)]
circuit2 = cirq.Circuit(ops)
print("\n方式2 - 从操作列表创建:")
print(circuit2)

# 方式3：使用+运算符连接电路
h_circuit = cirq.Circuit(cirq.H(q0))
cnot_circuit = cirq.Circuit(cirq.CNOT(q0, q1))
combined_circuit = h_circuit + cnot_circuit
print("\n方式3 - 使用+运算符连接电路:")
print(combined_circuit)

# 方式4：使用cirq.Moment来控制并行执行
moments = [
    cirq.Moment([cirq.H(q0), cirq.X(q1)]),  # 并行执行H和X
    cirq.Moment([cirq.CNOT(q0, q1)])        # 然后执行CNOT
]
moment_circuit = cirq.Circuit(moments)
print("\n方式4 - 使用Moment控制并行执行:")
print(moment_circuit)

# 3.2 电路插入和修改
print("\n3.2 电路插入和修改")
q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.measure(q0, q1, key='result')
)
print("原始电路:")
print(circuit)

# 在H和CNOT之间插入X门
modified_circuit = cirq.Circuit()
modified_circuit.append(cirq.H(q0))
modified_circuit.append(cirq.X(q0))  # 插入X门
modified_circuit.append(cirq.CNOT(q0, q1))
modified_circuit.append(cirq.measure(q0, q1, key='result'))
print("\n修改后的电路 (插入X门):")
print(modified_circuit)

# 3.3 电路变换和操作
print("\n3.3 电路变换和操作")

# 创建一个复杂一点的电路
q0, q1, q2 = cirq.LineQubit.range(3)
complex_circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.CNOT(q1, q2),
    cirq.X(q2),
    cirq.measure(q0, q1, q2, key='result')
)
print("复杂电路:")
print(complex_circuit)

# 获取电路的操作列表
operations = list(complex_circuit.all_operations())
print("\n电路包含的所有操作:", [str(op) for op in operations])

# 获取特定类型的操作
cnot_ops = [op for op in operations if isinstance(op.gate, cirq.ops.common_gates.CXPowGate)]
print("\nCNOT操作:", [str(op) for op in cnot_ops])

# 电路的矩量结构
moments = list(complex_circuit.moments)
print(f"\n电路包含 {len(moments)} 个矩量 (Moment)")
for i, moment in enumerate(moments):
    print(f"矩量 {i}: {moment}")

# 4. 参数化电路和电路变换
print("\n4. 参数化电路和电路变换")

# 4.1 创建带符号参数的电路
print("\n4.1 带符号参数的电路")
q0, q1 = cirq.LineQubit.range(2)

# 定义符号参数
theta = sympy.Symbol('θ')
phi = sympy.Symbol('φ')

# 创建参数化电路
param_circuit = cirq.Circuit(
    cirq.rx(theta).on(q0),
    cirq.rz(phi).on(q0),
    cirq.CNOT(q0, q1),
    cirq.rx(theta).on(q1),
    cirq.measure(q0, q1, key='result')
)

print("参数化电路:")
print(param_circuit)

# 4.2 参数解析和绑定
print("\n4.2 参数解析和绑定")
# 创建参数字典
params1 = {theta: np.pi/4, phi: np.pi/2}
resolved_circuit1 = cirq.resolve_parameters(param_circuit, params1)
print(f"绑定参数后的电路 (θ=π/4, φ=π/2):")
print(resolved_circuit1)

# 使用不同参数
params2 = {theta: np.pi/2, phi: np.pi}
resolved_circuit2 = cirq.resolve_parameters(param_circuit, params2)
print(f"\n绑定不同参数后的电路 (θ=π/2, φ=π):")
print(resolved_circuit2)

# 4.3 电路转换
print("\n4.3 电路转换")
q0, q1 = cirq.LineQubit.range(2)
original_circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.measure(q0, q1, key='result')
)
print("原始电路:")
print(original_circuit)

# 逆转电路（不包括测量）
inverse_circuit = cirq.inverse(original_circuit[:-1])
print("\n逆转的电路 (不包括测量):")
print(inverse_circuit)

# 完整电路：原始电路 + 逆转电路
complete_circuit = original_circuit[:-1] + inverse_circuit
print("\n完整电路 (应该返回到初始状态):")
print(complete_circuit)

# 5. 电路可视化与分析
print("\n5. 电路可视化与分析")

# 5.1 文本图表示
q0, q1, q2 = cirq.LineQubit.range(3)
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.CNOT(q1, q2),
    cirq.H(q2),
    cirq.measure(q0, q1, q2, key='result')
)

print("\n5.1 默认文本图表示:")
print(circuit)

print("\n横向文本图表示:")
print(circuit.to_text_diagram(transpose=True))

# 5.2 电路统计和分析
print("\n5.2 电路统计和分析")
# 计算电路中的门数量
ops = list(circuit.all_operations())
gate_counts = {}
for op in ops:
    gate_name = op.gate.__class__.__name__
    gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1

print("电路门统计:")
for gate, count in gate_counts.items():
    print(f"  {gate}: {count}")

# 计算电路深度
depth = len(list(circuit.moments))
print(f"电路深度 (矩量数): {depth}")

# 计算所需的量子比特数量
qubits = circuit.all_qubits()
print(f"使用的量子比特: {sorted(qubits, key=str)}")

print("\n总结:")
print("1. Cirq提供丰富的量子门集合")
print("2. 电路构建非常灵活，支持多种创建和组合方式")
print("3. 参数化电路允许创建可变的量子算法模板")
print("4. 电路分析和可视化工具有助于理解和优化量子算法")
print("5. Moment结构使得并行操作控制更为精确") 