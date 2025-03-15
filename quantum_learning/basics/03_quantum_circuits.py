#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
量子计算基础 3：量子电路模型
本文件介绍量子电路模型的基本概念和构建方法
"""

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram

print("===== 量子电路模型 =====")

print("\n1. 量子电路的基本概念")
print("   量子电路是量子计算的标准模型")
print("   电路由一系列量子比特和对它们的操作（量子门）组成")
print("   电路从左到右读取，表示时间的流逝")
print("   量子电路模型是图灵完备的，可以表示任何计算")

# 2. 创建基本量子电路
print("\n2. 创建基本量子电路")

# 2.1 电路初始化的不同方法
print("\n2.1 电路初始化的不同方法")

# 方法1：直接指定量子比特和经典比特数量
print("\n方法1：直接指定量子比特和经典比特数量")
qc1 = QuantumCircuit(2, 2)
print(f"电路1: {qc1.num_qubits} 个量子比特, {qc1.num_clbits} 个经典比特")
print(qc1.draw())

# 方法2：使用寄存器
print("\n方法2：使用寄存器")
qreg = QuantumRegister(2, 'q')  # 创建量子寄存器
creg = ClassicalRegister(2, 'c')  # 创建经典寄存器
qc2 = QuantumCircuit(qreg, creg)
print(f"电路2: {qc2.num_qubits} 个量子比特, {qc2.num_clbits} 个经典比特")
print(qc2.draw())

# 方法3：使用多个寄存器
print("\n方法3：使用多个寄存器")
qreg1 = QuantumRegister(2, 'q1')
qreg2 = QuantumRegister(1, 'q2')
creg1 = ClassicalRegister(2, 'c1')
creg2 = ClassicalRegister(1, 'c2')
qc3 = QuantumCircuit(qreg1, qreg2, creg1, creg2)
print(f"电路3: {qc3.num_qubits} 个量子比特, {qc3.num_clbits} 个经典比特")
print(qc3.draw())

# 2.2 添加量子门
print("\n2.2 添加量子门")
qc = QuantumCircuit(3, 3)

# 添加单量子比特门
qc.h(0)  # Hadamard门作用于量子比特0
qc.x(1)  # X门（NOT门）作用于量子比特1
qc.z(2)  # Z门作用于量子比特2

# 添加多量子比特门
qc.cx(0, 1)  # CNOT门，控制位是量子比特0，目标位是量子比特1
qc.swap(1, 2)  # SWAP门，交换量子比特1和2

# 测量
qc.measure([0, 1, 2], [0, 1, 2])  # 将量子比特0,1,2的测量结果保存到经典比特0,1,2

print("基本量子电路：")
print(qc.draw())

# 3. 电路的组合和重用
print("\n3. 电路的组合和重用")

# 3.1 创建子电路
print("\n3.1 创建子电路")
bell_pair = QuantumCircuit(2)
bell_pair.h(0)
bell_pair.cx(0, 1)
print("Bell对子电路：")
print(bell_pair.draw())

# 3.2 复制和组合电路
print("\n3.2 复制和组合电路")
qc_combined = QuantumCircuit(4, 4)
# 将Bell对子电路应用到前两个量子比特
qc_combined = qc_combined.compose(bell_pair, qubits=[0, 1])
# 将Bell对子电路应用到后两个量子比特
qc_combined = qc_combined.compose(bell_pair, qubits=[2, 3])
# 添加测量
qc_combined.measure([0, 1, 2, 3], [0, 1, 2, 3])

print("组合后的电路：")
print(qc_combined.draw())

# 4. 参数化量子电路
print("\n4. 参数化量子电路")
from qiskit.circuit import Parameter

# 创建参数
theta = Parameter('θ')
phi = Parameter('φ')

# 创建参数化电路
param_qc = QuantumCircuit(1, 1)
param_qc.rx(theta, 0)  # 绕X轴旋转角度theta
param_qc.rz(phi, 0)    # 绕Z轴旋转角度phi
param_qc.measure(0, 0)

print("参数化量子电路：")
print(param_qc.draw())

# 绑定参数
bound_qc = param_qc.assign_parameters({theta: np.pi/4, phi: np.pi/2})
print("\n绑定参数后的电路（θ = π/4, φ = π/2）：")
print(bound_qc.draw())

# 5. 条件操作
print("\n5. 条件操作（基于测量结果）")
# 在Qiskit中，条件操作通常通过中间测量和条件重置来实现
qc_condition = QuantumCircuit(2, 1)
qc_condition.h(0)
qc_condition.measure(0, 0)  # 测量第一个量子比特
qc_condition.x(1).c_if(qc_condition.cregs[0], 1)  # 如果测量结果为1，则对第二个量子比特应用X门

print("条件电路：")
print(qc_condition.draw())

# 6. 电路优化和转译
print("\n6. 电路优化和转译")
# 创建一个测试电路
test_qc = QuantumCircuit(3, 3)
test_qc.h(0)
test_qc.cx(0, 1)
test_qc.cx(1, 2)
test_qc.measure([0, 1, 2], [0, 1, 2])

print("原始电路：")
print(test_qc.draw())

# 转译电路
simulator = Aer.get_backend('qasm_simulator')
trans_qc = transpile(test_qc, simulator, optimization_level=3)
print("\n优化后的电路（优化级别3）：")
print(trans_qc.draw())

# 7. 运行量子电路
print("\n7. 运行量子电路")
# 创建一个GHZ态电路
ghz_qc = QuantumCircuit(3, 3)
ghz_qc.h(0)
ghz_qc.cx(0, 1)
ghz_qc.cx(0, 2)
ghz_qc.measure([0, 1, 2], [0, 1, 2])

print("GHZ态电路：")
print(ghz_qc.draw())

# 在模拟器上运行
simulator = Aer.get_backend('qasm_simulator')
compiled_circuit = transpile(ghz_qc, simulator)
job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
counts = result.get_counts(compiled_circuit)
print(f"\nGHZ态测量结果: {counts}")
print("注意：GHZ态测量结果应该只有|000⟩和|111⟩，因为这三个量子比特处于完全纠缠状态")

print("\n总结：")
print("1. 量子电路是量子计算的标准模型，由量子比特和量子门组成")
print("2. 可以通过不同方式创建和初始化量子电路")
print("3. 量子电路可以组合、参数化，以及包含条件操作")
print("4. 真实的量子计算需要考虑优化和转译步骤")
print("5. 量子电路可以在模拟器或真实量子设备上运行") 