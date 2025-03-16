#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IBM Qiskit框架学习 2：量子电路创建和可视化
本文件详细介绍Qiskit中创建、组合和可视化量子电路的方法
"""

# 导入必要的库
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, Aer
from qiskit.visualization import plot_histogram, plot_bloch_multivector, plot_circuit_layout
from qiskit.visualization import plot_state_city, plot_state_qsphere, plot_state_hinton
from qiskit.circuit.library import EfficientSU2
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator
import numpy as np
import matplotlib.pyplot as plt

print("===== Qiskit量子电路创建和可视化 =====")

# 1. 基本电路创建方法
print("\n1. 基本电路创建方法")

# 方法1：直接创建
print("方法1：直接指定量子比特和经典比特数量")
qc1 = QuantumCircuit(3, 3)  # 3个量子比特，3个经典比特
qc1.h(0)
qc1.cx(0, 1)
qc1.cx(0, 2)
qc1.measure([0, 1, 2], [0, 1, 2])
print(qc1.draw())

# 方法2：使用寄存器
print("\n方法2：使用量子寄存器和经典寄存器")
qr = QuantumRegister(2, name='q')
cr = ClassicalRegister(2, name='c')
qc2 = QuantumCircuit(qr, cr)
qc2.h(qr[0])
qc2.cx(qr[0], qr[1])
qc2.measure(qr, cr)
print(qc2.draw())

# 方法3：从空电路添加寄存器
print("\n方法3：从空电路添加寄存器")
qc3 = QuantumCircuit()
qr1 = QuantumRegister(2, 'q1')
qr2 = QuantumRegister(1, 'q2')
cr1 = ClassicalRegister(2, 'c1')
cr2 = ClassicalRegister(1, 'c2')
qc3.add_register(qr1)
qc3.add_register(qr2)
qc3.add_register(cr1)
qc3.add_register(cr2)
print(f"量子比特总数: {qc3.num_qubits}")
print(f"经典比特总数: {qc3.num_clbits}")
print(f"寄存器总数: {len(qc3.qregs) + len(qc3.cregs)}")

# 2. 电路构建和操作
print("\n2. 电路构建和操作")

# 创建基本电路
qc = QuantumCircuit(3)

# 添加基本门
print("添加基本量子门:")
qc.h(0)       # Hadamard门
qc.x(1)       # X门（NOT门）
qc.z(2)       # Z门
qc.cx(0, 1)   # CNOT门

# 添加旋转门
qc.rx(np.pi/4, 0)  # 绕X轴旋转π/4
qc.ry(np.pi/2, 1)  # 绕Y轴旋转π/2
qc.rz(np.pi/6, 2)  # 绕Z轴旋转π/6

print(qc.draw())

# 3. 电路组合
print("\n3. 电路组合")

# 创建两个小电路
bell_pair = QuantumCircuit(2)
bell_pair.h(0)
bell_pair.cx(0, 1)
print("Bell对电路:")
print(bell_pair.draw())

ghz = QuantumCircuit(3)
ghz.h(0)
ghz.cx(0, 1)
ghz.cx(0, 2)
print("\nGHZ态电路:")
print(ghz.draw())

# 组合电路 - 添加门
print("\n组合电路 - 添加门:")
combined = QuantumCircuit(3)
combined.append(bell_pair, [0, 1])  # 将bell_pair应用于量子比特0和1
combined.x(2)
print(combined.draw())

# 组合电路 - 电路复合
print("\n组合电路 - 电路复合:")
# 创建一个3量子比特的电路
circuit1 = QuantumCircuit(3)
circuit1.h([0, 1, 2])

# 创建一个3量子比特的电路
circuit2 = QuantumCircuit(3)
circuit2.cx(0, 1)
circuit2.cx(1, 2)

# 复合两个电路
composed = circuit1.compose(circuit2)
print("复合电路:")
print(composed.draw())

# 4. 电路可视化
print("\n4. 电路可视化")

# 创建复杂点的电路进行演示
vis_circuit = QuantumCircuit(5, 5)
vis_circuit.h(0)
vis_circuit.cx(0, range(1, 5))
vis_circuit.barrier()
vis_circuit.x([0, 2])
vis_circuit.z([1, 3, 4])
vis_circuit.barrier()
vis_circuit.measure(range(5), range(5))

print("复杂电路绘制:")
print(vis_circuit.draw())

# 保存电路图到不同格式
print("\n保存电路图到不同格式")
print("- 电路以文本格式输出")
print("- 电路图会保存为图片文件")

# 绘制并保存matplotlib格式
print("绘制电路图(matplotlib格式)并保存到文件")
fig = vis_circuit.draw(output='mpl', style={'backgroundcolor': '#EEEEEE'})
fig.savefig('vis_circuit_mpl.png')
plt.close(fig)

# 5. 电路的层次结构
print("\n5. 电路的层次结构")

# 创建一个电路，展示时刻(Moment)和指令(Instruction)
moment_circuit = QuantumCircuit(3)
moment_circuit.h(0)
moment_circuit.h(1)  # 这与H(0)在同一时刻
moment_circuit.cx(0, 2)  # 这与上面的H门不在同一时刻
moment_circuit.z(1)  # 这与CNOT门可以在同一时刻

print("电路的时刻结构:")
print(moment_circuit.draw())

print("\n电路指令分析:")
for i, instruction in enumerate(moment_circuit.data):
    print(f"指令 {i+1}: {instruction.operation.name} 在量子比特 {[qubit.index for qubit in instruction.qubits]}")

# 6. 参数化电路
print("\n6. 参数化电路")

# 创建参数化电路
from qiskit.circuit import Parameter

theta = Parameter('θ')
phi = Parameter('φ')

param_circuit = QuantumCircuit(2)
param_circuit.rx(theta, 0)
param_circuit.ry(phi, 1)
param_circuit.cx(0, 1)

print("参数化电路:")
print(param_circuit.draw())

# 为参数赋值
bound_circuit = param_circuit.bind_parameters({theta: np.pi/2, phi: np.pi/4})
print("\n绑定参数后的电路:")
print(bound_circuit.draw())

# 7. 使用电路库
print("\n7. 使用电路库")

# 使用EfficientSU2库电路，创建一个参数化的变分量子电路
print("使用电路库创建变分量子电路:")
var_form = EfficientSU2(4, entanglement='linear', reps=1)
print(var_form.draw())

print(f"\n参数数量: {var_form.num_parameters}")
print(f"参数名称: {var_form.parameters}")

# 8. 模拟和结果可视化
print("\n8. 模拟和结果可视化")

# 创建一个Bell状态电路
bell = QuantumCircuit(2, 2)
bell.h(0)
bell.cx(0, 1)
bell.measure([0, 1], [0, 1])

# 使用QASM模拟器
simulator = Aer.get_backend('qasm_simulator')
job = simulator.run(transpile(bell, simulator), shots=1000)
result = job.result()
counts = result.get_counts()

print("Bell状态测量结果:")
print(counts)

# 绘制结果直方图
print("结果直方图将保存到文件")
fig = plot_histogram(counts)
fig.savefig('bell_histogram.png')
plt.close(fig)

# 使用状态向量模拟器
sv_sim = Aer.get_backend('statevector_simulator')
# 创建没有测量的Bell态
bell_sv = QuantumCircuit(2)
bell_sv.h(0)
bell_sv.cx(0, 1)
job = sv_sim.run(transpile(bell_sv, sv_sim))
state = job.result().get_statevector()

# 不同的状态可视化方法
print("\n使用不同方法可视化量子态（结果将保存到文件）")

# Bloch球多向量表示
fig = plot_bloch_multivector(state)
fig.savefig('bell_bloch.png')
plt.close(fig)

# 城市图表示
fig = plot_state_city(state)
fig.savefig('bell_city.png')
plt.close(fig)

# Q球表示
fig = plot_state_qsphere(state)
fig.savefig('bell_qsphere.png')
plt.close(fig)

# Hinton图表示
fig = plot_state_hinton(state)
fig.savefig('bell_hinton.png')
plt.close(fig)

# 9. 电路与后端布局
print("\n9. 电路与后端布局")
print("在真实量子设备上运行时，需要考虑量子比特的物理布局")

# 创建一个简单的量子电路
layout_circuit = QuantumCircuit(5)
layout_circuit.h(0)
layout_circuit.cx(0, 1)
layout_circuit.cx(0, 2)
layout_circuit.cx(2, 3)
layout_circuit.cx(3, 4)

# 获取后端
backend = Aer.get_backend('qasm_simulator')

# 转译电路
transpiled_circuit = transpile(layout_circuit, backend, optimization_level=3)

print("原始电路:")
print(layout_circuit.draw())

print("\n转译后的电路:")
print(transpiled_circuit.draw())

# 10. 总结
print("\n10. 总结")
print("1. Qiskit提供了多种创建量子电路的方式")
print("2. 可以使用寄存器灵活组织量子比特和经典比特")
print("3. 电路可以通过多种方式组合和扩展")
print("4. Qiskit提供丰富的可视化工具")
print("5. 支持参数化电路，适用于变分算法")
print("6. 电路库提供了常用的量子电路模板")
print("7. 多种方法可以可视化量子态和测量结果")

print("\n下一步学习:")
print("- 深入了解各种量子门的性质和应用")
print("- 实现经典量子算法")
print("- 探索量子电路优化技术")
print("- 在真实量子设备上运行程序") 