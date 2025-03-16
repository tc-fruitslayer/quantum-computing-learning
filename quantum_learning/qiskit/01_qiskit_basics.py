#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IBM Qiskit框架学习 1：Qiskit基础和特性
本文件介绍Qiskit的基本概念、架构和使用方式
"""

# 导入Qiskit库
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.tools.monitor import job_monitor
from qiskit.providers.aer import QasmSimulator
import numpy as np
import matplotlib.pyplot as plt

print("===== Qiskit基础和特性 =====")

# 检查Qiskit版本
import qiskit
print(f"Qiskit版本: {qiskit.__version__}")

# 1. Qiskit架构概述
print("\n1. Qiskit架构概述")
print("Qiskit是一个用于量子计算的开源软件开发套件，包含以下主要组件:")
print("- Qiskit Terra: 核心组件，提供构建量子电路和执行的基础")
print("- Qiskit Aer: 模拟器组件，用于模拟量子电路")
print("- Qiskit Ignis: 错误表征和缓解组件")
print("- Qiskit Aqua: 跨领域量子算法库")
print("- Qiskit Machine Learning: 量子机器学习库")
print("- Qiskit Nature: 量子化学和物理模拟库")
print("- Qiskit Finance: 量子金融应用库")
print("- Qiskit Optimization: 量子优化算法库")

# 2. 创建第一个量子电路
print("\n2. 创建第一个量子电路")
print("在Qiskit中，量子电路是使用QuantumCircuit类创建的")
print("以下是一个2量子比特的Bell态电路")

# 创建一个有2个量子比特和2个经典比特的量子电路
qc = QuantumCircuit(2, 2)

# 对第一个量子比特应用Hadamard门
qc.h(0)

# 对第一个和第二个量子比特应用CNOT门 (受控非门)
qc.cx(0, 1)

# 测量量子比特
qc.measure([0, 1], [0, 1])

# 显示电路图
print("Bell态电路:")
print(qc.draw())

# 3. 量子寄存器和经典寄存器
print("\n3. 量子寄存器和经典寄存器")
print("Qiskit使用量子寄存器(QuantumRegister)和经典寄存器(ClassicalRegister)来组织量子比特和经典比特")

from qiskit import QuantumRegister, ClassicalRegister

# 创建两个量子寄存器
qr1 = QuantumRegister(2, name="q1")  # 2个量子比特，名称为q1
qr2 = QuantumRegister(1, name="q2")  # 1个量子比特，名称为q2

# 创建两个经典寄存器
cr1 = ClassicalRegister(2, name="c1")  # 2个经典比特，名称为c1
cr2 = ClassicalRegister(1, name="c2")  # 1个经典比特，名称为c2

# 使用寄存器创建量子电路
qc_registers = QuantumCircuit(qr1, qr2, cr1, cr2)

# 对第一个寄存器的第一个量子比特应用H门
qc_registers.h(qr1[0])

# 在两个不同寄存器的量子比特之间应用CNOT门
qc_registers.cx(qr1[0], qr2[0])

# 测量第一个量子寄存器到第一个经典寄存器
qc_registers.measure(qr1, cr1)

# 测量第二个量子寄存器到第二个经典寄存器
qc_registers.measure(qr2, cr2)

print("使用多个寄存器的电路:")
print(qc_registers.draw())

# 4. 模拟量子电路
print("\n4. 模拟量子电路")
print("Qiskit提供多种模拟器，最基本的是状态向量模拟器和QASM模拟器")

# 获取状态向量模拟器后端
simulator = Aer.get_backend('statevector_simulator')

# 创建一个简单的Bell态电路（不包括测量）
bell = QuantumCircuit(2)
bell.h(0)
bell.cx(0, 1)

# 执行模拟
job = simulator.run(transpile(bell, simulator))
result = job.result()

# 获取状态向量
statevector = result.get_statevector()
print("Bell态的状态向量:")
print(statevector)

# 通过状态向量可视化Bell态（在交互环境中才能显示）
print("状态向量可视化结果将保存到文件")
fig = plot_bloch_multivector(statevector)
fig.savefig('bell_state_bloch.png')
plt.close(fig)

# 使用QASM模拟器运行包含测量的电路
qasm_simulator = Aer.get_backend('qasm_simulator')

# 使用前面创建的Bell态电路（包含测量）
job = qasm_simulator.run(transpile(qc, qasm_simulator), shots=1024)
result = job.result()

# 获取计数结果
counts = result.get_counts()
print("\n模拟1024次测量的结果:")
print(counts)

# 结果可视化
print("计数结果可视化将保存到文件")
fig = plot_histogram(counts)
fig.savefig('bell_state_histogram.png')
plt.close(fig)

# 5. Qiskit中的电路编译和优化
print("\n5. Qiskit中的电路编译和优化")
print("量子电路在执行前，需要通过transpile转换为特定后端支持的门集")

# 创建一个简单电路
qc_original = QuantumCircuit(2)
qc_original.h(0)
qc_original.cx(0, 1)
qc_original.z(1)
qc_original.x(0)

print("原始电路:")
print(qc_original.draw())

# 对不同后端的编译优化
backend_sim = Aer.get_backend('qasm_simulator')
qc_transpiled = transpile(qc_original, backend_sim, optimization_level=1)

print("\n编译后的电路 (optimization_level=1):")
print(qc_transpiled.draw())

# 更高级的优化
qc_optimized = transpile(qc_original, backend_sim, optimization_level=3)
print("\n高度优化的电路 (optimization_level=3):")
print(qc_optimized.draw())

# 6. Qiskit Provider体系
print("\n6. Qiskit Provider体系")
print("Qiskit使用Provider模型来管理不同的量子后端")
print("主要包括:")
print("- Aer: 模拟器提供者")
print("- IBMQ: IBM真实量子设备提供者")
print("- 第三方提供者: 其他供应商的量子设备")

# 获取可用的Aer模拟器
print("\nAer提供的模拟器:")
for backend in Aer.backends():
    print(f"- {backend.name()}")

# 连接IBMQ需要账号和API密钥
# 这里只展示代码结构，不实际运行
print("\n连接IBMQ的代码示例 (需要API密钥):")
print("""
# 加载已保存的账号
IBMQ.load_account()

# 获取提供者
provider = IBMQ.get_provider(hub='ibm-q')

# 获取可用后端
for backend in provider.backends():
    print(backend.name())
    
# 选择一个后端
backend = provider.get_backend('ibmq_qasm_simulator')
""")

# 7. 自定义Qiskit组件
print("\n7. 自定义Qiskit组件")
print("Qiskit允许您创建自定义门、电路和其他组件")

# 示例：创建一个自定义电路函数，生成一个GHZ状态
def create_ghz_circuit(num_qubits):
    """创建一个生成GHZ状态的电路"""
    qc = QuantumCircuit(num_qubits)
    
    # 对第一个量子比特应用H门
    qc.h(0)
    
    # 对所有其他量子比特使用CNOT门
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    
    return qc

# 创建3量子比特GHZ态
ghz = create_ghz_circuit(3)
print("3量子比特GHZ电路:")
print(ghz.draw())

# 模拟GHZ态
simulator = Aer.get_backend('statevector_simulator')
result = simulator.run(transpile(ghz, simulator)).result()
statevector = result.get_statevector()

# 打印状态向量（理论上应该是|000⟩和|111⟩的均匀叠加）
print("\n3量子比特GHZ态的状态向量:")
print(statevector)

# 8. 总结
print("\n8. 总结")
print("1. Qiskit是一个全面的量子计算软件开发工具包")
print("2. 它提供了创建、转译和执行量子电路的工具")
print("3. 多种模拟器可用于不同类型的量子计算任务")
print("4. 可以连接到IBM真实量子设备运行电路")
print("5. 支持高级的电路优化和分析功能")
print("6. 提供了丰富的量子算法和应用库")

print("\n下一步学习:")
print("- 创建更复杂的量子电路")
print("- 深入了解各种量子门")
print("- 探索量子算法实现")
print("- 使用可视化工具分析结果")
print("- 连接到真实量子计算机")
print("- 学习量子误差缓解技术") 