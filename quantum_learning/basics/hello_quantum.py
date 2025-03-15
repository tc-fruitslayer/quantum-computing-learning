#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
量子计算入门示例：创建一个简单的量子电路
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator

# 创建一个有2个量子比特的量子电路
qc = QuantumCircuit(2, 2)

# 将第一个量子比特置于叠加态
qc.h(0)

# 使用CNOT门将两个量子比特纠缠在一起
qc.cx(0, 1)

# 测量两个量子比特
qc.measure([0, 1], [0, 1])

# 打印电路
print("量子电路:")
print(qc)

# 使用Aer模拟器运行电路
simulator = AerSimulator()
compiled_circuit = transpile(qc, simulator)
job = simulator.run(compiled_circuit, shots=1000)
result = job.result()

# 获取结果并打印
counts = result.get_counts(compiled_circuit)
print("\n测量结果:", counts)
print("\n这是一个Bell态，展示了量子纠缠的特性。")
print("当测量一个量子比特时，另一个量子比特的状态会立即确定，无论它们相距多远。") 