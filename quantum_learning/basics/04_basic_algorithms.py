#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
量子计算基础 4：量子算法基础
本文件介绍几种基本的量子算法，包括量子干涉、Deutsch-Jozsa算法和量子相位估计
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram

print("===== 量子算法基础 =====")

# 1. 量子干涉
print("\n1. 量子干涉")
print("   量子干涉是许多量子算法的基础")
print("   通过Hadamard门创建叠加态，执行不同的操作，然后再次应用Hadamard门")
print("   使波幅相长干涉或相消干涉，从而提取信息")

# 创建一个展示量子干涉的简单电路
interference_qc = QuantumCircuit(2, 2)

# 在两个量子比特上应用Hadamard门，创建叠加态
interference_qc.h(0)
interference_qc.h(1)

# 在第二个量子比特上应用Z门，引入相位
interference_qc.z(1)

# 再次应用Hadamard门
interference_qc.h(0)
interference_qc.h(1)

# 测量结果
interference_qc.measure([0, 1], [0, 1])

print("量子干涉电路：")
print(interference_qc.draw())

# 运行电路
simulator = Aer.get_backend('qasm_simulator')
compiled_circuit = transpile(interference_qc, simulator)
job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
counts = result.get_counts(compiled_circuit)
print(f"测量结果: {counts}")
print("注意观察测量结果中的干涉模式")

# 2. Deutsch-Jozsa算法
print("\n2. Deutsch-Jozsa算法")
print("   这是量子计算优于经典计算的最简单例子之一")
print("   问题：给定一个黑盒函数f: {0,1}^n -> {0,1}，判断它是常数函数还是平衡函数")
print("   经典算法：最坏情况下需要2^(n-1)+1次查询")
print("   量子算法：只需1次查询")

# 实现Deutsch-Jozsa算法 - 以1量子比特为例（即Deutsch算法）
print("\nDeutsch算法演示（单比特情况）:")

# 创建常数函数f(x) = 0的Oracle
def constant_oracle():
    qc = QuantumCircuit(2)  # 2个量子比特：1个查询比特，1个辅助比特
    # 不执行任何操作 - f(x) = 0
    return qc

# 创建常数函数f(x) = 1的Oracle
def constant_oracle_1():
    qc = QuantumCircuit(2)
    qc.x(1)  # 将辅助比特从|0⟩变为|1⟩ - f(x) = 1
    return qc

# 创建平衡函数f(x) = x的Oracle（身份函数）
def balanced_oracle_1():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)  # CNOT门，将辅助比特设置为输入值 - f(x) = x
    return qc

# 创建平衡函数f(x) = NOT(x)的Oracle
def balanced_oracle_2():
    qc = QuantumCircuit(2)
    qc.x(0)      # 翻转输入
    qc.cx(0, 1)  # CNOT门 - f(x) = NOT(x)
    qc.x(0)      # 恢复输入
    return qc

# 选择一个Oracle进行演示 - 这里使用平衡函数
oracle = balanced_oracle_1()

# 创建Deutsch电路
deutsch_qc = QuantumCircuit(2, 1)

# 初始化
deutsch_qc.x(1)
deutsch_qc.h(0)
deutsch_qc.h(1)

# 应用Oracle
deutsch_qc = deutsch_qc.compose(oracle)

# 最终的Hadamard门和测量
deutsch_qc.h(0)
deutsch_qc.measure(0, 0)

print("Deutsch算法电路（判断函数是常数还是平衡）：")
print(deutsch_qc.draw())

# 运行电路
simulator = Aer.get_backend('qasm_simulator')
compiled_circuit = transpile(deutsch_qc, simulator)
job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
counts = result.get_counts(compiled_circuit)
print(f"测量结果: {counts}")
print("解释：如果测量结果为|0⟩，则函数是常数；如果为|1⟩，则函数是平衡")

# 3. 量子相位估计
print("\n3. 量子相位估计")
print("   量子相位估计是许多重要量子算法的核心，如Shor因数分解算法")
print("   目标：估计幺正算子U的特征值e^(2πiθ)中的相位θ")
print("   算法使用量子傅里叶变换（QFT）将相位信息从特征向量转移到计算基上")

# 实现一个简单的相位估计示例 - 估计Z门的相位
print("\n量子相位估计演示（估计Z门的相位）:")

# Z门的特征值是：+1（对应相位θ=0）和-1（对应相位θ=1/2）
# 我们使用|1⟩作为Z门的特征向量，对应特征值-1

# 创建相位估计电路 - 3个精度比特
phase_qc = QuantumCircuit(4, 3)  # 3个精度比特 + 1个特征向量比特

# 准备特征向量|1⟩
phase_qc.x(3)

# 在精度比特上应用Hadamard门
for i in range(3):
    phase_qc.h(i)

# 应用受控U^(2^j)操作
# 对Z门，U^(2^j)仍然是Z门，只是相位被放大了2^j倍
phase_qc.cp(np.pi, 0, 3)  # 受控Z门（控制比特0，目标比特3）- 相当于U^(2^0)
phase_qc.cp(2*np.pi, 1, 3)  # 相当于U^(2^1)
phase_qc.cp(4*np.pi, 2, 3)  # 相当于U^(2^2)

# 应用逆量子傅里叶变换（这里直接使用QFT_dagger）
phase_qc.h(2)
phase_qc.cp(-np.pi/2, 1, 2)
phase_qc.h(1)
phase_qc.cp(-np.pi/4, 0, 1)
phase_qc.cp(-np.pi/2, 0, 2)
phase_qc.h(0)

# 测量精度比特
phase_qc.measure(range(3), range(3))

print("量子相位估计电路（估计Z门的相位）：")
print(phase_qc.draw())

# 运行电路
simulator = Aer.get_backend('qasm_simulator')
compiled_circuit = transpile(phase_qc, simulator)
job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
counts = result.get_counts(compiled_circuit)
print(f"测量结果: {counts}")
print("解释：测量结果应当接近'100'，表示相位θ≈0.5（转换为二进制是0.100...）")
print("      这与Z门作用于|1⟩得到特征值-1=e^(iπ)（相位θ=0.5）相符")

# 4. 总结和对比
print("\n4. 算法对比和量子加速")
print("   Deutsch-Jozsa算法：指数级加速（从O(2^n)到O(1)）")
print("   Grover搜索算法：二次加速（从O(N)到O(√N)）")
print("   Shor因数分解算法：指数级加速（从超多项式时间到多项式时间）")
print("\n这些算法展示了量子计算的三种主要技术：")
print("1. 量子并行性 - 通过叠加态同时处理多个输入")
print("2. 量子干涉 - 通过相位操作增强正确答案的概率振幅")
print("3. 量子纠缠 - 创建多量子比特之间的相关性")

print("\n总结：")
print("1. 量子算法可以为特定问题提供显著的速度提升")
print("2. 量子干涉是量子算法的基本工具")
print("3. 量子算法通常遵循相似的模式：初始化叠加态→应用特定变换→测量")
print("4. 量子相位估计是许多高级量子算法的核心子程序")
print("5. 随着量子硬件的发展，这些算法将在实际问题上展示优势") 