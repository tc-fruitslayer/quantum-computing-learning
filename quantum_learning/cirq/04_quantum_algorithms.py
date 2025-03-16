#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Google Cirq框架学习 4：量子算法实现
本文件详细介绍如何使用Cirq实现经典量子算法
"""

import cirq
import numpy as np
import matplotlib.pyplot as plt
import sympy
from typing import List, Dict, Tuple

print("===== Cirq中的量子算法实现 =====")

# 1. Deutsch-Jozsa算法
print("\n1. Deutsch-Jozsa算法")
print("目标：确定一个黑盒函数是常数函数还是平衡函数")
print("常数函数: f(x) 对所有输入返回相同的值(全0或全1)")
print("平衡函数: f(x) 对一半的输入返回0，另一半输入返回1")

# 1.1 实现常数函数的Oracle
print("\n1.1 常数函数的Oracle")
def deutsch_jozsa_constant_oracle(qubits, target, constant_value=0):
    """创建一个常数函数的量子Oracle
    
    Args:
        qubits: 输入量子比特列表
        target: 目标/输出量子比特
        constant_value: 常数值（0或1）
    
    Returns:
        包含Oracle操作的cirq.Circuit
    """
    circuit = cirq.Circuit()
    
    # 如果常数值为1，则对目标比特应用X门
    if constant_value == 1:
        circuit.append(cirq.X(target))
    
    return circuit

# 1.2 实现平衡函数的Oracle
print("\n1.2 平衡函数的Oracle")
def deutsch_jozsa_balanced_oracle(qubits, target):
    """创建一个平衡函数的量子Oracle
    平衡函数将一半的输入映射到0，另一半映射到1
    在这个例子中，我们创建一个将目标比特与第一个输入比特进行XOR的Oracle
    
    Args:
        qubits: 输入量子比特列表
        target: 目标/输出量子比特
    
    Returns:
        包含Oracle操作的cirq.Circuit
    """
    circuit = cirq.Circuit()
    # 简单的平衡函数：f(x) = x_0
    circuit.append(cirq.CNOT(qubits[0], target))
    return circuit

# 1.3 完整的Deutsch-Jozsa算法
print("\n1.3 完整的Deutsch-Jozsa算法")
def deutsch_jozsa_algorithm(n_qubits, oracle_type='constant', constant_value=0):
    """实现Deutsch-Jozsa算法
    
    Args:
        n_qubits: 输入量子比特的数量
        oracle_type: Oracle类型 ('constant' 或 'balanced')
        constant_value: 如果使用常数函数，指定常数值（0或1）
    
    Returns:
        包含完整算法的cirq.Circuit
    """
    # 创建量子比特
    input_qubits = [cirq.LineQubit(i) for i in range(n_qubits)]
    output_qubit = cirq.LineQubit(n_qubits)
    
    circuit = cirq.Circuit()
    
    # 初始化输出量子比特为|1>
    circuit.append(cirq.X(output_qubit))
    
    # 对所有量子比特应用Hadamard门
    circuit.append(cirq.H.on_each(input_qubits))
    circuit.append(cirq.H(output_qubit))
    
    # 应用Oracle
    if oracle_type == 'constant':
        oracle_circuit = deutsch_jozsa_constant_oracle(input_qubits, output_qubit, constant_value)
    else:  # balanced
        oracle_circuit = deutsch_jozsa_balanced_oracle(input_qubits, output_qubit)
    
    circuit.append(oracle_circuit)
    
    # 再次对输入量子比特应用Hadamard门
    circuit.append(cirq.H.on_each(input_qubits))
    
    # 测量所有输入量子比特
    circuit.append(cirq.measure(*input_qubits, key='result'))
    
    return circuit

# 演示Deutsch-Jozsa算法
n_qubits = 3  # 使用3个输入量子比特

# 常数函数的情况
constant_circuit = deutsch_jozsa_algorithm(n_qubits, 'constant', 0)
print("\n常数函数(f(x) = 0)的Deutsch-Jozsa电路:")
print(constant_circuit)

# 平衡函数的情况
balanced_circuit = deutsch_jozsa_algorithm(n_qubits, 'balanced')
print("\n平衡函数的Deutsch-Jozsa电路:")
print(balanced_circuit)

# 模拟并解释结果
simulator = cirq.Simulator()

# 常数函数的结果
constant_result = simulator.run(constant_circuit, repetitions=10)
print("\n常数函数的结果:")
print(constant_result)

# 平衡函数的结果
balanced_result = simulator.run(balanced_circuit, repetitions=10)
print("\n平衡函数的结果:")
print(balanced_result)

print("\nDeutsch-Jozsa算法解释:")
print("- 如果所有输入量子比特的测量结果都是0，函数是常数的")
print("- 如果有任何非0的测量结果，函数是平衡的")

# 2. Grover搜索算法
print("\n2. Grover搜索算法")
print("目标：在未排序数据中找到满足条件的元素")

# 2.1 Grover Oracle
print("\n2.1 创建Grover Oracle")
def grover_oracle(qubits, marked_states):
    """为Grover算法创建一个Oracle
    
    Args:
        qubits: 量子比特列表
        marked_states: 标记状态的列表（以二进制字符串表示）
    
    Returns:
        Oracle操作
    """
    # 将标记的状态转换为整数
    marked_indices = [int(state, 2) for state in marked_states]
    
    # 创建多控制Z门
    return cirq.Z.on_each([qubits[i] for i in range(len(qubits))])

# 2.2 Grover扩散算子
print("\n2.2 Grover扩散算子")
def grover_diffusion(qubits):
    """创建Grover扩散算子
    
    Args:
        qubits: 量子比特列表
    
    Returns:
        包含扩散算子的cirq.Circuit
    """
    n = len(qubits)
    circuit = cirq.Circuit()
    
    # 对所有量子比特应用Hadamard门
    circuit.append(cirq.H.on_each(qubits))
    
    # 应用多控制Z门 (或等效操作)
    # 先对所有量子比特应用X门
    circuit.append(cirq.X.on_each(qubits))
    
    # 添加多控制Z门（这里简化为CZ和受控操作）
    if n > 1:
        control_qubits = qubits[:-1]
        target_qubit = qubits[-1]
        circuit.append(cirq.H(target_qubit))
        circuit.append(cirq.CNOT(control_qubits[0], target_qubit))
        
        if n > 2:
            for i in range(1, len(control_qubits)):
                circuit.append(cirq.CNOT(control_qubits[i], target_qubit))
        
        circuit.append(cirq.H(target_qubit))
    
    # 再次对所有量子比特应用X门
    circuit.append(cirq.X.on_each(qubits))
    
    # 再次对所有量子比特应用Hadamard门
    circuit.append(cirq.H.on_each(qubits))
    
    return circuit

# 2.3 完整的Grover算法
print("\n2.3 完整的Grover算法")
def grover_algorithm(n_qubits, marked_states, num_iterations=None):
    """实现Grover搜索算法
    
    Args:
        n_qubits: 量子比特数量
        marked_states: 标记状态列表（以二进制字符串表示）
        num_iterations: Grover迭代次数，如果为None则使用最优迭代次数
    
    Returns:
        包含完整Grover算法的cirq.Circuit
    """
    # 创建量子比特
    qubits = [cirq.LineQubit(i) for i in range(n_qubits)]
    
    # 计算最优迭代次数
    N = 2**n_qubits
    M = len(marked_states)
    if num_iterations is None:
        num_iterations = int(np.round(np.pi/4 * np.sqrt(N/M)))
    
    circuit = cirq.Circuit()
    
    # 初始化：对所有量子比特应用Hadamard门
    circuit.append(cirq.H.on_each(qubits))
    
    # Grover迭代
    for _ in range(num_iterations):
        # Oracle
        oracle_circuit = grover_oracle(qubits, marked_states)
        circuit.append(oracle_circuit)
        
        # 扩散算子
        diffusion_circuit = grover_diffusion(qubits)
        circuit.append(diffusion_circuit)
    
    # 测量所有量子比特
    circuit.append(cirq.measure(*qubits, key='result'))
    
    return circuit

# 演示Grover算法
n_qubits = 3  # 使用3个量子比特（8个可能的状态）
marked_states = ['101']  # 标记状态 |101⟩
num_iterations = 2  # Grover迭代次数

print("\n为搜索元素'101'创建Grover电路:")
grover_circuit = grover_algorithm(n_qubits, marked_states, num_iterations)
print(grover_circuit)

# 模拟并解释结果
simulator = cirq.Simulator()
grover_result = simulator.run(grover_circuit, repetitions=100)

# 分析结果
results = grover_result.measurements['result']
counts = {}
for bits in results:
    bits_str = ''.join(str(int(bit)) for bit in bits)
    counts[bits_str] = counts.get(bits_str, 0) + 1

print("\nGrover搜索结果:")
for state, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
    print(f"|{state}⟩: {count} 次 ({count/100:.2f})")

print("\nGrover算法解释:")
print(f"- 理论上，{num_iterations}次Grover迭代后，标记状态|{marked_states[0]}⟩的概率应该显著增加")
print("- 模拟结果显示标记状态的测量频率明显高于其他状态")

# 3. 量子相位估计算法
print("\n3. 量子相位估计算法")
print("目标：估计酉算子的特征值的相位")

# 3.1 创建受控U操作
print("\n3.1 受控U操作")
def controlled_u_power(control, target, u_gate, power):
    """创建受控U^power操作
    
    Args:
        control: 控制量子比特
        target: 目标量子比特
        u_gate: 要控制的U门
        power: U门的幂
    
    Returns:
        受控U^power操作
    """
    # 对于简单的情况，我们可以使用内置的支持
    # 例如，对于U=Z，我们可以使用受控Z门
    if isinstance(u_gate, cirq.ZPowGate):
        return cirq.CZ(control, target)**(power)
    elif isinstance(u_gate, cirq.XPowGate):
        return cirq.CNOT(control, target)
    else:
        # 对于一般情况，我们需要更复杂的构造
        controlled_u = cirq.ControlledGate(sub_gate=u_gate**power)
        return controlled_u.on(control, target)

# 3.2 量子相位估计算法
print("\n3.2 量子相位估计算法实现")
def quantum_phase_estimation(unitary, precision_qubits, target_qubits):
    """实现量子相位估计算法
    
    Args:
        unitary: 要估计其特征值的酉矩阵（作为门操作）
        precision_qubits: 用于精度的量子比特列表
        target_qubits: 特征向量量子比特
    
    Returns:
        包含量子相位估计算法的cirq.Circuit
    """
    n = len(precision_qubits)
    circuit = cirq.Circuit()
    
    # 准备特征向量
    # 注意：在实际使用中，你需要确保target_qubits处于unitary的特征向量状态
    # 这里我们简化为已经在正确状态
    
    # 对精度量子比特应用Hadamard门
    circuit.append(cirq.H.on_each(precision_qubits))
    
    # 应用受控U^(2^j)操作
    for j in range(n):
        power = 2**j
        for target in target_qubits:
            circuit.append(controlled_u_power(precision_qubits[j], target, unitary, power))
    
    # 逆量子傅里叶变换 (QFT†)
    circuit.append(cirq.qft(*precision_qubits, inverse=True))
    
    # 测量精度量子比特
    circuit.append(cirq.measure(*precision_qubits, key='phase'))
    
    return circuit

# 演示量子相位估计
print("\n演示量子相位估计算法:")
# 使用Z门作为我们要估计相位的酉算子
# Z门的特征值是 e^(i*pi) (-1) 对应相位 0.5
unitary = cirq.Z

# 设置量子比特
precision_qubits = [cirq.LineQubit(i) for i in range(4)]  # 4位精度
target_qubit = cirq.LineQubit(4)  # 目标量子比特

# 创建电路
qpe_circuit = cirq.Circuit()

# 初始化目标量子比特为|1⟩（Z门的特征向量）
qpe_circuit.append(cirq.X(target_qubit))

# 添加相位估计算法的主要部分
qpe_circuit += quantum_phase_estimation(unitary, precision_qubits, [target_qubit])

print(qpe_circuit)

# 模拟并解释结果
simulator = cirq.Simulator()
qpe_result = simulator.run(qpe_circuit, repetitions=100)

# 分析结果
results = qpe_result.measurements['phase']
counts = {}
for bits in results:
    bits_str = ''.join(str(int(bit)) for bit in bits)
    decimal = int(bits_str, 2) / (2**len(precision_qubits))
    rounded = round(decimal, 3)
    counts[rounded] = counts.get(rounded, 0) + 1

print("\n量子相位估计结果:")
for phase, count in sorted(counts.items()):
    print(f"相位 {phase}: {count} 次 ({count/100:.2f})")

print("\n量子相位估计解释:")
print("- Z门的特征值是 -1 = e^(i*pi)，对应的相位是 0.5")
print("- 我们使用4个精度量子比特，理论上能够分辨 2^4 = 16 个不同的相位")
print("- 结果应该集中在相位 0.5 附近")

# 4. 量子傅里叶变换 (QFT)
print("\n4. 量子傅里叶变换")
print("目标：实现量子版本的傅里叶变换")

# 4.1 自定义QFT函数
print("\n4.1 自定义量子傅里叶变换")
def custom_qft(qubits, inverse=False):
    """实现量子傅里叶变换
    
    Args:
        qubits: 量子比特列表
        inverse: 如果为True，则实现逆变换
    
    Returns:
        包含QFT或逆QFT的cirq.Circuit
    """
    n = len(qubits)
    circuit = cirq.Circuit()
    
    # 如果是逆变换，反转量子比特顺序
    if inverse:
        qubits = qubits[::-1]
    
    # 实现QFT
    for i in range(n):
        # Hadamard门
        circuit.append(cirq.H(qubits[i]))
        
        # 条件旋转门
        for j in range(i+1, n):
            k = j - i
            theta = np.pi / (2**k)
            if inverse:
                theta = -theta
            circuit.append(cirq.CZ(qubits[i], qubits[j])**(theta/(np.pi)))
    
    # 如果是逆变换，再次反转量子比特顺序以恢复原始顺序
    if inverse:
        for i in range(n//2):
            circuit.append(cirq.SWAP(qubits[i], qubits[n-i-1]))
    
    return circuit

# 4.2 演示QFT
print("\n4.2 演示量子傅里叶变换")
# 设置量子比特
qft_qubits = [cirq.LineQubit(i) for i in range(4)]

# 创建一个简单的初始状态
qft_circuit = cirq.Circuit()
# 设置初始状态为|0001⟩
qft_circuit.append(cirq.X(qft_qubits[3]))

# 应用QFT
qft_circuit.append(custom_qft(qft_qubits))
print("应用QFT前后的电路:")
print(qft_circuit)

# 模拟QFT结果
simulator = cirq.Simulator()
qft_result = simulator.simulate(qft_circuit)

print("\nQFT后的状态向量:")
state = qft_result.final_state_vector
for i, amplitude in enumerate(state):
    if abs(amplitude) > 1e-10:
        binary = format(i, f'0{len(qft_qubits)}b')
        print(f"|{binary}⟩: {amplitude}")

# 应用逆QFT验证变换的正确性
inverse_qft_circuit = qft_circuit.copy()
inverse_qft_circuit.append(custom_qft(qft_qubits, inverse=True))

# 模拟逆QFT结果
inverse_result = simulator.simulate(inverse_qft_circuit)

print("\nQFT后接逆QFT的结果:")
inverse_state = inverse_result.final_state_vector
for i, amplitude in enumerate(inverse_state):
    if abs(amplitude) > 1e-10:
        binary = format(i, f'0{len(qft_qubits)}b')
        print(f"|{binary}⟩: {amplitude}")

print("\n量子傅里叶变换解释:")
print("- QFT是经典FFT的量子版本，但在量子计算中使用指数更少的操作")
print("- QFT将基态|x⟩转换为各相位的均匀叠加")
print("- 逆QFT将这种叠加状态转换回原始基态")
print("- QFT是Shor算法等许多量子算法的关键组件")

# 5. Shor算法（简化版）
print("\n5. Shor算法概述（简化版）")
print("目标：分解大整数为质因数")
print("注意：完整的Shor算法实现非常复杂，这里只提供概念性的演示")

print("\nShor算法主要步骤:")
print("1. 选择随机数a，并确保它与要分解的数N互质")
print("2. 找到a^r mod N = 1的最小正整数r（周期）")
print("3. 如果r是偶数且a^(r/2) mod N ≠ -1，计算gcd(a^(r/2)±1, N)")
print("4. 这些最大公约数很可能是N的非平凡因子")

print("\n量子部分主要用于高效找到周期r:")
print("- 创建两个量子寄存器")
print("- 对第一个寄存器应用Hadamard门")
print("- 实现模幂函数f(x) = a^x mod N作为量子门")
print("- 对第一个寄存器应用逆QFT")
print("- 测量并后处理得到周期r")

print("\n完整的Shor算法需要更复杂的电路构造")
print("特别是模幂函数的量子实现非常复杂")
print("建议参考专门的资源来深入理解和实现Shor算法")

# 总结
print("\n总结:")
print("1. Deutsch-Jozsa算法展示了量子计算的速度优势")
print("2. Grover搜索提供了在未排序数据中查找元素的二次加速")
print("3. 量子相位估计是许多高级量子算法的基础")
print("4. 量子傅里叶变换是经典FFT的量子版本，效率更高")
print("5. Shor算法（完整实现）可以有效分解大整数，对经典密码学构成威胁") 