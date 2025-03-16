#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IBM Qiskit框架学习 5：量子算法实现
本文件详细介绍Qiskit中经典量子算法的实现和应用
"""

# 导入必要的库
from qiskit import QuantumCircuit, Aer, execute, transpile
from qiskit.visualization import plot_histogram
from qiskit.algorithms import Grover, AmplificationProblem, Shor, PhaseEstimation
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import QFT, PhaseEstimation as PhaseEstimationCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import Operator
from qiskit.opflow import X, Z, I
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction

print("===== Qiskit量子算法实现 =====")

# 1. Deutsch-Jozsa算法
print("\n1. Deutsch-Jozsa算法")
print("Deutsch-Jozsa算法可以判断一个黑盒函数是常数函数还是平衡函数")

# 实现常数函数（全0或全1）的Oracle
def dj_constant_oracle(n):
    """返回一个n量子比特的常数Oracle"""
    oracle = QuantumCircuit(n+1)
    # 如果要输出1，则在目标量子比特上添加X门
    # 这里我们选择输出0，所以不需要添加额外的门
    return oracle

# 实现平衡函数Oracle
def dj_balanced_oracle(n):
    """返回一个n量子比特的平衡Oracle"""
    oracle = QuantumCircuit(n+1)
    # 对每个输入量子比特施加CNOT门，以控制目标量子比特
    for i in range(n):
        oracle.cx(i, n)
    return oracle

# 实现Deutsch-Jozsa算法
def deutsch_jozsa_algorithm(oracle, n):
    """实现Deutsch-Jozsa算法
    
    参数:
        oracle (QuantumCircuit): Oracle电路
        n (int): 量子比特数量
    
    返回:
        QuantumCircuit: 完整的Deutsch-Jozsa电路
    """
    dj_circuit = QuantumCircuit(n+1, n)
    
    # 初始化目标量子比特到|1⟩
    dj_circuit.x(n)
    
    # 对所有量子比特应用H门
    for qubit in range(n+1):
        dj_circuit.h(qubit)
    
    # 添加Oracle
    dj_circuit = dj_circuit.compose(oracle)
    
    # 再次对输入量子比特应用H门
    for qubit in range(n):
        dj_circuit.h(qubit)
    
    # 测量
    dj_circuit.measure(range(n), range(n))
    
    return dj_circuit

# 执行常数函数测试
n = 3  # 使用3个量子比特
constant_oracle = dj_constant_oracle(n)
dj_constant_circuit = deutsch_jozsa_algorithm(constant_oracle, n)

print("Deutsch-Jozsa电路 (常数函数):")
print(dj_constant_circuit.draw())

# 执行平衡函数测试
balanced_oracle = dj_balanced_oracle(n)
dj_balanced_circuit = deutsch_jozsa_algorithm(balanced_oracle, n)

print("\nDeutsch-Jozsa电路 (平衡函数):")
print(dj_balanced_circuit.draw())

# 模拟电路执行
simulator = Aer.get_backend('qasm_simulator')
constant_result = execute(dj_constant_circuit, simulator, shots=1024).result()
constant_counts = constant_result.get_counts()

balanced_result = execute(dj_balanced_circuit, simulator, shots=1024).result()
balanced_counts = balanced_result.get_counts()

print("\n常数函数结果:")
print(constant_counts)
print("全0结果表示函数是常数函数")

print("\n平衡函数结果:")
print(balanced_counts)
print("其他结果表示函数是平衡函数")

# 2. Bernstein-Vazirani算法
print("\n2. Bernstein-Vazirani算法")
print("Bernstein-Vazirani算法可以一次性确定一个黑盒函数的隐藏位串")

# 实现具有隐藏位串的Oracle
def bv_oracle(hidden_string):
    """返回一个具有隐藏位串的Oracle
    
    参数:
        hidden_string (str): 隐藏的位串，如'101'
    
    返回:
        QuantumCircuit: Oracle电路
    """
    n = len(hidden_string)
    oracle = QuantumCircuit(n+1)
    
    # 将目标量子比特置于|-⟩态
    oracle.x(n)
    oracle.h(n)
    
    # 对于隐藏串中为1的每个位，添加一个CNOT门
    for i in range(n):
        if hidden_string[i] == '1':
            oracle.cx(i, n)
    
    return oracle

# 实现Bernstein-Vazirani算法
def bernstein_vazirani_algorithm(oracle, n):
    """实现Bernstein-Vazirani算法
    
    参数:
        oracle (QuantumCircuit): Oracle电路
        n (int): 量子比特数量
    
    返回:
        QuantumCircuit: 完整的Bernstein-Vazirani电路
    """
    bv_circuit = QuantumCircuit(n+1, n)
    
    # 初始化目标量子比特到|1⟩
    bv_circuit.x(n)
    
    # 对所有量子比特应用H门
    for qubit in range(n+1):
        bv_circuit.h(qubit)
    
    # 添加Oracle
    bv_circuit = bv_circuit.compose(oracle)
    
    # 再次对输入量子比特应用H门
    for qubit in range(n):
        bv_circuit.h(qubit)
    
    # 测量
    bv_circuit.measure(range(n), range(n))
    
    return bv_circuit

# 执行Bernstein-Vazirani算法
hidden_string = '101'  # 隐藏的位串
n = len(hidden_string)
bv_oracle_circuit = bv_oracle(hidden_string)
bv_circuit = bernstein_vazirani_algorithm(bv_oracle_circuit, n)

print("Bernstein-Vazirani电路:")
print(bv_circuit.draw())

# 模拟电路执行
simulator = Aer.get_backend('qasm_simulator')
bv_result = execute(bv_circuit, simulator, shots=1024).result()
bv_counts = bv_result.get_counts()

print("\nBernstein-Vazirani结果:")
print(bv_counts)
print(f"最频繁的结果应该与隐藏位串{hidden_string}匹配")

# 3. Grover搜索算法
print("\n3. Grover搜索算法")
print("Grover算法是一种量子搜索算法，可以在O(√N)时间内在无序数据库中找到目标项")

# 创建一个简单的Grover Oracle，标记指定的状态
def grover_oracle(marked_states, n_qubits):
    """创建一个标记指定状态的Oracle
    
    参数:
        marked_states (list): 要标记的状态列表，如['101']
        n_qubits (int): 量子比特数
    
    返回:
        Operator: Oracle算子
    """
    # 创建一个零矩阵
    oracle_matrix = np.zeros((2**n_qubits, 2**n_qubits))
    
    # 对角线上全部设为1
    for i in range(2**n_qubits):
        oracle_matrix[i, i] = 1
    
    # 对标记的状态反转符号
    for state in marked_states:
        # 将二进制字符串转换为整数
        idx = int(state, 2)
        oracle_matrix[idx, idx] = -1
    
    return Operator(oracle_matrix)

# 实现Grover算法
def grover_algorithm(oracle, n_qubits, n_iterations=1):
    """实现Grover搜索算法
    
    参数:
        oracle (Operator): Oracle算子
        n_qubits (int): 量子比特数
        n_iterations (int): Grover迭代次数
    
    返回:
        QuantumCircuit: 完整的Grover电路
    """
    # 初始化电路
    grover_circuit = QuantumCircuit(n_qubits, n_qubits)
    
    # 初始化为均匀叠加态
    grover_circuit.h(range(n_qubits))
    
    # 实现指定次数的Grover迭代
    for _ in range(n_iterations):
        # 应用Oracle
        grover_circuit.append(oracle, range(n_qubits))
        
        # 应用扩散算子
        grover_circuit.h(range(n_qubits))
        grover_circuit.x(range(n_qubits))
        
        # 多控制Z门
        grover_circuit.h(n_qubits-1)
        grover_circuit.mct(list(range(n_qubits-1)), n_qubits-1)  # 多控制Toffoli门
        grover_circuit.h(n_qubits-1)
        
        grover_circuit.x(range(n_qubits))
        grover_circuit.h(range(n_qubits))
    
    # 测量所有量子比特
    grover_circuit.measure(range(n_qubits), range(n_qubits))
    
    return grover_circuit

# 执行Grover算法
n_qubits = 3
marked_states = ['101']  # 要搜索的状态

# 计算最优迭代次数
n_iterations = int(np.pi/4 * np.sqrt(2**n_qubits / len(marked_states)))
print(f"最优Grover迭代次数: {n_iterations}")

# 创建Oracle
oracle = grover_oracle(marked_states, n_qubits)

# 创建并执行Grover电路
grover_circuit = grover_algorithm(oracle, n_qubits, n_iterations)

print("Grover搜索电路:")
print(grover_circuit.draw())

# 模拟电路执行
simulator = Aer.get_backend('qasm_simulator')
grover_result = execute(grover_circuit, simulator, shots=1024).result()
grover_counts = grover_result.get_counts()

print("\nGrover搜索结果:")
print(grover_counts)
print(f"结果应该集中在标记的状态{marked_states}上")

# 使用Qiskit的内置Grover实现
print("\n使用Qiskit内置Grover实现:")

# 定义要搜索的布尔函数
def oracle_function(x):
    return x == '101'

# 定义搜索问题
problem = AmplificationProblem(
    oracle=oracle_function,
    state_preparation=QuantumCircuit(n_qubits).h(range(n_qubits))
)

# 创建Grover算法实例
grover = Grover(iterations=n_iterations)

# 执行Grover算法
result = grover.amplify(problem)
print(f"测量结果: {result.top_measurement}")
print(f"成功概率: {result.assignment_probability:.4f}")

# 4. 量子相位估计
print("\n4. 量子相位估计")
print("量子相位估计是许多量子算法的基础，如Shor算法")

# 实现量子相位估计
def phase_estimation_example(phase, n_counting_qubits):
    """使用量子相位估计电路估计相位
    
    参数:
        phase (float): 要估计的相位 (0到1之间)
        n_counting_qubits (int): 相位估计使用的量子比特数
    
    返回:
        QuantumCircuit: 量子相位估计电路
    """
    # 创建量子相位估计电路
    qpe_circuit = QuantumCircuit(n_counting_qubits + 1, n_counting_qubits)
    
    # 准备目标量子比特的特征态 |1⟩
    qpe_circuit.x(n_counting_qubits)
    
    # 对相位估计寄存器应用H门
    for qubit in range(n_counting_qubits):
        qpe_circuit.h(qubit)
    
    # 应用受控相位旋转
    for i in range(n_counting_qubits):
        angle = phase * 2*np.pi * 2**(n_counting_qubits-1-i)
        qpe_circuit.cp(angle, i, n_counting_qubits)
    
    # 应用逆QFT
    qpe_circuit.append(QFT(n_counting_qubits).inverse(), range(n_counting_qubits))
    
    # 测量相位估计寄存器
    qpe_circuit.measure(range(n_counting_qubits), range(n_counting_qubits))
    
    return qpe_circuit

# 执行量子相位估计
phase = 0.25  # 要估计的相位 (这里是1/4)
n_counting_qubits = 4  # 相位估计使用的量子比特数

qpe_circuit = phase_estimation_example(phase, n_counting_qubits)

print("量子相位估计电路:")
print(qpe_circuit.draw())

# 模拟电路执行
simulator = Aer.get_backend('qasm_simulator')
qpe_result = execute(qpe_circuit, simulator, shots=1024).result()
qpe_counts = qpe_result.get_counts()

# 打印结果并分析
print("\n量子相位估计结果:")
for bitstring, count in qpe_counts.items():
    decimal = int(bitstring, 2) / (2**n_counting_qubits)
    print(f"测量值: {bitstring} -> 相位估计: {decimal:.4f}, 计数: {count}")

# 使用Qiskit的内置QPE实现
print("\n使用Qiskit内置QPE实现:")

# 创建一个简单的酉算子，其特征值的相位是我们要估计的
theta = phase * 2 * np.pi
unitary = np.array([[np.exp(1j * theta), 0], [0, np.exp(-1j * theta)]])
u_gate = Operator(unitary)

# 创建目标状态准备电路
state_preparation = QuantumCircuit(1)
state_preparation.x(0)

# 创建相位估计电路
pe = PhaseEstimation(n_counting_qubits, state_preparation, u_gate)

# 执行相位估计
result = pe.run(simulator)
print(f"估计的相位: {result.phase}")
print(f"最接近的分数: {Fraction(result.phase).limit_denominator(100)}")

# 5. 量子傅里叶变换
print("\n5. 量子傅里叶变换")
print("量子傅里叶变换是多个量子算法的核心组件")

# 创建QFT电路
def create_qft_circuit(n_qubits):
    """创建n个量子比特的QFT电路
    
    参数:
        n_qubits (int): 量子比特数
    
    返回:
        QuantumCircuit: QFT电路
    """
    qft_circuit = QuantumCircuit(n_qubits)
    
    # 实现QFT
    for i in range(n_qubits):
        qft_circuit.h(i)
        for j in range(i+1, n_qubits):
            qft_circuit.cp(np.pi/float(2**(j-i)), j, i)
    
    # 交换量子比特顺序
    for i in range(n_qubits//2):
        qft_circuit.swap(i, n_qubits-1-i)
    
    return qft_circuit

# 创建一个示例电路，先准备一个状态然后应用QFT
def qft_example():
    """创建一个QFT示例电路"""
    n_qubits = 3
    
    # 创建电路
    qft_example_circuit = QuantumCircuit(n_qubits, n_qubits)
    
    # 准备一个简单的状态
    qft_example_circuit.x(0)  # |001⟩
    
    # 应用QFT
    qft = create_qft_circuit(n_qubits)
    qft_example_circuit = qft_example_circuit.compose(qft)
    
    # 测量
    qft_example_circuit.measure(range(n_qubits), range(n_qubits))
    
    return qft_example_circuit

# 执行QFT示例
qft_example_circuit = qft_example()

print("QFT示例电路:")
print(qft_example_circuit.draw())

# 模拟电路执行
simulator = Aer.get_backend('qasm_simulator')
qft_result = execute(qft_example_circuit, simulator, shots=1024).result()
qft_counts = qft_result.get_counts()

print("\nQFT结果:")
print(qft_counts)

# 使用Qiskit的内置QFT
print("\n使用Qiskit内置QFT:")
qiskit_qft = QFT(3)
print(qiskit_qft.draw())

# 6. VQE (变分量子特征值求解器)
print("\n6. VQE (变分量子特征值求解器)")
print("VQE是一种混合量子-经典算法，用于找到哈密顿量的最低特征值")

# 创建一个简单的哈密顿量
hamiltonian = Z ^ I + I ^ Z + 0.5 * X ^ X

print("哈密顿量:")
print(hamiltonian)

# 创建一个简单的变分形式
ansatz = EfficientSU2(2, reps=1)

print("变分形式电路:")
print(ansatz.draw())

# 实际的VQE实现需要经典优化器和多次测量，这里为了简化，我们手动测试一些参数

# 定义一个函数来计算期望值
def compute_expectation(parameters):
    """计算给定参数下的哈密顿量期望值
    
    参数:
        parameters (list): 变分形式的参数
    
    返回:
        float: 期望值
    """
    # 绑定参数
    bound_circuit = ansatz.bind_parameters(parameters)
    
    # 模拟电路
    simulator = Aer.get_backend('statevector_simulator')
    result = execute(bound_circuit, simulator).result()
    statevector = result.get_statevector()
    
    # 计算哈密顿量的期望值
    from qiskit.quantum_info import Statevector
    sv = Statevector(statevector)
    expectation = sv.expectation_value(hamiltonian)
    
    return expectation.real

# 测试一些参数
test_parameters = [
    [0, 0, 0, 0],
    [np.pi/4, 0, 0, 0],
    [np.pi/2, 0, 0, 0],
    [np.pi/4, np.pi/4, 0, 0]
]

print("\nVQE参数测试:")
for params in test_parameters:
    exp_val = compute_expectation(params)
    print(f"参数: {params} -> 期望值: {exp_val:.6f}")

# 7. 总结
print("\n7. 总结")
print("1. Deutsch-Jozsa算法可以一次性判断函数是常数函数还是平衡函数")
print("2. Bernstein-Vazirani算法可以一次性找到隐藏位串")
print("3. Grover搜索算法可以在无序数据库中实现平方加速搜索")
print("4. 量子相位估计是Shor算法等高级量子算法的基础")
print("5. 量子傅里叶变换在多个量子算法中扮演关键角色")
print("6. VQE是一种混合量子-经典算法，用于解决量子化学等领域的问题")

print("\n下一步学习:")
print("- 实现更复杂的量子算法，如Shor算法和HHL算法")
print("- 探索量子机器学习算法")
print("- 学习如何将实际问题映射到量子算法")
print("- 在真实量子硬件上运行量子算法") 