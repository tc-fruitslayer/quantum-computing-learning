#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IBM Qiskit框架学习 - 基础练习
本文件包含一系列帮助理解Qiskit基础概念的练习题
"""

# 导入必要的库
from qiskit import QuantumCircuit, Aer, execute, transpile
from qiskit.visualization import plot_histogram, plot_bloch_multivector
import matplotlib.pyplot as plt
import numpy as np

print("===== Qiskit基础练习 =====")
print("完成以下练习来测试您对Qiskit基础的理解")
print("每个练习都有一个或多个任务，请尝试独立完成")
print("练习后有提示和参考解答")

# --------------------------------
# 练习1: 创建和运行第一个量子电路
# --------------------------------
print("\n练习1: 创建和运行第一个量子电路")
print("任务1: 创建一个包含1个量子比特和1个经典比特的量子电路")
print("任务2: 对量子比特应用一个X门(NOT门)")
print("任务3: 测量量子比特并将结果存储到经典比特")
print("任务4: 使用QASM模拟器运行电路1000次并打印结果")

# 提示
print("\n提示:")
print("- 使用QuantumCircuit(1, 1)创建电路")
print("- 使用circuit.x(0)应用X门")
print("- 使用circuit.measure(0, 0)进行测量")
print("- 使用simulator = Aer.get_backend('qasm_simulator')获取模拟器")
print("- 使用execute(circuit, simulator, shots=1000)执行电路")

# 参考解答
def exercise1_solution():
    # 任务1: 创建电路
    circuit = QuantumCircuit(1, 1)
    
    # 任务2: 应用X门
    circuit.x(0)
    
    # 任务3: 测量
    circuit.measure(0, 0)
    
    # 任务4: 运行电路
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(circuit, simulator, shots=1000)
    result = job.result()
    counts = result.get_counts()
    
    print("\n参考解答 - 练习1:")
    print("电路:")
    print(circuit.draw())
    print("\n测量结果:")
    print(counts)
    print("\n解释: 应用X门将|0⟩转换为|1⟩，因此所有测量结果都是'1'")
    
    return circuit, counts

# 取消注释下面的行以查看参考解答
# circuit1, counts1 = exercise1_solution()

# --------------------------------
# 练习2: 创建叠加态
# --------------------------------
print("\n练习2: 创建叠加态")
print("任务1: 创建一个包含1个量子比特和1个经典比特的量子电路")
print("任务2: 将量子比特初始化为叠加态(|0⟩+|1⟩)/√2")
print("任务3: 测量量子比特")
print("任务4: 运行电路1000次并分析结果")

# 提示
print("\n提示:")
print("- 使用Hadamard门(circuit.h(0))创建叠加态")
print("- 叠加态测量时，应该有约50%概率得到0，50%概率得到1")

# 参考解答
def exercise2_solution():
    # 任务1: 创建电路
    circuit = QuantumCircuit(1, 1)
    
    # 任务2: 创建叠加态
    circuit.h(0)
    
    # 任务3: 测量
    circuit.measure(0, 0)
    
    # 任务4: 运行电路
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(circuit, simulator, shots=1000)
    result = job.result()
    counts = result.get_counts()
    
    print("\n参考解答 - 练习2:")
    print("电路:")
    print(circuit.draw())
    print("\n测量结果:")
    print(counts)
    print("\n解释: Hadamard门将|0⟩转换为(|0⟩+|1⟩)/√2，测量结果应该接近50%/50%分布")
    
    # 可视化结果
    fig = plot_histogram(counts)
    plt.title("叠加态测量结果")
    plt.savefig('exercise2_histogram.png')
    plt.close(fig)
    print("直方图已保存为'exercise2_histogram.png'")
    
    return circuit, counts

# 取消注释下面的行以查看参考解答
# circuit2, counts2 = exercise2_solution()

# --------------------------------
# 练习3: 创建Bell态
# --------------------------------
print("\n练习3: 创建Bell态")
print("任务1: 创建一个包含2个量子比特和2个经典比特的量子电路")
print("任务2: 创建Bell态 (|00⟩+|11⟩)/√2")
print("任务3: 测量两个量子比特")
print("任务4: 运行电路1000次并分析结果")

# 提示
print("\n提示:")
print("- 使用Hadamard门创建第一个量子比特的叠加态")
print("- 使用CNOT门(circuit.cx(0, 1))使两个量子比特纠缠")
print("- Bell态测量时，应该有约50%概率得到00，50%概率得到11")

# 参考解答
def exercise3_solution():
    # 任务1: 创建电路
    circuit = QuantumCircuit(2, 2)
    
    # 任务2: 创建Bell态
    circuit.h(0)
    circuit.cx(0, 1)
    
    # 任务3: 测量
    circuit.measure([0, 1], [0, 1])
    
    # 任务4: 运行电路
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(circuit, simulator, shots=1000)
    result = job.result()
    counts = result.get_counts()
    
    print("\n参考解答 - 练习3:")
    print("电路:")
    print(circuit.draw())
    print("\n测量结果:")
    print(counts)
    print("\n解释: Bell态是(|00⟩+|11⟩)/√2，测量结果应该只有'00'和'11'，且接近50%/50%分布")
    
    # 可视化结果
    fig = plot_histogram(counts)
    plt.title("Bell态测量结果")
    plt.savefig('exercise3_histogram.png')
    plt.close(fig)
    print("直方图已保存为'exercise3_histogram.png'")
    
    # 可视化Bell态的状态向量
    statevector_sim = Aer.get_backend('statevector_simulator')
    # 创建不带测量的电路以获取状态向量
    bell_circuit = QuantumCircuit(2)
    bell_circuit.h(0)
    bell_circuit.cx(0, 1)
    
    result = execute(bell_circuit, statevector_sim).result()
    statevector = result.get_statevector()
    
    fig = plot_bloch_multivector(statevector)
    plt.savefig('exercise3_bloch.png')
    plt.close(fig)
    print("Bloch球表示已保存为'exercise3_bloch.png'")
    
    return circuit, counts

# 取消注释下面的行以查看参考解答
# circuit3, counts3 = exercise3_solution()

# --------------------------------
# 练习4: 量子比特的相位
# --------------------------------
print("\n练习4: 量子比特的相位")
print("任务1: 创建一个包含1个量子比特和1个经典比特的量子电路")
print("任务2: 将量子比特初始化为|+⟩状态 (|0⟩+|1⟩)/√2")
print("任务3: 应用一个Z门以改变相位")
print("任务4: 应用一个Hadamard门将相位信息转换为振幅")
print("任务5: 测量并分析结果")

# 提示
print("\n提示:")
print("- 使用H门创建|+⟩状态")
print("- Z门将|+⟩转换为|-⟩ (|0⟩-|1⟩)/√2")
print("- 再次应用H门将|-⟩转换为|1⟩")
print("- 最终测量应该几乎总是得到'1'")

# 参考解答
def exercise4_solution():
    # 任务1: 创建电路
    circuit = QuantumCircuit(1, 1)
    
    # 任务2: 创建|+⟩状态
    circuit.h(0)
    
    # 任务3: 应用Z门
    circuit.z(0)
    
    # 任务4: 应用H门
    circuit.h(0)
    
    # 任务5: 测量
    circuit.measure(0, 0)
    
    # 运行电路
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(circuit, simulator, shots=1000)
    result = job.result()
    counts = result.get_counts()
    
    print("\n参考解答 - 练习4:")
    print("电路:")
    print(circuit.draw())
    print("\n测量结果:")
    print(counts)
    print("\n解释: H门将|0⟩转换为|+⟩，Z门将|+⟩转换为|-⟩，再次应用H门将|-⟩转换为|1⟩")
    
    # 可视化中间状态
    # 步骤1: 初始状态 |0⟩
    init_circuit = QuantumCircuit(1)
    
    # 步骤2: 应用H门 -> |+⟩
    h_circuit = QuantumCircuit(1)
    h_circuit.h(0)
    
    # 步骤3: 应用Z门 -> |-⟩
    z_circuit = QuantumCircuit(1)
    z_circuit.h(0)
    z_circuit.z(0)
    
    # 步骤4: 再次应用H门 -> |1⟩
    hz_circuit = QuantumCircuit(1)
    hz_circuit.h(0)
    hz_circuit.z(0)
    hz_circuit.h(0)
    
    # 获取所有状态向量
    statevector_sim = Aer.get_backend('statevector_simulator')
    result0 = execute(init_circuit, statevector_sim).result()
    state0 = result0.get_statevector()
    
    result1 = execute(h_circuit, statevector_sim).result()
    state1 = result1.get_statevector()
    
    result2 = execute(z_circuit, statevector_sim).result()
    state2 = result2.get_statevector()
    
    result3 = execute(hz_circuit, statevector_sim).result()
    state3 = result3.get_statevector()
    
    print("\n量子态的演化:")
    print(f"初始态 |0⟩: {state0}")
    print(f"应用H门后 |+⟩: {state1}")
    print(f"应用Z门后 |-⟩: {state2}")
    print(f"再次应用H门后 |1⟩: {state3}")
    
    return circuit, counts

# 取消注释下面的行以查看参考解答
# circuit4, counts4 = exercise4_solution()

# --------------------------------
# 练习5: 创建GHZ态
# --------------------------------
print("\n练习5: 创建GHZ态")
print("任务1: 创建一个包含3个量子比特和3个经典比特的量子电路")
print("任务2: 创建GHZ态 (|000⟩+|111⟩)/√2")
print("任务3: 测量所有量子比特")
print("任务4: 运行电路1000次并分析结果")

# 提示
print("\n提示:")
print("- GHZ态是Bell态的扩展")
print("- 对第一个量子比特应用H门，然后用CNOT门连接其他量子比特")
print("- 测量结果应该只有'000'和'111'，且接近50%/50%分布")

# 参考解答
def exercise5_solution():
    # 任务1: 创建电路
    circuit = QuantumCircuit(3, 3)
    
    # 任务2: 创建GHZ态
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    
    # 任务3: 测量
    circuit.measure([0, 1, 2], [0, 1, 2])
    
    # 任务4: 运行电路
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(circuit, simulator, shots=1000)
    result = job.result()
    counts = result.get_counts()
    
    print("\n参考解答 - 练习5:")
    print("电路:")
    print(circuit.draw())
    print("\n测量结果:")
    print(counts)
    print("\n解释: GHZ态是(|000⟩+|111⟩)/√2，测量结果应该只有'000'和'111'，且接近50%/50%分布")
    
    # 可视化结果
    fig = plot_histogram(counts)
    plt.title("GHZ态测量结果")
    plt.savefig('exercise5_histogram.png')
    plt.close(fig)
    print("直方图已保存为'exercise5_histogram.png'")
    
    return circuit, counts

# 取消注释下面的行以查看参考解答
# circuit5, counts5 = exercise5_solution()

# --------------------------------
# 练习6: 检测纠缠
# --------------------------------
print("\n练习6: 检测纠缠")
print("任务1: 创建一个Bell态 (|00⟩+|11⟩)/√2")
print("任务2: 使用状态向量模拟器获取完整的量子态")
print("任务3: 检查两个量子比特的纠缠")

# 提示
print("\n提示:")
print("- 使用Aer.get_backend('statevector_simulator')获取状态向量")
print("- 纠缠态不能表示为单个量子比特状态的张量积")
print("- 可以计算归约密度矩阵并检查其纯度")

# 参考解答
def exercise6_solution():
    # 任务1: 创建Bell态
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    
    # 任务2: 获取状态向量
    simulator = Aer.get_backend('statevector_simulator')
    result = execute(circuit, simulator).result()
    statevector = result.get_statevector()
    
    print("\n参考解答 - 练习6:")
    print("电路:")
    print(circuit.draw())
    print("\nBell态的状态向量:")
    print(statevector)
    
    # 任务3: 检查纠缠
    # 为了简化，我们计算一个量子比特的约化密度矩阵
    from qiskit.quantum_info import partial_trace, DensityMatrix
    
    # 创建密度矩阵
    rho = DensityMatrix(statevector)
    
    # 计算第一个量子比特的约化密度矩阵
    rho_0 = partial_trace(rho, [1])
    
    # 计算纯度，对于纯态应该是1，对于最大混合态是0.5
    purity = rho_0.purity()
    
    print("\n第一个量子比特的约化密度矩阵:")
    print(rho_0)
    print(f"\n纯度: {purity}")
    
    if abs(purity - 0.5) < 0.01:
        print("\n结论: 两个量子比特是最大纠缠的")
    else:
        print("\n结论: 两个量子比特不是最大纠缠的")
        
    # 可视化完整密度矩阵
    from qiskit.visualization import plot_state_city
    fig = plot_state_city(rho)
    plt.savefig('exercise6_density_matrix.png')
    plt.close(fig)
    print("密度矩阵表示已保存为'exercise6_density_matrix.png'")
    
    return circuit, statevector

# 取消注释下面的行以查看参考解答
# circuit6, statevector6 = exercise6_solution()

# --------------------------------
# 练习7: 量子随机数发生器
# --------------------------------
print("\n练习7: 量子随机数发生器")
print("任务1: 创建一个包含8个量子比特和8个经典比特的量子电路")
print("任务2: 对每个量子比特应用H门以创建均匀叠加态")
print("任务3: 测量所有量子比特获取一个随机字节")
print("任务4: 运行电路多次并验证随机性")

# 提示
print("\n提示:")
print("- 量子测量的随机性可以用来生成随机数")
print("- 可以使用每次运行的结果生成一个随机字节")
print("- 使用直方图或其他统计方法验证分布的均匀性")

# 参考解答
def exercise7_solution():
    # 任务1: 创建电路
    num_qubits = 8
    circuit = QuantumCircuit(num_qubits, num_qubits)
    
    # 任务2: 创建均匀叠加态
    for i in range(num_qubits):
        circuit.h(i)
    
    # 任务3: 测量所有量子比特
    circuit.measure(range(num_qubits), range(num_qubits))
    
    # 任务4: 运行电路多次
    num_shots = 10  # 只生成10个随机字节作为示例
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(circuit, simulator, shots=num_shots)
    result = job.result()
    counts = result.get_counts()
    
    print("\n参考解答 - 练习7:")
    print("电路:")
    print(circuit.draw())
    
    # 将结果转换为整数
    random_bytes = []
    for bitstring in counts.keys():
        val = int(bitstring, 2)
        random_bytes.extend([val] * counts[bitstring])
    
    print("\n生成的随机字节:")
    for byte in random_bytes:
        print(f"{byte} (二进制: {byte:08b})")
    
    # 生成更多随机数并验证分布
    verification_shots = 1000
    job = execute(circuit, simulator, shots=verification_shots)
    result = job.result()
    counts = result.get_counts()
    
    # 简单的随机性检验
    print(f"\n生成了{verification_shots}个随机字节")
    print(f"得到了{len(counts)}种不同的值")
    
    # 如果分布均匀，我们期望每个值出现的次数大约相同
    # 在8量子比特情况下，有256种可能的值，每个值出现频率约为 1/256
    expected_count = verification_shots / 256
    all_counts = list(counts.values())
    min_count = min(all_counts)
    max_count = max(all_counts)
    avg_count = sum(all_counts) / len(all_counts)
    
    print(f"理论上每个值应出现约{expected_count:.2f}次")
    print(f"实际最小出现次数: {min_count}")
    print(f"实际最大出现次数: {max_count}")
    print(f"实际平均出现次数: {avg_count:.2f}")
    
    # 对于真正的随机数发生器，应该进行更严格的统计测试
    
    return circuit, random_bytes

# 取消注释下面的行以查看参考解答
# circuit7, random_bytes7 = exercise7_solution()

# --------------------------------
# 总结
# --------------------------------
print("\n==== 练习总结 ====")
print("完成这些练习后，您应该已经掌握了Qiskit的基本用法，包括:")
print("1. 创建量子电路并应用基本量子门")
print("2. 创建叠加态和纠缠态")
print("3. 在模拟器上运行量子电路并分析结果")
print("4. 理解量子比特的相位和测量")
print("5. 实现简单的量子算法")

print("\n要查看参考解答，请取消注释相应的函数调用")
print("练习是学习的关键部分，建议先尝试自己解决，然后再参考解答")
print("祝您在量子计算的学习道路上取得进步！") 