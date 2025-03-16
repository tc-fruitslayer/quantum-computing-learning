#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IBM Qiskit框架学习 - 基础练习解答
本文件包含对应练习的完整解答
"""

# 导入必要的库
from qiskit import QuantumCircuit, Aer, execute, transpile
from qiskit.visualization import plot_histogram, plot_bloch_multivector, plot_state_city
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------
# 练习1解答: 创建和运行第一个量子电路
# --------------------------------
def exercise1_solution():
    """创建贝尔态电路的解答"""
    # 创建2量子比特电路
    circuit = QuantumCircuit(2, 2)
    
    # 对第一个量子比特应用Hadamard门
    circuit.h(0)
    
    # 添加CNOT门，从量子比特0到量子比特1
    circuit.cx(0, 1)
    
    # 测量量子比特
    circuit.measure([0, 1], [0, 1])
    
    # 运行电路
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(circuit, simulator, shots=1024)
    result = job.result()
    counts = result.get_counts(circuit)
    
    # 打印结果
    print("\n练习1解答 - 贝尔态电路:")
    print(circuit.draw())
    print("\n测量结果:")
    print(counts)
    
    # 可视化结果
    plt.figure(figsize=(8, 6))
    plot_histogram(counts)
    plt.title("贝尔态测量结果")
    plt.savefig('bell_state_histogram.png')
    plt.close()
    
    return circuit, counts

# --------------------------------
# 练习2解答: 制备不同的量子态
# --------------------------------
def exercise2_solution():
    """制备不同量子态的解答"""
    # 创建电路
    qc1 = QuantumCircuit(1)  # |0⟩状态，不需要额外操作
    
    qc2 = QuantumCircuit(1)  # |1⟩状态
    qc2.x(0)
    
    qc3 = QuantumCircuit(1)  # |+⟩状态
    qc3.h(0)
    
    qc4 = QuantumCircuit(1)  # |−⟩状态
    qc4.x(0)
    qc4.h(0)
    
    qc5 = QuantumCircuit(1)  # |+i⟩状态
    qc5.h(0)
    qc5.s(0)
    
    qc6 = QuantumCircuit(1)  # |−i⟩状态
    qc6.h(0)
    qc6.sdg(0)
    
    # 获取所有状态向量
    simulator = Aer.get_backend('statevector_simulator')
    states = {}
    
    for i, qc in enumerate([qc1, qc2, qc3, qc4, qc5, qc6]):
        job = execute(qc, simulator)
        result = job.result()
        statevector = result.get_statevector()
        states[f"state{i+1}"] = statevector
    
    # 打印结果
    print("\n练习2解答 - 制备不同量子态:")
    state_names = ["$|0\\rangle$", "$|1\\rangle$", "$|+\\rangle$", "$|-\\rangle$", "$|+i\\rangle$", "$|-i\\rangle$"]
    
    for i, (state_key, statevector) in enumerate(states.items()):
        print(f"\n{state_names[i]}态的电路:")
        if i == 0:
            print(qc1.draw())
        elif i == 1:
            print(qc2.draw())
        elif i == 2:
            print(qc3.draw())
        elif i == 3:
            print(qc4.draw())
        elif i == 4:
            print(qc5.draw())
        else:
            print(qc6.draw())
        print(f"状态向量: {statevector}")
    
    # 可视化Bloch球
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (state_key, statevector) in enumerate(states.items()):
        plot_bloch_multivector(statevector, ax=axes[i], title=state_names[i])
    
    plt.tight_layout()
    plt.savefig('quantum_states_bloch.png')
    plt.close(fig)
    
    return states

# --------------------------------
# 练习3解答: 测量不同基底
# --------------------------------
def exercise3_solution():
    """在不同基底测量的解答"""
    # 创建电路
    qc1 = QuantumCircuit(1, 1)  # Z基测量
    qc1.h(0)  # 准备|+⟩态
    qc1.measure(0, 0)  # 在Z基测量
    
    qc2 = QuantumCircuit(1, 1)  # X基测量
    qc2.h(0)  # 准备|+⟩态
    qc2.h(0)  # 在测量前应用H，将X基转换为Z基
    qc2.measure(0, 0)  # 在Z基测量(实际上是X基)
    
    qc3 = QuantumCircuit(1, 1)  # Y基测量
    qc3.h(0)  # 准备|+⟩态
    qc3.sdg(0)  # 在测量前应用S†和H
    qc3.h(0)
    qc3.measure(0, 0)  # 在Z基测量(实际上是Y基)
    
    # 运行电路
    simulator = Aer.get_backend('qasm_simulator')
    results = {}
    
    for i, qc in enumerate([qc1, qc2, qc3]):
        job = execute(qc, simulator, shots=1024)
        result = job.result()
        counts = result.get_counts(qc)
        results[f"basis{i+1}"] = counts
    
    # 打印结果
    print("\n练习3解答 - 在不同基底测量:")
    basis_names = ["Z基(计算基)", "X基", "Y基"]
    
    for i, (basis_key, counts) in enumerate(results.items()):
        print(f"\n在{basis_names[i]}中测量|+⟩态:")
        if i == 0:
            print(qc1.draw())
        elif i == 1:
            print(qc2.draw())
        else:
            print(qc3.draw())
        print(f"测量结果: {counts}")
    
    # 可视化结果
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (basis_key, counts) in enumerate(results.items()):
        plot_histogram(counts, ax=axes[i], title=f"在{basis_names[i]}中测量|+⟩态")
    
    plt.tight_layout()
    plt.savefig('measurement_bases.png')
    plt.close(fig)
    
    return results

# --------------------------------
# 练习4解答: 制备GHZ态
# --------------------------------
def exercise4_solution():
    """制备GHZ态的解答"""
    # 创建3量子比特GHZ电路
    ghz_circuit = QuantumCircuit(3, 3)
    
    # 应用H门到第一个量子比特
    ghz_circuit.h(0)
    
    # 应用CNOT门扩展叠加
    ghz_circuit.cx(0, 1)
    ghz_circuit.cx(0, 2)
    
    # 测量
    ghz_circuit.measure([0, 1, 2], [0, 1, 2])
    
    # 运行电路
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(ghz_circuit, simulator, shots=1024)
    result = job.result()
    counts = result.get_counts(ghz_circuit)
    
    # 打印结果
    print("\n练习4解答 - GHZ态:")
    print(ghz_circuit.draw())
    print("\n测量结果:")
    print(counts)
    
    # 获取状态向量
    sv_simulator = Aer.get_backend('statevector_simulator')
    ghz_sv_circuit = QuantumCircuit(3)
    ghz_sv_circuit.h(0)
    ghz_sv_circuit.cx(0, 1)
    ghz_sv_circuit.cx(0, 2)
    
    job_sv = execute(ghz_sv_circuit, sv_simulator)
    statevector = job_sv.result().get_statevector()
    
    print("\nGHZ态状态向量:")
    print(statevector)
    
    # 可视化结果
    plt.figure(figsize=(8, 6))
    plot_histogram(counts)
    plt.title("GHZ态测量结果")
    plt.savefig('ghz_histogram.png')
    plt.close()
    
    # 使用城市图可视化状态向量
    fig = plt.figure(figsize=(10, 8))
    plot_state_city(statevector)
    plt.title("GHZ态的状态向量表示")
    plt.savefig('ghz_statevector.png')
    plt.close(fig)
    
    return ghz_circuit, counts, statevector

# --------------------------------
# 练习5解答: 纠缠与贝尔不等式
# --------------------------------
def exercise5_solution():
    """纠缠与贝尔不等式的解答"""
    # 创建纠缠态电路
    bell_circuit = QuantumCircuit(2)
    bell_circuit.h(0)
    bell_circuit.cx(0, 1)
    
    # 在不同角度测量并计算关联
    angle_pairs = [
        (0, np.pi/4),       # A1,B1
        (0, 3*np.pi/4),     # A1,B2
        (np.pi/2, np.pi/4), # A2,B1
        (np.pi/2, 3*np.pi/4) # A2,B2
    ]
    
    results = {}
    correlations = {}
    
    # 测量每一对角度
    for i, (theta_a, theta_b) in enumerate(angle_pairs):
        # 创建纠缠后的测量电路
        meas_circuit = QuantumCircuit(2, 2)
        meas_circuit.h(0)
        meas_circuit.cx(0, 1)
        
        # 在特定角度测量
        # 实际中需要旋转测量基底
        meas_circuit.ry(-theta_a, 0)
        meas_circuit.ry(-theta_b, 1)
        
        meas_circuit.measure([0, 1], [0, 1])
        
        # 运行模拟
        simulator = Aer.get_backend('qasm_simulator')
        job = execute(meas_circuit, simulator, shots=1024)
        counts = job.result().get_counts()
        results[f"angle_pair_{i}"] = counts
        
        # 计算关联值 E = P(00) + P(11) - P(01) - P(10)
        correlation = 0
        total_shots = 0
        for outcome, count in counts.items():
            total_shots += count
            # 检查结果是否一致
            if outcome == '00' or outcome == '11':
                correlation += count
            else:
                correlation -= count
        
        # 归一化
        correlation /= total_shots
        correlations[f"E({i})"] = correlation
    
    # 计算CHSH不等式 S = E(A1,B1) - E(A1,B2) + E(A2,B1) + E(A2,B2)
    chsh_value = correlations["E(0)"] - correlations["E(1)"] + correlations["E(2)"] + correlations["E(3)"]
    
    # 打印结果
    print("\n练习5解答 - 纠缠与贝尔不等式:")
    print("贝尔态电路:")
    print(bell_circuit.draw())
    
    print("\n在不同角度对测量结果:")
    for i, (angle_pair, counts) in enumerate(results.items()):
        theta_a, theta_b = angle_pairs[i]
        print(f"\n角度对 θA={theta_a:.4f}, θB={theta_b:.4f}:")
        print(f"测量结果: {counts}")
        print(f"关联值 E = {correlations[f'E({i})']:.4f}")
    
    print(f"\nCHSH不等式值 S = {chsh_value:.4f}")
    print(f"经典极限为2，量子力学极限为2√2≈2.82")
    if abs(chsh_value) > 2:
        print("结果违反了贝尔不等式，证明量子纠缠的非局域性!")
    
    # 可视化结果
    labels = [f"(θA={theta_a:.2f}, θB={theta_b:.2f})" for theta_a, theta_b in angle_pairs]
    correlation_values = [correlations[f"E({i})"] for i in range(len(angle_pairs))]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, correlation_values)
    plt.axhline(y=1, color='r', linestyle='--', label='经典极限')
    plt.axhline(y=-1, color='r', linestyle='--')
    plt.ylabel('关联值 E')
    plt.title('不同角度对的量子关联')
    plt.ylim(-1.1, 1.1)
    plt.savefig('bell_correlations.png')
    plt.close()
    
    return results, correlations, chsh_value

# 执行解答
if __name__ == "__main__":
    print("===== Qiskit基础练习解答 =====")
    
    # 取消注释以运行特定练习的解答
    # circuit1, counts1 = exercise1_solution()
    # states = exercise2_solution()
    # measurement_results = exercise3_solution()
    # ghz_circuit, ghz_counts, ghz_statevector = exercise4_solution()
    # bell_results, bell_correlations, chsh_value = exercise5_solution()
    
    print("\n所有练习解答已准备就绪!")
    print("取消注释相应的函数调用以运行特定练习的解答") 