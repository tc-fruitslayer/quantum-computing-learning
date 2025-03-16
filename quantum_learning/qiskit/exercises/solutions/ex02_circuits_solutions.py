#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IBM Qiskit框架学习 - 量子电路练习解答
本文件包含对应练习的完整解答
"""

# 导入必要的库
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute, transpile
from qiskit.visualization import plot_histogram, plot_bloch_multivector, plot_state_city
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFT
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------
# 练习1解答: 电路组合与复合
# --------------------------------
def exercise1_solution():
    """电路组合与复合的解答"""
    # 任务1: 创建两个独立电路
    # 第一个电路: 对第一个量子比特应用H门，对第二个量子比特应用X门
    circuit1 = QuantumCircuit(2)
    circuit1.h(0)
    circuit1.x(1)
    
    # 第二个电路: 应用CNOT门和Z门
    circuit2 = QuantumCircuit(2)
    circuit2.cx(0, 1)
    circuit2.z(0)
    
    # 任务2: 组合电路
    combined_circuit = circuit1.compose(circuit2)
    
    # 任务3: 使用状态向量模拟器
    simulator = Aer.get_backend('statevector_simulator')
    
    # 模拟第一个电路
    job1 = execute(circuit1, simulator)
    state1 = job1.result().get_statevector()
    
    # 模拟第二个电路
    init_state = Statevector.from_label('00')  # 初始状态 |00⟩
    job2 = execute(circuit2, simulator)
    state2 = init_state.evolve(circuit2)
    
    # 模拟组合电路
    job_combined = execute(combined_circuit, simulator)
    state_combined = job_combined.result().get_statevector()
    
    # 打印结果
    print("\n练习1解答 - 电路组合与复合:")
    print("电路1:")
    print(circuit1.draw())
    print("\n电路1的状态向量:")
    print(state1)
    
    print("\n电路2:")
    print(circuit2.draw())
    
    print("\n组合后的电路:")
    print(combined_circuit.draw())
    print("\n组合电路的状态向量:")
    print(state_combined)
    print("\n解释: 组合电路执行了电路1和电路2的操作序列，最终结果是两个电路效果的结合")
    
    # 可视化最终状态
    fig = plot_bloch_multivector(state_combined)
    plt.savefig('exercise1_bloch.png')
    plt.close(fig)
    print("Bloch球表示已保存为'exercise1_bloch.png'")
    
    return combined_circuit, state_combined

# --------------------------------
# 练习2解答: 使用量子寄存器
# --------------------------------
def exercise2_solution():
    """使用量子寄存器的解答"""
    # 任务1: 创建寄存器
    qr1 = QuantumRegister(2, 'q1')  # 第一个量子寄存器，2个量子比特
    qr2 = QuantumRegister(1, 'q2')  # 第二个量子寄存器，1个量子比特
    cr = ClassicalRegister(3, 'c')  # 经典寄存器，3个经典比特
    
    # 任务2: 创建电路
    circuit = QuantumCircuit(qr1, qr2, cr)
    
    # 任务3: 应用门
    circuit.h(qr1[0])  # 对第一个寄存器的第一个量子比特应用H门
    circuit.h(qr1[1])  # 对第一个寄存器的第二个量子比特应用H门
    circuit.x(qr2[0])  # 对第二个寄存器的量子比特应用X门
    
    # 任务4: 应用受控门
    circuit.cx(qr1[0], qr2[0])  # 从qr1[0]到qr2[0]的CNOT门
    circuit.cz(qr1[1], qr2[0])  # 从qr1[1]到qr2[0]的CZ门
    
    # 任务5: 测量和运行
    circuit.measure(qr1, cr[0:2])  # 测量第一个寄存器到前两个经典比特
    circuit.measure(qr2, cr[2])    # 测量第二个寄存器到第三个经典比特
    
    # 运行电路
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(circuit, simulator, shots=1024)
    result = job.result()
    counts = result.get_counts()
    
    # 打印结果
    print("\n练习2解答 - 使用量子寄存器:")
    print("电路:")
    print(circuit.draw())
    print("\n测量结果:")
    print(counts)
    print("\n解释:")
    print("- 第一个寄存器的两个量子比特都处于叠加态")
    print("- 第二个寄存器的量子比特初始为|1⟩，但受到第一个寄存器的控制")
    print("- 结果是多种可能态的叠加，由于量子比特间的纠缠")
    
    # 可视化结果
    fig = plot_histogram(counts)
    plt.title("量子寄存器电路测量结果")
    plt.savefig('exercise2_histogram.png')
    plt.close(fig)
    
    # 检查寄存器信息
    print("\n电路信息:")
    print(f"量子比特总数: {circuit.num_qubits}")
    print(f"经典比特总数: {circuit.num_clbits}")
    print(f"量子寄存器: {circuit.qregs}")
    print(f"经典寄存器: {circuit.cregs}")
    
    return circuit, counts

# --------------------------------
# 练习3解答: 创建和应用栅栏
# --------------------------------
def exercise3_solution():
    """创建和应用栅栏的解答"""
    # 任务1: 创建电路
    circuit = QuantumCircuit(3, 3)
    
    # 任务2: 应用门和栅栏
    # 第一阶段: 初始化
    circuit.h(0)
    circuit.x(1)
    circuit.h(2)
    
    # 添加栅栏，表示初始化结束
    circuit.barrier()
    
    # 第二阶段: 纠缠操作
    circuit.cx(0, 1)
    circuit.cx(2, 1)
    
    # 只在量子比特0和1之间添加栅栏
    circuit.barrier([0, 1])
    
    # 第三阶段: 最终操作
    circuit.h(0)
    circuit.z(2)
    
    # 添加栅栏，表示操作结束
    circuit.barrier()
    
    # 测量
    circuit.measure([0, 1, 2], [0, 1, 2])
    
    # 运行电路
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(circuit, simulator, shots=1024)
    result = job.result()
    counts = result.get_counts()
    
    # 打印结果
    print("\n练习3解答 - 创建和应用栅栏:")
    print("带栅栏的电路:")
    print(circuit.draw())
    print("\n测量结果:")
    print(counts)
    
    print("\n解释:")
    print("- 栅栏帮助将电路分成逻辑部分，使电路更容易理解")
    print("- 栅栏不影响量子电路的行为，但可以影响电路优化")
    print("- 在实际设备上，栅栏可以防止优化器跨越不同的逻辑部分进行优化")
    
    # 查看带栅栏电路的转译
    transpiled_circuit = transpile(circuit, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=1)
    print("\n转译后的电路:")
    print(transpiled_circuit.draw())
    print("注意栅栏如何在转译过程中保持电路的逻辑结构")
    
    # 可视化结果
    fig = plot_histogram(counts)
    plt.title("带栅栏电路的测量结果")
    plt.savefig('exercise3_histogram.png')
    plt.close(fig)
    
    return circuit, transpiled_circuit

# --------------------------------
# 练习4解答: 参数化电路
# --------------------------------
def exercise4_solution():
    """参数化电路的解答"""
    # 任务1: 创建参数化电路
    from qiskit.circuit import Parameter
    
    # 创建参数
    theta = Parameter('θ')
    phi = Parameter('φ')
    
    # 创建电路
    circuit = QuantumCircuit(2, 2)
    
    # 应用参数化门
    circuit.rx(theta, 0)
    circuit.ry(phi, 1)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])
    
    # 任务2: 绑定参数并执行
    simulator = Aer.get_backend('qasm_simulator')
    
    # 定义要测试的参数值
    theta_values = [0, np.pi/4, np.pi/2, np.pi]
    phi_values = [0, np.pi/2]
    
    results = {}
    
    # 尝试不同参数组合
    for t in theta_values:
        for p in phi_values:
            # 绑定参数
            bound_circuit = circuit.bind_parameters({theta: t, phi: p})
            
            # 执行电路
            job = execute(bound_circuit, simulator, shots=1024)
            result = job.result()
            counts = result.get_counts()
            
            # 存储结果
            param_key = f"θ={t:.2f}, φ={p:.2f}"
            results[param_key] = counts
    
    # 打印结果
    print("\n练习4解答 - 参数化电路:")
    print("参数化电路:")
    print(circuit.draw())
    
    print("\n不同参数值的测量结果:")
    for param, counts in results.items():
        print(f"{param}: {counts}")
    
    print("\n解释:")
    print("- 参数θ控制第一个量子比特绕X轴的旋转角度")
    print("- 参数φ控制第二个量子比特绕Y轴的旋转角度")
    print("- θ=0时，第一个量子比特保持在|0⟩状态")
    print("- θ=π时，第一个量子比特翻转到|1⟩状态")
    print("- θ=π/2时，第一个量子比特处于均匀叠加态")
    print("- CNOT门将这种改变传递给第二个量子比特，导致纠缠")
    
    # 可视化特定参数组合的结果
    specific_params = {
        "θ=0.00, φ=0.00": results["θ=0.00, φ=0.00"],
        "θ=1.57, φ=1.57": results["θ=1.57, φ=1.57"]
    }
    
    fig = plot_histogram(specific_params)
    plt.title("不同参数值的测量结果对比")
    plt.savefig('exercise4_histogram.png')
    plt.close(fig)
    
    return circuit, results

# --------------------------------
# 练习5解答: 使用电路库组件
# --------------------------------
def exercise5_solution():
    """使用电路库组件的解答"""
    # 任务1: 创建使用QFT的电路
    # 我们将测试4种不同的初始态
    initial_states = {
        '00': QuantumCircuit(2),
        '01': QuantumCircuit(2),
        '10': QuantumCircuit(2),
        '11': QuantumCircuit(2)
    }
    
    # 准备初始态
    initial_states['00']  # |00⟩态不需要额外操作
    initial_states['01'].x(1)  # |01⟩态
    initial_states['10'].x(0)  # |10⟩态
    initial_states['11'].x([0, 1])  # |11⟩态
    
    # 创建QFT和逆QFT电路
    qft = QFT(2)
    inverse_qft = QFT(2).inverse()
    
    # 任务2: 应用QFT和逆QFT
    qft_circuits = {}
    qft_inverse_circuits = {}
    
    for state_label, init_circuit in initial_states.items():
        # 应用QFT
        qft_circuit = init_circuit.copy()
        qft_circuit = qft_circuit.compose(qft)
        qft_circuits[state_label] = qft_circuit
        
        # 应用QFT然后逆QFT
        qft_inverse_circuit = qft_circuit.copy()
        qft_inverse_circuit = qft_inverse_circuit.compose(inverse_qft)
        qft_inverse_circuits[state_label] = qft_inverse_circuit
    
    # 任务3: 分析结果
    simulator = Aer.get_backend('statevector_simulator')
    
    # 存储结果
    initial_states_sv = {}
    qft_states_sv = {}
    qft_inverse_states_sv = {}
    
    for state_label in initial_states:
        # 获取初始态的状态向量
        init_job = execute(initial_states[state_label], simulator)
        initial_states_sv[state_label] = init_job.result().get_statevector()
        
        # 获取QFT后的状态向量
        qft_job = execute(qft_circuits[state_label], simulator)
        qft_states_sv[state_label] = qft_job.result().get_statevector()
        
        # 获取QFT然后逆QFT后的状态向量
        qft_inv_job = execute(qft_inverse_circuits[state_label], simulator)
        qft_inverse_states_sv[state_label] = qft_inv_job.result().get_statevector()
    
    # 打印结果
    print("\n练习5解答 - 使用电路库组件:")
    print("QFT电路:")
    print(qft.draw())
    
    print("\n逆QFT电路:")
    print(inverse_qft.draw())
    
    print("\n不同初始态经过QFT后的状态向量:")
    for state_label, sv in qft_states_sv.items():
        print(f"初始态 |{state_label}⟩ -> QFT后: {sv}")
    
    print("\n应用QFT然后逆QFT后的状态向量:")
    for state_label, sv in qft_inverse_states_sv.items():
        print(f"初始态 |{state_label}⟩ -> QFT -> 逆QFT后: {sv}")
        # 计算保真度，检查是否回到初始态
        fidelity = abs(np.dot(sv.conjugate(), initial_states_sv[state_label]))**2
        print(f"  与初始态的保真度: {fidelity:.6f}")
    
    # 可视化状态向量
    fig = plot_state_city(qft_states_sv['00'])
    plt.title("初始态|00⟩经过QFT后的状态")
    plt.savefig('exercise5_qft_00.png')
    plt.close(fig)
    
    fig = plot_state_city(qft_states_sv['01'])
    plt.title("初始态|01⟩经过QFT后的状态")
    plt.savefig('exercise5_qft_01.png')
    plt.close(fig)
    
    return qft_circuits, qft_states_sv

# --------------------------------
# 练习6解答: 创建多控制门电路
# --------------------------------
def exercise6_solution():
    """创建多控制门电路的解答"""
    # 任务1: 创建电路
    n_qubits = 5
    circuit = QuantumCircuit(n_qubits, n_qubits)
    
    # 任务2: 应用多控制X门
    # 设置控制比特为0, 1, 2，目标比特为4
    control_qubits = [0, 1, 2]
    target_qubit = 4
    ancilla_qubit = 3  # 辅助量子比特
    
    # 首先，将所有控制比特置为|1⟩
    circuit.x(control_qubits)
    
    # 应用多控制X门
    circuit.mcx(control_qubits, target_qubit)
    
    # 再次应用多控制X门，但这次使用辅助量子比特
    circuit.mcx(control_qubits, ancilla_qubit, mode='recursion')
    
    # 任务3: 测量并分析
    circuit.measure(range(n_qubits), range(n_qubits))
    
    # 运行电路
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(circuit, simulator, shots=1024)
    result = job.result()
    counts = result.get_counts()
    
    # 打印结果
    print("\n练习6解答 - 创建多控制门电路:")
    print("多控制X门电路:")
    print(circuit.draw())
    print("\n测量结果:")
    print(counts)
    
    print("\n解释:")
    print("- 多控制X门(或多控制Toffoli门)只有在所有控制比特都是|1⟩时才翻转目标比特")
    print("- 我们将控制比特(0, 1, 2)初始化为|1⟩，所以目标比特(4)和辅助比特(3)被翻转为|1⟩")
    print("- 最终，我们应该看到所有5个量子比特都是|1⟩(对应二进制'11111')")
    
    # 转译为基本门的实现
    basic_circuit = transpile(circuit, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=0)
    print(f"\n转译为基本门后的门数量: {len(basic_circuit)}")
    print("多控制门通常会转译为多个基本门")
    
    # 可视化结果
    fig = plot_histogram(counts)
    plt.title("多控制X门电路测量结果")
    plt.savefig('exercise6_histogram.png')
    plt.close(fig)
    
    return circuit, counts

# --------------------------------
# 练习7解答: 创建相位估计电路
# --------------------------------
def exercise7_solution():
    """创建相位估计电路的解答"""
    # 任务1: 创建相位估计电路
    # 我们将估计相位θ=1/4 (对应U|1⟩=e^{2πiθ}|1⟩)
    
    # 定义参数
    n_counting = 3  # 相位估计寄存器的量子比特数
    theta = 0.25    # 要估计的相位 (1/4)
    
    # 创建电路
    phase_est_circuit = QuantumCircuit(n_counting + 1, n_counting)
    
    # 任务2: 相位估计实现
    # 第1步: 准备估计寄存器为均匀叠加态
    for i in range(n_counting):
        phase_est_circuit.h(i)
    
    # 第2步: 准备目标寄存器的特征态 (这里是|1⟩)
    phase_est_circuit.x(n_counting)
    
    # 第3步: 应用受控U^{2^j}门
    # U = diag(1, e^{2πiθ})，我们用相位门P来实现
    for j in range(n_counting):
        # 对于每个j，应用CP门，控制位是估计寄存器的量子比特j
        # 旋转角度是2π*θ*2^j
        angle = 2 * np.pi * theta * 2**(n_counting-1-j)
        phase_est_circuit.cp(angle, j, n_counting)
    
    # 第4步: 应用逆QFT到估计寄存器
    phase_est_circuit.append(QFT(n_counting).inverse(), range(n_counting))
    
    # 第5步: 测量估计寄存器
    phase_est_circuit.measure(range(n_counting), range(n_counting))
    
    # 任务3: 分析不同相位的估计准确度
    # 测试几个不同的相位值
    phase_values = [0.125, 0.25, 0.375, 0.5]
    phase_results = {}
    
    simulator = Aer.get_backend('qasm_simulator')
    
    for phase in phase_values:
        # 创建相位估计电路
        circuit = QuantumCircuit(n_counting + 1, n_counting)
        
        # 估计寄存器准备
        for i in range(n_counting):
            circuit.h(i)
        
        # 目标寄存器准备
        circuit.x(n_counting)
        
        # 受控相位旋转
        for j in range(n_counting):
            angle = 2 * np.pi * phase * 2**(n_counting-1-j)
            circuit.cp(angle, j, n_counting)
        
        # 逆QFT
        circuit.append(QFT(n_counting).inverse(), range(n_counting))
        
        # 测量
        circuit.measure(range(n_counting), range(n_counting))
        
        # 运行电路
        job = execute(circuit, simulator, shots=1024)
        result = job.result()
        counts = result.get_counts()
        
        phase_results[phase] = counts
    
    # 打印结果
    print("\n练习7解答 - 创建相位估计电路:")
    print("相位估计电路 (θ=0.25):")
    print(phase_est_circuit.draw())
    
    print("\n不同相位值的估计结果:")
    for phase, counts in phase_results.items():
        print(f"\n相位 θ={phase}:")
        print(counts)
        
        # 解释测量结果
        # 对于每个测量结果比特串，计算对应的相位估计
        estimated_phases = {}
        for bitstring, count in counts.items():
            # 将二进制比特串转换为整数
            measured_int = int(bitstring, 2)
            # 计算估计的相位
            estimated_phase = measured_int / (2**n_counting)
            estimated_phases[estimated_phase] = count
        
        # 输出估计的相位
        print(f"估计的相位值及其计数:")
        for est_phase, count in estimated_phases.items():
            print(f"  {est_phase:.3f}: {count} (误差: {abs(est_phase - phase):.3f})")
    
    # 可视化特定相位的结果
    fig = plot_histogram(phase_results[0.25])
    plt.title("相位θ=0.25的估计结果")
    plt.savefig('exercise7_histogram.png')
    plt.close(fig)
    
    return phase_est_circuit, phase_results

# 执行解答
if __name__ == "__main__":
    print("===== Qiskit量子电路练习解答 =====")
    
    # 取消注释以运行特定练习的解答
    # combined_circuit, state_combined = exercise1_solution()
    # circuit2, counts2 = exercise2_solution()
    # circuit3, transpiled3 = exercise3_solution()
    # circuit4, results4 = exercise4_solution()
    # qft_circuits5, qft_states5 = exercise5_solution()
    # circuit6, counts6 = exercise6_solution()
    # phase_est_circuit7, phase_results7 = exercise7_solution()
    
    print("\n所有练习解答已准备就绪!")
    print("取消注释相应的函数调用以运行特定练习的解答") 