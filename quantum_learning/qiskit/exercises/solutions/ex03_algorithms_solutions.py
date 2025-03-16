#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IBM Qiskit框架学习 - 量子算法练习解答
本文件包含对应练习的完整解答
"""

# 导入必要的库
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute, transpile
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.quantum_info import Statevector
from qiskit.algorithms import Grover, AmplificationProblem
from qiskit.circuit.library import PhaseOracle, GroverOperator
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------
# 练习1解答: 实现Deutsch-Jozsa算法
# --------------------------------
def exercise1_solution():
    """实现Deutsch-Jozsa算法的解答"""
    # 我们将为3个量子比特的输入实现DJ算法
    n = 3  # 问题量子比特数量
    
    # 创建各种oracle函数
    # 常数函数f(x)=0: 不做任何操作
    def constant_0_oracle():
        oracle = QuantumCircuit(n+1)
        # 什么都不做，因为f(x)=0意味着y⊕f(x)=y
        return oracle
    
    # 常数函数f(x)=1: 对辅助量子比特应用X门
    def constant_1_oracle():
        oracle = QuantumCircuit(n+1)
        oracle.x(n)  # 翻转辅助量子比特
        return oracle
    
    # 平衡函数oracle示例1: 对一半输入返回0，一半返回1
    def balanced_oracle_1():
        oracle = QuantumCircuit(n+1)
        # 将第一个输入量子比特的值复制到辅助量子比特
        # 这意味着f(x)=x[0]
        oracle.cx(0, n)
        return oracle
    
    # 平衡函数oracle示例2: 另一种平衡函数
    def balanced_oracle_2():
        oracle = QuantumCircuit(n+1)
        # 这里我们实现f(x) = x[0] ⊕ x[1]
        oracle.cx(0, n)
        oracle.cx(1, n)
        return oracle
    
    # 完整Deutsch-Jozsa电路构建函数
    def deutsch_jozsa_circuit(oracle_function):
        # 创建量子电路
        dj_circuit = QuantumCircuit(n+1, n)
        
        # 初始化辅助量子比特为|1⟩
        dj_circuit.x(n)
        
        # 对所有量子比特应用H门，创建叠加态
        for qubit in range(n+1):
            dj_circuit.h(qubit)
        
        # 应用oracle
        oracle = oracle_function()
        dj_circuit = dj_circuit.compose(oracle)
        
        # 再次对输入量子比特应用H门
        for qubit in range(n):
            dj_circuit.h(qubit)
        
        # 测量输入寄存器
        dj_circuit.measure(range(n), range(n))
        
        return dj_circuit
    
    # 为每个oracle构建和运行电路
    simulator = Aer.get_backend('qasm_simulator')
    oracles = {
        "常数函数f(x)=0": constant_0_oracle,
        "常数函数f(x)=1": constant_1_oracle,
        "平衡函数1": balanced_oracle_1,
        "平衡函数2": balanced_oracle_2
    }
    
    results = {}
    circuits = {}
    
    for name, oracle_function in oracles.items():
        # 构建电路
        circuit = deutsch_jozsa_circuit(oracle_function)
        circuits[name] = circuit
        
        # 运行电路
        job = execute(circuit, simulator, shots=1024)
        counts = job.result().get_counts()
        results[name] = counts
    
    # 打印结果
    print("\n练习1解答 - 实现Deutsch-Jozsa算法:")
    
    for name, counts in results.items():
        print(f"\n{name}的结果:")
        print(f"电路:")
        print(circuits[name].draw(output='text', fold=80))
        print(f"测量结果: {counts}")
        
        # 判断函数类型
        all_zeros = '0' * n
        if all_zeros in counts and counts[all_zeros] > 0.9 * sum(counts.values()):
            conclusion = "常数函数"
        else:
            conclusion = "平衡函数"
        
        print(f"结论: 该函数是{conclusion}")
    
    # 可视化结果
    plt.figure(figsize=(15, 10))
    
    for i, (name, counts) in enumerate(results.items()):
        plt.subplot(2, 2, i+1)
        plot_histogram(counts)
        plt.title(f"{name}的测量结果")
    
    plt.tight_layout()
    plt.savefig('dj_algorithm_results.png')
    plt.close()
    
    return circuits, results

# --------------------------------
# 练习2解答: 实现Bernstein-Vazirani算法
# --------------------------------
def exercise2_solution():
    """实现Bernstein-Vazirani算法的解答"""
    # 我们将为4量子比特的问题实现BV算法
    n = 4  # 问题量子比特数量
    
    # 创建oracle函数
    def bv_oracle(hidden_string):
        oracle = QuantumCircuit(n+1)
        
        # 对于hidden_string中的每个1，添加CNOT门
        for i in range(n):
            if hidden_string[i] == '1':
                oracle.cx(i, n)
                
        return oracle
    
    # 构建Bernstein-Vazirani电路
    def bernstein_vazirani_circuit(hidden_string):
        # 创建量子电路
        bv_circuit = QuantumCircuit(n+1, n)
        
        # 初始化辅助量子比特为|1⟩
        bv_circuit.x(n)
        
        # 对所有量子比特应用H门
        for qubit in range(n+1):
            bv_circuit.h(qubit)
        
        # 应用oracle
        oracle = bv_oracle(hidden_string)
        bv_circuit = bv_circuit.compose(oracle)
        
        # 再次对输入量子比特应用H门
        for qubit in range(n):
            bv_circuit.h(qubit)
        
        # 测量
        bv_circuit.measure(range(n), range(n))
        
        return bv_circuit
    
    # 测试几个不同的隐藏字符串
    hidden_strings = ["0101", "1100", "1111", "1010"]
    simulator = Aer.get_backend('qasm_simulator')
    
    results = {}
    circuits = {}
    
    for hidden in hidden_strings:
        # 构建电路
        circuit = bernstein_vazirani_circuit(hidden)
        circuits[hidden] = circuit
        
        # 运行电路
        job = execute(circuit, simulator, shots=1024)
        counts = job.result().get_counts()
        results[hidden] = counts
    
    # 打印结果
    print("\n练习2解答 - 实现Bernstein-Vazirani算法:")
    
    for hidden_string, counts in results.items():
        print(f"\n隐藏字符串s={hidden_string}的结果:")
        
        # 找出出现次数最多的测量结果
        most_common = max(counts.items(), key=lambda x: x[1])
        recovered_string = most_common[0][::-1]  # 反转比特顺序
        
        print(f"测量结果: {counts}")
        print(f"恢复的字符串: {recovered_string}")
        print(f"与原始隐藏字符串匹配: {recovered_string == hidden_string}")
    
    # 可视化其中一个电路
    example = "1010"
    circuit = circuits[example]
    
    print(f"\n隐藏字符串s={example}的电路:")
    print(circuit.draw(output='text', fold=80))
    
    # 可视化所有结果
    plt.figure(figsize=(15, 10))
    
    for i, (hidden, counts) in enumerate(results.items()):
        plt.subplot(2, 2, i+1)
        plot_histogram(counts)
        plt.title(f"隐藏字符串s={hidden}的测量结果")
    
    plt.tight_layout()
    plt.savefig('bv_algorithm_results.png')
    plt.close()
    
    return circuits, results

# --------------------------------
# 练习3解答: 实现Grover搜索算法
# --------------------------------
def exercise3_solution():
    """实现Grover搜索算法的解答"""
    # 使用Qiskit提供的Grover算法实现
    def qiskit_grover(target_string, n_qubits):
        """使用Qiskit的Grover实现"""
        # 创建oracle
        oracle = PhaseOracle(f"0b{target_string} & (1)")
        
        # 创建量子求解问题
        problem = AmplificationProblem(oracle, is_good_state=oracle.evaluate_bitstring)
        
        # 使用Qiskit的Grover算法
        grover = Grover(iterations=None)  # 自动确定迭代次数
        
        # 获取Grover电路
        circuit = grover.construct_circuit(problem)
        
        return circuit, grover, problem
    
    # 手动实现Grover算法
    def manual_grover_circuit(n_qubits, target_string):
        """手动实现Grover搜索算法"""
        # 创建电路
        circuit = QuantumCircuit(n_qubits)
        
        # 初始化为均匀叠加态
        circuit.h(range(n_qubits))
        
        # 计算最优迭代次数
        iterations = int(np.pi/4 * np.sqrt(2**n_qubits))
        
        # 添加栅栏
        circuit.barrier()
        
        target_int = int(target_string, 2)
        
        # 迭代Grover算法
        for i in range(iterations):
            # Oracle - 标记目标状态
            # 这里我们简化实现，使用multi-controlled Z门
            if n_qubits <= 2:  # 对于小规模问题，使用简单方法
                # 创建目标态的二进制表示
                target_bits = [int(bit) for bit in target_string]
                
                # 对所有值为0的比特应用X门
                for qubit, bit in enumerate(target_bits):
                    if bit == 0:
                        circuit.x(qubit)
                
                # 应用多控制Z门
                if n_qubits == 2:
                    circuit.cz(0, 1)
                else:  # n_qubits == 1
                    circuit.z(0)
                
                # 再次应用X门以恢复原始状态
                for qubit, bit in enumerate(target_bits):
                    if bit == 0:
                        circuit.x(qubit)
            else:
                # 对于更大规模问题，我们可以使用更复杂的实现
                # 为简化起见，这里使用简化版本
                oracle_circuit = QuantumCircuit(n_qubits)
                
                # 对所有位使用X门将目标状态转换为|00...0⟩
                for qubit in range(n_qubits):
                    if target_string[qubit] == '0':
                        oracle_circuit.x(qubit)
                
                # 应用多控制Z门
                oracle_circuit.h(n_qubits-1)
                oracle_circuit.mcx(list(range(n_qubits-1)), n_qubits-1)
                oracle_circuit.h(n_qubits-1)
                
                # 恢复状态
                for qubit in range(n_qubits):
                    if target_string[qubit] == '0':
                        oracle_circuit.x(qubit)
                
                # 将oracle添加到主电路
                circuit = circuit.compose(oracle_circuit)
            
            # 添加栅栏
            circuit.barrier()
            
            # 扩散算子(反射绕平均振幅)
            circuit.h(range(n_qubits))
            circuit.x(range(n_qubits))
            
            # 应用多控制Z门
            if n_qubits <= 2:
                if n_qubits == 2:
                    circuit.cz(0, 1)
                else:  # n_qubits == 1
                    circuit.z(0)
            else:
                # 对于多量子比特，使用MCZ
                circuit.h(n_qubits-1)
                circuit.mcx(list(range(n_qubits-1)), n_qubits-1)
                circuit.h(n_qubits-1)
            
            circuit.x(range(n_qubits))
            circuit.h(range(n_qubits))
            
            # 添加栅栏
            circuit.barrier()
        
        # 添加测量
        circuit.measure_all()
        
        return circuit
    
    # 分析不同问题规模的性能
    problem_sizes = [2, 3]
    manual_results = {}
    qiskit_results = {}
    
    simulator = Aer.get_backend('qasm_simulator')
    
    for n in problem_sizes:
        # 为每个问题规模选择一个随机目标字符串
        target = bin(np.random.randint(2**n))[2:].zfill(n)
        
        # 手动实现
        manual_circuit = manual_grover_circuit(n, target)
        
        # Qiskit实现
        qiskit_circuit, grover, problem = qiskit_grover(target, n)
        
        # 运行手动实现
        manual_job = execute(manual_circuit, simulator, shots=1024)
        manual_counts = manual_job.result().get_counts()
        
        # 运行Qiskit实现
        qiskit_job = execute(qiskit_circuit, simulator, shots=1024)
        qiskit_counts = qiskit_job.result().get_counts()
        
        # 存储结果
        manual_results[f"{n}_qubits_target_{target}"] = {
            "target": target,
            "counts": manual_counts,
            "circuit": manual_circuit
        }
        
        qiskit_results[f"{n}_qubits_target_{target}"] = {
            "target": target,
            "counts": qiskit_counts,
            "circuit": qiskit_circuit,
            "optimal_iterations": grover._optimal_num_iterations(problem)
        }
    
    # 打印结果
    print("\n练习3解答 - 实现Grover搜索算法:")
    
    for key in manual_results:
        target = manual_results[key]["target"]
        manual_counts = manual_results[key]["counts"]
        qiskit_counts = qiskit_results[key]["counts"]
        optimal_iterations = qiskit_results[key]["optimal_iterations"]
        
        print(f"\n问题规模: {key}")
        print(f"目标字符串: {target}")
        print(f"最优迭代次数: {optimal_iterations}")
        
        print(f"\n手动实现测量结果: {manual_counts}")
        
        # 找出最可能的结果
        manual_most_common = max(manual_counts.items(), key=lambda x: x[1])
        manual_found_string = manual_most_common[0]
        
        print(f"手动实现找到的字符串: {manual_found_string}")
        print(f"成功: {manual_found_string == target}")
        
        print(f"\nQiskit实现测量结果: {qiskit_counts}")
        
        # 在Qiskit Grover结果中，状态可能包含辅助量子比特，需要提取
        qiskit_most_common = max(qiskit_counts.items(), key=lambda x: x[1])
        qiskit_found_string = qiskit_most_common[0]
        
        # 提取与目标长度相同的部分
        if '_' in qiskit_found_string:
            relevant_bits = qiskit_found_string[-len(target):]
        else:
            relevant_bits = qiskit_found_string
        
        print(f"Qiskit实现找到的字符串: {relevant_bits}")
        print(f"成功: {relevant_bits == target}")
    
    # 可视化结果
    plt.figure(figsize=(15, 10))
    
    row_idx = 0
    for key in manual_results:
        target = manual_results[key]["target"]
        manual_counts = manual_results[key]["counts"]
        qiskit_counts = qiskit_results[key]["counts"]
        
        # 手动实现结果
        plt.subplot(len(problem_sizes), 2, row_idx*2+1)
        plot_histogram(manual_counts)
        plt.title(f"手动实现: 目标={target}")
        
        # Qiskit实现结果
        plt.subplot(len(problem_sizes), 2, row_idx*2+2)
        plot_histogram(qiskit_counts)
        plt.title(f"Qiskit实现: 目标={target}")
        
        row_idx += 1
    
    plt.tight_layout()
    plt.savefig('grover_algorithm_results.png')
    plt.close()
    
    return manual_results, qiskit_results

# --------------------------------
# 练习4解答: 实现量子相位估计
# --------------------------------
def exercise4_solution():
    """实现量子相位估计的解答"""
    # 创建一个用于相位估计的函数
    def phase_estimation(unitary, precision, state_preparation=None):
        """
        实现量子相位估计算法
        
        参数:
        unitary - 要估计特征值的幺正矩阵(以电路形式)
        precision - 计数寄存器的量子比特数(精度)
        state_preparation - 可选的状态准备电路
        
        返回:
        量子相位估计电路
        """
        # 确定量子比特数
        n_counting = precision
        n_target = unitary.num_qubits
        
        # 创建电路
        qpe_circuit = QuantumCircuit(n_counting + n_target, n_counting)
        
        # 如果提供了状态准备电路，应用它
        if state_preparation:
            qpe_circuit.compose(state_preparation, qubits=range(n_counting, n_counting + n_target), inplace=True)
            
        # 对计数寄存器应用H门
        for qubit in range(n_counting):
            qpe_circuit.h(qubit)
            
        # 对于每个计数量子比特，应用受控-U^(2^j)
        for j in range(n_counting):
            # 计算要应用的U的幂次
            power = 2**(n_counting - j - 1)
            
            # 创建受控版本的U^power
            controlled_u_power = unitary.copy()
            
            # 循环应用power次unitary
            for _ in range(power - 1):
                controlled_u_power = controlled_u_power.compose(unitary)
            
            # 添加控制
            qpe_circuit.compose(controlled_u_power.control(), 
                              qubits=[j] + list(range(n_counting, n_counting + n_target)), 
                              inplace=True)
        
        # 应用逆QFT到计数寄存器
        from qiskit.circuit.library import QFT
        qpe_circuit.compose(QFT(n_counting, do_swaps=False).inverse(), qubits=range(n_counting), inplace=True)
        
        # 测量计数寄存器
        qpe_circuit.measure(range(n_counting), range(n_counting))
        
        return qpe_circuit
    
    # 创建一个简单的相位门作为要估计的幺正算子
    # 我们将估计S门的相位，其特征值是e^{iπ/2}，对应相位θ=1/4
    def s_gate():
        """创建S门电路"""
        s_circuit = QuantumCircuit(1)
        s_circuit.s(0)
        return s_circuit
    
    # 创建一个将目标寄存器准备为S门特征态的电路
    def prepare_eigenstate():
        """准备S门的特征态|1⟩"""
        prep_circuit = QuantumCircuit(1)
        prep_circuit.x(0)  # 将|0⟩转换为|1⟩
        return prep_circuit
    
    # 创建一个T门电路，其相位是π/4，对应θ=1/8
    def t_gate():
        """创建T门电路"""
        t_circuit = QuantumCircuit(1)
        t_circuit.t(0)
        return t_circuit
    
    # 尝试不同精度
    precision_levels = [3, 5, 7]
    unitary_gates = {
        "S门 (θ=1/4)": (s_gate(), 0.25),
        "T门 (θ=1/8)": (t_gate(), 0.125)
    }
    
    simulator = Aer.get_backend('qasm_simulator')
    
    results = {}
    circuits = {}
    
    for gate_name, (gate_circuit, true_phase) in unitary_gates.items():
        for precision in precision_levels:
            # 创建QPE电路
            qpe_circuit = phase_estimation(gate_circuit, precision, prepare_eigenstate())
            circuit_key = f"{gate_name}_precision_{precision}"
            circuits[circuit_key] = qpe_circuit
            
            # 运行电路
            job = execute(qpe_circuit, simulator, shots=1024)
            counts = job.result().get_counts()
            results[circuit_key] = {
                "counts": counts,
                "true_phase": true_phase,
                "precision": precision
            }
    
    # 打印结果
    print("\n练习4解答 - 实现量子相位估计:")
    
    for circuit_key, result_data in results.items():
        counts = result_data["counts"]
        true_phase = result_data["true_phase"]
        precision = result_data["precision"]
        
        print(f"\n{circuit_key}:")
        print(f"真实相位: {true_phase}")
        print(f"精度(量子比特数): {precision}")
        print(f"测量结果: {counts}")
        
        # 分析结果
        estimated_phases = {}
        for bitstring, count in counts.items():
            # 将二进制转换为十进制
            decimal = int(bitstring, 2)
            # 计算对应的相位估计
            phase = decimal / (2**precision)
            estimated_phases[phase] = count
        
        # 显示估计的相位和误差
        print("估计的相位值:")
        for phase, count in sorted(estimated_phases.items()):
            error = abs(phase - true_phase)
            probability = count / 1024
            print(f"  θ≈{phase:.6f}, 概率:{probability:.4f}, 误差:{error:.6f}")
        
        # 理论最小误差
        min_error = 1 / (2**precision)
        print(f"理论最小误差: {min_error:.6f}")
    
    # 可视化结果
    plt.figure(figsize=(15, 10))
    
    row = 0
    for gate_name, (gate_circuit, true_phase) in unitary_gates.items():
        for i, precision in enumerate(precision_levels):
            circuit_key = f"{gate_name}_precision_{precision}"
            counts = results[circuit_key]["counts"]
            
            # 处理结果以显示相位而不是二进制字符串
            phase_counts = {}
            for bitstring, count in counts.items():
                decimal = int(bitstring, 2)
                phase = decimal / (2**precision)
                phase_str = f"{phase:.4f}"
                phase_counts[phase_str] = count
            
            plt.subplot(len(unitary_gates), len(precision_levels), row*len(precision_levels)+i+1)
            plot_histogram(phase_counts)
            plt.title(f"{gate_name}, 精度={precision}")
            plt.xlabel("估计的相位值θ")
            plt.axvline(x=true_phase, color='r', linestyle='--', label=f'真实值θ={true_phase}')
            plt.legend()
        
        row += 1
    
    plt.tight_layout()
    plt.savefig('qpe_results.png')
    plt.close()
    
    return circuits, results

# --------------------------------
# 练习5解答: 实现量子傅里叶变换
# --------------------------------
def exercise5_solution():
    """实现量子傅里叶变换的解答"""
    # 手动构建QFT电路
    def qft_rotations(circuit, n):
        """对n个量子比特应用QFT旋转"""
        if n == 0:
            return circuit
        n -= 1
        circuit.h(n)
        for qubit in range(n):
            # 计算旋转角度
            angle = 2 * np.pi / (2**(n-qubit+1))
            # 应用受控旋转
            circuit.cp(angle, qubit, n)
        # 递归处理剩余量子比特
        qft_rotations(circuit, n)
        
    def swap_registers(circuit, n):
        """反转量子比特的顺序"""
        for qubit in range(n//2):
            circuit.swap(qubit, n-qubit-1)
        return circuit
    
    def manual_qft(n_qubits):
        """手动实现的QFT电路"""
        qft_circuit = QuantumCircuit(n_qubits)
        
        # 应用QFT旋转
        qft_rotations(qft_circuit, n_qubits)
        
        # 交换寄存器以匹配标准定义
        swap_registers(qft_circuit, n_qubits)
        
        return qft_circuit
    
    # 比较手动实现与Qiskit库实现
    n_qubits = 4
    
    # 手动实现
    manual_qft_circuit = manual_qft(n_qubits)
    
    # Qiskit库实现
    from qiskit.circuit.library import QFT
    library_qft_circuit = QFT(n_qubits)
    
    # 对不同输入态应用QFT
    input_states = {
        '0000': QuantumCircuit(n_qubits),
        '0001': QuantumCircuit(n_qubits),
        '0101': QuantumCircuit(n_qubits),
        '1111': QuantumCircuit(n_qubits)
    }
    
    # 准备输入态
    input_states['0000']  # |0000⟩不需要额外操作
    input_states['0001'].x(n_qubits-1)  # |0001⟩
    
    # |0101⟩准备
    input_states['0101'].x(1)
    input_states['0101'].x(3)
    
    # |1111⟩准备
    input_states['1111'].x(range(n_qubits))
    
    # 应用QFT并获取状态向量
    simulator = Aer.get_backend('statevector_simulator')
    
    results = {}
    
    for state_label, init_circuit in input_states.items():
        # 应用手动QFT
        qft_circuit = init_circuit.copy()
        qft_circuit = qft_circuit.compose(manual_qft_circuit)
        
        # 获取状态向量
        job = execute(qft_circuit, simulator)
        statevector = job.result().get_statevector()
        
        results[state_label] = statevector
    
    # 打印结果
    print("\n练习5解答 - 实现量子傅里叶变换:")
    
    print("\n手动实现的QFT电路:")
    print(manual_qft_circuit.draw())
    
    print("\nQiskit库的QFT电路:")
    print(library_qft_circuit.draw())
    
    print("\n比较:")
    if manual_qft_circuit.equiv(library_qft_circuit):
        print("手动实现和库实现的QFT电路等价")
    else:
        print("手动实现和库实现的QFT电路有差异")
        
        # 分析差异
        from qiskit import transpile
        basis_gates = ['h', 'cx', 'u1', 'u2', 'u3']
        
        manual_transpiled = transpile(manual_qft_circuit, basis_gates=basis_gates)
        library_transpiled = transpile(library_qft_circuit, basis_gates=basis_gates)
        
        print(f"手动实现门数量: {len(manual_transpiled)}")
        print(f"库实现门数量: {len(library_transpiled)}")
    
    # 可视化结果
    from qiskit.visualization import plot_state_city
    
    plt.figure(figsize=(15, 10))
    
    for i, (state_label, statevector) in enumerate(results.items()):
        plt.subplot(2, 2, i+1)
        plot_state_city(statevector, title=f"输入态|{state_label}⟩的QFT结果")
    
    plt.tight_layout()
    plt.savefig('qft_results.png')
    plt.close()
    
    return results

# 执行解答
if __name__ == "__main__":
    print("===== Qiskit量子算法练习解答 =====")
    
    # 取消注释以运行特定练习的解答
    # dj_circuits, dj_results = exercise1_solution()
    # bv_circuits, bv_results = exercise2_solution()
    # grover_manual, grover_qiskit = exercise3_solution()
    # qpe_circuits, qpe_results = exercise4_solution()
    # qft_results = exercise5_solution()
    
    print("\n所有练习解答已准备就绪!")
    print("取消注释相应的函数调用以运行特定练习的解答") 