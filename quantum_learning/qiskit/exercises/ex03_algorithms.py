#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IBM Qiskit框架学习 - 量子算法练习
本文件包含一系列帮助理解和实现量子算法的练习题
"""

# 导入必要的库
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute, transpile
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.quantum_info import Statevector
from qiskit.algorithms import Grover, AmplificationProblem
from qiskit.circuit.library import PhaseOracle, GroverOperator
import matplotlib.pyplot as plt
import numpy as np

print("===== Qiskit量子算法练习 =====")
print("完成以下练习来测试您对量子算法实现的理解")
print("每个练习都有一个或多个任务，请尝试独立完成")
print("练习后有提示和参考解答")

# --------------------------------
# 练习1: 实现Deutsch-Jozsa算法
# --------------------------------
print("\n练习1: 实现Deutsch-Jozsa算法")
print("任务1: 为常数函数f(x)=0和f(x)=1创建oracle电路")
print("任务2: 为平衡函数创建oracle电路")
print("任务3: 构建完整的Deutsch-Jozsa电路并运行")
print("任务4: 从测量结果判断函数是常数还是平衡")

# 提示
print("\n提示:")
print("- Deutsch-Jozsa算法用于判断黑盒函数是常数还是平衡的")
print("- 常数函数对所有输入返回相同值(0或1)")
print("- 平衡函数对一半输入返回0，对一半输入返回1")
print("- 传统算法需要测试一半以上的可能输入，而DJ算法只需一次查询")
print("- Oracle实现为Uf|x⟩|y⟩ = |x⟩|y⊕f(x)⟩")

# 参考解答
def exercise1_solution():
    # 任务1-3: 实现Deutsch-Jozsa算法的电路
    
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
    
    # 任务4: 分析结果
    print("\n参考解答 - 练习1:")
    
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
    
    print("\n解释:")
    print("- 如果所有量子比特的测量结果都是|0⟩，则函数是常数函数")
    print("- 如果至少有一个量子比特的测量结果不是|0⟩，则函数是平衡函数")
    print("- 经典算法需要2^(n-1)+1次查询来确定，而D-J算法只需1次查询")
    print("- 这展示了量子算法的指数级加速")
    
    return circuits, results

# 取消注释下面的行以查看参考解答
# dj_circuits, dj_results = exercise1_solution()

# --------------------------------
# 练习2: 实现Bernstein-Vazirani算法
# --------------------------------
print("\n练习2: 实现Bernstein-Vazirani算法")
print("任务1: 为隐藏字符串s创建oracle电路")
print("任务2: 构建完整的Bernstein-Vazirani电路")
print("任务3: 运行电路并从测量结果中恢复隐藏字符串")

# 提示
print("\n提示:")
print("- Bernstein-Vazirani算法用于查找隐藏的位串s")
print("- 给定一个函数f(x) = s·x mod 2，其中s·x是点积")
print("- 经典算法需要n次查询，量子算法只需1次")
print("- Oracle实现为Uf|x⟩|y⟩ = |x⟩|y⊕(s·x)⟩")
print("- 最终测量结果直接给出隐藏字符串s")

# 参考解答
def exercise2_solution():
    # 我们将为4量子比特的问题实现BV算法
    n = 4  # 问题量子比特数量
    
    # 任务1: 创建oracle函数
    def bv_oracle(hidden_string):
        oracle = QuantumCircuit(n+1)
        
        # 对于hidden_string中的每个1，添加CNOT门
        for i in range(n):
            if hidden_string[i] == '1':
                oracle.cx(i, n)
                
        return oracle
    
    # 任务2: 构建Bernstein-Vazirani电路
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
    
    # 任务3: 分析结果
    print("\n参考解答 - 练习2:")
    
    for hidden_string, counts in results.items():
        print(f"\n隐藏字符串s={hidden_string}的结果:")
        
        # 找出出现次数最多的测量结果
        most_common = max(counts.items(), key=lambda x: x[1])
        recovered_string = most_common[0][::-1]  # 反转比特顺序
        
        print(f"测量结果: {counts}")
        print(f"恢复的字符串: {recovered_string}")
        print(f"与原始隐藏字符串匹配: {recovered_string == hidden_string}")
    
    # 可视化其中一个电路和结果
    example = "1010"
    circuit = circuits[example]
    result = results[example]
    
    print(f"\n隐藏字符串s={example}的电路:")
    print(circuit.draw(output='text', fold=80))
    
    print("\n解释:")
    print("- Bernstein-Vazirani算法通过一次查询找到隐藏字符串s")
    print("- 这是通过将f(x)=s·x映射到量子电路，并使用量子并行性实现的")
    print("- 在理想情况下，测量结果直接给出隐藏字符串")
    print("- 经典算法需要n次查询，而BV算法只需1次")
    
    # 可视化结果
    fig = plot_histogram(result)
    plt.title(f"隐藏字符串s={example}的测量结果")
    plt.savefig('exercise2_histogram.png')
    plt.close(fig)
    print("直方图已保存为'exercise2_histogram.png'")
    
    return circuits, results

# 取消注释下面的行以查看参考解答
# bv_circuits, bv_results = exercise2_solution()

# --------------------------------
# 练习3: 实现Grover搜索算法
# --------------------------------
print("\n练习3: 实现Grover搜索算法")
print("任务1: 为搜索问题创建oracle电路")
print("任务2: 构建Grover扩散算子")
print("任务3: 实现完整的Grover搜索算法")
print("任务4: 分析不同问题规模下Grover算法的性能")

# 提示
print("\n提示:")
print("- Grover算法用于在未排序数据中搜索")
print("- 经典算法需要O(N)次查询，而Grover算法只需O(√N)次")
print("- Oracle标记目标状态，扩散算子放大这些标记状态的振幅")
print("- 迭代次数与问题规模有关，大约为π√N/4")
print("- 使用qiskit.algorithms.AmplificationProblem和Grover可以简化实现")

# 参考解答
def exercise3_solution():
    # 任务1-3: 实现Grover搜索算法
    
    # 首先，我们手动实现一个简单的Grover搜索
    def manual_grover(target_string, n_qubits):
        """手动实现Grover算法"""
        # 创建量子电路
        grover_circuit = QuantumCircuit(n_qubits)
        
        # 初始化为均匀叠加态
        grover_circuit.h(range(n_qubits))
        
        # 确定迭代次数
        iterations = int(np.pi/4 * np.sqrt(2**n_qubits))
        
        # 添加阻断，以便绘图时区分不同部分
        grover_circuit.barrier()
        
        # 实现Grover迭代
        for i in range(iterations):
            # Oracle: 标记目标状态
            # 如果状态等于目标状态，将相位反转
            target_int = int(target_string, 2)
            grover_circuit.z(n_qubits - 1).c_if(range(n_qubits), target_int)
            
            # 添加阻断
            grover_circuit.barrier()
            
            # 扩散: H -> Z0 -> H
            # 对所有量子比特应用H门
            grover_circuit.h(range(n_qubits))
            
            # 对|0⟩态应用相位反转
            grover_circuit.z(range(n_qubits))
            
            # 实现受控-Z门，除了|0⟩以外的所有状态都取反
            grover_circuit.x(range(n_qubits))
            grover_circuit.h(n_qubits - 1)
            grover_circuit.mct(list(range(n_qubits - 1)), n_qubits - 1)
            grover_circuit.h(n_qubits - 1)
            grover_circuit.x(range(n_qubits))
            
            # 再次对所有量子比特应用H门
            grover_circuit.h(range(n_qubits))
            
            # 添加阻断
            grover_circuit.barrier()
        
        # 测量
        grover_circuit.measure_all()
        
        return grover_circuit
    
    # 接下来，我们使用Qiskit提供的Grover算法实现
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
        
        return circuit
    
    # 任务4: 分析不同问题规模下的性能
    problem_sizes = [2, 3]
    results = {}
    
    simulator = Aer.get_backend('qasm_simulator')
    
    for n in problem_sizes:
        # 为每个问题规模选择一个随机目标字符串
        target = bin(np.random.randint(2**n))[2:].zfill(n)
        
        # 两种实现方式
        # circuit_manual = manual_grover(target, n)
        circuit_qiskit = qiskit_grover(target, n)
        
        # 运行Qiskit实现
        job_qiskit = execute(circuit_qiskit, simulator, shots=1024)
        counts_qiskit = job_qiskit.result().get_counts()
        
        results[f"{n}_qubits_target_{target}"] = {
            "target": target,
            "qiskit_counts": counts_qiskit
        }
    
    print("\n参考解答 - 练习3:")
    
    for key, data in results.items():
        target = data["target"]
        counts = data["qiskit_counts"]
        
        print(f"\n问题规模: {key}")
        print(f"目标字符串: {target}")
        print(f"测量结果: {counts}")
        
        # 找出最可能的结果
        most_common = max(counts.items(), key=lambda x: x[1])
        found_string = most_common[0]
        
        # 在Qiskit Grover结果中，状态可能包含辅助量子比特，需要提取
        if '_' in found_string:
            # 提取与目标长度相同的部分
            relevant_bits = found_string[-len(target):]
        else:
            relevant_bits = found_string
        
        print(f"找到的字符串: {relevant_bits}")
        print(f"成功: {relevant_bits == target}")
    
    print("\n解释:")
    print("- Grover算法通过迭代放大目标状态的振幅来查找解")
    print("- 对于N=2^n个元素的数据库，经典搜索需要O(N)次查询")
    print("- Grover算法只需O(√N)次查询，对于大型问题是显著的加速")
    print("- 最佳迭代次数约为π√N/4，太多或太少都会降低成功概率")
    print("- Qiskit提供了高级工具来简化Grover算法的实现")
    
    return results

# 取消注释下面的行以查看参考解答
# grover_results = exercise3_solution()

# --------------------------------
# 练习4: 实现量子相位估计
# --------------------------------
print("\n练习4: 实现量子相位估计")
print("任务1: 创建一个相位估计电路，估计幺正算子的特征值")
print("任务2: 执行相位估计并分析结果准确性")
print("任务3: 尝试不同精度(量子比特数)并比较结果")

# 提示
print("\n提示:")
print("- 量子相位估计是许多量子算法的关键子程序")
print("- 算法用于估计幺正算子U的特征值e^{2πiθ}中的θ")
print("- 算法需要两个寄存器:计数寄存器和特征值寄存器")
print("- 主要步骤:初始化、应用受控U^(2^j)操作、逆QFT、测量")
print("- 精度取决于计数寄存器的量子比特数")

# 参考解答
def exercise4_solution():
    # 任务1-2: 实现量子相位估计
    
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
    
    # 任务3: 尝试不同精度
    precision_levels = [3, 5, 7]
    simulator = Aer.get_backend('qasm_simulator')
    
    results = {}
    circuits = {}
    
    for precision in precision_levels:
        # 创建QPE电路
        qpe_circuit = phase_estimation(s_gate(), precision, prepare_eigenstate())
        circuits[precision] = qpe_circuit
        
        # 运行电路
        job = execute(qpe_circuit, simulator, shots=1024)
        counts = job.result().get_counts()
        results[precision] = counts
    
    print("\n参考解答 - 练习4:")
    
    # 真实相位值(S门对应θ=1/4)
    true_phase = 0.25
    
    for precision, counts in results.items():
        print(f"\n精度(量子比特数): {precision}")
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
    example_precision = 5
    circuit = circuits[example_precision]
    result = results[example_precision]
    
    print(f"\n精度为{example_precision}的QPE电路:")
    print(circuit.draw(output='text', fold=120))
    
    print("\n解释:")
    print("- 量子相位估计能够估计幺正算子的特征值e^{2πiθ}中的相位θ")
    print("- 精度取决于计数寄存器的量子比特数")
    print("- 对于n个量子比特，我们可以将θ估计到2^-n的精度")
    print("- 特征态必须已知并且能够准备")
    print("- 该算法是Shor算法、量子化学模拟等其他量子算法的核心")
    
    # 处理结果以显示相位而不是二进制字符串
    phase_counts = {}
    for bitstring, count in result.items():
        decimal = int(bitstring, 2)
        phase = decimal / (2**example_precision)
        phase_str = f"{phase:.4f}"
        phase_counts[phase_str] = count
    
    # 可视化相位估计结果
    fig = plot_histogram(phase_counts)
    plt.title(f"S门相位估计结果 (真实值θ=0.25)")
    plt.xlabel("估计的相位值θ")
    plt.savefig('exercise4_histogram.png')
    plt.close(fig)
    print("相位估计结果直方图已保存为'exercise4_histogram.png'")
    
    return circuits, results

# 取消注释下面的行以查看参考解答
# qpe_circuits, qpe_results = exercise4_solution()

# --------------------------------
# 练习5: 实现量子傅里叶变换
# --------------------------------
print("\n练习5: 实现量子傅里叶变换")
print("任务1: 手动构建量子傅里叶变换(QFT)电路")
print("任务2: 比较手动实现与Qiskit库中QFT的差异")
print("任务3: 对不同输入态应用QFT并分析结果")

# 提示
print("\n提示:")
print("- QFT是量子版本的离散傅里叶变换")
print("- QFT将计算基态|x⟩映射到傅里叶基态")
print("- 实现包括Hadamard门和受控旋转门")
print("- 可以使用circuit.qft()或qiskit.circuit.library.QFT")
print("- QFT在许多量子算法中是关键组件，如Shor算法和相位估计")

# 参考解答
def exercise5_solution():
    # 任务1: 手动构建QFT电路
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
    
    # 任务2: 比较手动实现与Qiskit库实现
    n_qubits = 4
    
    # 手动实现
    manual_qft_circuit = manual_qft(n_qubits)
    
    # Qiskit库实现
    from qiskit.circuit.library import QFT
    library_qft_circuit = QFT(n_qubits)
    
    # 任务3: 对不同输入态应用QFT
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
    
    print("\n参考解答 - 练习5:")
    
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
    
    print("\n不同输入态的QFT结果:")
    for state_label, statevector in results.items():
        print(f"\n输入态|{state_label}⟩的QFT后状态:")
        # 提取振幅和相位
        amplitudes = np.abs(statevector)
        phases = np.angle(statevector) / np.pi
        
        # 显示最大振幅的几个状态
        indices = np.argsort(-amplitudes)[:8]  # 取振幅最大的8个
        for idx in indices:
            if amplitudes[idx] > 0.01:  # 仅显示振幅较大的
                bin_idx = bin(idx)[2:].zfill(n_qubits)
                print(f"  |{bin_idx}⟩: 振幅={amplitudes[idx]:.4f}, 相位={phases[idx]:.4f}π")
    
    print("\n解释:")
    print("- QFT将计算基态转换到傅里叶基态")
    print("- |0⟩态在QFT下变为均匀叠加态")
    print("- 其他输入产生特定的振幅和相位模式")
    print("- QFT是逆QFT的共轭转置")
    print("- 经典FFT算法复杂度为O(n log n)，QFT为O(n²)")
    print("- 但QFT作用于叠加态，使其在特定算法中提供指数级加速")
    
    # 可视化|0101⟩的QFT结果
    from qiskit.visualization import plot_state_city
    
    fig = plot_state_city(results['0101'])
    plt.title("输入态|0101⟩的QFT结果")
    plt.savefig('exercise5_qft_state.png')
    plt.close(fig)
    print("状态可视化已保存为'exercise5_qft_state.png'")
    
    return results

# 取消注释下面的行以查看参考解答
# qft_results = exercise5_solution()

# --------------------------------
# 总结
# --------------------------------
print("\n==== 练习总结 ====")
print("完成这些练习后，您应该已经掌握了几种基本量子算法，包括:")
print("1. Deutsch-Jozsa算法 - 区分常数和平衡函数")
print("2. Bernstein-Vazirani算法 - 寻找隐藏字符串")
print("3. Grover搜索算法 - 在未排序数据中搜索")
print("4. 量子相位估计 - 估计幺正算子的特征值")
print("5. 量子傅里叶变换 - 量子信息的基本变换")

print("\n这些算法展示了量子计算的几种基本优势:")
print("- 量子并行性 - 同时处理多种可能性")
print("- 干涉效应 - 增强正确答案的概率")
print("- 纠缠 - 在量子比特之间建立相关性")

print("\n要查看参考解答，请取消注释相应的函数调用")
print("继续深入探索这些算法，并尝试将其应用到更复杂的问题中！") 