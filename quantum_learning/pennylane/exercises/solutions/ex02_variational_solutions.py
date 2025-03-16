#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Xanadu PennyLane框架 - 变分量子电路练习解答

本文件包含对PennyLane变分量子电路练习的参考解答。
如果您还没有尝试完成练习，建议先自行尝试再查看解答。
"""

import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

print("===== PennyLane变分量子电路练习解答 =====")

"""
练习1: 创建和优化基本变分电路
"""
print("\n练习1: 创建和优化基本变分电路 - 解答")

def exercise1_solution():
    # 创建量子设备
    dev = qml.device('default.qubit', wires=2)
    
    # 定义变分量子电路
    @qml.qnode(dev)
    def variational_circuit(params):
        # 编码层 - 旋转门
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=0)
        qml.RZ(params[2], wires=1)
        qml.RY(params[3], wires=1)
        
        # 纠缠层 - CNOT门
        qml.CNOT(wires=[0, 1])
        
        # 第二层变分门
        qml.RX(params[4], wires=0)
        qml.RY(params[5], wires=1)
        
        # 测量层 - 测量两个量子比特的Z算符的张量积
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
    
    # 定义成本函数 - 目标是使两个量子比特反相关，即Z0×Z1 = -1
    def cost_function(params):
        """
        当量子比特反相关时，PauliZ算符的期望值积为-1
        （即一个量子比特为|0⟩，另一个为|1⟩，或相反）
        所以我们希望最小化 1 + <Z0×Z1>
        """
        corr = variational_circuit(params)
        return 1 + corr  # 当完美反相关时，成本为0
    
    # 初始化随机参数
    params = np.random.uniform(0, 2*np.pi, 6)
    
    # 使用梯度下降优化器
    opt = qml.GradientDescentOptimizer(stepsize=0.1)
    
    # 优化过程
    cost_history = [cost_function(params)]
    
    print(f"初始成本: {cost_history[0]:.6f}")
    
    # 运行优化100步
    steps = 100
    for i in range(steps):
        params = opt.step(cost_function, params)
        cost = cost_function(params)
        cost_history.append(cost)
        
        if (i + 1) % 20 == 0:
            print(f"步骤 {i+1}: 成本 = {cost:.6f}")
    
    print(f"最终成本: {cost_history[-1]:.6f}")
    
    # 绘制优化过程
    plt.figure()
    plt.plot(range(steps + 1), cost_history)
    plt.xlabel('优化步骤')
    plt.ylabel('成本')
    plt.title('变分电路优化过程')
    plt.grid(True)
    plt.savefig('variational_circuit_optimization.png')
    plt.close()
    
    print("变分电路优化过程已绘制并保存为'variational_circuit_optimization.png'")
    
    return params, cost_history

params1, cost_history1 = exercise1_solution()

"""
练习2: 实现变分量子特征值求解器(VQE)
"""
print("\n练习2: 实现变分量子特征值求解器(VQE) - 解答")

def exercise2_solution():
    # 创建量子设备
    dev_vqe = qml.device('default.qubit', wires=2)
    
    # 创建哈密顿量
    def create_h2_hamiltonian():
        """创建简化的H2分子哈密顿量"""
        coeffs = [0.5, 0.5, 0.5, -0.5]
        obs = [
            qml.PauliI(0) @ qml.PauliI(1),
            qml.PauliZ(0) @ qml.PauliZ(1),
            qml.PauliX(0) @ qml.PauliX(1),
            qml.PauliY(0) @ qml.PauliY(1)
        ]
        return qml.Hamiltonian(coeffs, obs)
    
    H = create_h2_hamiltonian()
    print(f"H2分子哈密顿量:\n{H}")
    
    # 定义变分电路
    @qml.qnode(dev_vqe)
    def vqe_circuit(params, hamiltonian):
        """VQE试探态准备电路"""
        # 初始态准备 - 从|00⟩开始
        # 变分层
        qml.RY(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(params[2], wires=0)
        qml.RY(params[3], wires=1)
        
        # 返回哈密顿量的期望值
        return qml.expval(hamiltonian)
    
    # 定义成本函数
    def vqe_cost(params, hamiltonian):
        """VQE成本函数 - 哈密顿量的期望值"""
        return vqe_circuit(params, hamiltonian)
    
    # 优化VQE
    init_params = np.random.uniform(0, np.pi, 4)
    opt_vqe = qml.GradientDescentOptimizer(stepsize=0.4)
    params_vqe = init_params
    energy_history = [vqe_cost(params_vqe, H)]
    
    print(f"初始能量: {energy_history[0]:.6f}")
    
    # 运行优化
    steps = 100
    for i in range(steps):
        params_vqe = opt_vqe.step(lambda p: vqe_cost(p, H), params_vqe)
        energy = vqe_cost(params_vqe, H)
        energy_history.append(energy)
        
        if (i + 1) % 20 == 0:
            print(f"步骤 {i+1}: 能量 = {energy:.6f}")
    
    print(f"优化后的能量: {energy_history[-1]:.6f}")
    print(f"理论基态能量: -1.0")
    
    # 绘制能量收敛过程
    plt.figure()
    plt.plot(range(steps + 1), energy_history)
    plt.xlabel('优化步骤')
    plt.ylabel('能量')
    plt.title('VQE能量收敛过程')
    plt.grid(True)
    plt.savefig('vqe_convergence.png')
    plt.close()
    
    print("VQE能量收敛过程已绘制并保存为'vqe_convergence.png'")
    
    return params_vqe, energy_history

params2, energy_history2 = exercise2_solution()

"""
练习3: 量子近似优化算法(QAOA)求解最大割问题
"""
print("\n练习3: 量子近似优化算法(QAOA)求解最大割问题 - 解答")

def exercise3_solution():
    # 定义问题规模
    n_nodes = 4
    dev_qaoa = qml.device('default.qubit', wires=n_nodes)
    
    # 定义图的邻接矩阵
    adjacency_matrix = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 1, 0]
    ])
    
    print(f"图的邻接矩阵:\n{adjacency_matrix}")
    
    # 创建最大割哈密顿量
    def maxcut_hamiltonian(adj_matrix):
        """创建最大割问题的哈密顿量"""
        n = len(adj_matrix)
        coeffs = []
        obs = []
        
        for i in range(n):
            for j in range(i+1, n):
                if adj_matrix[i, j] == 1:
                    # 添加哈密顿量项
                    # 对于最大割问题，我们希望相邻节点分配不同的颜色
                    # 这对应于最小化 Σ_<i,j> Z_i Z_j
                    coeffs.append(0.5)
                    obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
                    
                    # 添加常数项使得基态能量为负
                    coeffs.append(-0.5)
                    obs.append(qml.Identity(0))
        
        return qml.Hamiltonian(coeffs, obs)
    
    H_maxcut = maxcut_hamiltonian(adjacency_matrix)
    print(f"最大割哈密顿量:\n{H_maxcut}")
    
    # 实现QAOA电路
    @qml.qnode(dev_qaoa)
    def qaoa_circuit(params, hamiltonian):
        """QAOA电路"""
        # 准备均匀叠加态
        for i in range(n_nodes):
            qml.Hadamard(wires=i)
        
        # 提取QAOA参数
        p = len(params) // 2  # QAOA深度
        gammas = params[:p]
        betas = params[p:]
        
        # QAOA层
        for i in range(p):
            # 问题哈密顿量演化
            qml.ApproxTimeEvolution(hamiltonian, gammas[i], 1)
            
            # 混合哈密顿量演化
            for j in range(n_nodes):
                qml.RX(2 * betas[i], wires=j)
        
        # 返回能量期望值
        return qml.expval(hamiltonian)
    
    # 定义成本函数
    def qaoa_cost(params, hamiltonian):
        """QAOA成本函数"""
        return qaoa_circuit(params, hamiltonian)
    
    # 优化QAOA
    p = 1  # QAOA深度
    init_params = np.random.uniform(0, np.pi, 2 * p)
    opt_qaoa = qml.AdamOptimizer(stepsize=0.1)
    params_qaoa = init_params
    cost_history_qaoa = [qaoa_cost(params_qaoa, H_maxcut)]
    
    print(f"初始成本: {cost_history_qaoa[0]:.6f}")
    
    # 运行优化
    steps = 100
    for i in range(steps):
        params_qaoa = opt_qaoa.step(lambda p: qaoa_cost(p, H_maxcut), params_qaoa)
        cost = qaoa_cost(params_qaoa, H_maxcut)
        cost_history_qaoa.append(cost)
        
        if (i + 1) % 20 == 0:
            print(f"步骤 {i+1}: 成本 = {cost:.6f}")
    
    print(f"优化后的成本: {cost_history_qaoa[-1]:.6f}")
    
    # 从优化结果中提取解决方案
    def get_maxcut_solution(params, adjacency_matrix):
        """从优化的QAOA参数中提取最大割解决方案"""
        # 创建一个量子电路来获取最优解
        @qml.qnode(dev_qaoa)
        def qaoa_state(optimized_params):
            # 准备均匀叠加态
            for i in range(n_nodes):
                qml.Hadamard(wires=i)
            
            # 提取QAOA参数
            p = len(optimized_params) // 2  # QAOA深度
            gammas = optimized_params[:p]
            betas = optimized_params[p:]
            
            # QAOA层
            for i in range(p):
                # 问题哈密顿量演化
                qml.ApproxTimeEvolution(H_maxcut, gammas[i], 1)
                
                # 混合哈密顿量演化
                for j in range(n_nodes):
                    qml.RX(2 * betas[i], wires=j)
            
            # 返回计算基测量结果
            return qml.probs(wires=range(n_nodes))
        
        # 获取概率分布
        probs = qaoa_state(params)
        
        # 找到最高概率的位串
        max_prob_idx = np.argmax(probs)
        
        # 将索引转换为二进制串
        max_bitstring = format(max_prob_idx, f'0{n_nodes}b')
        
        # 计算割的大小
        cut_size = 0
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if adjacency_matrix[i, j] == 1 and max_bitstring[i] != max_bitstring[j]:
                    cut_size += 1
        
        return max_bitstring, cut_size
    
    solution, cut_size = get_maxcut_solution(params_qaoa, adjacency_matrix)
    print(f"最大割解决方案: {solution}")
    print(f"割的大小: {cut_size}")
    
    # 绘制QAOA成本曲线
    plt.figure()
    plt.plot(range(steps + 1), cost_history_qaoa)
    plt.xlabel('优化步骤')
    plt.ylabel('成本')
    plt.title('QAOA优化过程')
    plt.grid(True)
    plt.savefig('qaoa_optimization.png')
    plt.close()
    
    print("QAOA优化过程已绘制并保存为'qaoa_optimization.png'")
    
    return params_qaoa, cost_history_qaoa, solution, cut_size

params3, cost_history3, solution3, cut_size3 = exercise3_solution()

"""
练习4: 参数移位规则和量子梯度计算
"""
print("\n练习4: 参数移位规则和量子梯度计算 - 解答")

def exercise4_solution():
    # 创建量子设备
    dev_grad = qml.device('default.qubit', wires=1)
    
    # 定义简单的参数化电路
    @qml.qnode(dev_grad)
    def circuit(params):
        """简单的参数化电路"""
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=0)
        return qml.expval(qml.PauliZ(0))
    
    # 实现参数移位规则
    def parameter_shift(circuit, params, idx, shift=np.pi/2):
        """
        使用参数移位规则计算梯度
        
        Args:
            circuit: 量子电路函数
            params: 参数数组
            idx: 要计算梯度的参数索引
            shift: 移位量
            
        Returns:
            参数的梯度
        """
        # 创建移位参数
        shifted_params_plus = params.copy()
        shifted_params_plus[idx] += shift
        
        shifted_params_minus = params.copy()
        shifted_params_minus[idx] -= shift
        
        # 计算移位后的函数值
        forward = circuit(shifted_params_plus)
        backward = circuit(shifted_params_minus)
        
        # 计算梯度
        gradient = (forward - backward) / (2 * np.sin(shift))
        
        return gradient
    
    # 比较手动梯度与自动梯度
    test_params = np.array([0.5, 0.8])
    
    manual_grad_0 = parameter_shift(circuit, test_params, 0)
    manual_grad_1 = parameter_shift(circuit, test_params, 1)
    
    auto_grad = qml.grad(circuit)(test_params)
    
    print(f"参数: {test_params}")
    print(f"手动计算的梯度: [{manual_grad_0:.6f}, {manual_grad_1:.6f}]")
    print(f"PennyLane计算的梯度: {auto_grad}")
    
    # 绘制不同参数值的梯度
    param_range = np.linspace(0, 2*np.pi, 50)
    gradients_0 = []
    gradients_1 = []
    
    for param in param_range:
        params = np.array([param, np.pi/4])  # 固定第二个参数
        grad = qml.grad(circuit)(params)
        gradients_0.append(grad[0])
        
        params = np.array([np.pi/4, param])  # 固定第一个参数
        grad = qml.grad(circuit)(params)
        gradients_1.append(grad[1])
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(param_range, gradients_0)
    plt.xlabel('参数值')
    plt.ylabel('梯度 ∂f/∂θ₀')
    plt.title('RX门参数的梯度')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(param_range, gradients_1)
    plt.xlabel('参数值')
    plt.ylabel('梯度 ∂f/∂θ₁')
    plt.title('RY门参数的梯度')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('parameter_shift_gradients.png')
    plt.close()
    
    print("参数移位梯度已绘制并保存为'parameter_shift_gradients.png'")
    
    # 检验梯度计算的正确性
    def analytical_gradient(params):
        """计算解析梯度进行对比"""
        x, y = params
        return [
            -np.sin(x) * np.cos(y),  # ∂f/∂x
            -np.cos(x) * np.sin(y)   # ∂f/∂y
        ]
    
    analytical = analytical_gradient(test_params)
    print(f"解析梯度: [{analytical[0]:.6f}, {analytical[1]:.6f}]")
    
    return manual_grad_0, manual_grad_1, auto_grad

manual_grad_0, manual_grad_1, auto_grad = exercise4_solution()

"""
练习5: 构建变分量子门
"""
print("\n练习5: 构建变分量子门 - 解答")

def exercise5_solution():
    # 创建量子设备
    n_qubits = 3
    dev_vqg = qml.device('default.qubit', wires=n_qubits)
    
    # 定义目标QFT电路
    @qml.qnode(dev_vqg)
    def target_qft():
        """标准QFT电路"""
        # 准备非平凡的初始态
        qml.PauliX(wires=0)
        
        # 应用QFT
        qml.QFT(wires=range(n_qubits))
        
        # 返回状态向量
        return qml.state()
    
    # 定义变分QFT电路
    @qml.qnode(dev_vqg)
    def variational_qft(params):
        """变分QFT电路"""
        # 准备与目标电路相同的初始态
        qml.PauliX(wires=0)
        
        # 变分层结构
        # QFT包含Hadamard门和受控旋转门
        # 第一层：Hadamard门
        for i in range(n_qubits):
            qml.RY(params[i], wires=i)
            qml.RZ(params[i + n_qubits], wires=i)
        
        # 第二层：受控旋转
        qubits = list(range(n_qubits))
        
        # 应用受控旋转模拟QFT
        p_idx = 2 * n_qubits  # 参数索引
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                # 应用参数化的受控旋转
                with qml.control(1):
                    qml.RZ(params[p_idx], wires=j)
                p_idx += 1
                
                # 添加CNOT以增加纠缠
                qml.CNOT(wires=[i, j])
                qml.CNOT(wires=[j, i])
                
                # 再添加一个参数化的旋转
                with qml.control(1):
                    qml.RZ(params[p_idx], wires=j)
                p_idx += 1
        
        # 第三层：最终旋转进行微调
        for i in range(n_qubits):
            qml.RY(params[p_idx], wires=i)
            p_idx += 1
            qml.RZ(params[p_idx], wires=i)
            p_idx += 1
        
        # 返回状态向量
        return qml.state()
    
    # 计算变分电路参数总数
    n_controlled_ops = n_qubits * (n_qubits - 1)  # 每对量子比特需要两个操作
    n_params = 2 * n_qubits + 2 * n_controlled_ops + 2 * n_qubits
    
    # 计算成本函数 - 量子态保真度
    def fidelity_cost(params):
        """计算变分电路与目标电路的保真度"""
        target_state = target_qft()
        variational_state = variational_qft(params)
        
        # 计算保真度 |<ψ|φ>|²
        fidelity = np.abs(np.vdot(target_state, variational_state))**2
        
        # 我们希望最大化保真度，所以返回负保真度作为成本
        return -fidelity
    
    # 优化变分QFT电路
    init_params = np.random.uniform(0, 2*np.pi, n_params)
    opt_vqft = qml.AdamOptimizer(stepsize=0.1)
    params_vqft = init_params
    fidelity_history = [1 + fidelity_cost(params_vqft)]  # 转换为保真度
    
    print(f"变分QFT电路参数数量: {n_params}")
    print(f"初始保真度: {fidelity_history[0]:.6f}")
    
    # 运行优化
    steps = 100
    for i in range(steps):
        params_vqft = opt_vqft.step(fidelity_cost, params_vqft)
        fidelity = 1 + fidelity_cost(params_vqft)  # 转换为保真度
        fidelity_history.append(fidelity)
        
        if (i + 1) % 20 == 0:
            print(f"步骤 {i+1}: 保真度 = {fidelity:.6f}")
    
    print(f"最终保真度: {fidelity_history[-1]:.6f}")
    
    # 绘制保真度收敛过程
    plt.figure()
    plt.plot(range(steps + 1), fidelity_history)
    plt.xlabel('优化步骤')
    plt.ylabel('保真度')
    plt.title('变分QFT电路优化')
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.savefig('variational_qft_fidelity.png')
    plt.close()
    
    print("变分QFT保真度已绘制并保存为'variational_qft_fidelity.png'")
    
    return params_vqft, fidelity_history

params5, fidelity_history5 = exercise5_solution()

"""
练习6: 集成不同优化器的比较
"""
print("\n练习6: 集成不同优化器的比较 - 解答")

def exercise6_solution():
    # 创建量子设备
    dev_opt = qml.device('default.qubit', wires=2)
    
    # 创建一个简单的变分电路
    @qml.qnode(dev_opt)
    def opt_circuit(params):
        """用于优化器比较的电路"""
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RZ(params[2], wires=0)
        qml.RX(params[3], wires=1)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
    
    # 定义成本函数
    def opt_cost(params):
        """优化的成本函数"""
        return 1 - opt_circuit(params)
    
    # 比较不同优化器
    init_params = np.random.uniform(0, 2*np.pi, 4)
    n_steps = 100
    
    # 创建优化器字典
    optimizers = {
        "GradientDescent": qml.GradientDescentOptimizer(stepsize=0.2),
        "Adam": qml.AdamOptimizer(stepsize=0.1),
        "Adagrad": qml.AdagradOptimizer(stepsize=0.5),
        "Momentum": qml.MomentumOptimizer(stepsize=0.1, momentum=0.9)
    }
    
    # 存储每个优化器的结果
    results = {}
    
    for name, opt in optimizers.items():
        params = init_params.copy()
        cost_history = [opt_cost(params)]
        
        for i in range(n_steps):
            # 优化步骤
            params = opt.step(opt_cost, params)
            
            # 存储成本
            cost_history.append(opt_cost(params))
        
        results[name] = {
            "final_params": params,
            "cost_history": cost_history,
            "final_cost": cost_history[-1]
        }
        
        print(f"{name}: 最终成本 = {cost_history[-1]:.6f}")
    
    # 绘制比较结果
    plt.figure(figsize=(10, 6))
    
    for name, result in results.items():
        plt.plot(result["cost_history"], label=name)
    
    plt.xlabel('优化步骤')
    plt.ylabel('成本')
    plt.title('不同优化器的性能比较')
    plt.legend()
    plt.grid(True)
    plt.savefig('optimizer_comparison.png')
    plt.close()
    
    print("优化器比较已绘制并保存为'optimizer_comparison.png'")
    
    # 分析结果
    print("\n优化器性能比较:")
    for name, result in sorted(results.items(), key=lambda x: x[1]["final_cost"]):
        print(f"{name}: 最终成本 = {result['final_cost']:.6f}")
    
    # 比较不同优化器的收敛速度
    convergence_step = {}
    threshold = 0.01  # 收敛阈值
    
    for name, result in results.items():
        costs = result["cost_history"]
        for i, cost in enumerate(costs):
            if cost < threshold:
                convergence_step[name] = i
                break
        else:
            convergence_step[name] = n_steps
    
    print("\n收敛到阈值 {threshold} 所需的步骤:")
    for name, step in sorted(convergence_step.items(), key=lambda x: x[1]):
        if step < n_steps:
            print(f"{name}: {step}步")
        else:
            print(f"{name}: 未收敛")
    
    return results

optimizer_results = exercise6_solution()

print("\n所有练习解答已完成。这些解答展示了PennyLane的变分量子电路和优化功能。")
print("建议尝试修改这些代码，探索不同的电路结构、成本函数和优化器，以加深理解。")
print("下一步: 学习量子机器学习技术和应用。") 