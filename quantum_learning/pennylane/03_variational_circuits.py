#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Xanadu PennyLane框架学习 3：变分量子电路
本文件深入介绍变分量子电路的结构、类型和应用
"""

# 导入必要的库
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

print("===== PennyLane变分量子电路 =====")

# 检查PennyLane版本
print(f"PennyLane版本: {qml.__version__}")

# 1. 变分量子电路概述
print("\n1. 变分量子电路概述")
print("变分量子电路(VQC)是一类参数化的量子电路，是众多量子算法的基础")
print("主要特点:")
print("- 包含可调节的参数")
print("- 可以通过经典优化方法进行训练")
print("- 适用于NISQ（嘈杂中等规模量子）设备")
print("- 可以执行各种计算任务，从模拟到机器学习")

# 2. 变分量子电路的基本结构
print("\n2. 变分量子电路的基本结构")
print("变分量子电路通常包含以下组件:")
print("1. 初始状态准备 - 通常是简单的状态如|0...0⟩")
print("2. 数据编码 - 将经典数据编码到量子态中")
print("3. 变分部分 - 包含参数化量子门的层")
print("4. 测量 - 获取计算结果")

# 创建一个简单的设备
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# 定义一个基本的变分量子电路
@qml.qnode(dev)
def basic_variational_circuit(x, params):
    # 1. 初始状态准备 (默认为|0...0⟩)
    
    # 2. 数据编码
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
    
    # 3. 变分部分: 3层，每层包含参数化旋转门和纠缠门
    for layer in range(3):
        # 参数化旋转门
        for i in range(n_qubits):
            qml.RX(params[layer][i][0], wires=i)
            qml.RZ(params[layer][i][1], wires=i)
        
        # 纠缠门
        for i in range(n_qubits-1):
            qml.CNOT(wires=[i, i+1])
        qml.CNOT(wires=[n_qubits-1, 0])  # 闭合链
    
    # 4. 测量
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# 准备随机输入数据和参数
x = np.random.uniform(0, np.pi, n_qubits)
params_shape = (3, n_qubits, 2)  # 3层, n_qubits个量子比特, 每个量子比特2个参数
params = np.random.uniform(-np.pi, np.pi, params_shape)

# 执行电路
result = basic_variational_circuit(x, params)
print("\n基本变分量子电路的结构:")
print(qml.draw(basic_variational_circuit)(x, params))
print(f"\n输出结果: {result}")

# 3. 常见变分电路结构
print("\n3. 常见变分电路结构")
print("PennyLane提供了多种预定义的变分电路结构:")

# 3.1 强纠缠层
print("\n3.1 强纠缠层 (StronglyEntanglingLayers)")
@qml.qnode(dev)
def strongly_entangling_circuit(params):
    qml.templates.StronglyEntanglingLayers(params, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# 参数形状取决于层数和量子比特数
se_params_shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=n_qubits)
se_params = np.random.uniform(-np.pi, np.pi, se_params_shape)

# 执行电路
print("强纠缠层电路结构:")
print(qml.draw(strongly_entangling_circuit)(se_params))

# 3.2 随机层
print("\n3.2 随机层 (RandomLayers)")
@qml.qnode(dev)
def random_layers_circuit(params):
    qml.templates.RandomLayers(params, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# 创建随机层参数
rl_params = np.random.uniform(-np.pi, np.pi, (3, n_qubits))

# 执行电路
print("随机层电路结构:")
print(qml.draw(random_layers_circuit)(rl_params))

# 3.3 基本纠缠层
print("\n3.3 基本纠缠层 (BasicEntanglerLayers)")
@qml.qnode(dev)
def basic_entangler_circuit(params):
    qml.templates.BasicEntanglerLayers(params, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# 参数形状取决于层数和量子比特数
be_params_shape = qml.templates.BasicEntanglerLayers.shape(n_layers=2, n_wires=n_qubits)
be_params = np.random.uniform(-np.pi, np.pi, be_params_shape)

# 执行电路
print("基本纠缠层电路结构:")
print(qml.draw(basic_entangler_circuit)(be_params))

# 4. 数据编码方法
print("\n4. 数据编码方法")
print("在变分量子电路中，数据编码是一个关键步骤，PennyLane提供多种编码方法:")

# 4.1 角度编码
print("\n4.1 角度编码 (AngleEmbedding)")
@qml.qnode(dev)
def angle_embedding_circuit(x):
    qml.templates.AngleEmbedding(x, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# 创建输入数据
angle_data = np.random.uniform(0, np.pi, n_qubits)

print("角度编码电路结构:")
print(qml.draw(angle_embedding_circuit)(angle_data))

# 4.2 振幅编码
print("\n4.2 振幅编码 (AmplitudeEmbedding)")
@qml.qnode(dev)
def amplitude_embedding_circuit(x):
    # 需要2^n_qubits个输入特征
    qml.templates.AmplitudeEmbedding(x, wires=range(n_qubits), normalize=True)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# 创建2^n_qubits个特征
n_features = 2**n_qubits
amp_data = np.random.uniform(-1, 1, n_features)
amp_data = amp_data / np.linalg.norm(amp_data)  # 归一化

print("振幅编码电路结构:")
print(qml.draw(amplitude_embedding_circuit)(amp_data))

# 4.3 IQP特征映射
print("\n4.3 IQP特征映射 (IQPEmbedding)")
@qml.qnode(dev)
def iqp_embedding_circuit(x):
    qml.templates.IQPEmbedding(x, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# 创建输入数据
iqp_data = np.random.uniform(-np.pi, np.pi, n_qubits)

print("IQP特征映射电路结构:")
print(qml.draw(iqp_embedding_circuit)(iqp_data))

# 5. 变分电路的表现力
print("\n5. 变分电路的表现力")
print("变分电路的表现力取决于其架构、层数和参数数量")

# 创建一个变分电路表现力实验
def circuit_expressivity_experiment():
    # 创建一个基于CNOT纠缠的变分电路
    
    @qml.qnode(dev)
    def circuit(params, depth):
        # 应用参数化层
        for d in range(depth):
            # 参数旋转层
            for i in range(n_qubits):
                qml.RX(params[d, i, 0], wires=i)
                qml.RY(params[d, i, 1], wires=i)
                qml.RZ(params[d, i, 2], wires=i)
            
            # 纠缠层
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
        
        # 返回每个量子比特的期望值
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    # 测试不同深度的电路
    for depth in [1, 2, 4]:
        # 随机参数
        params = np.random.uniform(-np.pi, np.pi, (depth, n_qubits, 3))
        
        # 运行电路
        result = circuit(params, depth)
        
        # 计算电路复杂度
        n_params = depth * n_qubits * 3
        n_gates = depth * (n_qubits * 3 + n_qubits - 1)  # 旋转门 + CNOT门
        
        print(f"\n深度 {depth}:")
        print(f"- 参数数量: {n_params}")
        print(f"- 门数量: {n_gates}")
        
        # 计算输出向量的范数作为复杂性的简单度量
        vector_norm = np.linalg.norm(result)
        print(f"- 输出向量范数: {vector_norm:.6f}")
        print(f"- 输出向量平均值: {np.mean(result):.6f}")

print("\n执行电路表现力实验:")
circuit_expressivity_experiment()

# 6. 常见的变分量子算法
print("\n6. 常见的变分量子算法")
print("变分量子电路是多种量子算法的基础:")

# 6.1 量子近似优化算法(QAOA)
print("\n6.1 量子近似优化算法(QAOA)")
print("QAOA用于解决组合优化问题，如MaxCut问题")

# 创建一个小型QAOA示例
# 定义一个简单的图（MaxCut问题）
n_nodes = 4
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]

# 创建成本哈密顿量
cost_h = qml.Hamiltonian(
    coeffs = [1.0 for _ in range(len(edges))],
    observables = [qml.PauliZ(i) @ qml.PauliZ(j) for i, j in edges]
)

# 创建混合哈密顿量
mix_h = qml.Hamiltonian(
    coeffs = [1.0 for _ in range(n_nodes)],
    observables = [qml.PauliX(i) for i in range(n_nodes)]
)

# 定义QAOA电路
def qaoa_layer(gamma, alpha):
    # 问题哈密顿量演化
    qml.exp(cost_h, gamma)
    # 混合哈密顿量演化
    qml.exp(mix_h, alpha)

# 创建具有深度p=2的QAOA电路
p = 2
dev_qaoa = qml.device("default.qubit", wires=n_nodes)

@qml.qnode(dev_qaoa)
def qaoa_circuit(params):
    # 初始状态：均匀叠加态
    for i in range(n_nodes):
        qml.Hadamard(wires=i)
    
    # 应用p层QAOA
    for i in range(p):
        qaoa_layer(params[2*i], params[2*i+1])
    
    # 返回成本哈密顿量的期望值
    return qml.expval(cost_h)

# 优化QAOA参数
def optimize_qaoa():
    # 随机初始参数
    params = np.random.uniform(0, np.pi, 2*p)
    
    # 定义目标函数（我们想要最小化）
    def objective(params):
        return qaoa_circuit(params)
    
    # 模拟优化过程（只显示几步）
    optimizer = qml.GradientDescentOptimizer(stepsize=0.1)
    params_history = [params]
    energy_history = [qaoa_circuit(params)]
    
    for i in range(5):
        params = optimizer.step(objective, params)
        params_history.append(params)
        energy_history.append(qaoa_circuit(params))
    
    return params_history, energy_history

# 运行优化并打印结果
params_history, energy_history = optimize_qaoa()

print("QAOA电路结构:")
print(qml.draw(qaoa_circuit)(params_history[-1]))
print("\nQAOA优化过程:")
for i, (params, energy) in enumerate(zip(params_history, energy_history)):
    print(f"迭代 {i}: 能量 = {energy:.6f}, 参数 = {params}")

# 6.2 变分量子特征求解器(VQE)
print("\n6.2 变分量子特征求解器(VQE)")
print("VQE用于估计哈密顿量的基态能量，特别是在量子化学中")

# 创建一个简单的H2分子哈密顿量的简化版本
h2_hamiltonian = qml.Hamiltonian(
    coeffs = [0.2252, 0.3435, -0.4347, 0.5716, 0.0910],
    observables = [
        qml.Identity(0),
        qml.PauliZ(0),
        qml.PauliZ(1),
        qml.PauliZ(0) @ qml.PauliZ(1),
        qml.PauliX(0) @ qml.PauliX(1)
    ]
)

# 创建一个用于VQE的简单变分形式
def vqe_ansatz(params):
    # 初始态 |01⟩
    qml.PauliX(wires=1)
    
    # 变分演化
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(params[2], wires=0)
    qml.RY(params[3], wires=1)

# 创建VQE电路
dev_vqe = qml.device("default.qubit", wires=2)

@qml.qnode(dev_vqe)
def vqe_circuit(params):
    vqe_ansatz(params)
    return qml.expval(h2_hamiltonian)

# 随机初始参数
init_params = np.random.uniform(0, 2*np.pi, 4)

# 优化VQE参数
def optimize_vqe():
    # 定义目标函数
    def objective(params):
        return vqe_circuit(params)
    
    # 随机初始参数
    params = np.random.uniform(0, 2*np.pi, 4)
    
    # 模拟优化过程（只显示几步）
    optimizer = qml.GradientDescentOptimizer(stepsize=0.2)
    params_history = [params]
    energy_history = [vqe_circuit(params)]
    
    for i in range(5):
        params = optimizer.step(objective, params)
        params_history.append(params)
        energy_history.append(vqe_circuit(params))
    
    return params_history, energy_history

# 运行优化并打印结果
vqe_params_history, vqe_energy_history = optimize_vqe()

print("VQE电路结构:")
print(qml.draw(vqe_circuit)(vqe_params_history[-1]))
print("\nVQE优化过程:")
for i, (params, energy) in enumerate(zip(vqe_params_history, vqe_energy_history)):
    print(f"迭代 {i}: 能量 = {energy:.6f}")

# 7. 变分量子电路中的条幺性(Unitarity)
print("\n7. 变分量子电路中的条幺性(Unitarity)")
print("量子电路必须是条幺的，这影响了参数化策略")

# 创建一个带有条幺性约束的参数化电路
dev_unit = qml.device("default.qubit", wires=2)

@qml.qnode(dev_unit)
def unitary_circuit(params):
    # 一个2量子比特的条幺变换
    # U = exp(-iH) 其中H是厄米算符
    
    # 构建一个参数化的厄米算符
    H = (
        params[0] * qml.PauliX(0) + 
        params[1] * qml.PauliY(0) + 
        params[2] * qml.PauliZ(0) +
        params[3] * qml.PauliX(1) + 
        params[4] * qml.PauliY(1) + 
        params[5] * qml.PauliZ(1) +
        params[6] * qml.PauliX(0) @ qml.PauliX(1) +
        params[7] * qml.PauliY(0) @ qml.PauliY(1) +
        params[8] * qml.PauliZ(0) @ qml.PauliZ(1)
    )
    
    # 应用条幺演化
    qml.exp(H, 1.0)
    
    return qml.state()

# 随机参数
unit_params = np.random.uniform(-0.1, 0.1, 9)

# 执行电路
unit_state = unitary_circuit(unit_params)

print("\n厄米生成的条幺变换:")
print(qml.draw(unitary_circuit)(unit_params))

# 8. 梯度消失问题与解决方案
print("\n8. 梯度消失问题与解决方案")
print("变分量子电路可能面临梯度消失/爆炸问题")

# 创建一个展示梯度消失的电路
dev_grad = qml.device("default.qubit", wires=1)

@qml.qnode(dev_grad)
def gradient_circuit(params):
    # 多个旋转门串联可能导致梯度消失
    for i in range(20):
        qml.RX(params[i], wires=0)
    return qml.expval(qml.PauliZ(0))

# 随机参数
grad_params = np.random.uniform(-np.pi, np.pi, 20)

# 计算整体梯度
gradient = qml.grad(gradient_circuit)(grad_params)

print("\n梯度值:")
print(f"前5个参数的梯度: {gradient[:5] if len(gradient) > 0 else '空梯度'}")
print(f"后5个参数的梯度: {gradient[-5:] if len(gradient) > 0 else '空梯度'}")

# 安全处理可能为空的梯度数组
if len(gradient) > 0 and np.any(gradient != 0):
    print(f"梯度最大绝对值: {np.max(np.abs(gradient)):.6e}")
    non_zero_grads = np.abs(gradient[gradient != 0])
    if len(non_zero_grads) > 0:
        print(f"梯度最小绝对值: {np.min(non_zero_grads):.6e}")
    else:
        print("所有梯度都为零")
else:
    print("梯度为空或全为零，无法计算最大/最小值")
    # 使用一个简单示例来展示梯度概念
    print("为演示目的，使用一个简单的示例梯度：[0.5, -0.3]")

# 8.1 参数移位规则
print("\n8.1 参数移位规则")
print("参数移位规则是计算变分量子电路梯度的一种方法")

@qml.qnode(dev_grad)
def shift_circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    return qml.expval(qml.PauliZ(0))

# 手动计算梯度
def parameter_shift_gradient(f, params, idx, shift=np.pi/2):
    shifted = params.copy()
    shifted[idx] += shift
    forward = f(shifted)
    
    shifted = params.copy()
    shifted[idx] -= shift
    backward = f(shifted)
    
    return (forward - backward) / (2 * np.sin(shift))

# 随机参数
shift_params = np.random.uniform(-np.pi, np.pi, 2)

# 计算自动梯度和参数移位梯度
auto_grad = qml.grad(shift_circuit)(shift_params)
shift_grad0 = parameter_shift_gradient(shift_circuit, shift_params, 0)
shift_grad1 = parameter_shift_gradient(shift_circuit, shift_params, 1)

print("\n参数移位梯度比较:")
print(f"自动梯度: {auto_grad}")
print(f"参数移位梯度 (手动计算): [{shift_grad0}, {shift_grad1}]")

# 9. 总结
print("\n9. 总结")
print("1. 变分量子电路是可调节参数的量子电路")
print("2. 它们是许多量子算法（如QAOA和VQE）的基础")
print("3. 特征编码是将经典数据引入量子电路的关键")
print("4. 变分电路的表现力取决于架构和深度")
print("5. 梯度计算对优化变分电路至关重要")
print("6. 参数移位规则是计算梯度的一种方法")

print("\n下一步学习:")
print("- 量子梯度和优化技术")
print("- 量子机器学习")
print("- 变分量子算法的实际应用") 