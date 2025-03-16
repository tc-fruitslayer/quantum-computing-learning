#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Xanadu PennyLane框架 - 变分量子电路练习

本文件包含关于变分量子电路和变分量子算法的练习。
完成这些练习将帮助您理解参数化量子电路和量子化学、优化问题的应用。
"""

import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

print("===== PennyLane变分量子电路练习 =====")

"""
练习1: 创建和优化基本变分电路
------------------------------
任务:
1. 创建一个包含2个量子比特的设备
2. 定义一个变分量子电路，包含旋转门（RX, RY, RZ）和纠缠门（CNOT）
3. 定义一个成本函数，目标是使两个量子比特的测量结果反相关（一个为|0⟩时，另一个为|1⟩）
4. 使用梯度下降优化器优化参数
5. 绘制优化过程中成本函数的变化
"""

print("\n练习1: 创建和优化基本变分电路")

# 您的代码:
# dev = ...

# def variational_circuit(params):
#     # 编码层 - 旋转门
#     ...
#     
#     # 纠缠层 - CNOT门
#     ...
#     
#     # 测量层
#     return ...

# def cost_function(params):
#     """
#     定义要优化的成本函数
#     提示: 考虑使用 qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
#     希望两个量子比特反相关时，期望值应该接近什么？
#     """
#     ...
#     return ...

# 优化过程
# params = ...
# opt = ...
# cost_history = []

# for i in range(...):
#     # 优化步骤
#     ...
#     
#     # 存储成本
#     ...
#     
#     # 打印进度
#     ...

# 绘制优化过程
# plt.figure()
# plt.plot(...)
# plt.xlabel(...)
# plt.ylabel(...)
# plt.title(...)
# plt.savefig('variational_circuit_optimization.png')
# plt.close()

"""
练习2: 实现变分量子特征值求解器(VQE)
------------------------------------
任务:
1. 创建一个2量子比特系统的简化氢分子哈密顿量
   H = 0.5*I⊗I + 0.5*Z⊗Z + 0.5*X⊗X - 0.5*Y⊗Y
2. 定义一个变分量子电路作为VQE的试探态
3. 计算电路产生的试探态在哈密顿量下的期望值
4. 使用优化器找到基态能量
5. 比较优化结果与理论基态能量（应为-1.0）
"""

print("\n练习2: 实现变分量子特征值求解器(VQE)")

# 您的代码:
# dev_vqe = ...

# 创建哈密顿量
# def create_h2_hamiltonian():
#     """创建简化的H2分子哈密顿量"""
#     coeffs = ...
#     obs = [
#         ...
#     ]
#     return qml.Hamiltonian(coeffs, obs)

# H = create_h2_hamiltonian()
# print(f"H2分子哈密顿量:\n{H}")

# 定义变分电路
# @qml.qnode(dev_vqe)
# def vqe_circuit(params, hamiltonian):
#     """VQE试探态准备电路"""
#     # 初始态准备
#     ...
#     
#     # 变分层
#     ...
#     
#     # 返回期望值
#     return ...

# 定义成本函数
# def vqe_cost(params, hamiltonian):
#     """VQE成本函数 - 哈密顿量的期望值"""
#     ...

# 优化VQE
# init_params = ...
# opt_vqe = ...
# params_vqe = init_params
# energy_history = [vqe_cost(params_vqe, H)]

# print(f"初始能量: {energy_history[0]:.6f}")

# for i in range(...):
#     # 优化步骤
#     ...
#     
#     # 存储能量
#     ...
#     
#     # 打印进度
#     ...

# print(f"优化后的能量: {energy_history[-1]:.6f}")
# print(f"理论基态能量: -1.0")

# 绘制能量收敛过程
# plt.figure()
# plt.plot(...)
# plt.xlabel(...)
# plt.ylabel(...)
# plt.title(...)
# plt.grid(True)
# plt.savefig('vqe_convergence.png')
# plt.close()

"""
练习3: 量子近似优化算法(QAOA)求解最大割问题
-------------------------------------------
任务:
1. 定义一个4节点的图（使用邻接矩阵表示）
2. 创建最大割问题的哈密顿量
3. 实现QAOA电路，包括问题哈密顿量演化和混合哈密顿量演化
4. 优化QAOA参数
5. 从优化结果中提取最大割解决方案
"""

print("\n练习3: 量子近似优化算法(QAOA)求解最大割问题")

# 您的代码:
# n_nodes = 4
# dev_qaoa = ...

# 定义图的邻接矩阵
# adjacency_matrix = np.array([
#     ...
# ])

# print(f"图的邻接矩阵:\n{adjacency_matrix}")

# 创建最大割哈密顿量
# def maxcut_hamiltonian(adj_matrix):
#     """创建最大割问题的哈密顿量"""
#     n = len(adj_matrix)
#     coeffs = []
#     obs = []
#     
#     for i in range(n):
#         for j in range(i+1, n):
#             if adj_matrix[i, j] == 1:
#                 # 添加哈密顿量项
#                 ...
#     
#     return qml.Hamiltonian(coeffs, obs)

# H_maxcut = maxcut_hamiltonian(adjacency_matrix)
# print(f"最大割哈密顿量:\n{H_maxcut}")

# 实现QAOA电路
# @qml.qnode(dev_qaoa)
# def qaoa_circuit(params, hamiltonian):
#     """QAOA电路"""
#     # 准备均匀叠加态
#     ...
#     
#     # 提取QAOA参数
#     p = len(params) // 2  # QAOA深度
#     gammas = params[:p]
#     betas = params[p:]
#     
#     # QAOA层
#     for i in range(p):
#         # 问题哈密顿量演化
#         ...
#         
#         # 混合哈密顿量演化
#         ...
#     
#     # 返回能量期望值
#     return ...

# 定义成本函数
# def qaoa_cost(params, hamiltonian):
#     """QAOA成本函数"""
#     ...

# 优化QAOA
# p = 1  # QAOA深度
# init_params = ...
# opt_qaoa = ...
# params_qaoa = init_params
# cost_history_qaoa = [qaoa_cost(params_qaoa, H_maxcut)]

# print(f"初始成本: {cost_history_qaoa[0]:.6f}")

# for i in range(...):
#     # 优化步骤
#     ...
#     
#     # 存储成本
#     ...
#     
#     # 打印进度
#     ...

# print(f"优化后的成本: {cost_history_qaoa[-1]:.6f}")

# 从优化结果中提取解决方案
# def get_maxcut_solution(params, adjacency_matrix):
#     """从优化的QAOA参数中提取最大割解决方案"""
#     # 创建一个量子电路来获取最优解
#     @qml.qnode(dev_qaoa)
#     def qaoa_state(optimized_params):
#         # 准备均匀叠加态
#         ...
#         
#         # QAOA层（与上面相同）
#         ...
#         
#         # 返回计算基测量结果
#         return qml.probs(wires=range(n_nodes))
#     
#     # 获取概率分布
#     probs = ...
#     
#     # 找到最高概率的位串
#     max_prob_idx = ...
#     max_bitstring = ...
#     
#     # 计算割的大小
#     cut_size = 0
#     for i in range(n_nodes):
#         for j in range(i+1, n_nodes):
#             if adjacency_matrix[i, j] == 1 and max_bitstring[i] != max_bitstring[j]:
#                 cut_size += 1
#     
#     return max_bitstring, cut_size

# solution, cut_size = get_maxcut_solution(params_qaoa, adjacency_matrix)
# print(f"最大割解决方案: {solution}")
# print(f"割的大小: {cut_size}")

"""
练习4: 参数移位规则和量子梯度计算
----------------------------------
任务:
1. 创建一个简单的参数化量子电路
2. 手动实现参数移位规则计算梯度
3. 比较手动计算的梯度与PennyLane自动计算的梯度
4. 为不同参数值计算梯度，并绘制梯度曲线
"""

print("\n练习4: 参数移位规则和量子梯度计算")

# 您的代码:
# dev_grad = ...

# @qml.qnode(dev_grad)
# def circuit(params):
#     """简单的参数化电路"""
#     qml.RX(params[0], wires=0)
#     qml.RY(params[1], wires=0)
#     return qml.expval(qml.PauliZ(0))

# 实现参数移位规则
# def parameter_shift(circuit, params, idx, shift=np.pi/2):
#     """
#     使用参数移位规则计算梯度
#     
#     Args:
#         circuit: 量子电路函数
#         params: 参数数组
#         idx: 要计算梯度的参数索引
#         shift: 移位量
#         
#     Returns:
#         参数的梯度
#     """
#     shifted_params_plus = ...
#     shifted_params_minus = ...
#     
#     forward = circuit(shifted_params_plus)
#     backward = circuit(shifted_params_minus)
#     
#     gradient = ...
#     
#     return gradient

# 比较手动梯度与自动梯度
# test_params = ...
# 
# manual_grad_0 = ...
# manual_grad_1 = ...
# 
# auto_grad = qml.grad(circuit)(test_params)
# 
# print(f"参数: {test_params}")
# print(f"手动计算的梯度: [{manual_grad_0:.6f}, {manual_grad_1:.6f}]")
# print(f"PennyLane计算的梯度: {auto_grad}")

# 绘制不同参数值的梯度
# param_range = np.linspace(0, 2*np.pi, 50)
# gradients_0 = []
# gradients_1 = []

# for param in param_range:
#     params = np.array([param, np.pi/4])  # 固定第二个参数
#     grad = qml.grad(circuit)(params)
#     gradients_0.append(grad[0])
#     
#     params = np.array([np.pi/4, param])  # 固定第一个参数
#     grad = qml.grad(circuit)(params)
#     gradients_1.append(grad[1])

# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.plot(...)
# plt.xlabel(...)
# plt.ylabel(...)
# plt.title(...)
# plt.grid(True)

# plt.subplot(1, 2, 2)
# plt.plot(...)
# plt.xlabel(...)
# plt.ylabel(...)
# plt.title(...)
# plt.grid(True)

# plt.tight_layout()
# plt.savefig('parameter_shift_gradients.png')
# plt.close()

"""
练习5: 构建变分量子门
---------------------
任务:
1. 创建一个实现量子傅里叶变换(QFT)的变分电路
2. 定义参数化的QFT电路，用单量子门和CNOT门近似QFT
3. 定义成本函数，衡量变分电路与真实QFT的近似程度
4. 优化参数以使变分电路尽可能接近真实QFT
"""

print("\n练习5: 构建变分量子门")

# 您的代码:
# n_qubits = 3
# dev_vqg = ...

# 定义目标QFT电路
# @qml.qnode(dev_vqg)
# def target_qft():
#     """标准QFT电路"""
#     # 准备非平凡的初始态
#     qml.PauliX(wires=0)
#     
#     # 应用QFT
#     qml.QFT(wires=range(n_qubits))
#     
#     # 返回状态向量
#     return qml.state()

# 定义变分QFT电路
# @qml.qnode(dev_vqg)
# def variational_qft(params):
#     """变分QFT电路"""
#     # 准备与目标电路相同的初始态
#     qml.PauliX(wires=0)
#     
#     # 变分层结构
#     # 提示: 考虑使用旋转门和CNOT门的组合
#     # 参数可以用于旋转角度
#     ...
#     
#     # 返回状态向量
#     return qml.state()

# 计算成本函数 - 量子态保真度
# def fidelity_cost(params):
#     """计算变分电路与目标电路的保真度"""
#     target_state = ...
#     variational_state = ...
#     
#     # 计算保真度
#     fidelity = ...
#     
#     # 我们希望最大化保真度，所以返回负保真度作为成本
#     return ...

# 优化变分QFT电路
# n_layers = 5  # 变分电路的层数
# n_params = ...  # 计算参数总数
# init_params = ...
# opt_vqft = ...
# params_vqft = init_params
# fidelity_history = [1 + fidelity_cost(params_vqft)]  # 转换为保真度

# print(f"初始保真度: {fidelity_history[0]:.6f}")

# for i in range(...):
#     # 优化步骤
#     ...
#     
#     # 存储保真度
#     ...
#     
#     # 打印进度
#     ...

# print(f"最终保真度: {fidelity_history[-1]:.6f}")

# 绘制保真度收敛过程
# plt.figure()
# plt.plot(...)
# plt.xlabel(...)
# plt.ylabel(...)
# plt.title(...)
# plt.grid(True)
# plt.ylim(0, 1.05)
# plt.savefig('variational_qft_fidelity.png')
# plt.close()

"""
练习6: 集成不同优化器的比较
---------------------------
任务:
1. 使用相同的变分电路和初始参数
2. 比较不同优化器的性能：GradientDescent、Adam、Adagrad和QNSPSA
3. 绘制不同优化器的收敛曲线
4. 分析哪种优化器在特定问题上表现最佳
"""

print("\n练习6: 集成不同优化器的比较")

# 您的代码:
# dev_opt = ...

# 创建一个简单的变分电路
# @qml.qnode(dev_opt)
# def opt_circuit(params):
#     """用于优化器比较的电路"""
#     qml.RX(params[0], wires=0)
#     qml.RY(params[1], wires=1)
#     qml.CNOT(wires=[0, 1])
#     qml.RZ(params[2], wires=0)
#     qml.RX(params[3], wires=1)
#     return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

# 定义成本函数
# def opt_cost(params):
#     """优化的成本函数"""
#     return 1 - opt_circuit(params)

# 比较不同优化器
# init_params = ...
# n_steps = 100

# 创建优化器字典
# optimizers = {
#     "GradientDescent": ...,
#     "Adam": ...,
#     "Adagrad": ...,
#     "Momentum": ...
# }

# 存储每个优化器的结果
# results = {}

# for name, opt in optimizers.items():
#     params = init_params.copy()
#     cost_history = [opt_cost(params)]
#     
#     for i in range(n_steps):
#         # 优化步骤
#         ...
#         
#         # 存储成本
#         ...
#     
#     results[name] = {
#         "final_params": params,
#         "cost_history": cost_history,
#         "final_cost": cost_history[-1]
#     }
#     
#     print(f"{name}: 最终成本 = {cost_history[-1]:.6f}")

# 绘制比较结果
# plt.figure(figsize=(10, 6))

# for name, result in results.items():
#     plt.plot(result["cost_history"], label=f"{name}")

# plt.xlabel(...)
# plt.ylabel(...)
# plt.title(...)
# plt.legend()
# plt.grid(True)
# plt.savefig('optimizer_comparison.png')
# plt.close()

# 分析结果
# print("\n优化器性能比较:")
# for name, result in sorted(results.items(), key=lambda x: x[1]["final_cost"]):
#     print(f"{name}: 最终成本 = {result['final_cost']:.6f}")

print("\n完成所有练习后，请查看解决方案文件以比较您的实现。")
print("下一步: 学习量子机器学习技术和应用。") 