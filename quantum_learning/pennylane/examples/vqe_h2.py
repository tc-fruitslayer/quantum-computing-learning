#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VQE模拟氢分子(H₂)能量
=====================

这个示例展示如何使用PennyLane的变分量子本征求解器(VQE)来模拟氢分子的基态能量。
VQE是一种混合量子-经典算法，使用经典优化器来最小化量子波函数的能量期望值。

作者: (c) 量子计算学习
日期: 2023
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（如果有需要）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
except:
    pass

print("VQE模拟氢分子(H₂)基态能量")
print("========================")
print()

# 设置随机数种子以保证结果可重现
np.random.seed(42)

# 定义模拟设备
dev = qml.device("default.qubit", wires=2)

# 定义氢分子的汉密尔顿量
def hydrogen_hamiltonian(bond_length):
    """
    为给定的键长创建氢分子的分子汉密尔顿量
    
    Args:
        bond_length (float): 氢分子的键长，单位为埃(Å)
        
    Returns:
        qml.Hamiltonian: 分子汉密尔顿量
    """
    # 电子积分常数
    a = 0.5 / bond_length
    
    # 旋转角度（用于从原子轨道到分子轨道的转换）
    theta = np.pi / 4
    
    # 能量常数
    e_core = 1.0 / bond_length
    
    # 轨道能量
    e_1 = a + 1
    e_2 = a - 1
    
    # 轨道-轨道相互作用
    g = 0.25 / bond_length
    
    # 定义泡利算符
    I  = qml.Identity(0)
    Z0 = qml.PauliZ(0)
    Z1 = qml.PauliZ(1)
    X0 = qml.PauliX(0)
    X1 = qml.PauliX(1)
    Y0 = qml.PauliY(0)
    Y1 = qml.PauliY(1)
    
    # 构建汉密尔顿量
    H = (e_core * I @ I + 
         0.5 * (e_1 + e_2) * (I @ I - Z0 @ Z1) +
         0.5 * (e_1 - e_2) * (Z0 @ I - I @ Z1) +
         g * (X0 @ X1 + Y0 @ Y1))
    
    # 返回汉密尔顿量对象
    return H

# 定义变分量子电路(Ansatz)
@qml.qnode(dev)
def ansatz_circuit(params, bond_length=0.7414):
    """
    用于VQE的参数化量子电路，针对氢分子的基态近似
    
    Args:
        params (ndarray): 变分参数
        bond_length (float): 氢分子键长，单位为埃
        
    Returns:
        float: 汉密尔顿量的期望值（能量）
    """
    # 初始化氢分子的哈特里-福克态：|01>
    qml.PauliX(wires=1)
    
    # 应用单比特旋转层
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    
    # 应用CNOT门产生纠缠
    qml.CNOT(wires=[0, 1])
    
    # 应用第二个单比特旋转层
    qml.RY(params[2], wires=0)
    qml.RY(params[3], wires=1)
    
    # 计算并返回能量期望值
    H = hydrogen_hamiltonian(bond_length)
    return qml.expval(H)

# 定义VQE优化函数
def optimize_vqe(bond_length=0.7414, steps=100, init_params=None):
    """
    运行VQE优化以找到基态能量
    
    Args:
        bond_length (float): 氢分子键长，单位为埃
        steps (int): 优化步数
        init_params (ndarray): 初始参数，如果为None则随机初始化
        
    Returns:
        tuple: (优化后的参数, 能量历史, 最终能量)
    """
    # 如果没有提供初始参数，则随机初始化
    if init_params is None:
        init_params = np.random.uniform(0, 2*np.pi, size=4)
    
    # 定义本次优化的目标函数
    def cost(params):
        return ansatz_circuit(params, bond_length)
    
    # 选择优化器
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    
    # 存储优化历史
    params = init_params
    energy_history = []
    
    # 运行优化
    for i in range(steps):
        params = opt.step(cost, params)
        energy = cost(params)
        energy_history.append(energy)
        
        # 每10步打印一次进度
        if (i+1) % 10 == 0:
            print(f"步骤 {i+1}: 能量 = {energy:.6f} Ha")
    
    return params, energy_history, energy_history[-1]

# 扫描不同键长的基态能量
def bond_length_scan(bond_lengths):
    """
    对一系列键长运行VQE，创建氢分子的势能面
    
    Args:
        bond_lengths (ndarray): 要扫描的键长数组，单位为埃
        
    Returns:
        ndarray: 对应的基态能量数组
    """
    energies = []
    opt_params = None
    
    for bond_length in bond_lengths:
        print(f"\n计算键长 {bond_length:.4f} Å 的基态能量:")
        # 使用前一个优化结果作为下一个键长的初始值（热启动）
        opt_params, _, energy = optimize_vqe(bond_length, steps=50, init_params=opt_params)
        energies.append(energy)
        print(f"键长 {bond_length:.4f} Å 的基态能量: {energy:.6f} Ha")
    
    return np.array(energies)

# 主函数
def main():
    # 第1部分：优化单一键长的基态能量
    print("\n第1部分: 优化平衡键长的基态能量")
    print("------------------------------")
    
    # 氢分子的平衡键长约为0.7414埃
    equilibrium_bond_length = 0.7414
    
    # 运行VQE优化
    opt_params, energy_history, final_energy = optimize_vqe(
        bond_length=equilibrium_bond_length, 
        steps=100
    )
    
    print("\n优化结果:")
    print(f"最终基态能量: {final_energy:.6f} 哈特里")
    print(f"优化后的参数: {opt_params}")
    
    # 绘制优化过程中的能量变化
    plt.figure(figsize=(10, 6))
    plt.plot(energy_history, 'o-')
    plt.title(f'氢分子VQE优化过程 (键长 = {equilibrium_bond_length} Å)')
    plt.xlabel('优化步骤')
    plt.ylabel('能量 (哈特里)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../images/vqe_h2_optimization.png', dpi=300)
    
    # 第2部分：扫描不同键长的势能面
    print("\n第2部分: 绘制氢分子的势能面")
    print("--------------------------")
    
    # 定义要扫描的键长范围
    bond_lengths = np.linspace(0.5, 2.0, 8)
    
    # 运行键长扫描
    energies = bond_length_scan(bond_lengths)
    
    # 绘制势能面
    plt.figure(figsize=(10, 6))
    plt.plot(bond_lengths, energies, 'o-')
    plt.title('氢分子势能面 (VQE)')
    plt.xlabel('键长 (Å)')
    plt.ylabel('能量 (哈特里)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../images/vqe_h2_potential_curve.png', dpi=300)
    
    print("\nVQE模拟完成！图表已保存到images目录。")

if __name__ == "__main__":
    main() 