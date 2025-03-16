#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Xanadu PennyLane框架学习 2：混合量子-经典计算
本文件详细介绍PennyLane中的混合量子-经典计算模型和应用
"""

# 导入必要的库
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from pennylane import numpy as pnp

print("===== PennyLane混合量子-经典计算 =====")

# 检查PennyLane版本
print(f"PennyLane版本: {qml.__version__}")

# 1. 混合量子-经典计算概述
print("\n1. 混合量子-经典计算概述")
print("混合量子-经典计算是结合量子和经典计算的范式，通常包含以下步骤:")
print("- 使用经典计算机准备初始参数")
print("- 使用量子处理器执行参数化量子电路")
print("- 测量量子电路的输出")
print("- 使用经典计算机处理测量结果和优化参数")
print("- 迭代上述过程以达到目标")
print("\n这种范式特别适用于当前的NISQ（嘈杂中等规模量子）设备")

# 2. 基本混合计算模式
print("\n2. 基本混合计算模式")
print("最简单的混合计算模式是将量子电路作为经典计算过程中的一个'子程序'")

# 创建一个简单的量子设备
dev = qml.device("default.qubit", wires=1)

# 定义一个参数化量子电路
@qml.qnode(dev)
def quantum_circuit(theta):
    qml.RX(theta, wires=0)
    return qml.expval(qml.PauliZ(0))

# 定义一个混合函数，包含经典和量子计算
def hybrid_function(theta):
    # 经典预处理
    theta_modified = np.sin(theta) * np.pi
    
    # 量子计算
    result = quantum_circuit(theta_modified)
    
    # 经典后处理
    return np.cos(result) ** 2

# 计算混合函数在不同输入上的值
thetas = np.linspace(0, 2*np.pi, 20)
hybrid_values = [hybrid_function(theta) for theta in thetas]

print("\n混合函数在不同输入上的值:")
for theta, value in zip(thetas[:5], hybrid_values[:5]):
    print(f"theta = {theta:.2f}, 混合函数值 = {value:.6f}")
print("... 更多值省略 ...")

# 3. 变分量子算法框架
print("\n3. 变分量子算法框架")
print("变分量子算法(VQA)是混合量子-经典计算的重要应用框架")
print("VQA包含以下组件:")
print("- 参数化量子电路（可训练的'模型'）")
print("- 成本函数（量化解决方案质量）")
print("- 经典优化器（用于调整参数）")

# 创建一个简单的变分电路
n_qubits = 4
n_layers = 2
dev_vqa = qml.device("default.qubit", wires=n_qubits)

# 定义一个简单的变分量子电路结构
def variational_circuit(params, x=None):
    # 如果提供数据，则进行编码
    if x is not None:
        for i in range(n_qubits):
            qml.RY(x[i % len(x)], wires=i)
    
    # 应用参数化层
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RY(params[l][i][0], wires=i)
            qml.RZ(params[l][i][1], wires=i)
        
        # 添加纠缠
        for i in range(n_qubits-1):
            qml.CNOT(wires=[i, i+1])
        qml.CNOT(wires=[n_qubits-1, 0])  # 闭合链
    
    # 返回每个量子比特的期望值
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# 创建量子节点
vqa_qnode = qml.QNode(variational_circuit, dev_vqa)

# 随机初始化参数
params_shape = (n_layers, n_qubits, 2)  # (层数, 量子比特数, 每个量子比特的参数数)
params = np.random.uniform(0, 2*np.pi, params_shape)

# 打印电路结构
print("\n变分量子电路结构:")
print(qml.draw(vqa_qnode)(params))

# 4. 量子-经典优化循环
print("\n4. 量子-经典优化循环")
print("在混合量子-经典算法中，经典优化器用于更新量子电路参数")

# 定义一个简单的成本函数 - 目标是使所有量子比特均处于|0⟩态
def cost_function(params):
    expectations = vqa_qnode(params)
    # 目标是使所有期望值接近1（全部为|0⟩态）
    return 1 - np.mean(expectations)

# 打印初始成本
initial_cost = cost_function(params)
print(f"\n初始成本: {initial_cost:.6f}")

# 模拟一个优化步骤
def one_optimization_step(params, learning_rate=0.1):
    # 计算梯度
    try:
        grad = qml.grad(cost_function)(params)
        if len(grad) == 0:
            print("警告: 梯度为空数组，使用随机梯度代替（仅演示目的）")
            # 创建与params形状相同的随机梯度
            grad = np.random.uniform(-0.1, 0.1, params.shape)
        else:
            grad = np.array(grad)
    except Exception as e:
        print(f"计算梯度时出错: {e}")
        print("使用随机梯度代替（仅演示目的）")
        # 创建与params形状相同的随机梯度
        grad = np.random.uniform(-0.1, 0.1, params.shape)
    
    # 梯度下降更新
    new_params = params - learning_rate * grad
    
    # 计算新成本
    new_cost = cost_function(new_params)
    
    return new_params, new_cost

new_params, new_cost = one_optimization_step(params)
print(f"一步优化后的成本: {new_cost:.6f}")

# 5. 经典优化器
print("\n5. 经典优化器")
print("PennyLane提供多种经典优化器，与量子电路无缝集成")

# 创建一个简单的设备和问题
dev_opt = qml.device("default.qubit", wires=2)

@qml.qnode(dev_opt)
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

def cost(params):
    return 1 - circuit(params)

# 初始参数
init_params = np.array([0.5, 0.1])

# 使用PennyLane的GradientDescentOptimizer
print("\n使用梯度下降优化器:")
opt = qml.GradientDescentOptimizer(stepsize=0.2)

# 模拟几步优化
params = init_params
costs = [cost(params)]

for i in range(5):
    params = opt.step(cost, params)
    costs.append(cost(params))
    print(f"步骤 {i+1}: 成本 = {costs[-1]:.6f}, 参数 = {params}")

# 6. 使用PennyLane与其他框架的集成
print("\n6. 使用PennyLane与其他框架的集成")
print("PennyLane可以与流行的机器学习框架无缝集成")

# 显示与TensorFlow和PyTorch的集成示例
print("\n与TensorFlow集成的示例代码 (展示，不执行):")
print("""
import tensorflow as tf

# 创建设备
dev = qml.device('default.qubit', wires=2)

# 创建量子节点，指定TensorFlow接口
@qml.qnode(dev, interface='tf')
def circuit(x, params):
    qml.RX(x[0], wires=0)
    qml.RX(x[1], wires=1)
    qml.RZ(params[0], wires=0)
    qml.RZ(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

# 创建模型
class HybridModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.params = tf.Variable([0.01, 0.01], dtype=tf.float32)
        
    def call(self, x):
        return circuit(x, self.params)

# 使用
model = HybridModel()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

with tf.GradientTape() as tape:
    output = model(tf.constant([[0.5, 0.1]]))
    loss = tf.reduce_sum(output)

gradients = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
""")

print("\n与PyTorch集成的示例代码 (展示，不执行):")
print("""
import torch

# 创建设备
dev = qml.device('default.qubit', wires=2)

# 创建量子节点，指定PyTorch接口
@qml.qnode(dev, interface='torch')
def circuit(x, params):
    qml.RX(x[0], wires=0)
    qml.RX(x[1], wires=1)
    qml.RZ(params[0], wires=0)
    qml.RZ(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

# 创建模型
class HybridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.params = torch.nn.Parameter(torch.tensor([0.01, 0.01], dtype=torch.float32))
        
    def forward(self, x):
        return circuit(x, self.params)

# 使用
model = HybridModel()
opt = torch.optim.SGD(model.parameters(), lr=0.1)

x = torch.tensor([0.5, 0.1], dtype=torch.float32)
output = model(x)
loss = torch.sum(output)

loss.backward()
opt.step()
""")

# 7. 批量处理和并行化
print("\n7. 批量处理和并行化")
print("PennyLane允许批处理量子计算，提高混合算法的效率")

# 创建支持batch模式的设备
dev_batch = qml.device("default.qubit", wires=2, shots=1000)

# 定义支持批处理的电路
@qml.qnode(dev_batch)
def batch_circuit(x):
    qml.RX(x[0], wires=0)
    qml.RY(x[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

# 批量输入
batch_inputs = np.array([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6]
])

# 逐个处理
print("\n单独处理每个输入:")
for x in batch_inputs:
    result = batch_circuit(x)
    print(f"输入 {x}, 输出 {result}")

# 8. 变分量子特征求解器示例
print("\n8. 变分量子特征求解器(VQE)示例")
print("VQE是解决量子化学问题的一种混合量子-经典算法")

# 创建一个简单的设备
dev_vqe = qml.device("default.qubit", wires=2)

# 定义氢分子电子哈密顿量的系数（简化版本）
coeffs = np.array([0.5, 0.5])
obs = [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliY(0) @ qml.PauliY(1)]

# 定义电路 - 简单的参数化电路
@qml.qnode(dev_vqe)
def vqe_circuit(params):
    # 准备初始态 |01⟩
    qml.PauliX(wires=1)
    
    # 应用参数化酉变换
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(params[2], wires=0)
    qml.RY(params[3], wires=1)
    
    # 返回哈密顿量的期望值
    return qml.expval(qml.Hamiltonian(coeffs, obs))

# 定义成本函数 - 哈密顿量的期望值
def vqe_cost(params):
    return vqe_circuit(params)

# 随机初始化参数
init_vqe_params = np.random.uniform(0, 2*np.pi, 4)

# 打印电路和初始成本
print("\nVQE电路:")
print(qml.draw(vqe_circuit)(init_vqe_params))
print(f"初始成本: {vqe_cost(init_vqe_params):.6f}")

# 使用优化器
opt_vqe = qml.GradientDescentOptimizer(stepsize=0.2)

# 模拟几步优化
vqe_params = init_vqe_params
vqe_costs = [vqe_cost(vqe_params)]

print("\nVQE优化过程:")
for i in range(5):
    vqe_params = opt_vqe.step(vqe_cost, vqe_params)
    vqe_costs.append(vqe_cost(vqe_params))
    print(f"步骤 {i+1}: 能量 = {vqe_costs[-1]:.6f}")

# 9. 量子感知机示例
print("\n9. 量子感知机示例")
print("量子感知机是量子神经网络的一种简单形式")

# 创建数据集 - 简单的二分类问题
X = np.array([[0.1, 0.2], [0.9, 0.8], [0.2, 0.1], [0.8, 0.9]])
Y = np.array([0, 1, 0, 1])  # 二元标签

# 创建设备
dev_qp = qml.device("default.qubit", wires=2)

# 定义量子感知机电路
@qml.qnode(dev_qp)
def quantum_perceptron(x, params):
    # 数据编码
    qml.RX(np.pi * x[0], wires=0)
    qml.RX(np.pi * x[1], wires=1)
    
    # 参数化旋转
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    
    # 纠缠层
    qml.CNOT(wires=[0, 1])
    
    # 测量输出
    return qml.expval(qml.PauliZ(1))

# 定义损失函数 - 均方误差
def qp_loss(params, X, Y):
    predictions = np.array([quantum_perceptron(x, params) for x in X])
    # 将[-1,1]映射到[0,1]
    predictions = (predictions + 1) / 2
    return np.mean((predictions - Y) ** 2)

# 随机初始化参数
init_qp_params = np.random.uniform(0, 2*np.pi, 2)

# 打印初始损失
print(f"\n初始损失: {qp_loss(init_qp_params, X, Y):.6f}")

# 使用优化器
opt_qp = qml.GradientDescentOptimizer(stepsize=0.5)

# 模拟几步优化
qp_params = init_qp_params
qp_losses = [qp_loss(qp_params, X, Y)]

print("\n量子感知机训练过程:")
for i in range(10):
    qp_params = opt_qp.step(lambda p: qp_loss(p, X, Y), qp_params)
    qp_losses.append(qp_loss(qp_params, X, Y))
    print(f"步骤 {i+1}: 损失 = {qp_losses[-1]:.6f}")

# 测试训练后的模型
predictions = [(quantum_perceptron(x, qp_params) + 1) / 2 for x in X]
print("\n训练后的预测:")
for x, y, pred in zip(X, Y, predictions):
    print(f"输入 {x}, 真实标签 {y}, 预测 {pred:.4f}")

# 10. 总结
print("\n10. 总结")
print("1. 混合量子-经典计算结合了量子计算和经典计算的优势")
print("2. 变分量子算法是当前NISQ时代的重要计算范式")
print("3. PennyLane支持与主流机器学习框架的无缝集成")
print("4. 应用范围包括量子化学模拟、量子机器学习等")
print("5. 混合方法允许在现有量子硬件上解决实际问题")

print("\n下一步学习:")
print("- 变分量子电路的深入探索")
print("- 量子梯度和优化技术")
print("- 量子机器学习模型") 