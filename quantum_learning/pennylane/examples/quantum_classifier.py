#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量子分类器 - 二分类问题示例
========================

这个示例展示如何使用量子电路构建一个简单的二分类器模型。
我们将使用PennyLane构建一个变分量子分类器(VQC)，
并在模拟的数据集上进行训练和测试。

作者: (c) 量子计算学习
日期: 2023
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# 设置中文字体（如果有需要）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
except:
    pass

print("量子分类器 - 二分类问题示例")
print("=========================")
print()

# 设置随机种子以确保结果可重现
np.random.seed(42)

# 生成月牙形数据集
def generate_data(n_samples=200):
    """
    生成二分类的月牙形数据集
    
    Args:
        n_samples (int): 样本数量
        
    Returns:
        tuple: (特征数据, 标签)
    """
    X, y = make_moons(n_samples=n_samples, noise=0.15)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

# 定义量子设备
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# 定义数据编码电路
def data_embedding(x):
    """
    将经典数据编码到量子态
    使用振幅编码 (Amplitude Encoding) 方式
    
    Args:
        x (ndarray): 2维特征向量
    """
    # 扩展特征以使用所有量子比特
    features = np.zeros(2**n_qubits)
    features[0] = x[0]
    features[1] = x[1]
    # 归一化
    features = features / np.linalg.norm(features)
    
    # 振幅编码
    qml.AmplitudeEmbedding(features=features, wires=range(n_qubits), normalize=True)

# 定义变分量子电路
def variational_circuit(params):
    """
    创建变分量子分类器的电路结构
    
    Args:
        params (ndarray): 变分参数
    """
    # 参数形状: (layers, n_qubits, 3)
    n_layers = params.shape[0]
    
    # 实现一个强表达能力的可训练电路
    for layer in range(n_layers):
        # 单比特旋转层
        for qubit in range(n_qubits):
            qml.RX(params[layer, qubit, 0], wires=qubit)
            qml.RY(params[layer, qubit, 1], wires=qubit)
            qml.RZ(params[layer, qubit, 2], wires=qubit)
        
        # 纠缠层 - 环形结构
        for qubit in range(n_qubits):
            qml.CNOT(wires=[qubit, (qubit + 1) % n_qubits])

# 定义量子节点（量子电路）
@qml.qnode(dev)
def quantum_circuit(params, x):
    """
    完整的量子分类器电路
    
    Args:
        params (ndarray): 变分参数
        x (ndarray): 输入特征
        
    Returns:
        float: |1>状态的概率，用作分类预测
    """
    # 数据编码
    data_embedding(x)
    
    # 可训练的变分电路
    variational_circuit(params)
    
    # 测量第一个量子比特的期望值作为预测结果
    return qml.expval(qml.PauliZ(0))

# 定义分类器和损失函数
def classifier_predict(params, x):
    """
    基于量子电路的输出进行二分类预测
    
    Args:
        params (ndarray): 模型参数
        x (ndarray): 特征数据
        
    Returns:
        int: 预测的类别 (0 或 1)
    """
    # 使用量子电路的输出
    prediction = quantum_circuit(params, x)
    # 将连续输出转换为二分类结果
    return int(prediction <= 0.0)

def square_loss(labels, predictions):
    """
    计算平方损失
    
    Args:
        labels (ndarray): 真实标签
        predictions (ndarray): 模型预测值
        
    Returns:
        float: 平均平方损失
    """
    return np.mean((labels - predictions) ** 2)

def cost(params, X, y):
    """
    计算模型在数据集上的总损失
    
    Args:
        params (ndarray): 模型参数
        X (ndarray): 特征数据
        y (ndarray): 标签
        
    Returns:
        float: 平均损失值
    """
    # 获取量子电路原始输出
    predictions = [quantum_circuit(params, x) for x in X]
    
    # 将输出转换到 0-1 范围
    predictions = [(p + 1) / 2 for p in predictions]
    
    # 计算损失
    return square_loss(y, predictions)

# 训练分类器
def train_classifier(X_train, y_train, n_layers=2, steps=200):
    """
    训练量子分类器
    
    Args:
        X_train (ndarray): 训练特征
        y_train (ndarray): 训练标签
        n_layers (int): 变分电路的层数
        steps (int): 优化步数
        
    Returns:
        tuple: (优化后的参数, 损失历史)
    """
    # 初始化随机参数
    params = np.random.uniform(
        low=0, high=2*np.pi, 
        size=(n_layers, n_qubits, 3)
    )
    
    # 定义优化器
    opt = qml.AdamOptimizer(stepsize=0.05)
    
    # 存储损失历史
    loss_history = []
    
    # 迭代优化
    for i in range(steps):
        params, loss = opt.step_and_cost(
            lambda p: cost(p, X_train, y_train), params
        )
        
        loss_history.append(loss)
        
        # 每20步打印进度
        if (i+1) % 20 == 0:
            accuracy = accuracy_score(
                y_train, 
                [classifier_predict(params, x) for x in X_train]
            )
            print(f"步骤 {i+1}: 损失 = {loss:.4f}, 准确率 = {accuracy:.4f}")
    
    return params, loss_history

# 可视化决策边界
def plot_decision_boundary(params, X, y, title="量子分类器决策边界"):
    """
    绘制分类器的决策边界
    
    Args:
        params (ndarray): 模型参数
        X (ndarray): 数据特征
        y (ndarray): 数据标签
        title (str): 图表标题
    """
    h = 0.05  # 网格步长
    
    # 创建网格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )
    
    # 为网格中的每个点预测类别
    Z = np.array([classifier_predict(params, np.array([x, y])) 
                  for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界和散点图
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor='k', cmap=plt.cm.RdBu)
    plt.title(title)
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.tight_layout()
    plt.savefig('../images/quantum_classifier_boundary.png', dpi=300)

# 主函数
def main():
    # 生成数据集
    X, y = generate_data(n_samples=200)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    print(f"数据集大小: {len(X)} 样本")
    print(f"训练集: {len(X_train)} 样本")
    print(f"测试集: {len(X_test)} 样本")
    
    # 训练模型
    print("\n开始训练量子分类器...")
    params, loss_history = train_classifier(X_train, y_train, n_layers=3, steps=100)
    
    # 在测试集上评估模型
    y_pred = [classifier_predict(params, x) for x in X_test]
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print("\n模型评估:")
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    # 计算并显示混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("\n混淆矩阵:")
    print(cm)
    
    # 绘制训练损失
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('量子分类器训练损失')
    plt.xlabel('优化步骤')
    plt.ylabel('损失')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../images/quantum_classifier_loss.png', dpi=300)
    
    # 绘制决策边界
    plot_decision_boundary(params, X, y)
    
    print("\n量子分类器训练完成！图表已保存到images目录。")

if __name__ == "__main__":
    main() 