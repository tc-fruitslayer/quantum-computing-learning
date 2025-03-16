#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Xanadu PennyLane框架 - 量子机器学习练习

本文件包含关于量子机器学习(QML)的练习。
完成这些练习将帮助您理解如何使用量子计算进行机器学习任务。
"""

import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# 尝试导入scikit-learn，用于数据生成和评估
try:
    import sklearn
    from sklearn.datasets import make_moons, make_circles
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("警告: scikit-learn未安装，某些练习可能无法完成")

# 尝试导入torch，用于混合量子-经典模型
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch未安装，某些练习可能无法完成")

print("===== PennyLane量子机器学习练习 =====")

"""
练习1: 数据编码方法
------------------
任务:
1. 实现三种不同的量子数据编码方法:
   a. 角度编码 - 将经典数据编码为旋转角度
   b. 振幅编码 - 将经典数据编码为量子态的振幅
   c. 基于IQP的编码 - 使用交替的旋转层和纠缠层
2. 比较不同编码方法的量子电路结构和特性
"""

print("\n练习1: 数据编码方法")

# 创建量子设备
n_qubits = 4
# dev_encoding = ...

# 角度编码
# @qml.qnode(dev_encoding)
# def angle_encoding(x):
#     """
#     角度编码 - 将数据编码为旋转角度
#     
#     Args:
#         x: 输入数据，长度应等于量子比特数
#         
#     Returns:
#         量子态的测量结果
#     """
#     # 使用RX, RY或RZ门将数据编码为旋转角度
#     ...
#     
#     # 返回所有量子比特的期望值
#     return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# 振幅编码
# @qml.qnode(dev_encoding)
# def amplitude_encoding(x):
#     """
#     振幅编码 - 将数据编码为量子态的振幅
#     
#     Args:
#         x: 输入数据，长度应为2^n_qubits，并且已归一化
#         
#     Returns:
#         量子态的测量结果
#     """
#     # 使用振幅编码将数据编码为量子态
#     # 提示: 使用 qml.AmplitudeEmbedding
#     ...
#     
#     # 返回所有量子比特的期望值
#     return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# IQP编码
# @qml.qnode(dev_encoding)
# def iqp_encoding(x):
#     """
#     IQP编码 - 使用交替的旋转和纠缠层
#     
#     Args:
#         x: 输入数据
#         
#     Returns:
#         量子态的测量结果
#     """
#     # 首先应用Hadamard门创建叠加态
#     ...
#     
#     # 使用RZ门应用旋转层
#     ...
#     
#     # 应用纠缠层（例如ZZ相互作用）
#     ...
#     
#     # 再次应用旋转层
#     ...
#     
#     # 返回所有量子比特的期望值
#     return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# 测试编码方法
# x_test = np.random.random(n_qubits)
# x_test_normalized = x_test / np.linalg.norm(x_test)
# x_amplitude = np.random.random(2**n_qubits)
# x_amplitude = x_amplitude / np.linalg.norm(x_amplitude)

# print("测试数据:", x_test)
# print("角度编码结果:", angle_encoding(x_test))
# print("振幅编码结果:", amplitude_encoding(x_amplitude))
# print("IQP编码结果:", iqp_encoding(x_test))

# 绘制编码电路
# fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# qml.draw_mpl(angle_encoding, expansion_strategy="device")(x_test)
# ax[0].set_title("角度编码电路")

# qml.draw_mpl(amplitude_encoding, expansion_strategy="device")(x_amplitude)
# ax[1].set_title("振幅编码电路")

# qml.draw_mpl(iqp_encoding, expansion_strategy="device")(x_test)
# ax[2].set_title("IQP编码电路")

# plt.tight_layout()
# plt.savefig('encoding_circuits.png')
# plt.close()

"""
练习2: 量子神经网络(QNN)
-----------------------
任务:
1. 设计一个包含以下组件的量子神经网络:
   a. 数据编码层
   b. 变分层(多个旋转和纠缠门)
   c. 测量层
2. 使用QNN解决简单的二分类问题
3. 实现训练循环，优化神经网络参数
"""

print("\n练习2: 量子神经网络(QNN)")

if SKLEARN_AVAILABLE:
    print("scikit-learn可用，可以完成量子神经网络分类练习")
    # 以下代码被注释，仅供参考
    # 生成简单的二分类数据集
    # X, y = ...
    # X_train, X_test, y_train, y_test = ...
    
    # 创建量子设备
    # n_qubits = 2
    # dev_qnn = ...
    
    # 定义量子神经网络
    # @qml.qnode(dev_qnn)
    # def quantum_neural_network(inputs, weights):
    #     """
    #     量子神经网络
    #     
    #     Args:
    #         inputs: 输入数据特征
    #         weights: 变分参数
    #         
    #     Returns:
    #         输出
    #     """
    #     # 数据编码层
    #     ...
    #     
    #     # 变分层
    #     ...
    #     
    #     # 测量输出
    #     return qml.expval(qml.PauliZ(0))
else:
    print("未安装scikit-learn，跳过量子神经网络分类练习")

"""
练习3: 量子核方法
---------------
任务:
1. 实现量子核函数，使用量子电路计算两个数据点之间的相似度
2. 使用量子核函数构建核矩阵
3. 将量子核与经典核方法(如SVM)结合
"""

print("\n练习3: 量子核方法")

if SKLEARN_AVAILABLE:
    print("scikit-learn可用，可以完成量子核方法练习")
    # 以下代码被注释，仅供参考
    # 创建量子设备
    # n_qubits = 2
    # dev_kernel = ...
    
    # 定义特征映射电路
    # @qml.qnode(dev_kernel)
    # def feature_map(x):
    #     """
    #     量子特征映射电路
    #     
    #     Args:
    #         x: 输入数据
    #         
    #     Returns:
    #         量子态
    #     """
    #     # 数据编码
    #     ...
    #     
    #     # 非线性变换
    #     ...
    #     
    #     # 返回量子态
    #     return qml.state()
    
    # 定义量子核函数
    # def quantum_kernel(x1, x2):
    #     """
    #     计算两个数据点的量子核
    #     
    #     Args:
    #         x1, x2: 两个数据点
    #         
    #     Returns:
    #         核值 (内积)
    #     """
    #     state1 = feature_map(x1)
    #     state2 = feature_map(x2)
    #     
    #     # 计算量子态的内积
    #     return np.abs(np.vdot(state1, state2))**2
    
    # 生成数据集
    # X, y = ...  # 例如make_circles或make_moons
    # X_train, X_test, y_train, y_test = ...
    
    # 计算核矩阵
    # def compute_kernel_matrix(X1, X2):
    #     """计算两组数据点之间的核矩阵"""
    #     n1 = len(X1)
    #     n2 = len(X2)
    #     kernel_matrix = np.zeros((n1, n2))
    #     
    #     for i in range(n1):
    #         for j in range(n2):
    #             kernel_matrix[i, j] = quantum_kernel(X1[i], X2[j])
    #     
    #     return kernel_matrix
    
    # # 计算训练集和测试集的核矩阵
    # K_train = ...
    # K_test = ...
    
    # # 使用核矩阵进行分类
    # from sklearn.svm import SVC
    # 
    # # 创建预计算核的SVM
    # qsvm = SVC(kernel='precomputed')
    # 
    # # 训练模型
    # qsvm.fit(K_train, y_train)
    # 
    # # 预测
    # y_pred = qsvm.predict(K_test)
    # 
    # # 计算准确率
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"量子核SVM的测试准确率: {accuracy:.4f}")
    
    # # 可视化结果
    # plt.figure(figsize=(10, 8))
    # 
    # # 绘制核矩阵
    # plt.subplot(2, 2, 1)
    # plt.imshow(K_train)
    # plt.title('量子核矩阵 (训练集)')
    # plt.colorbar()
    # 
    # # 绘制决策边界
    # plt.subplot(2, 2, 2)
    # 
    # # 创建网格
    # h = 0.05
    # x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    # y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    #                      np.arange(y_min, y_max, h))
    # 
    # # 计算所有网格点与训练点的核
    # grid_points = np.c_[xx.ravel(), yy.ravel()]
    # K_grid = compute_kernel_matrix(grid_points, X_train)
    # 
    # # 预测网格点
    # Z = qsvm.predict(K_grid)
    # Z = Z.reshape(xx.shape)
    # 
    # plt.contourf(xx, yy, Z, alpha=0.3)
    # plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k')
    # plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', s=60)
    # plt.title(f'量子核SVM决策边界 (准确率: {accuracy:.4f})')
    # 
    # # 与经典RBF核比较
    # from sklearn.svm import SVC as ClassicalSVC
    # cls_svm = ClassicalSVC(kernel='rbf')
    # cls_svm.fit(X_train, y_train)
    # cls_pred = cls_svm.predict(X_test)
    # cls_accuracy = accuracy_score(y_test, cls_pred)
    # 
    # plt.subplot(2, 2, 3)
    # Z_cls = cls_svm.predict(np.c_[xx.ravel(), yy.ravel()])
    # Z_cls = Z_cls.reshape(xx.shape)
    # 
    # plt.contourf(xx, yy, Z_cls, alpha=0.3)
    # plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k')
    # plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', s=60)
    # plt.title(f'经典RBF核SVM (准确率: {cls_accuracy:.4f})')
    # 
    # plt.tight_layout()
    # plt.savefig('quantum_kernel.png')
    # plt.close()
    # 
    # print(f"经典RBF核SVM测试准确率: {cls_accuracy:.4f}")
    # print(f"量子核SVM测试准确率: {accuracy:.4f}")
else:
    print("未安装scikit-learn，跳过量子核方法练习")

print("""
练习4: 量子转移学习
-----------------
任务:
1. 准备一个预训练的经典模型（例如小型MLP）
2. 设计一个混合量子-经典转移学习模型，其中:
   - 经典部分用于特征提取
   - 量子部分用于分类
3. 比较纯经典模型和混合模型的性能
""")

print("\n练习4: 量子转移学习")

if TORCH_AVAILABLE and SKLEARN_AVAILABLE:
    print("PyTorch和scikit-learn可用，可以完成量子转移学习练习")
    # 以下代码被注释，仅供参考
    # 创建量子设备
    # n_qubits = 2
    # dev_transfer = ...
    
    # 创建数据集
    # X, y = ...
    # X_train, X_test, y_train, y_test = ...
else:
    print("未安装PyTorch或scikit-learn，跳过量子转移学习练习")

print("\n完成所有练习后，请查看解决方案文件以比较您的实现。")
print("祝贺您完成了PennyLane量子机器学习练习！这些技术是量子计算应用的前沿。") 