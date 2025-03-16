#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Xanadu PennyLane框架 - 量子机器学习练习解答

本文件包含对PennyLane量子机器学习练习的参考解答。
如果您还没有尝试完成练习，建议先自行尝试再查看解答。
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
    print("警告: scikit-learn未安装，某些练习无法完成")

# 尝试导入torch，用于混合量子-经典模型
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch未安装，某些练习无法完成")

print("===== PennyLane量子机器学习练习解答 =====")

"""
练习1: 数据编码方法
"""
print("\n练习1: 数据编码方法 - 解答")

def exercise1_solution():
    # 创建量子设备
    n_qubits = 4
    dev_encoding = qml.device('default.qubit', wires=n_qubits)
    
    # 角度编码
    @qml.qnode(dev_encoding)
    def angle_encoding(x):
        """
        角度编码 - 将数据编码为旋转角度
        
        Args:
            x: 输入数据，长度应等于量子比特数
            
        Returns:
            量子态的测量结果
        """
        # 使用RX, RY或RZ门将数据编码为旋转角度
        for i in range(n_qubits):
            qml.RX(x[i], wires=i)
            qml.RY(x[i], wires=i)
        
        # 返回所有量子比特的期望值
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    # 振幅编码
    @qml.qnode(dev_encoding)
    def amplitude_encoding(x):
        """
        振幅编码 - 将数据编码为量子态的振幅
        
        Args:
            x: 输入数据，长度应为2^n_qubits，并且已归一化
            
        Returns:
            量子态的测量结果
        """
        # 使用振幅编码将数据编码为量子态
        qml.AmplitudeEmbedding(x, wires=range(n_qubits), normalize=True)
        
        # 返回所有量子比特的期望值
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    # IQP编码
    @qml.qnode(dev_encoding)
    def iqp_encoding(x):
        """
        IQP编码 - 使用交替的旋转和纠缠层
        
        Args:
            x: 输入数据
            
        Returns:
            量子态的测量结果
        """
        # 首先应用Hadamard门创建叠加态
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        
        # 使用RZ门应用旋转层
        for i in range(n_qubits):
            qml.RZ(x[i], wires=i)
        
        # 应用纠缠层（例如ZZ相互作用）
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                qml.CNOT(wires=[i, j])
                qml.RZ(x[i] * x[j], wires=j)
                qml.CNOT(wires=[i, j])
        
        # 再次应用旋转层
        for i in range(n_qubits):
            qml.RZ(x[i], wires=i)
            qml.Hadamard(wires=i)
        
        # 返回所有量子比特的期望值
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    # 测试编码方法
    x_test = np.random.random(n_qubits)
    x_test_normalized = x_test / np.linalg.norm(x_test)
    x_amplitude = np.random.random(2**n_qubits)
    x_amplitude = x_amplitude / np.linalg.norm(x_amplitude)
    
    print("测试数据:", x_test)
    print("角度编码结果:", angle_encoding(x_test))
    print("振幅编码结果:", amplitude_encoding(x_amplitude))
    print("IQP编码结果:", iqp_encoding(x_test))
    
    # 绘制编码电路
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    qml.draw_mpl(angle_encoding, expansion_strategy="device")(x_test)
    ax[0].set_title("角度编码电路")
    
    qml.draw_mpl(amplitude_encoding, expansion_strategy="device")(x_amplitude)
    ax[1].set_title("振幅编码电路")
    
    qml.draw_mpl(iqp_encoding, expansion_strategy="device")(x_test)
    ax[2].set_title("IQP编码电路")
    
    plt.tight_layout()
    plt.savefig('encoding_circuits.png')
    plt.close()
    
    print("编码电路已绘制并保存为'encoding_circuits.png'")
    
    return angle_encoding, amplitude_encoding, iqp_encoding

if SKLEARN_AVAILABLE and TORCH_AVAILABLE:
    angle_encoding, amplitude_encoding, iqp_encoding = exercise1_solution()
else:
    print("跳过练习1因为需要完整的依赖项")

"""
练习2: 量子神经网络(QNN)
"""
print("\n练习2: 量子神经网络(QNN) - 解答")

if SKLEARN_AVAILABLE:
    def exercise2_solution():
        # 生成简单的二分类数据集
        X, y = make_moons(n_samples=100, noise=0.15)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 创建量子设备
        n_qubits = 2
        dev_qnn = qml.device('default.qubit', wires=n_qubits)
        
        # 定义量子神经网络
        @qml.qnode(dev_qnn)
        def quantum_neural_network(inputs, weights):
            """
            量子神经网络
            
            Args:
                inputs: 输入数据特征
                weights: 变分参数
                
            Returns:
                输出
            """
            # 数据编码层
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            
            # 变分层 - 重复层结构
            n_layers = len(weights) // (n_qubits * 2 + 1)
            
            for l in range(n_layers):
                # 单量子比特旋转
                for i in range(n_qubits):
                    qml.RX(weights[l * (n_qubits * 2 + 1) + i], wires=i)
                    qml.RY(weights[l * (n_qubits * 2 + 1) + n_qubits + i], wires=i)
                
                # 纠缠门
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                
                # 最后一个旋转门
                qml.RZ(weights[l * (n_qubits * 2 + 1) + 2 * n_qubits], wires=0)
            
            # 测量输出
            return qml.expval(qml.PauliZ(0))
        
        # 定义二分类模型
        def binary_classifier(inputs, weights):
            """将QNN输出转换为二分类结果"""
            return quantum_neural_network(inputs, weights) > 0.0
        
        # 定义成本函数
        def cost(weights, X, y):
            """
            计算均方误差成本
            
            Args:
                weights: 模型权重
                X: 输入特征
                y: 目标标签 (0/1)
                
            Returns:
                成本值
            """
            # 将标签y从{0,1}转换为{-1,1}
            y_mapped = 2 * y - 1
            
            # 计算预测
            predictions = np.array([quantum_neural_network(x, weights) for x in X])
            
            # 计算均方误差
            return np.mean((predictions - y_mapped) ** 2)
        
        # 训练QNN
        # 初始化随机权重
        n_layers = 2
        weight_shape = n_layers * (n_qubits * 2 + 1)  # 每层有2*n_qubits个旋转门和1个额外旋转
        weights = np.random.uniform(-np.pi, np.pi, weight_shape)
        
        # 使用梯度下降优化器
        opt = qml.AdamOptimizer(stepsize=0.1)
        steps = 100
        batch_size = 5
        cost_history = []
        
        print(f"开始训练QNN, 数据形状: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"初始权重形状: {weights.shape}，层数: {n_layers}")
        
        for step in range(steps):
            # 随机选择批次
            batch_indices = np.random.randint(0, len(X_train), batch_size)
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            # 更新权重
            weights = opt.step(lambda w: cost(w, X_batch, y_batch), weights)
            
            # 计算全部数据的成本
            current_cost = cost(weights, X_train, y_train)
            cost_history.append(current_cost)
            
            if (step + 1) % 10 == 0:
                print(f"步骤 {step+1}: 成本 = {current_cost:.6f}")
        
        # 评估模型性能
        y_pred = [binary_classifier(x, weights) for x in X_test]
        accuracy = accuracy_score(y_test, y_pred)
        print(f"测试准确率: {accuracy:.4f}")
        
        # 绘制决策边界
        plt.figure(figsize=(12, 5))
        
        # 绘制训练过程
        plt.subplot(1, 2, 1)
        plt.plot(cost_history)
        plt.xlabel('优化步骤')
        plt.ylabel('成本')
        plt.title('QNN训练过程')
        plt.grid(True)
        
        # 绘制决策边界
        plt.subplot(1, 2, 2)
        
        # 创建网格
        h = 0.01
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # 预测网格点
        Z = np.array([binary_classifier(np.array([x, y]), weights) 
                      for x, y in zip(xx.ravel(), yy.ravel())])
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k')
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', s=60)
        plt.title(f'QNN决策边界 (准确率: {accuracy:.4f})')
        
        plt.tight_layout()
        plt.savefig('qnn_classification.png')
        plt.close()
        
        print("QNN分类结果已绘制并保存为'qnn_classification.png'")
        
        return weights, cost_history, accuracy
    
    weights2, cost_history2, accuracy2 = exercise2_solution()
else:
    print("未安装scikit-learn，跳过量子神经网络分类练习")

"""
练习3: 量子核方法
"""
print("\n练习3: 量子核方法 - 解答")

if SKLEARN_AVAILABLE:
    def exercise3_solution():
        # 创建量子设备
        n_qubits = 2
        dev_kernel = qml.device('default.qubit', wires=n_qubits)
        
        # 定义特征映射电路
        @qml.qnode(dev_kernel)
        def feature_map(x):
            """
            量子特征映射电路
            
            Args:
                x: 输入数据
                
            Returns:
                量子态
            """
            # 数据编码
            for i in range(n_qubits):
                qml.RX(x[i], wires=i)
            
            # 非线性变换
            for i in range(n_qubits):
                qml.RZ(x[i] ** 2, wires=i)
            
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(x[i] * x[i + 1], wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
            
            # 返回量子态
            return qml.state()
        
        # 定义量子核函数
        def quantum_kernel(x1, x2):
            """
            计算两个数据点的量子核
            
            Args:
                x1, x2: 两个数据点
                
            Returns:
                核值 (内积)
            """
            state1 = feature_map(x1)
            state2 = feature_map(x2)
            
            # 计算量子态的内积
            return np.abs(np.vdot(state1, state2))**2
        
        # 生成数据集
        X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 计算核矩阵
        def compute_kernel_matrix(X1, X2):
            """计算两组数据点之间的核矩阵"""
            n1 = len(X1)
            n2 = len(X2)
            kernel_matrix = np.zeros((n1, n2))
            
            for i in range(n1):
                for j in range(n2):
                    kernel_matrix[i, j] = quantum_kernel(X1[i], X2[j])
            
            return kernel_matrix
        
        # 计算训练集和测试集的核矩阵
        print("计算训练集的核矩阵...")
        K_train = compute_kernel_matrix(X_train, X_train)
        print("计算测试集的核矩阵...")
        K_test = compute_kernel_matrix(X_test, X_train)
        
        # 使用核矩阵进行分类
        from sklearn.svm import SVC
        
        # 创建预计算核的SVM
        qsvm = SVC(kernel='precomputed')
        
        # 训练模型
        print("训练量子核SVM...")
        qsvm.fit(K_train, y_train)
        
        # 预测
        y_pred = qsvm.predict(K_test)
        
        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        print(f"量子核SVM的测试准确率: {accuracy:.4f}")
        
        # 可视化结果
        plt.figure(figsize=(10, 8))
        
        # 绘制核矩阵
        plt.subplot(2, 2, 1)
        plt.imshow(K_train)
        plt.title('量子核矩阵 (训练集)')
        plt.colorbar()
        
        # 绘制决策边界
        plt.subplot(2, 2, 2)
        
        # 创建网格
        h = 0.05
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # 计算所有网格点与训练点的核
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        print("计算网格点的核矩阵以绘制决策边界...")
        K_grid = compute_kernel_matrix(grid_points, X_train)
        
        # 预测网格点
        Z = qsvm.predict(K_grid)
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k')
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', s=60)
        plt.title(f'量子核SVM决策边界 (准确率: {accuracy:.4f})')
        
        # 与经典RBF核比较
        from sklearn.svm import SVC as ClassicalSVC
        cls_svm = ClassicalSVC(kernel='rbf')
        print("训练经典RBF核SVM...")
        cls_svm.fit(X_train, y_train)
        cls_pred = cls_svm.predict(X_test)
        cls_accuracy = accuracy_score(y_test, cls_pred)
        
        plt.subplot(2, 2, 3)
        Z_cls = cls_svm.predict(np.c_[xx.ravel(), yy.ravel()])
        Z_cls = Z_cls.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z_cls, alpha=0.3)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k')
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', s=60)
        plt.title(f'经典RBF核SVM (准确率: {cls_accuracy:.4f})')
        
        plt.tight_layout()
        plt.savefig('quantum_kernel.png')
        plt.close()
        
        print(f"经典RBF核SVM测试准确率: {cls_accuracy:.4f}")
        print(f"量子核SVM测试准确率: {accuracy:.4f}")
        print("量子核比较结果已绘制并保存为'quantum_kernel.png'")
        
        return K_train, accuracy, cls_accuracy
    
    K_train3, q_accuracy3, c_accuracy3 = exercise3_solution()
else:
    print("未安装scikit-learn，跳过量子核方法练习")

"""
练习4: 量子转移学习
"""
print("\n练习4: 量子转移学习 - 解答")

if TORCH_AVAILABLE and SKLEARN_AVAILABLE:
    def exercise4_solution():
        # 创建量子设备
        n_qubits = 2
        dev_transfer = qml.device('default.qubit', wires=n_qubits)
        
        # 创建数据集
        X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 数据预处理
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
        
        # 预训练的经典模型
        class ClassicalMLP(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, output_dim)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        # 训练经典模型
        input_dim = X_train.shape[1]
        hidden_dim = 4
        output_dim = 2  # 特征提取的输出维度
        
        classical_model = ClassicalMLP(input_dim, hidden_dim, output_dim)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(classical_model.parameters(), lr=0.01)
        
        # 训练循环
        epochs = 50
        print("训练经典模型...")
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = classical_model(X_train_tensor)
            loss = criterion(outputs[:, 0].reshape(-1, 1), y_train_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"预训练经典模型，Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        # 量子部分 - 将经典模型的特征用作量子电路的输入
        @qml.qnode(dev_transfer)
        def quantum_circuit(inputs, weights):
            """
            量子分类器电路
            
            Args:
                inputs: 经典模型提取的特征
                weights: 量子电路的权重
                
            Returns:
                分类预测
            """
            # 编码经典特征
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            
            # 变分层
            for j in range(len(weights) // 3):
                idx = j * 3
                # 旋转层
                qml.RX(weights[idx], wires=0)
                qml.RY(weights[idx + 1], wires=1)
                
                # 纠缠层
                qml.CNOT(wires=[0, 1])
                qml.RZ(weights[idx + 2], wires=1)
            
            # 测量
            return qml.expval(qml.PauliZ(0))
        
        # 混合量子-经典模型
        class HybridModel(nn.Module):
            def __init__(self, classical_model, n_qubits):
                super().__init__()
                self.classical_model = classical_model
                self.q_weights = nn.Parameter(torch.randn(n_qubits * 3))  # 量子电路参数
                
            def forward(self, x):
                # 冻结经典模型权重
                with torch.no_grad():
                    features = self.classical_model(x)
                
                # 归一化特征
                features_normalized = features / (torch.norm(features, dim=1, keepdim=True) + 1e-8)
                
                # 使用量子电路处理特征
                q_out = torch.tensor([
                    quantum_circuit(
                        features_normalized[i].detach().numpy(), 
                        self.q_weights.detach().numpy()
                    )
                    for i in range(len(features_normalized))
                ], requires_grad=True)
                
                return q_out.reshape(-1, 1)
        
        # 训练混合模型
        hybrid_model = HybridModel(classical_model, n_qubits)
        
        # 仅训练量子部分的参数
        hybrid_optimizer = optim.Adam([hybrid_model.q_weights], lr=0.1)
        
        # 训练循环
        hybrid_epochs = 30
        hybrid_losses = []
        
        print("训练混合量子-经典模型...")
        for epoch in range(hybrid_epochs):
            hybrid_optimizer.zero_grad()
            
            # 前向传播
            hybrid_outputs = hybrid_model(X_train_tensor)
            
            # 将输出转换为0-1之间
            hybrid_outputs = (hybrid_outputs + 1) / 2
            
            # 计算损失
            hybrid_loss = nn.functional.binary_cross_entropy(
                hybrid_outputs, 
                y_train_tensor
            )
            
            # 反向传播
            hybrid_loss.backward()
            hybrid_optimizer.step()
            
            hybrid_losses.append(hybrid_loss.item())
            
            if (epoch + 1) % 5 == 0:
                print(f"混合模型训练，Epoch {epoch+1}/{hybrid_epochs}, Loss: {hybrid_loss.item():.4f}")
        
        # 评估模型
        with torch.no_grad():
            # 评估经典模型
            classical_outputs = classical_model(X_test_tensor)
            classical_preds = (torch.sigmoid(classical_outputs[:, 0]) > 0.5).numpy().astype(int)
            classical_accuracy = accuracy_score(y_test, classical_preds)
            
            # 评估混合模型
            hybrid_outputs = hybrid_model(X_test_tensor)
            hybrid_preds = (hybrid_outputs > 0.5).numpy().astype(int).flatten()
            hybrid_accuracy = accuracy_score(y_test, hybrid_preds)
        
        print(f"经典模型测试准确率: {classical_accuracy:.4f}")
        print(f"混合量子-经典模型测试准确率: {hybrid_accuracy:.4f}")
        
        # 可视化结果
        plt.figure(figsize=(12, 5))
        
        # 绘制训练损失
        plt.subplot(1, 2, 1)
        plt.plot(hybrid_losses)
        plt.xlabel('优化步骤')
        plt.ylabel('损失')
        plt.title('混合模型训练损失')
        plt.grid(True)
        
        # 绘制决策边界比较
        plt.subplot(1, 2, 2)
        
        # 创建网格
        h = 0.05
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # 网格点张量
        grid_tensor = torch.tensor(
            np.c_[xx.ravel(), yy.ravel()], 
            dtype=torch.float32
        )
        
        # 混合模型预测
        with torch.no_grad():
            Z_hybrid = hybrid_model(grid_tensor).numpy()
        Z_hybrid = (Z_hybrid > 0.5).astype(int).reshape(xx.shape)
        
        plt.contourf(xx, yy, Z_hybrid, alpha=0.3, levels=1)
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', marker='o')
        plt.scatter(X_test[:, 0], X_test[:, 1], 
                  c=hybrid_preds, alpha=0.2, marker='x', s=100)
        plt.title(f'混合模型决策边界 (准确率: {hybrid_accuracy:.4f})')
        
        plt.tight_layout()
        plt.savefig('quantum_transfer_learning.png')
        plt.close()
        
        print("量子转移学习结果已绘制并保存为'quantum_transfer_learning.png'")
        
        return classical_accuracy, hybrid_accuracy, hybrid_losses
    
    classical_accuracy4, hybrid_accuracy4, hybrid_losses4 = exercise4_solution()
else:
    print("未安装PyTorch或scikit-learn，跳过量子转移学习练习")

print("\n所有练习解答已完成。这些解答展示了PennyLane的量子机器学习功能。")
print("祝贺您完成了PennyLane框架的学习！接下来，您可以考虑尝试真实量子硬件上的实验。") 