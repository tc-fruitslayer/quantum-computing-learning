#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Xanadu PennyLane框架学习 5：量子机器学习
本文件详细介绍量子机器学习的概念、模型和实现
"""

# 导入必要的库
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# 尝试导入经典机器学习库（可选）
try:
    import sklearn
    from sklearn.datasets import make_moons, load_iris
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("sklearn未安装，某些示例可能无法运行")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch未安装，某些示例可能无法运行")

print("===== PennyLane量子机器学习 =====")

# 检查PennyLane版本
print(f"PennyLane版本: {qml.__version__}")

# 1. 量子机器学习概述
print("\n1. 量子机器学习概述")
print("量子机器学习(QML)是机器学习和量子计算的交叉领域")
print("主要方向包括:")
print("- 量子增强机器学习：使用量子算法加速经典机器学习")
print("- 量子模型：使用量子电路作为机器学习模型")
print("- 量子数据：处理量子数据的机器学习方法")
print("- 经典辅助量子学习：使用经典优化器训练量子模型")

# 2. 量子机器学习的优势
print("\n2. 量子机器学习的优势")
print("为什么要使用量子机器学习？")
print("- 处理指数级特征空间：量子计算可以有效地表示和处理大型特征空间")
print("- 量子并行性：量子叠加使某些计算可以并行执行")
print("- 量子纠缠：可以捕获复杂的特征相关性")
print("- 量子隧穿效应：可能帮助优化器逃离局部极小值")
print("- 量子增强核方法：某些量子电路可以实现经典算法难以计算的核函数")

# 3. 数据编码
print("\n3. 数据编码")
print("将经典数据编码到量子态是量子机器学习的第一步")

# 创建一个量子设备
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# 3.1 角度编码
print("\n3.1 角度编码")
@qml.qnode(dev)
def angle_encoding(features):
    qml.templates.AngleEmbedding(features, wires=range(n_qubits))
    return qml.state()

# 3.2 振幅编码
print("\n3.2 振幅编码")
@qml.qnode(dev)
def amplitude_encoding(features):
    qml.templates.AmplitudeEmbedding(features, wires=range(n_qubits), normalize=True)
    return qml.state()

# 3.3 基于量子傅里叶变换的编码
print("\n3.3 基于量子傅里叶变换的编码")
@qml.qnode(dev)
def qft_encoding(features):
    # 先编码初始角度
    qml.templates.AngleEmbedding(features, wires=range(n_qubits))
    # 应用QFT
    qml.QFT(wires=range(n_qubits))
    return qml.state()

# 示例数据
features = np.random.uniform(0, np.pi, n_qubits)
normalized_features = features / np.linalg.norm(features)  # 用于振幅编码

print("\n编码方法比较:")
print("角度编码结构:")
print(qml.draw(angle_encoding)(features))

print("\n振幅编码结构:")
print(qml.draw(amplitude_encoding)(normalized_features))

print("\nQFT编码结构:")
print(qml.draw(qft_encoding)(features))

# 4. 量子神经网络
print("\n4. 量子神经网络")
print("量子神经网络(QNN)是使用参数化量子电路实现的神经网络")

# 4.1 连续变量量子神经网络
print("\n4.1 变分量子电路作为神经网络")

# 创建一个简单的变分量子神经网络
@qml.qnode(dev)
def variational_circuit(inputs, weights):
    # 编码输入
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    
    # 变分层
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    
    # 测量
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# 获取权重张量形状
weight_shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=n_qubits)
weights = np.random.uniform(-np.pi, np.pi, weight_shape)

# 运行电路
inputs = np.random.uniform(-np.pi, np.pi, n_qubits)
outputs = variational_circuit(inputs, weights)

print(f"\n变分量子神经网络结构:")
print(qml.draw(variational_circuit)(inputs, weights))
print(f"\n输入: {inputs}")
print(f"输出: {outputs}")

# 5. 量子分类器
print("\n5. 量子分类器")
print("量子分类器使用量子电路进行分类任务")

# 5.1 变分量子分类器 (VQC)
print("\n5.1 变分量子分类器 (VQC)")

def variational_classifier(inputs, weights):
    """变分量子分类器"""
    # 计算变分量子电路的输出
    qnn_outputs = variational_circuit(inputs, weights)
    
    # 聚合输出以获得2类分类输出
    # 使用第一个量子比特的期望值作为类别得分
    return qnn_outputs[0]

# 5.2 基于核的量子分类器
print("\n5.2 基于核的量子分类器")

def quantum_kernel(x1, x2):
    """量子核函数：计算两个输入向量之间的量子相似度"""
    # 创建内积设备
    dev_kern = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev_kern)
    def kernel_circuit(x1, x2):
        # 编码第一个输入
        qml.templates.AngleEmbedding(x1, wires=range(n_qubits))
        
        # 应用逆变换
        qml.adjoint(qml.templates.AngleEmbedding)(x2, wires=range(n_qubits))
        
        # 返回测量|0...0⟩的概率（融洽度）
        return qml.probs(wires=range(n_qubits))[0]
    
    return kernel_circuit(x1, x2)

# 测试核函数
x1 = np.random.uniform(0, np.pi, n_qubits)
x2 = np.random.uniform(0, np.pi, n_qubits)
kernel_value = quantum_kernel(x1, x2)

print(f"\n量子核值 K(x1, x2) = {kernel_value:.6f}")
print(f"相同输入的核值 K(x1, x1) = {quantum_kernel(x1, x1):.6f}")

# 6. 量子生成对抗网络
print("\n6. 量子生成对抗网络 (QGAN)")
print("QGAN结合了量子电路和GAN的思想")

# 创建一个简单的量子生成器
@qml.qnode(dev)
def quantum_generator(noise, weights):
    # 编码噪声
    qml.templates.AngleEmbedding(noise, wires=range(n_qubits))
    
    # 应用参数化电路
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    
    # 返回测量结果（生成的"假"数据）
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# 创建一个简单的量子判别器
@qml.qnode(dev)
def quantum_discriminator(data, weights):
    # 编码数据（真实的或生成的）
    qml.templates.AngleEmbedding(data, wires=range(n_qubits))
    
    # 应用参数化电路
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    
    # 返回判别结果（1代表真实，0代表伪造）
    return qml.expval(qml.PauliZ(0))

# 7. 量子卷积神经网络
print("\n7. 量子卷积神经网络 (QCNN)")
print("QCNN结合了量子电路和CNN的思想")

def quantum_conv_layer(inputs, weights, wires):
    """量子卷积层"""
    # 编码输入
    qml.templates.AngleEmbedding(inputs, wires=wires)
    
    # 应用卷积式量子操作
    for i in range(len(wires)-1):
        # 对相邻量子比特应用参数化操作
        qml.RY(weights[i, 0], wires=wires[i])
        qml.RZ(weights[i, 1], wires=wires[i+1])
        qml.CNOT(wires=[wires[i], wires[i+1]])

# 8. 实际分类问题示例
print("\n8. 实际分类问题示例")

if SKLEARN_AVAILABLE:
    print("使用sklearn数据集进行量子分类示例")
    
    # 创建一个简单的二分类数据集（月牙形）
    X, y = make_moons(n_samples=200, noise=0.1)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # 创建一个2量子比特分类器
    n_qubits_class = 2
    dev_class = qml.device("default.qubit", wires=n_qubits_class)
    
    @qml.qnode(dev_class)
    def quantum_classifier(inputs, weights):
        # 缩放输入到合适的范围
        scaled_inputs = np.array([inputs[0], inputs[1]])
        
        # 编码输入
        qml.RY(scaled_inputs[0], wires=0)
        qml.RY(scaled_inputs[1], wires=1)
        qml.CNOT(wires=[0, 1])
        
        # 变分层
        qml.RX(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(weights[2], wires=0)
        qml.RY(weights[3], wires=1)
        
        # 返回测量结果
        return qml.expval(qml.PauliZ(0))
    
    # 定义成本函数（二分类交叉熵损失的简化版本）
    def cost(weights, X, y):
        predictions = [quantum_classifier(x, weights) for x in X]
        # 将[-1,1]范围映射到[0,1]
        predictions = [(p + 1) / 2 for p in predictions]
        # 计算二元交叉熵
        loss = 0
        for pred, target in zip(predictions, y):
            # 避免数值不稳定性
            pred = np.clip(pred, 1e-10, 1 - 1e-10)
            loss += -(target * np.log(pred) + (1 - target) * np.log(1 - pred))
        return loss / len(y)
    
    # 训练分类器
    def train_classifier(X_train, y_train, n_epochs=50):
        # 初始化权重
        weights = np.random.uniform(-np.pi, np.pi, 4)
        
        # 使用Adam优化器
        opt = qml.AdamOptimizer(stepsize=0.1)
        
        # 训练循环
        loss_history = []
        
        for epoch in range(n_epochs):
            # 单步优化
            weights = opt.step(lambda w: cost(w, X_train, y_train), weights)
            
            # 计算当前损失
            loss = cost(weights, X_train, y_train)
            loss_history.append(loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.6f}")
        
        return weights, loss_history
    
    # 运行训练（仅训练几个轮次用于演示）
    trained_weights, loss_history = train_classifier(X_train, y_train, n_epochs=20)
    
    # 评估模型
    def evaluate_classifier(weights, X, y):
        predictions = [quantum_classifier(x, weights) for x in X]
        # 映射到[0,1]然后转换为二进制预测
        binary_predictions = [1 if (p + 1) / 2 > 0.5 else 0 for p in predictions]
        # 计算准确率
        accuracy = np.mean(np.array(binary_predictions) == y)
        return accuracy
    
    train_accuracy = evaluate_classifier(trained_weights, X_train, y_train)
    test_accuracy = evaluate_classifier(trained_weights, X_test, y_test)
    
    print(f"\n训练准确率: {train_accuracy:.4f}")
    print(f"测试准确率: {test_accuracy:.4f}")
    
    # 绘制损失曲线
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('量子分类器训练损失')
    plt.grid(True)
    plt.savefig('quantum_classifier_loss.png')
    plt.close()
    
    # 绘制决策边界
    def plot_decision_boundary():
        h = 0.02  # 步长
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        # 为网格中的每个点获取预测
        Z = np.array([quantum_classifier(np.array([x, y]), trained_weights) for x, y in zip(xx.ravel(), yy.ravel())])
        Z = (Z + 1) / 2  # 映射到[0,1]
        Z = Z.reshape(xx.shape)
        
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdBu, edgecolors='k')
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.RdBu, edgecolors='k', marker='^')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title('量子分类器决策边界')
        plt.savefig('quantum_classifier_boundary.png')
        plt.close()
    
    # 生成决策边界图（可能计算量较大）
    print("\n绘制决策边界...")
    plot_decision_boundary()
    print("决策边界已保存到'quantum_classifier_boundary.png'")
else:
    print("sklearn未安装，跳过分类示例")

# 9. 量子传递学习
print("\n9. 量子传递学习")
print("量子传递学习结合了预训练的经典模型和量子模型")

# 创建一个量子传递学习示例框架
def quantum_transfer_learning_example():
    print("\n量子传递学习示例流程:")
    print("1. 使用预训练的经典网络(如ResNet)提取特征")
    print("2. 将提取的特征编码到量子态中")
    print("3. 应用量子变分电路进行进一步处理")
    print("4. 测量得到最终分类结果")

    # 假设的预训练网络特征
    classical_features = np.random.uniform(-1, 1, n_qubits)
    print(f"\n预训练网络提取的特征: {classical_features}")
    
    # 定义量子传递学习电路
    @qml.qnode(dev)
    def qtl_circuit(features, weights):
        # 编码经典特征
        qml.templates.AngleEmbedding(features, wires=range(n_qubits))
        
        # 应用量子变分层
        qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        
        # 测量
        return qml.expval(qml.PauliZ(0))
    
    # 随机初始化权重
    qtl_weights = np.random.uniform(-np.pi, np.pi, weight_shape)
    
    # 运行电路
    qtl_output = qtl_circuit(classical_features, qtl_weights)
    
    print(f"量子传递学习输出: {qtl_output:.6f}")
    print("注：完整实现需要与PyTorch或TensorFlow集成")

# 执行量子传递学习示例
quantum_transfer_learning_example()

# 10. 量子卷积神经网络完整示例
print("\n10. 量子卷积神经网络完整示例")

def quantum_cnn_example():
    # 创建专用设备
    n_qubits_cnn = 4
    dev_cnn = qml.device("default.qubit", wires=n_qubits_cnn)
    
    @qml.qnode(dev_cnn)
    def qcnn_circuit(inputs, conv_weights, pool_weights, fc_weights):
        # 编码输入
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits_cnn))
        
        # 卷积层 1
        # 在相邻量子比特对上应用卷积操作
        for i in range(n_qubits_cnn - 1):
            qml.RY(conv_weights[0, i, 0], wires=i)
            qml.RY(conv_weights[0, i, 1], wires=i+1)
            qml.CNOT(wires=[i, i+1])
            qml.RY(conv_weights[0, i, 2], wires=i)
            qml.RY(conv_weights[0, i, 3], wires=i+1)
        
        # 池化层 1（通过测量和条件重置实现）
        # 简化版本：只保留偶数量子比特
        qubits_after_pool = [0, 2]
        
        # 卷积层 2
        qml.RY(conv_weights[1, 0, 0], wires=qubits_after_pool[0])
        qml.RY(conv_weights[1, 0, 1], wires=qubits_after_pool[1])
        qml.CNOT(wires=qubits_after_pool)
        qml.RY(conv_weights[1, 0, 2], wires=qubits_after_pool[0])
        qml.RY(conv_weights[1, 0, 3], wires=qubits_after_pool[1])
        
        # 全连接层
        for i, qubit in enumerate(qubits_after_pool):
            qml.RY(fc_weights[i], wires=qubit)
        
        # 最终测量
        return qml.expval(qml.PauliZ(0))
    
    # 初始化随机权重
    conv_weights = np.random.uniform(-np.pi, np.pi, (2, n_qubits_cnn-1, 4))  # 2个卷积层
    pool_weights = np.random.uniform(-np.pi, np.pi, (1, 2))  # 1个池化层
    fc_weights = np.random.uniform(-np.pi, np.pi, 2)  # 全连接层
    
    # 随机输入
    inputs = np.random.uniform(-np.pi, np.pi, n_qubits_cnn)
    
    # 运行电路
    output = qcnn_circuit(inputs, conv_weights, pool_weights, fc_weights)
    
    print(f"量子CNN输出: {output:.6f}")
    print(f"电路结构:")
    print(qml.draw(qcnn_circuit)(inputs, conv_weights, pool_weights, fc_weights))

# 执行量子CNN示例
quantum_cnn_example()

# 11. 与PyTorch集成 - 混合经典-量子模型
print("\n11. 与PyTorch集成 - 混合经典-量子模型")

if TORCH_AVAILABLE:
    print("创建PyTorch-PennyLane混合模型示例")
    
    # 创建量子设备
    n_qubits_torch = 4
    dev_torch = qml.device("default.qubit", wires=n_qubits_torch)
    
    # 定义量子电路
    @qml.qnode(dev_torch, interface="torch")
    def torch_quantum_circuit(inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits_torch))
        qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits_torch))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits_torch)]
    
    # 定义PyTorch量子层
    class TorchQuantumLayer(torch.nn.Module):
        def __init__(self, n_qubits, n_layers):
            super().__init__()
            self.n_qubits = n_qubits
            shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
            weight_shapes = {"weights": shape}
            self.qlayer = qml.qnn.TorchLayer(torch_quantum_circuit, weight_shapes)
            
        def forward(self, x):
            return self.qlayer(x)
    
    # 定义混合模型
    class HybridModel(torch.nn.Module):
        def __init__(self, n_qubits, n_layers, input_size, hidden_size):
            super().__init__()
            self.pre_net = torch.nn.Sequential(
                torch.nn.Linear(input_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, n_qubits)
            )
            self.quantum_layer = TorchQuantumLayer(n_qubits, n_layers)
            self.post_net = torch.nn.Linear(n_qubits, 1)
            
        def forward(self, x):
            x = self.pre_net(x)
            x = self.quantum_layer(x)
            return torch.sigmoid(self.post_net(x))
    
    # 创建模型实例
    model = HybridModel(n_qubits=n_qubits_torch, n_layers=2, input_size=10, hidden_size=20)
    
    # 示例输入
    x = torch.randn(5, 10)
    
    # 前向传播
    output = model(x)
    
    print(f"\n混合模型输出形状: {output.shape}")
    print(f"输出值: {output.detach().numpy()}")
    
    print("\n混合模型架构:")
    print(model)
    
    print("\n注：完整训练需要定义损失函数、优化器和训练循环")
else:
    print("PyTorch未安装，跳过混合模型示例")

# 12. 量子机器学习的挑战和前景
print("\n12. 量子机器学习的挑战和前景")
print("挑战:")
print("- 量子噪声和退相干：限制了当前量子设备的能力")
print("- 有限的量子比特数量：限制了可处理的问题规模")
print("- 优化难度：梯度消失、局部极小值等问题")
print("- 量子-经典接口开销：频繁的量子-经典通信可能抵消量子优势")
print("- 理论理解有限：量子机器学习的理论基础仍在发展中")

print("\n前景:")
print("- 量子启发的经典算法：即使在经典硬件上也可能带来改进")
print("- 特定应用领域的优势：如量子化学、材料科学等")
print("- 新型量子算法：可能带来指数级加速")
print("- 量子特征映射：提供经典难以实现的特征空间")
print("- 随着量子硬件改进，更多应用将变得可行")

# 13. 总结
print("\n13. 总结")
print("1. 量子机器学习结合了量子计算和机器学习的优势")
print("2. 数据编码是量子机器学习的关键步骤")
print("3. 量子神经网络和分类器为经典任务提供了新方法")
print("4. 变分量子算法是当前NISQ时代的主要范式")
print("5. 混合经典-量子模型可结合两种计算范式的优势")

print("\n下一步学习:")
print("- 更复杂的量子机器学习模型")
print("- 实际量子数据处理任务")
print("- 量子增强算法的理论保证")
print("- 特定领域应用（如量子化学、量子金融）") 