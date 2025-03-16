#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Google Cirq框架学习 5：TensorFlow Quantum集成
本文件详细介绍如何结合Cirq与TensorFlow Quantum进行量子机器学习
注意：运行本文件需要安装tensorflow-quantum包
"""

import cirq
import sympy
import numpy as np
import matplotlib.pyplot as plt

# 尝试导入tensorflow和tensorflow_quantum
try:
    import tensorflow as tf
    import tensorflow_quantum as tfq
    print("===== Cirq与TensorFlow Quantum集成 =====")
    print(f"TensorFlow版本: {tf.__version__}")
    print(f"TensorFlow Quantum版本: {tfq.__version__}")
    tfq_available = True
except ImportError:
    print("===== TensorFlow Quantum未安装 =====")
    print("本文件需要安装tensorflow和tensorflow-quantum")
    print("安装命令: pip install tensorflow tensorflow-quantum")
    print("将继续执行部分代码，但无法运行TFQ相关功能")
    tfq_available = False

# 1. TensorFlow Quantum简介
print("\n1. TensorFlow Quantum简介")
print("TensorFlow Quantum (TFQ) 是一个用于混合量子-经典机器学习的库")
print("它结合了Google的Cirq量子编程框架和TensorFlow机器学习库")
print("TFQ让开发者能够创建量子数据和量子模型，并在经典神经网络中使用它们")

# 2. 基本组件和概念
print("\n2. 基本组件和概念")

# 2.1 量子电路与张量
print("\n2.1 量子电路与张量")
print("TFQ将Cirq电路转换为TensorFlow张量，以便在TensorFlow计算图中使用")

# 创建一个简单的Bell状态电路
q0, q1 = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
bell_circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1)
)
print("Bell状态电路:")
print(bell_circuit)

if tfq_available:
    # 将电路转换为张量
    circuit_tensor = tfq.convert_to_tensor([bell_circuit])
    print("\n转换为TensorFlow张量后的形状:", circuit_tensor.shape)

# 2.2 参数化量子电路
print("\n2.2 参数化量子电路")
print("TFQ支持使用符号参数创建参数化量子电路，这些参数可以在训练过程中优化")

# 创建一个参数化电路
theta = sympy.Symbol('theta')
phi = sympy.Symbol('phi')

param_circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.rx(theta).on(q0),
    cirq.rx(phi).on(q1),
    cirq.CNOT(q0, q1)
)
print("参数化电路:")
print(param_circuit)

if tfq_available:
    # 将参数化电路转换为张量
    param_tensor = tfq.convert_to_tensor([param_circuit])
    print("\n参数化电路张量的形状:", param_tensor.shape)
    
    # 创建参数值字典
    param_values = tf.convert_to_tensor([[0.5, 0.25]])  # 批量大小=1，参数数量=2
    print("参数值张量的形状:", param_values.shape)

# 3. TFQ中的量子电路评估
print("\n3. TFQ中的量子电路评估")
if tfq_available:
    # 创建一个简单电路进行演示
    q0 = cirq.GridQubit(0, 0)
    simple_circuit = cirq.Circuit(cirq.X(q0)**0.5, cirq.measure(q0, key='m'))
    print("带测量的简单电路:")
    print(simple_circuit)
    
    # 3.1 使用TFQ执行电路
    print("\n3.1 使用TFQ执行电路")
    # 将电路转换为张量
    circuit_tensor = tfq.convert_to_tensor([simple_circuit])
    
    # 执行电路并获取测量结果
    result = tfq.layers.Sample()(circuit_tensor)
    print("采样结果:", result.numpy())
    
    # 3.2 期望值计算
    print("\n3.2 计算泡利算子的期望值")
    # 创建一个不带测量的电路
    q0 = cirq.GridQubit(0, 0)
    circuit = cirq.Circuit(cirq.X(q0)**0.5)
    circuit_tensor = tfq.convert_to_tensor([circuit])
    
    # 定义要测量的算子
    operators = tfq.convert_to_tensor([[cirq.Z(q0)]])
    
    # 计算期望值
    expectations = tfq.layers.Expectation()(circuit_tensor, operators=operators)
    print("Z算子的期望值:", expectations.numpy())
else:
    print("TensorFlow Quantum未安装，无法执行电路评估示例")

# 4. 构建量子-经典混合模型
print("\n4. 构建量子-经典混合模型")
if tfq_available:
    # 4.1 量子神经网络层
    print("\n4.1 量子神经网络层")
    print("TFQ提供了量子层，可以嵌入到普通的TensorFlow模型中")
    
    # 创建一个简单的参数化电路
    q0 = cirq.GridQubit(0, 0)
    theta = sympy.Symbol('theta')
    circuit = cirq.Circuit(cirq.rx(theta).on(q0))
    
    # 展示量子层的设计
    print("参数化单量子比特电路:")
    print(circuit)
    
    # 创建一个量子层
    model = tf.keras.Sequential([
        # 输入层
        tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string),
        # 量子层
        tfq.layers.PQC(
            circuit, # 参数化电路
            cirq.Z(q0)  # 测量算子
        )
    ])
    
    print("\n量子-经典混合模型结构:")
    model.summary()
    
    # 4.2 使用控制器模式构建模型
    print("\n4.2 使用控制器模式构建模型")
    # 创建一个经典控制器，生成量子电路的参数
    controller = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    print("经典控制器模型结构:")
    controller.build((None, 5))  # 假设输入特征维度为5
    controller.summary()
    
    print("\n量子-经典混合模型通常由以下部分组成:")
    print("1. 经典神经网络：预处理输入数据并生成量子电路参数")
    print("2. 量子电路层：使用参数执行量子操作")
    print("3. 测量层：计算量子态的期望值")
    print("4. 后处理层：将量子结果转换为最终输出")
else:
    print("TensorFlow Quantum未安装，无法构建量子-经典混合模型示例")

# 5. 量子数据编码和分类
print("\n5. 量子数据编码和分类")
if tfq_available:
    print("\n5.1 量子数据编码")
    print("数据编码是将经典数据转换为量子态的过程")
    print("常见编码策略包括:")
    print("- 基编码: 将经典bit直接映射到计算基上")
    print("- 振幅编码: 将数据编码到量子态的振幅中")
    print("- 角度编码: 使用旋转门的角度表示数据")
    
    # 演示角度编码
    def angle_encoding(features, qubits):
        """使用旋转门将特征编码到量子态中"""
        circuit = cirq.Circuit()
        for i, qubit in enumerate(qubits):
            # 将特征值用作旋转角度
            circuit.append(cirq.rx(features[i % len(features)]).on(qubit))
        return circuit
    
    # 创建一个示例特征向量和量子比特
    features = np.array([0.1, 0.2, 0.3, 0.4])
    qubits = [cirq.GridQubit(0, i) for i in range(4)]
    
    # 编码电路
    encoding_circuit = angle_encoding(features, qubits)
    print("\n角度编码电路:")
    print(encoding_circuit)
    
    print("\n5.2 量子分类")
    print("量子分类使用参数化量子电路来对数据进行分类")
    
    # 创建简单的量子分类器
    def create_quantum_classifier(n_qubits):
        """创建一个简单的量子分类器电路"""
        qubits = [cirq.GridQubit(0, i) for i in range(n_qubits)]
        
        # 创建编码和变分电路
        circuit = cirq.Circuit()
        
        # 为每个量子比特添加参数
        params = sympy.symbols(f'theta0:{n_qubits}')
        params2 = sympy.symbols(f'theta1:{n_qubits}')
        
        # 编码层 - 这里假设输入已经被编码
        for i, qubit in enumerate(qubits):
            circuit.append(cirq.H(qubit))
        
        # 第一个变分层 - 单量子比特旋转
        for i, qubit in enumerate(qubits):
            circuit.append(cirq.rx(params[i]).on(qubit))
        
        # 纠缠层
        for i in range(n_qubits - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
        
        # 第二个变分层 - 单量子比特旋转
        for i, qubit in enumerate(qubits):
            circuit.append(cirq.rx(params2[i]).on(qubit))
        
        return circuit, qubits
    
    # 创建一个2量子比特的分类器
    classifier_circuit, qubits = create_quantum_classifier(2)
    print("\n量子分类器电路:")
    print(classifier_circuit)
    
    # 创建要观测的算子 - 使用第一个量子比特的Z测量作为输出
    readout_operator = cirq.Z(qubits[0])
    
    print("\n示例分类模型构建:")
    # 构建量子分类模型
    quantum_model = tf.keras.Sequential([
        # 输入电路
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        # 参数化量子电路层
        tfq.layers.PQC(classifier_circuit, readout_operator)
    ])
    
    # 编译模型
    quantum_model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        metrics=['accuracy']
    )
    
    print("量子分类模型结构:")
    quantum_model.summary()
else:
    print("TensorFlow Quantum未安装，无法执行量子数据编码和分类示例")

# 6. 量子变分电路和优化
print("\n6. 量子变分电路和优化")
if tfq_available:
    print("\n6.1 变分量子电路 (VQC)")
    print("变分量子电路是参数化的量子电路，可以通过梯度下降进行优化")
    print("VQC是许多量子机器学习算法的核心，如VQE和QAOA")
    
    # 创建一个简单的变分电路
    q0, q1 = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
    
    # 定义符号参数
    theta1 = sympy.Symbol('theta1')
    theta2 = sympy.Symbol('theta2')
    theta3 = sympy.Symbol('theta3')
    
    vqc = cirq.Circuit([
        cirq.rx(theta1).on(q0),
        cirq.rx(theta2).on(q1),
        cirq.CNOT(q0, q1),
        cirq.rx(theta3).on(q0)
    ])
    
    print("\n变分量子电路:")
    print(vqc)
    
    # 定义测量
    operators = tfq.convert_to_tensor([[
        cirq.Z(q0) * cirq.Z(q1)  # 测量ZZ关联
    ]])
    
    print("\n6.2 变分参数优化")
    # 创建一个包装层用于优化
    expectation_layer = tfq.layers.Expectation()
    
    # 设置初始参数
    initializer = tf.constant_initializer(np.random.random(3))
    
    # 创建优化变量
    var_params = tf.Variable(
        initializer(shape=(1, 3), dtype=tf.float32),
        trainable=True
    )
    
    # 定义损失函数 - 我们想要最小化ZZ关联的期望值
    def loss_function():
        # 以张量形式传递电路和参数
        circuits_tensor = tfq.convert_to_tensor([vqc])
        expectation = expectation_layer(
            circuits_tensor, 
            operators=operators,
            symbol_names=['theta1', 'theta2', 'theta3'],
            symbol_values=var_params
        )
        # 最小化期望值
        return tf.reduce_sum(expectation)
    
    # 创建优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    
    # 定义优化步骤
    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            loss = loss_function()
        grads = tape.gradient(loss, [var_params])
        optimizer.apply_gradients(zip(grads, [var_params]))
        return loss
    
    print("\n变分参数优化示例:")
    print("初始参数:", var_params.numpy())
    
    # 执行几个优化步骤
    for i in range(5):
        loss = train_step()
        print(f"步骤 {i+1}, 损失: {loss.numpy()[0]}, 参数: {var_params.numpy()[0]}")
else:
    print("TensorFlow Quantum未安装，无法执行量子变分电路和优化示例")

# 7. 实际应用示例
print("\n7. 实际应用示例")
if tfq_available:
    print("\n7.1 QAOA解决Max-Cut问题")
    print("量子近似优化算法 (QAOA) 是一种混合量子-经典算法")
    print("可用于解决组合优化问题，如Max-Cut")
    
    # 定义一个简单的图
    print("\n定义一个简单的图:")
    print("节点: A, B, C, D")
    print("边: (A,B), (B,C), (C,D), (D,A), (A,C)")
    
    # 对应的权重矩阵
    adjacency_matrix = np.array([
        [0, 1, 1, 1],  # A的连接
        [1, 0, 1, 0],  # B的连接
        [1, 1, 0, 1],  # C的连接
        [1, 0, 1, 0]   # D的连接
    ])
    
    # 创建量子比特
    qubits = [cirq.GridQubit(0, i) for i in range(4)]
    
    # QAOA电路参数
    gamma = sympy.Symbol('gamma')
    beta = sympy.Symbol('beta')
    
    # 创建QAOA电路
    def create_qaoa_circuit(qubits, adjacency_matrix, gamma, beta):
        """创建用于Max-Cut的QAOA电路"""
        n = len(qubits)
        circuit = cirq.Circuit()
        
        # 初始化为均匀叠加态
        circuit.append(cirq.H.on_each(qubits))
        
        # 问题Hamiltonian
        for i in range(n):
            for j in range(i+1, n):
                if adjacency_matrix[i, j] == 1:
                    # ZZ交互项对应边
                    circuit.append(cirq.ZZ(qubits[i], qubits[j])**(gamma))
        
        # 混合Hamiltonian
        for i in range(n):
            circuit.append(cirq.X(qubits[i])**(beta))
        
        return circuit
    
    # 创建QAOA电路
    qaoa_circuit = create_qaoa_circuit(qubits, adjacency_matrix, gamma, beta)
    
    print("\nQAOA电路 (单层):")
    print(qaoa_circuit)
    
    # 定义目标函数 - 计算所有边的切割值的和
    def max_cut_objective(adjacency_matrix, measurement):
        """计算Max-Cut目标函数值"""
        cut_value = 0
        n = len(measurement)
        for i in range(n):
            for j in range(i+1, n):
                if adjacency_matrix[i, j] == 1:
                    # 如果节点在不同子集中，边被切割
                    if measurement[i] != measurement[j]:
                        cut_value += 1
        return cut_value
    
    print("\n7.2 量子机器学习用于分类")
    print("量子机器学习可用于分类任务，如MNIST或其他数据集")
    print("由于空间限制，仅展示概念性代码")
    
    def create_dress_qnn_model(n_qubits, n_layers):
        """创建一个'穿着式'量子神经网络模型"""
        qubits = [cirq.GridQubit(0, i) for i in range(n_qubits)]
        
        # 数据编码电路
        data_symbols = sympy.symbols(f'x0:{n_qubits}')
        data_circuit = cirq.Circuit()
        for i, qubit in enumerate(qubits):
            data_circuit.append(cirq.rx(data_symbols[i]).on(qubit))
        
        # 变分电路
        model_symbols = []
        model_circuit = cirq.Circuit()
        
        # 多层变分电路
        for l in range(n_layers):
            # 第l层的参数
            layer_symbols = sympy.symbols(f'theta{l}_0:{2*n_qubits}')
            model_symbols.extend(layer_symbols)
            
            # 单量子比特旋转
            for i, qubit in enumerate(qubits):
                model_circuit.append(cirq.rx(layer_symbols[i]).on(qubit))
                model_circuit.append(cirq.rz(layer_symbols[i+n_qubits]).on(qubit))
            
            # 纠缠层
            for i in range(n_qubits - 1):
                model_circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
            model_circuit.append(cirq.CNOT(qubits[n_qubits-1], qubits[0]))
        
        # 测量第一个量子比特
        readout_operator = cirq.Z(qubits[0])
        
        return data_circuit, model_circuit, model_symbols, readout_operator
    
    print("\n'穿着式'量子神经网络参数:")
    print("量子比特数: 4")
    print("变分层数: 2")
    
    # 创建模型
    data_circuit, model_circuit, model_symbols, readout_operator = create_dress_qnn_model(4, 2)
    
    combined_circuit = data_circuit + model_circuit
    print("\n完整量子神经网络电路 (数据编码 + 变分):")
    print(combined_circuit)
else:
    print("TensorFlow Quantum未安装，无法执行实际应用示例")

# 8. TFQ的优势和限制
print("\n8. TFQ的优势和限制")

print("\n8.1 TFQ的优势:")
print("- 与TensorFlow生态系统无缝集成")
print("- 支持自动微分和梯度下降优化")
print("- 批处理和并行化能力强")
print("- 可以利用GPU加速经典部分的计算")
print("- 支持高级模型设计模式，如控制器-变分结构")

print("\n8.2 TFQ的限制:")
print("- 量子模拟在经典硬件上仍受到规模限制")
print("- 当前版本的编程模型和功能还在发展中")
print("- 与实际量子硬件的接口较为有限")
print("- 由于量子计算领域快速发展，API可能会变化")

# 9. 结论
print("\n9. 结论:")
print("- TensorFlow Quantum将量子计算和机器学习结合在一起")
print("- 它提供了一个实验量子机器学习算法的统一框架")
print("- 适用于研究不同的量子电路架构和训练方法")
print("- 为经典-量子混合模型提供了一个良好的起点")
print("- 随着量子硬件的发展，TFQ的实用性将进一步提高")

print("\n推荐学习路径:")
print("1. 掌握基本的Cirq和TensorFlow概念")
print("2. 学习如何设计和优化参数化量子电路")
print("3. 实验不同的数据编码策略")
print("4. 尝试构建混合量子-经典模型解决实际问题")
print("5. 探索变分算法如QAOA和VQE") 