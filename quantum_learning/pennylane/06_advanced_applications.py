#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Xanadu PennyLane框架学习 6：高级应用
本文件详细介绍PennyLane在量子化学、量子金融和量子优化等领域的高级应用
"""

# 导入必要的库
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# 导入中文字体支持
try:
    from mpl_zhfonts import set_chinese_font
    set_chinese_font()
    print("已启用中文字体支持")
except ImportError:
    print("警告: 未找到中文字体支持模块，图表中的中文可能无法正确显示")

# 尝试导入专业领域相关的库
try:
    import openfermion
    from openfermionpennylane import qchem
    OPENFERMION_AVAILABLE = True
except ImportError:
    OPENFERMION_AVAILABLE = False
    print("未安装OpenFermion和PennyLane-Chemistry，某些量子化学示例将无法运行")

print("===== PennyLane高级应用 =====")

# 检查PennyLane版本
print(f"PennyLane版本: {qml.__version__}")

# 1. 量子化学应用
print("\n1. 量子化学应用")
print("量子计算机被认为是模拟量子系统的理想工具")
print("PennyLane提供了多种用于量子化学模拟的功能")

# 1.1 分子哈密顿量
print("\n1.1 分子哈密顿量")
print("电子哈密顿量描述了分子中电子的相互作用")

# 创建一个简化的氢分子哈密顿量（用于演示）
def simplified_hydrogen_hamiltonian():
    """创建一个简化的氢分子哈密顿量"""
    # 只考虑最简单的项
    coefficients = np.array([0.7, 0.2, 0.2, 0.15, 0.08])
    observables = [
        qml.Identity(0) @ qml.Identity(1),
        qml.PauliZ(0),
        qml.PauliZ(1),
        qml.PauliZ(0) @ qml.PauliZ(1),
        qml.PauliX(0) @ qml.PauliX(1)
    ]
    
    # 创建哈密顿量
    H = qml.Hamiltonian(coefficients, observables)
    return H

H_simplified = simplified_hydrogen_hamiltonian()
print(f"\n简化的H2分子哈密顿量:\n{H_simplified}")

# 如果有OpenFermion，则演示更真实的分子构建
if OPENFERMION_AVAILABLE:
    print("\n使用OpenFermion构建真实分子哈密顿量（代码示例）:")
    print("""
    # 定义分子几何结构
    symbols = ["H", "H"]
    coordinates = np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])
    
    # 计算电子积分
    h, coulomb = qchem.molecular_hamiltonian(
        symbols, coordinates, basis="sto-3g"
    )
    
    # 转换为泡利表示的哈密顿量
    hamiltonian = qchem.decompose_hamiltonian(h, coulomb)
    print(hamiltonian)
    """)
else:
    print("\n要使用真实分子，请安装OpenFermion和PennyLane-Chemistry插件:")
    print("pip install openfermion openfermion-pennylane")

# 1.2 变分量子特征值求解器 (VQE)
print("\n1.2 变分量子特征值求解器 (VQE)")
print("VQE是寻找哈密顿量基态能量的量子算法")

# 创建量子设备
dev_vqe = qml.device("default.qubit", wires=2)

# 定义VQE电路
@qml.qnode(dev_vqe)
def vqe_circuit(params, hamiltonian):
    # 准备试探态
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(params[2], wires=0)
    qml.RY(params[3], wires=1)
    
    # 返回哈密顿量的期望值
    return qml.expval(hamiltonian)

# 执行VQE优化
def run_vqe(hamiltonian, init_params=None, steps=100):
    """运行VQE优化"""
    # 初始化参数
    if init_params is None:
        init_params = np.random.uniform(0, np.pi, 4)
    
    # 定义成本函数
    def cost(params):
        return vqe_circuit(params, hamiltonian)
    
    # 使用优化器
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    
    # 优化过程
    params = init_params
    energies = [cost(params)]
    
    for i in range(steps):
        params = opt.step(cost, params)
        energies.append(cost(params))
        
        if (i + 1) % 20 == 0:
            print(f"步骤 {i+1}: 能量 = {energies[-1]:.6f}")
    
    return params, energies

# 运行VQE（只执行少量步骤用于演示）
init_params = np.array([0.1, 0.2, 0.3, 0.4])
final_params, energies = run_vqe(H_simplified, init_params, steps=60)

print(f"\nVQE优化后的能量: {energies[-1]:.6f}")
print(f"理论基态能量: -1.0")

# 绘制能量收敛过程
plt.figure(figsize=(8, 5))
plt.plot(energies)
plt.xlabel('优化步骤')
plt.ylabel('能量')
plt.title('VQE能量收敛过程')
plt.grid(True)
plt.savefig('vqe_convergence.png')
plt.close()

print("已绘制VQE能量收敛曲线并保存为'vqe_convergence.png'")

# 1.3 激发态计算 - 量子亚空间扩展 (SSVQE)
print("\n1.3 激发态计算 - 量子亚空间扩展")
print("除了基态，我们还可以使用量子算法计算激发态")

# 定义SSVQE电路
@qml.qnode(dev_vqe)
def ssvqe_circuit(params, init_state, hamiltonian):
    # 准备初始态
    if init_state == 1:
        qml.PauliX(wires=0)
    elif init_state == 2:
        qml.PauliX(wires=1)
    elif init_state == 3:
        qml.PauliX(wires=0)
        qml.PauliX(wires=1)
    
    # 应用变分电路
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(params[2], wires=0)
    qml.RY(params[3], wires=1)
    
    # 返回哈密顿量的期望值
    return qml.expval(hamiltonian)

# 1.4 量子动力学模拟
print("\n1.4 量子动力学模拟")
print("量子计算机可以有效地模拟量子系统的时间演化")

# 创建用于时间演化的量子设备
dev_dyn = qml.device("default.qubit", wires=2)

# 定义量子动力学模拟电路
@qml.qnode(dev_dyn)
def time_evolution_circuit(time, hamiltonian):
    """模拟哈密顿量驱动下的量子态时间演化"""
    # 初始态：|+⟩|+⟩
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    
    # 时间演化 - 使用近似时间演化
    qml.ApproxTimeEvolution(hamiltonian, time, 3)
    
    # 返回测量结果
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

# 计算不同时间点的演化结果
times = np.linspace(0, 10, 10)  # 只使用10个时间点
expectations = []

# 计算时间点的演化结果
for t in times:
    z0, z1 = time_evolution_circuit(t, H_simplified)
    expectations.append([z0, z1])

expectations = np.array(expectations)

# 绘制时间演化
plt.figure(figsize=(8, 5))
plt.plot(times, expectations[:, 0], 'b-', label='<Z₀>')
plt.plot(times, expectations[:, 1], 'r-', label='<Z₁>')
plt.xlabel('时间')
plt.ylabel('期望值')
plt.title('量子比特在H2有效哈密顿量下的时间演化')
plt.legend()
plt.grid(True)
plt.savefig('time_evolution.png')
plt.close()

print("已绘制时间演化图并保存为'time_evolution.png'")

# 2. 量子金融应用
print("\n2. 量子金融应用")
print("量子计算在金融领域的应用包括优化投资组合、期权定价和风险分析")

# 2.1 投资组合优化
print("\n2.1 投资组合优化")
print("使用QAOA算法优化股票投资组合")

# 创建用于QAOA的量子设备
n_assets = 4  # 资产数量
dev_finance = qml.device("default.qubit", wires=n_assets)

# 生成示例数据：回报率和相关性矩阵
# 在真实应用中，这些数据来自市场分析
np.random.seed(42)
returns = np.random.uniform(0.05, 0.15, n_assets)  # 年回报率
cov_matrix = np.random.uniform(0, 0.1, (n_assets, n_assets))  # 初始随机矩阵
cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)  # 使其对称
np.fill_diagonal(cov_matrix, np.random.uniform(0.1, 0.2, n_assets))  # 对角线为方差

print(f"\n资产回报率: {returns}")
print(f"协方差矩阵:\n{cov_matrix}")

# 定义投资组合优化的成本哈密顿量
def portfolio_hamiltonian(returns, cov_matrix, risk_aversion=0.5):
    """创建投资组合优化的哈密顿量"""
    n_assets = len(returns)
    
    # 收益项（负号是因为我们要最大化收益）
    return_coeffs = [-risk_aversion * returns[i] for i in range(n_assets)]
    return_obs = [qml.PauliZ(i) for i in range(n_assets)]
    
    # 风险项
    risk_coeffs = []
    risk_obs = []
    
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            risk_coeffs.append((1 - risk_aversion) * cov_matrix[i, j])
            risk_obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
    
    # 组合为哈密顿量
    all_coeffs = return_coeffs + risk_coeffs
    all_obs = return_obs + risk_obs
    
    return qml.Hamiltonian(all_coeffs, all_obs)

# 构建投资组合哈密顿量
portfolio_H = portfolio_hamiltonian(returns, cov_matrix, risk_aversion=0.7)
print(f"\n投资组合优化哈密顿量:\n{portfolio_H}")

# 定义QAOA电路
@qml.qnode(dev_finance)
def qaoa_circuit(params, hamiltonian):
    """用于投资组合优化的QAOA电路"""
    # 准备均匀叠加态
    for i in range(n_assets):
        qml.Hadamard(wires=i)
    
    # QAOA层
    p = len(params) // 2  # QAOA的深度
    
    for i in range(p):
        # 问题哈密顿量演化
        qml.ApproxTimeEvolution(hamiltonian, params[i], 1)
        
        # 混合哈密顿量演化
        for j in range(n_assets):
            qml.RX(params[p + i], wires=j)
    
    # 返回哈密顿量的期望值
    return qml.expval(hamiltonian)

# 运行QAOA优化
def run_qaoa(hamiltonian, p=1, steps=100):
    """运行QAOA优化"""
    # 初始化参数（2*p个参数）
    init_params = np.random.uniform(0, 2*np.pi, 2*p)
    
    # 定义成本函数
    def cost(params):
        return qaoa_circuit(params, hamiltonian)
    
    # 使用优化器
    opt = qml.AdamOptimizer(stepsize=0.1)
    
    # 优化过程
    params = init_params
    costs = [cost(params)]
    
    for i in range(steps):
        params = opt.step(cost, params)
        costs.append(cost(params))
        
        if (i + 1) % 20 == 0:
            print(f"步骤 {i+1}: 成本 = {costs[-1]:.6f}")
    
    return params, costs

# 运行QAOA（少量步骤用于演示）
qaoa_params, qaoa_costs = run_qaoa(portfolio_H, p=1, steps=40)

print(f"\nQAOA优化后的投资组合成本: {qaoa_costs[-1]:.6f}")

# 2.2 蒙特卡洛定价
print("\n2.2 量子蒙特卡洛定价")
print("量子计算提供了加速蒙特卡洛模拟的潜力")

# 定义一个简单的量子振幅估计电路
n_eval = 3  # 需要3个量子比特进行评估
n_state = 1  # 1个量子比特表示状态
dev_amp = qml.device("default.qubit", wires=n_eval + n_state)

@qml.qnode(dev_amp)
def amplitude_estimation_circuit(theta):
    """简化的量子振幅估计电路"""
    # 准备态|1⟩，表示我们感兴趣的状态
    qml.PauliX(wires=n_eval)
    
    # 编码要估计的概率振幅（在实际应用中，这将是某种期权支付）
    qml.RY(2 * np.arcsin(np.sqrt(theta)), wires=n_eval)
    
    # 评估量子比特
    for i in range(n_eval):
        qml.Hadamard(wires=i)
    
    # 应用受控量子算子
    for i in range(n_eval):
        qml.ctrl(qml.PauliX, control=i)(wires=n_eval)
    
    # 应用QFT†
    qml.adjoint(qml.QFT)(wires=range(n_eval))
    
    # 返回测量概率
    return qml.probs(wires=range(n_eval))

# 估计一个示例概率
true_prob = 0.36
estimated_probs = amplitude_estimation_circuit(true_prob)

print(f"\n真实概率: {true_prob}")
print(f"量子振幅估计结果: {estimated_probs}")
print("振幅估计可用于期权定价，通过估计期权支付的期望值")

# 3. 量子优化问题
print("\n3. 量子优化问题")
print("量子计算在复杂优化问题上展现出潜力")

# 3.1 最大割问题
print("\n3.1 最大割问题")
print("最大割问题是一个NP难问题，适合用量子算法解决")

# 创建用于最大割的量子设备
n_nodes = 4  # 节点数量
dev_maxcut = qml.device("default.qubit", wires=n_nodes)

# 创建示例图
adjacency_matrix = np.array([
    [0, 1, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [0, 1, 1, 0]
])

print(f"\n图的邻接矩阵:\n{adjacency_matrix}")

# 定义最大割哈密顿量
def maxcut_hamiltonian(adjacency_matrix):
    """创建最大割问题的哈密顿量"""
    n_nodes = len(adjacency_matrix)
    coeffs = []
    obs = []
    
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if adjacency_matrix[i, j] == 1:
                coeffs.append(0.5)
                obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
                
                coeffs.append(-0.5)
                obs.append(qml.Identity(0))
    
    return qml.Hamiltonian(coeffs, obs)

# 构建最大割哈密顿量
maxcut_H = maxcut_hamiltonian(adjacency_matrix)
print(f"\n最大割问题哈密顿量:\n{maxcut_H}")

# 定义QAOA电路
@qml.qnode(dev_maxcut)
def maxcut_qaoa_circuit(params, hamiltonian):
    """用于最大割问题的QAOA电路"""
    # 准备均匀叠加态
    for i in range(n_nodes):
        qml.Hadamard(wires=i)
    
    # QAOA层
    p = len(params) // (2 * n_nodes)  # QAOA的深度
    
    for i in range(p):
        # 问题哈密顿量演化
        qml.ApproxTimeEvolution(hamiltonian, params[i], 1)
        
        # 混合哈密顿量演化
        for j in range(n_nodes):
            qml.RX(params[p + i*n_nodes + j], wires=j)
    
    # 返回哈密顿量的期望值
    return qml.expval(hamiltonian)

# 3.2 旅行商问题
print("\n3.2 旅行商问题")
print("旅行商问题是一个著名的组合优化问题")

# 创建城市距离矩阵
n_cities = 4
np.random.seed(123)
distances = np.random.uniform(1, 10, (n_cities, n_cities))
distances = (distances + distances.T) / 2  # 确保对称
np.fill_diagonal(distances, 0)  # 对角线为0

print(f"\n城市间距离矩阵:\n{distances}")

print("\n旅行商问题需要更复杂的编码和约束条件")
print("通常需要使用QUBO（二次无约束二进制优化）表示")
print("这超出了基础演示的范围，但方法类似于最大割问题")

# 4. 量子机器学习高级应用
print("\n4. 量子机器学习高级应用")
print("量子机器学习结合了量子计算和机器学习的优势")

# 4.1 量子卷积神经网络
print("\n4.1 量子卷积神经网络 (QCNN)")
print("QCNN结合了量子电路和卷积神经网络的思想")

# 创建用于QCNN的设备
n_qubits_qcnn = 8
dev_qcnn = qml.device("default.qubit", wires=n_qubits_qcnn)

# 定义卷积层
def quantum_conv_layer(params, wires):
    """量子卷积层"""
    # 应用旋转门
    for i, wire in enumerate(wires):
        qml.RY(params[i, 0], wires=wire)
        qml.RZ(params[i, 1], wires=wire)
    
    # 应用纠缠门连接相邻量子比特
    for i in range(len(wires) - 1):
        qml.CNOT(wires=[wires[i], wires[i+1]])
    
    # 闭合边界条件（可选）
    # qml.CNOT(wires=[wires[-1], wires[0]])

# 定义池化层
def quantum_pool_layer(params, wires):
    """量子池化层 - 将成对量子比特映射到单个量子比特"""
    for i in range(0, len(wires), 2):
        if i + 1 < len(wires):
            # 每两个量子比特参数化操作
            qml.RY(params[i//2, 0], wires=wires[i])
            qml.RY(params[i//2, 1], wires=wires[i+1])
            qml.CNOT(wires=[wires[i], wires[i+1]])
            qml.RY(params[i//2, 2], wires=wires[i])
            qml.RY(params[i//2, 3], wires=wires[i+1])

# 定义完整的QCNN结构
@qml.qnode(dev_qcnn)
def qcnn_circuit(params, features):
    """完整的QCNN电路"""
    # 编码输入特征
    qml.templates.AngleEmbedding(features, wires=range(n_qubits_qcnn))
    
    # 第一个卷积层
    quantum_conv_layer(params[0], wires=range(n_qubits_qcnn))
    
    # 第一个池化层 - 从8个量子比特到4个
    active_wires = list(range(n_qubits_qcnn))
    pooled_wires = active_wires[::2]  # 取偶数位置的量子比特
    quantum_pool_layer(params[1], wires=active_wires)
    active_wires = pooled_wires
    
    # 第二个卷积层
    quantum_conv_layer(params[2], wires=active_wires)
    
    # 第二个池化层 - 从4个量子比特到2个
    pooled_wires = active_wires[::2]
    quantum_pool_layer(params[3], wires=active_wires)
    active_wires = pooled_wires
    
    # 最后的卷积层
    quantum_conv_layer(params[4], wires=active_wires)
    
    # 简单的全连接层 - 测量第一个量子比特
    return qml.expval(qml.PauliZ(0))

# 创建随机参数
num_conv1_params = (n_qubits_qcnn, 2)
num_pool1_params = (n_qubits_qcnn // 2, 4)
num_conv2_params = (n_qubits_qcnn // 2, 2)
num_pool2_params = (n_qubits_qcnn // 4, 4)
num_conv3_params = (n_qubits_qcnn // 4, 2)

# 构建参数
params = [
    np.random.uniform(-np.pi, np.pi, num_conv1_params),
    np.random.uniform(-np.pi, np.pi, num_pool1_params),
    np.random.uniform(-np.pi, np.pi, num_conv2_params),
    np.random.uniform(-np.pi, np.pi, num_pool2_params),
    np.random.uniform(-np.pi, np.pi, num_conv3_params)
]

# 创建随机特征
features = np.random.uniform(-np.pi, np.pi, n_qubits_qcnn)

# 执行QCNN
result = qcnn_circuit(params, features)
print(f"\nQCNN输出: {result:.6f}")

# 4.2 量子生成对抗网络
print("\n4.2 量子生成对抗网络 (QGAN)")
print("QGAN结合了量子电路和GAN的思想")

print("\nQGAN通常包含:")
print("1. 量子生成器 - 创建量子态")
print("2. 量子或经典判别器 - 区分真实和生成的量子态")
print("3. 对抗训练过程")

print("\n一个完整的QGAN实现需要更复杂的训练循环，这里只展示基本结构")

# 5. 量子误差缓解技术
print("\n5. 量子误差缓解技术")
print("在NISQ时代，处理量子噪声至关重要")

# 5.1 噪声外推法
print("5.1 噪声外推法")
print("通过在不同噪声水平下执行电路，可以外推到零噪声结果")

# 创建一个支持噪声操作的设备
dev_noisy = qml.device('default.mixed', wires=1)

# 定义一个带噪声的量子电路
@qml.qnode(dev_noisy)
def noisy_circuit(params, noise_level=0.0):
    """带噪声的量子电路"""
    # 应用旋转门
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    
    # 添加去极化噪声
    if noise_level > 0:
        qml.DepolarizingChannel(noise_level, wires=0)
    
    # 测量
    return qml.expval(qml.PauliZ(0))

# 使用零噪声外推
def zero_noise_extrapolation(circuit, params):
    """零噪声外推方法"""
    # 使用不同的噪声级别
    noise_levels = [0.0, 0.1, 0.2, 0.3]
    results = []
    
    # 在不同噪声级别下运行电路
    for noise in noise_levels:
        res = circuit(params, noise_level=noise)
        results.append(res)
        print(f"噪声级别 {noise:.1f}: 结果 = {res:.6f}")
    
    # 使用线性回归外推到零噪声
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    X = np.array(noise_levels).reshape(-1, 1)
    y = np.array(results)
    model.fit(X, y)
    
    # 外推到零噪声
    extrapolated = model.predict(np.array([0.0]).reshape(-1, 1))[0]
    return extrapolated

# 生成随机参数并应用零噪声外推
params = np.random.normal(0, np.pi, 2)
zne_result = zero_noise_extrapolation(noisy_circuit, params)
print(f"零噪声外推结果: {zne_result:.6f}")

# 5.2 其他量子误差缓解技术
print("\n5.2 其他量子误差缓解技术")
print("- 概率误差消除法")
print("- 动态解耦技术")
print("- 量子纠错码")
print("- 后选择方法")
print("- 镀金方法")

# 6. 实际部署和量子云服务
print("\n6. 实际部署和量子云服务")
print("PennyLane可以与多个量子云服务集成")

print("\n支持的量子硬件提供商:")
print("- IBM Quantum")
print("- Amazon Braket")
print("- Microsoft Azure Quantum")
print("- Rigetti")
print("- Xanadu Cloud")

print("\n连接到真实量子设备的示例代码:")
print("""
# IBM Quantum 集成示例
# 需要先安装 pennylane-qiskit
# dev = qml.device('qiskit.ibmq', wires=2, backend='ibm_oslo', ibmqx_token="YOUR_TOKEN")

# Amazon Braket 集成示例
# 需要先安装 pennylane-braket 和设置 AWS 凭证
# dev = qml.device('braket.aws.qubit', wires=2, device_arn='arn:aws:braket:::device/quantum-simulator/amazon/sv1')

# 创建量子节点
@qml.qnode(dev)
def cloud_circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

# 执行云上的量子计算
result = cloud_circuit([0.1, 0.2])
""")

# 7. 总结
print("\n7. 总结")
print("1. 量子计算在化学、金融和优化等领域有广泛应用")
print("2. VQE是量子化学中最重要的量子算法之一")
print("3. 量子优化算法如QAOA可以解决组合优化问题")
print("4. 高级量子机器学习模型如QCNN和QGAN展现了量子计算的潜力")
print("5. 量子误差缓解技术对于在当前噪声设备上运行算法至关重要")
print("6. PennyLane与多个量子云服务的集成使实际部署成为可能")

print("\n下一步学习:")
print("- 深入探索特定领域的应用")
print("- 量子算法的伸缩性和资源分析")
print("- 更复杂的量子误差缓解和纠错技术") 