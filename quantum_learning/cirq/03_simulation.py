#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Google Cirq框架学习 3：量子模拟与测量
本文件详细介绍Cirq中的量子模拟器类型、测量方法和结果分析
"""

import cirq
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

print("===== Cirq中的量子模拟与测量 =====")

# 1. Cirq中的模拟器类型
print("\n1. Cirq中的模拟器类型")

# 1.1 状态向量模拟器
print("\n1.1 状态向量模拟器（Statevector Simulator）")
print("状态向量模拟器跟踪量子系统的完整状态向量，适用于小型电路的精确模拟")

# 创建一个简单的Bell状态电路
q0, q1 = cirq.LineQubit.range(2)
bell_circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1)
)
print("Bell状态电路:")
print(bell_circuit)

# 创建状态向量模拟器
sv_simulator = cirq.Simulator()
print("状态向量模拟器创建完成:", sv_simulator)

# 获取最终状态向量
result = sv_simulator.simulate(bell_circuit)
print("\n最终状态向量:")
print(result.final_state_vector)

# 以更友好的方式显示状态向量
def print_state(state, qubits, decimals=3):
    """以量子态表示法打印状态向量"""
    n_qubits = len(qubits)
    state_dict = {}
    for i, amplitude in enumerate(state):
        if abs(amplitude) > 1e-6:  # 忽略非常小的振幅
            # 转换整数索引为二进制形式
            binary = format(i, f'0{n_qubits}b')
            # 创建态表示
            label = '|' + binary + '⟩'
            state_dict[label] = amplitude
    
    # 打印状态
    print("状态向量（ket形式）:")
    for label, amplitude in state_dict.items():
        real = np.real(amplitude)
        imag = np.imag(amplitude)
        if abs(imag) < 1e-10:  # 实数
            print(f"  {real:.{decimals}f} {label}")
        else:  # 复数
            sign = '+' if imag >= 0 else ''
            print(f"  {real:.{decimals}f}{sign}{imag:.{decimals}f}i {label}")

print_state(result.final_state_vector, [q0, q1])

# 1.2 密度矩阵模拟器
print("\n1.2 密度矩阵模拟器（Density Matrix Simulator）")
print("密度矩阵模拟器跟踪量子系统的密度矩阵，可以模拟混合态和开放系统")

# 创建密度矩阵模拟器
dm_simulator = cirq.DensityMatrixSimulator()
print("密度矩阵模拟器创建完成:", dm_simulator)

# 模拟Bell状态电路
dm_result = dm_simulator.simulate(bell_circuit)
print("\n最终密度矩阵:")
print(dm_result.final_density_matrix)

# 1.3 Clifford模拟器
print("\n1.3 Clifford模拟器")
print("Clifford模拟器专门模拟仅包含Clifford门的电路，效率更高")

# 创建仅包含Clifford门的电路
clifford_circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.S(q1)
)
print("Clifford电路:")
print(clifford_circuit)

# 创建Clifford模拟器
clifford_simulator = cirq.CliffordSimulator()
print("Clifford模拟器创建完成:", clifford_simulator)

# 模拟Clifford电路
clifford_result = clifford_simulator.simulate(clifford_circuit)
print("\nClifford模拟结果:")
print(clifford_result)

# 1.4 分解模拟器（ZX计算）
print("\n1.4 分解模拟器（ZX计算）")
try:
    # 提示：这需要安装cirq-core[contrib]
    from cirq.contrib import routing
    # 使用ZX计算进行电路分解和模拟...
    print("这需要安装ZX计算相关的包")
except ImportError:
    print("这需要安装额外的ZX计算相关的包，在此不演示")

# 2. 量子测量和采样
print("\n2. 量子测量和采样")

# 2.1 在电路中添加测量
print("\n2.1 在电路中添加测量")
q0, q1 = cirq.LineQubit.range(2)
circuit_with_measurement = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    # 添加测量，并指定结果的键
    cirq.measure(q0, q1, key='bell_result')
)
print("带测量的电路:")
print(circuit_with_measurement)

# 2.2 运行电路并获取多次测量结果
print("\n2.2 运行电路并获取多次测量结果")
simulator = cirq.Simulator()
repetitions = 1000  # 测量次数

# 运行多次并收集结果
results = simulator.run(circuit_with_measurement, repetitions=repetitions)
print(f"执行了 {repetitions} 次测量")

# 获取特定键的结果
bell_results = results.measurements['bell_result']
print(f"测量结果形状: {bell_results.shape}")  # 形状是 (repetitions, 2)
print("前10次测量结果:")
print(bell_results[:10])

# 统计结果的频率
# 将每一行的二进制数组合成一个十进制结果
decimal_results = []
for measurement in bell_results:
    # 将二进制数组转换为字符串然后为十进制数
    result_str = ''.join(str(int(bit)) for bit in measurement)
    decimal_results.append(int(result_str, 2))

# 使用Counter计算频率
counter = Counter(decimal_results)
print("\n结果频率分布:")
for result, count in sorted(counter.items()):
    binary = format(result, f'0{len(circuit_with_measurement.all_qubits())}b')
    probability = count / repetitions
    print(f"|{binary}⟩: {count} 次 ({probability:.4f})")

# 2.3 可视化测量结果
print("\n2.3 可视化测量结果")
# 绘制测量结果的条形图
plt.figure(figsize=(10, 6))
labels = [format(result, f'0{len(circuit_with_measurement.all_qubits())}b') for result in sorted(counter.keys())]
values = [counter[result] / repetitions for result in sorted(counter.keys())]

plt.bar(labels, values)
plt.xlabel('测量结果')
plt.ylabel('概率')
plt.title(f'Bell状态 {repetitions} 次测量的概率分布')
plt.ylim(0, 1)
# 保存图片而不是显示，因为在终端环境中不能显示
plt.savefig('bell_state_measurement.png')
print("测量结果图表已保存为：bell_state_measurement.png")

# 3. 部分测量和中间测量
print("\n3. 部分测量和中间测量")

# 3.1 部分量子比特的测量
print("\n3.1 部分量子比特的测量")
q0, q1, q2 = cirq.LineQubit.range(3)
partial_circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.H(q1),
    cirq.CNOT(q0, q2),
    # 只测量q0
    cirq.measure(q0, key='q0_result'),
    # 在q1和q2上继续操作
    cirq.CNOT(q1, q2),
    # 最后测量q1和q2
    cirq.measure(q1, q2, key='q1q2_result')
)
print("部分测量电路:")
print(partial_circuit)

# 运行部分测量电路
partial_results = simulator.run(partial_circuit, repetitions=100)
print("\n部分测量结果:")
print("q0测量结果:", partial_results.measurements['q0_result'][:5])
print("q1和q2测量结果:", partial_results.measurements['q1q2_result'][:5])

# 3.2 测量和反馈
print("\n3.2 测量和反馈")
print("注意：直接的测量反馈在Cirq中不如在Qiskit中直观，但可以通过有条件的操作实现")

# 创建一个模拟量子隐形传态的电路
q0, q1, q2 = cirq.LineQubit.range(3)

# 准备要传送的状态（q0上的任意状态）
theta, phi = np.pi/4, np.pi/3
teleport_circuit = cirq.Circuit(
    # 准备要传送的状态
    cirq.rx(theta).on(q0),
    cirq.rz(phi).on(q0),
    
    # 创建Bell对（q1和q2）
    cirq.H(q1),
    cirq.CNOT(q1, q2),
    
    # 纠缠源量子比特和Bell对的一半
    cirq.CNOT(q0, q1),
    cirq.H(q0),
    
    # 测量q0和q1
    cirq.measure(q0, key='m0'),
    cirq.measure(q1, key='m1'),
)

print("量子隐形传态电路 (部分):")
print(teleport_circuit)

# 在真实设备上，可以基于测量结果应用门
# 在模拟中，我们可以使用后选择来查看不同测量结果下的最终状态
print("\n注：在真实设备上，我们可以基于测量结果应用门")
print("在这个模拟中，我们省略了基于测量结果的反馈操作")

# 4. 非理想模拟：噪声和退相干
print("\n4. 非理想模拟：噪声和退相干")

# 4.1 在门上添加噪声
print("\n4.1 在门上添加噪声")
q0, q1 = cirq.LineQubit.range(2)

# 创建理想电路
ideal_circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.measure(q0, q1, key='result')
)
print("理想电路:")
print(ideal_circuit)

# 创建带噪声的电路
noise_level = 0.05  # 5%的噪声
noisy_circuit = cirq.Circuit()

# 添加带噪声的H门
noisy_h = cirq.H.on(q0).with_noise(cirq.depolarize(noise_level))
noisy_circuit.append(noisy_h)

# 添加带噪声的CNOT门
noisy_cnot = cirq.CNOT.on(q0, q1).with_noise(cirq.depolarize(noise_level))
noisy_circuit.append(noisy_cnot)

# 添加测量
noisy_circuit.append(cirq.measure(q0, q1, key='result'))

print("\n带噪声的电路:")
print(noisy_circuit)

# 4.2 运行带噪声的电路
print("\n4.2 运行带噪声的电路，并与理想电路比较")
# 运行理想电路
ideal_results = simulator.run(ideal_circuit, repetitions=1000)
ideal_counter = Counter()
for bits in ideal_results.measurements['result']:
    result_str = ''.join(str(int(bit)) for bit in bits)
    ideal_counter[result_str] += 1

# 运行带噪声的电路
noisy_results = simulator.run(noisy_circuit, repetitions=1000)
noisy_counter = Counter()
for bits in noisy_results.measurements['result']:
    result_str = ''.join(str(int(bit)) for bit in bits)
    noisy_counter[result_str] += 1

# 比较结果
print("\n理想电路结果分布:")
for result, count in sorted(ideal_counter.items()):
    print(f"|{result}⟩: {count/1000:.4f}")

print("\n带噪声电路结果分布:")
for result, count in sorted(noisy_counter.items()):
    print(f"|{result}⟩: {count/1000:.4f}")

# 4.3 使用噪声模型
print("\n4.3 使用噪声模型")
try:
    # 创建一个噪声模型
    # 这需要特定版本的cirq或额外的库
    noise_model = cirq.NoiseModel.from_noise_model_like(
        cirq.ConstantQubitNoiseModel(cirq.depolarize(0.01))
    )
    # 使用噪声模型创建带噪声的模拟器
    noisy_simulator = cirq.Simulator(noise=noise_model)
    print("已创建带噪声模型的模拟器")
except Exception as e:
    print(f"创建噪声模型需要特定版本的cirq: {str(e)}")

# 5. 高级模拟特性
print("\n5. 高级模拟特性")

# 5.1 梯度计算和参数优化
print("\n5.1 梯度计算和参数优化")
theta = sympy.Symbol('θ')

# 创建参数化电路
q = cirq.LineQubit(0)
param_circuit = cirq.Circuit(
    cirq.rx(theta).on(q),
    cirq.measure(q, key='m')
)

# 定义一个函数，计算特定角度下的期望值
def expectation(angle_value):
    # 绑定参数
    bound_circuit = cirq.resolve_parameters(
        param_circuit, {theta: angle_value}
    )
    # 运行模拟
    result = simulator.run(bound_circuit, repetitions=1000)
    # 计算期望值 (|0⟩概率 - |1⟩概率)
    measurements = result.measurements['m']
    zeros = sum(1 for m in measurements if m[0] == 0)
    ones = sum(1 for m in measurements if m[0] == 1)
    return (zeros - ones) / 1000

# 计算不同角度的期望值
angles = np.linspace(0, 2*np.pi, 20)
expectations = [expectation(angle) for angle in angles]

# 绘制期望值曲线
plt.figure(figsize=(10, 6))
plt.plot(angles, expectations, 'o-')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.xlabel('角度θ')
plt.ylabel('期望值 ⟨Z⟩')
plt.title('单量子比特旋转的期望值')
plt.grid(True)
plt.savefig('expectation_values.png')
print("期望值曲线已保存为：expectation_values.png")

# 5.2 特定组件期望值
print("\n5.2 特定组件期望值")
q0, q1 = cirq.LineQubit.range(2)
bell_circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1)
)

# 计算各种泡利矩阵的期望值
pauli_observables = [
    cirq.Z(q0),                  # 测量q0的Z分量
    cirq.Z(q1),                  # 测量q1的Z分量 
    cirq.Z(q0) * cirq.Z(q1),     # 测量ZZ关联
    cirq.X(q0) * cirq.X(q1),     # 测量XX关联
    cirq.Y(q0) * cirq.Y(q1)      # 测量YY关联
]

print("Bell态的各种观测量期望值:")
for observable in pauli_observables:
    # 使用状态向量模拟器计算期望值
    result = sv_simulator.simulate_expectation_values(
        bell_circuit, observables=[observable]
    )
    print(f"⟨{observable}⟩ = {result[0]:.4f}")

print("\n总结:")
print("1. Cirq提供多种类型的模拟器，适用于不同的模拟需求")
print("2. 量子测量可以添加到电路中，并可以多次运行收集统计结果")
print("3. 可以通过添加噪声模型来模拟真实量子设备上的退相干效应")
print("4. 高级模拟功能支持参数化电路的优化和梯度计算")
print("5. 可以计算各种观测量的期望值，帮助理解量子态的性质") 