#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IBM Qiskit框架学习 7：量子误差纠正和降噪
本文件详细介绍量子计算中的误差源、量子误差纠正码和降噪技术
"""

# 导入必要的库
from qiskit import QuantumCircuit, Aer, execute, transpile
from qiskit.visualization import plot_histogram
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error, thermal_relaxation_error
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.quantum_info import Statevector, state_fidelity
import matplotlib.pyplot as plt
import numpy as np

print("===== 量子误差纠正和降噪 =====")

# 1. 量子噪声的来源
print("\n1. 量子噪声的来源")
print("量子计算机中的噪声有多种来源，包括:")

print("\n主要噪声源:")
print("- 量子退相干 (Decoherence): 量子比特与环境的相互作用导致量子态的损失")
print("- 门操作误差 (Gate Errors): 量子门执行时的不精确性")
print("- 测量误差 (Measurement Errors): 读取量子态时的不准确性")
print("- 准备态误差 (State Preparation Errors): 初始化量子比特时的误差")
print("- 串扰 (Crosstalk): 操作一个量子比特对附近量子比特的意外影响")
print("- 热噪声 (Thermal Noise): 由非零温度导致的能量波动")

# 2. 在Qiskit中模拟噪声
print("\n2. 在Qiskit中模拟噪声")
print("Qiskit Aer允许我们模拟不同类型的量子噪声")

# 创建一个简单的贝尔态电路
bell_circuit = QuantumCircuit(2, 2)
bell_circuit.h(0)
bell_circuit.cx(0, 1)
bell_circuit.measure([0, 1], [0, 1])

print("贝尔态电路:")
print(bell_circuit.draw())

# 定义不同类型的噪声模型

# 1. 退相干噪声模型 (比特翻转错误)
print("\n2.1 退相干噪声模型 (比特翻转错误)")
bit_flip_noise_model = NoiseModel()
# 单量子比特比特翻转误差
p_bit_flip = 0.05  # 比特翻转的概率
bit_flip = pauli_error([('X', p_bit_flip), ('I', 1 - p_bit_flip)])
bit_flip_noise_model.add_all_qubit_quantum_error(bit_flip, ['u1', 'u2', 'u3'])

# 执行带噪声的模拟
simulator = Aer.get_backend('qasm_simulator')
job = execute(bell_circuit, simulator, noise_model=bit_flip_noise_model, shots=1024)
bit_flip_result = job.result()
bit_flip_counts = bit_flip_result.get_counts()

print("\n比特翻转噪声模拟结果:")
print(bit_flip_counts)

# 2. 去极化噪声模型 (多种错误)
print("\n2.2 去极化噪声模型 (多种错误)")
depol_noise_model = NoiseModel()
# 单量子比特去极化误差
p_depol_1 = 0.05  # 单量子比特去极化概率
error_1q = depolarizing_error(p_depol_1, 1)
depol_noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3'])

# 双量子比特去极化误差
p_depol_2 = 0.1  # 双量子比特去极化概率
error_2q = depolarizing_error(p_depol_2, 2)
depol_noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

# 执行带噪声的模拟
job = execute(bell_circuit, simulator, noise_model=depol_noise_model, shots=1024)
depol_result = job.result()
depol_counts = depol_result.get_counts()

print("\n去极化噪声模拟结果:")
print(depol_counts)

# 3. 热弛豫噪声模型 (T1/T2衰减)
print("\n2.3 热弛豫噪声模型 (T1/T2衰减)")
t1_noise_model = NoiseModel()
# 设置T1和T2参数
t1 = 50  # T1时间 (微秒)
t2 = 30  # T2时间 (微秒，注意T2 <= T1)
gate_time = 0.1  # 门操作时间 (微秒)

# 创建热弛豫误差
thermal_error = thermal_relaxation_error(t1, t2, gate_time)
t1_noise_model.add_all_qubit_quantum_error(thermal_error, ['u1', 'u2', 'u3'])

# CNOT门的热弛豫误差 (为简化，我们使用相同的参数)
cx_gate_time = 0.3  # CNOT门时间
cx_thermal_error = thermal_relaxation_error(t1, t2, cx_gate_time, 2)
t1_noise_model.add_all_qubit_quantum_error(cx_thermal_error, ['cx'])

# 执行带噪声的模拟
job = execute(bell_circuit, simulator, noise_model=t1_noise_model, shots=1024)
t1_result = job.result()
t1_counts = t1_result.get_counts()

print("\n热弛豫噪声模拟结果:")
print(t1_counts)

# 可视化不同噪声模型的结果
print("\n不同噪声模型结果对比 (图像将保存到文件):")
fig = plot_histogram([bit_flip_counts, depol_counts, t1_counts],
                      legend=['比特翻转', '去极化', '热弛豫'])
fig.savefig('noise_models_comparison.png')
plt.close(fig)

# 3. 测量误差缓解
print("\n3. 测量误差缓解")
print("测量误差是量子电路噪声的主要来源之一，可以通过校准进行缓解")

# 创建测量误差模型
meas_noise_model = NoiseModel()
# 添加测量误差 (0->1的概率为0.1，1->0的概率为0.05)
p01 = 0.1  # |0⟩错误测量为|1⟩的概率
p10 = 0.05  # |1⟩错误测量为|0⟩的概率

# 使用完全测量误差校准和过滤
# 创建校准电路
qc = QuantumCircuit(2)
meas_calibs, state_labels = complete_meas_cal(qr=qc.qregs[0], circlabel='mcal')

# 对校准电路应用测量误差
for i in range(len(meas_calibs)):
    meas_calibs[i].measure_all()

# 模拟带测量误差的校准电路执行
noise_model = NoiseModel()
# 对每个量子比特添加自定义的读取误差
for qubit in range(2):
    read_err = pauli_error([('X', p01), ('I', 1 - p01)])  # |0⟩ -> |1⟩ 错误
    noise_model.add_quantum_error(read_err, ['measure'], [qubit])
    read_err = pauli_error([('X', p10), ('I', 1 - p10)])  # |1⟩ -> |0⟩ 错误
    noise_model.add_quantum_error(read_err, ['reset'], [qubit])

print("\n测量校准电路:")
print(meas_calibs[0].draw())

# 执行校准电路
calib_job = execute(meas_calibs, simulator, 
                   shots=1024,
                   noise_model=noise_model)
calib_result = calib_job.result()

# 构建测量过滤器
meas_fitter = CompleteMeasFitter(calib_result, state_labels, circlabel='mcal')
meas_filter = meas_fitter.filter

print("\n测量误差校准矩阵:")
print(meas_fitter.cal_matrix)

# 应用相同的噪声模型到Bell状态电路
noisy_bell_job = execute(bell_circuit, simulator, 
                        shots=1024,
                        noise_model=noise_model)
noisy_bell_result = noisy_bell_job.result()
noisy_bell_counts = noisy_bell_result.get_counts()

# 应用测量误差缓解
mitigated_result = meas_filter.apply(noisy_bell_result)
mitigated_counts = mitigated_result.get_counts()

print("\n有噪声的Bell状态测量结果:")
print(noisy_bell_counts)
print("\n误差缓解后的Bell状态测量结果:")
print(mitigated_counts)

# 可视化误差缓解前后的结果
print("\n测量误差缓解前后对比 (图像将保存到文件):")
fig = plot_histogram([noisy_bell_counts, mitigated_counts],
                      legend=['有噪声', '误差缓解后'])
fig.savefig('measurement_error_mitigation.png')
plt.close(fig)

# 4. 量子误差纠正码
print("\n4. 量子误差纠正码")
print("量子纠错码是抵抗量子噪声的关键技术")

# 4.1 比特翻转码
print("\n4.1 比特翻转码")
print("比特翻转码可以纠正X错误 (比特翻转)")

def bit_flip_code_encode():
    """比特翻转码编码电路"""
    qc = QuantumCircuit(3, 1)
    
    # 将第一个量子比特编码到逻辑态
    qc.x(0)  # 将第一个量子比特初始化为|1⟩
    
    # 使用CNOT门将状态复制到其他量子比特
    qc.cx(0, 1)
    qc.cx(0, 2)
    
    return qc

def bit_flip_code_correct():
    """比特翻转码纠错电路"""
    qc = QuantumCircuit(3, 1)
    
    # 使用辅助量子比特执行多数投票
    qc.cx(0, 1)  # 第一个和第二个比特的奇偶性
    qc.cx(0, 2)  # 第一个和第三个比特的奇偶性
    
    # 如果第二个和第三个量子比特都是1，则第一个量子比特有错误
    qc.ccx(1, 2, 0)  # 双控制X门，如果辅助比特都是1，则翻转目标比特
    
    # 重置辅助量子比特
    qc.cx(0, 1)
    qc.cx(0, 2)
    
    # 测量结果
    qc.measure(0, 0)
    
    return qc

# 创建完整的比特翻转码电路
def complete_bit_flip_code(error_qubit=None, error_type='X'):
    """完整的比特翻转码电路，包括编码、错误注入和纠错"""
    # 编码
    qc = bit_flip_code_encode()
    
    # 注入错误
    if error_qubit is not None:
        if error_type == 'X':
            qc.x(error_qubit)
        elif error_type == 'Z':
            qc.z(error_qubit)
    
    # 纠错
    qc = qc.compose(bit_flip_code_correct())
    
    return qc

# 无错误情况
no_error_circuit = complete_bit_flip_code()
print("比特翻转码 (无错误):")
print(no_error_circuit.draw())

# 对量子比特1注入X错误
x_error_circuit = complete_bit_flip_code(error_qubit=1, error_type='X')
print("\n比特翻转码 (量子比特1上的X错误):")
print(x_error_circuit.draw())

# 模拟电路
simulator = Aer.get_backend('qasm_simulator')
no_error_job = execute(no_error_circuit, simulator, shots=1024)
no_error_result = no_error_job.result()
no_error_counts = no_error_result.get_counts()

x_error_job = execute(x_error_circuit, simulator, shots=1024)
x_error_result = x_error_job.result()
x_error_counts = x_error_result.get_counts()

print("\n无错误结果:")
print(no_error_counts)
print("\nX错误纠正后结果:")
print(x_error_counts)

# 4.2 相位翻转码
print("\n4.2 相位翻转码")
print("相位翻转码可以纠正Z错误 (相位翻转)")

def phase_flip_code_encode():
    """相位翻转码编码电路"""
    qc = QuantumCircuit(3, 1)
    
    # 将第一个量子比特初始化为|+⟩
    qc.x(0)  # 先初始化为|1⟩
    qc.h(0)  # 变为|+⟩或|-⟩
    
    # 创建纠缠态
    qc.h(1)
    qc.h(2)
    qc.cx(0, 1)
    qc.cx(0, 2)
    
    return qc

def phase_flip_code_correct():
    """相位翻转码纠错电路"""
    qc = QuantumCircuit(3, 1)
    
    # 对所有量子比特应用Hadamard门
    qc.h(0)
    qc.h(1)
    qc.h(2)
    
    # 使用辅助量子比特执行相位奇偶校验
    qc.cx(0, 1)
    qc.cx(0, 2)
    
    # 纠正相位错误
    qc.ccx(1, 2, 0)
    
    # 重置辅助量子比特
    qc.cx(0, 1)
    qc.cx(0, 2)
    
    # 转回计算基
    qc.h(0)
    qc.h(1)
    qc.h(2)
    
    # 测量结果
    qc.measure(0, 0)
    
    return qc

# 创建完整的相位翻转码电路
def complete_phase_flip_code(error_qubit=None, error_type='Z'):
    """完整的相位翻转码电路，包括编码、错误注入和纠错"""
    # 编码
    qc = phase_flip_code_encode()
    
    # 注入错误
    if error_qubit is not None:
        if error_type == 'X':
            qc.x(error_qubit)
        elif error_type == 'Z':
            qc.z(error_qubit)
    
    # 纠错
    qc = qc.compose(phase_flip_code_correct())
    
    return qc

# 无错误情况
no_error_phase_circuit = complete_phase_flip_code()
print("相位翻转码 (无错误):")
print(no_error_phase_circuit.draw())

# 对量子比特1注入Z错误
z_error_phase_circuit = complete_phase_flip_code(error_qubit=1, error_type='Z')
print("\n相位翻转码 (量子比特1上的Z错误):")
print(z_error_phase_circuit.draw())

# 模拟电路
no_error_phase_job = execute(no_error_phase_circuit, simulator, shots=1024)
no_error_phase_result = no_error_phase_job.result()
no_error_phase_counts = no_error_phase_result.get_counts()

z_error_phase_job = execute(z_error_phase_circuit, simulator, shots=1024)
z_error_phase_result = z_error_phase_job.result()
z_error_phase_counts = z_error_phase_result.get_counts()

print("\n无错误结果:")
print(no_error_phase_counts)
print("\nZ错误纠正后结果:")
print(z_error_phase_counts)

# 5. Shor码 - 通用量子纠错码
print("\n5. Shor码 - 通用量子纠错码")
print("Shor码是一种可以纠正任意单量子比特错误的量子纠错码")
print("完整的Shor码实现较为复杂，需要9个量子比特")
print("Shor码是通过结合比特翻转码和相位翻转码来实现的")
print("它可以纠正X、Z和Y错误，其中Y错误等同于X和Z错误的组合")

# 6. 动态解耦技术
print("\n6. 动态解耦技术")
print("动态解耦是一种降低退相干噪声影响的技术")

# 实现简单的动态解耦序列
def spin_echo_sequence():
    """自旋回波序列，可减轻退相干影响"""
    qc = QuantumCircuit(1, 1)
    
    # 创建叠加态
    qc.h(0)
    
    # 等待一段时间 (这里用身份门替代)
    qc.id(0)
    
    # 插入X门，翻转相位
    qc.x(0)
    
    # 再等待相同的时间
    qc.id(0)
    
    # 再次应用X门，回到原始状态
    qc.x(0)
    
    # 测量
    qc.h(0)
    qc.measure(0, 0)
    
    return qc

# 创建自旋回波电路
spin_echo_circuit = spin_echo_sequence()
print("自旋回波序列:")
print(spin_echo_circuit.draw())

# 带噪声和不带自旋回波的对比
def no_decoupling_sequence():
    """没有解耦的参考电路"""
    qc = QuantumCircuit(1, 1)
    
    # 创建叠加态
    qc.h(0)
    
    # 等待一段时间 (用多个身份门替代)
    qc.id(0)
    qc.id(0)
    qc.id(0)
    qc.id(0)
    
    # 测量
    qc.h(0)
    qc.measure(0, 0)
    
    return qc

# 创建不带解耦的电路
no_decoupling_circuit = no_decoupling_sequence()
print("\n没有解耦的参考电路:")
print(no_decoupling_circuit.draw())

# 使用热弛豫模型模拟退相干
t1 = 20  # T1时间
t2 = 10  # T2时间 (T2 <= T1)
gate_time = 1  # 每个门的时间

# 创建噪声模型
decoh_noise_model = NoiseModel()
thermal_err = thermal_relaxation_error(t1, t2, gate_time)
decoh_noise_model.add_all_qubit_quantum_error(thermal_err, ['id', 'u1', 'u2', 'u3'])

# 模拟带噪声的电路
spin_echo_job = execute(spin_echo_circuit, simulator, 
                      noise_model=decoh_noise_model, shots=1024)
spin_echo_result = spin_echo_job.result()
spin_echo_counts = spin_echo_result.get_counts()

no_decoupling_job = execute(no_decoupling_circuit, simulator, 
                          noise_model=decoh_noise_model, shots=1024)
no_decoupling_result = no_decoupling_job.result()
no_decoupling_counts = no_decoupling_result.get_counts()

print("\n带自旋回波的结果:")
print(spin_echo_counts)
print("\n没有解耦的结果:")
print(no_decoupling_counts)

# 可视化比较
print("\n动态解耦效果对比 (图像将保存到文件):")
fig = plot_histogram([spin_echo_counts, no_decoupling_counts],
                      legend=['自旋回波', '无解耦'])
fig.savefig('dynamic_decoupling_comparison.png')
plt.close(fig)

# 7. 其他降噪技术
print("\n7. 其他降噪技术")
print("除了上述方法，还有其他几种降噪和错误缓解技术:")

print("\n- 零噪声外推法 (Zero-Noise Extrapolation):")
print("  通过在不同噪声水平下运行电路并外推到零噪声点来减轻误差")

print("\n- 概率错误消除 (Probabilistic Error Cancellation):")
print("  通过添加额外的门操作来消除特定噪声的效果")

print("\n- 变分量子误差缓解 (Variational Quantum Error Mitigation):")
print("  使用变分方法优化电路参数，使其对噪声更加稳健")

print("\n- 量子重启 (Quantum Restart):")
print("  在出现错误时重新开始计算，适用于有中间测量的电路")

# 8. 总结
print("\n8. 总结")
print("1. 量子计算机面临多种噪声源，包括退相干、门误差和测量误差")
print("2. Qiskit提供了丰富的工具来模拟各种噪声模型")
print("3. 测量误差缓解是最容易实现的错误缓解技术之一")
print("4. 量子纠错码如比特翻转码和相位翻转码可以纠正特定类型的错误")
print("5. Shor码等通用量子纠错码可以纠正任意单量子比特错误")
print("6. 动态解耦等技术可以减轻退相干噪声的影响")

print("\n下一步学习:")
print("- 实现更复杂的量子纠错码，如Steane码和表面码")
print("- 学习容错量子计算的基本原理")
print("- 研究针对特定量子硬件优化的误差缓解策略")
print("- 探索量子纠错码与量子算法的结合") 