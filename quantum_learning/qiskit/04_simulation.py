#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IBM Qiskit框架学习 4：模拟和测量
本文件详细介绍Qiskit中的各种模拟器、测量方法和结果分析
"""

# 导入必要的库
from qiskit import QuantumCircuit, Aer, execute, transpile, assemble
from qiskit.visualization import plot_histogram, plot_bloch_multivector, plot_state_city
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit.result import marginal_counts
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator, UnitarySimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
import numpy as np
import matplotlib.pyplot as plt

print("===== Qiskit模拟和测量 =====")

# 1. Qiskit中的模拟器类型
print("\n1. Qiskit中的模拟器类型")
print("Qiskit提供多种类型的模拟器，适合不同的模拟需求:")

print("\n可用的Aer模拟器:")
for backend_name in Aer.backends():
    print(f"- {backend_name}")

# 2. 状态向量模拟器
print("\n2. 状态向量模拟器 (Statevector Simulator)")
print("此模拟器直接计算量子态的状态向量表示")

# 创建一个简单的叠加态电路
bell_circuit = QuantumCircuit(2)
bell_circuit.h(0)
bell_circuit.cx(0, 1)

# 使用状态向量模拟器
sv_sim = Aer.get_backend('statevector_simulator')
result = execute(bell_circuit, sv_sim).result()
statevector = result.get_statevector()

print("贝尔态电路:")
print(bell_circuit.draw())
print("\n贝尔态的状态向量:")
print(statevector)

# 可视化状态向量
print("\n状态向量可视化（图像将保存到文件）:")
fig = plot_bloch_multivector(statevector)
fig.savefig('statevector_bloch.png')
plt.close(fig)

fig = plot_state_city(statevector)
fig.savefig('statevector_city.png')
plt.close(fig)

# 直接创建和操作状态向量
print("\n直接创建和操作状态向量:")
# 创建|0⟩状态
psi = Statevector.from_label('0')
print(f"|0⟩状态向量: {psi}")

# 应用Hadamard门
psi = psi.evolve(bell_circuit)
print(f"应用Bell电路后的状态向量: {psi}")

# 3. QASM模拟器
print("\n3. QASM模拟器 (QASM Simulator)")
print("此模拟器执行量子电路的多次测量，模拟真实量子计算机的行为")

# 创建带测量的贝尔态电路
meas_bell = QuantumCircuit(2, 2)
meas_bell.h(0)
meas_bell.cx(0, 1)
meas_bell.measure([0, 1], [0, 1])

print("带测量的贝尔态电路:")
print(meas_bell.draw())

# 使用QASM模拟器执行
qasm_sim = Aer.get_backend('qasm_simulator')
shots = 1024
qasm_job = execute(meas_bell, qasm_sim, shots=shots)
qasm_result = qasm_job.result()
counts = qasm_result.get_counts()

print(f"\n用{shots}次测量得到的计数结果:")
print(counts)

# 可视化计数结果
print("\n计数结果的可视化（图像将保存到文件）:")
fig = plot_histogram(counts)
fig.savefig('qasm_counts_histogram.png')
plt.close(fig)

# 4. 酉模拟器
print("\n4. 酉模拟器 (Unitary Simulator)")
print("此模拟器计算电路的酉矩阵表示")

# 使用酉模拟器
unit_sim = Aer.get_backend('unitary_simulator')
unit_result = execute(bell_circuit, unit_sim).result()
unitary = unit_result.get_unitary()

print("贝尔态电路的酉矩阵表示:")
print(unitary)

# 5. 扩展的QASM模拟器功能
print("\n5. 扩展的QASM模拟器功能")
print("QASM模拟器提供了多种高级功能，如噪声模拟和保存额外数据")

# 创建一个更复杂的电路
complex_circuit = QuantumCircuit(3, 3)
complex_circuit.h(0)
complex_circuit.cx(0, 1)
complex_circuit.cx(0, 2)
complex_circuit.measure([0, 1, 2], [0, 1, 2])

print("GHZ状态电路:")
print(complex_circuit.draw())

# 使用高级选项运行QASM模拟器
aer_sim = Aer.get_backend('aer_simulator')
options = {
    "method": "statevector",
    "device": "CPU",
    "shots": 1024,
    "save_state": True  # 保存最终状态
}

aer_result = execute(complex_circuit, aer_sim, **options).result()
aer_counts = aer_result.get_counts()

print(f"\nGHZ状态的测量结果:")
print(aer_counts)

# 6. 部分测量和边缘计数
print("\n6. 部分测量和边缘计数")
print("可以对特定的量子比特子集进行测量或计算边缘分布")

# 创建一个3量子比特电路
three_qubits = QuantumCircuit(3, 3)
three_qubits.h(0)
three_qubits.cx(0, 1)
three_qubits.cx(1, 2)
three_qubits.measure([0, 1, 2], [0, 1, 2])

# 运行模拟器
result = execute(three_qubits, qasm_sim, shots=1024).result()
counts = result.get_counts()

print("3量子比特电路测量结果:")
print(counts)

# 计算0和1量子比特的边缘分布
marginal_01 = marginal_counts(counts, [0, 1])
print("\n量子比特0和1的边缘分布:")
print(marginal_01)

# 计算量子比特2的边缘分布
marginal_2 = marginal_counts(counts, [2])
print("\n量子比特2的边缘分布:")
print(marginal_2)

# 7. 密度矩阵模拟
print("\n7. 密度矩阵模拟")
print("密度矩阵可以表示混合状态，对于噪声模拟很有用")

# 创建一个纯态的密度矩阵
bell_statevector = Statevector.from_instruction(bell_circuit)
bell_dm = DensityMatrix(bell_statevector)

print("贝尔态的密度矩阵:")
print(bell_dm)

# 模拟退相干噪声
noisy_dm = bell_dm.evolve(pauli_error([('X', 0.1), ('I', 0.9)]), qargs=[0])
print("\n施加退相干噪声后的密度矩阵:")
print(noisy_dm)

# 8. 噪声模拟
print("\n8. 噪声模拟")
print("Qiskit Aer支持各种噪声模型，可以模拟真实量子计算机的噪声")

# 创建一个简单的噪声模型
noise_model = NoiseModel()

# 添加单量子比特去极化噪声
p1q = 0.05  # 单量子比特噪声概率
error1 = depolarizing_error(p1q, 1)
noise_model.add_all_qubit_quantum_error(error1, ['u1', 'u2', 'u3'])

# 添加双量子比特去极化噪声
p2q = 0.1  # 双量子比特噪声概率
error2 = depolarizing_error(p2q, 2)
noise_model.add_all_qubit_quantum_error(error2, ['cx'])

print("噪声模型:")
print(noise_model)

# 使用噪声模型运行电路
noisy_result = execute(meas_bell, 
                        qasm_sim, 
                        noise_model=noise_model, 
                        shots=1024).result()
noisy_counts = noisy_result.get_counts()

print("\n无噪声测量结果:")
print(counts)
print("\n有噪声测量结果:")
print(noisy_counts)

# 可视化对比
print("\n无噪声vs有噪声结果对比（图像将保存到文件）:")
fig = plot_histogram([counts, noisy_counts], 
                     legend=['无噪声', '有噪声'])
fig.savefig('noise_comparison.png')
plt.close(fig)

# 9. 测量误差缓解
print("\n9. 测量误差缓解")
print("Qiskit提供了测量误差缓解技术，可以减轻测量过程中的误差")

# 创建一个简单的电路用于校准
qr = QuantumCircuit(2)
meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='mcal')

# 运行校准电路
calibration_result = execute(meas_calibs, qasm_sim, 
                            shots=1024, 
                            noise_model=noise_model).result()

# 创建校准矩阵
meas_fitter = CompleteMeasFitter(calibration_result, state_labels, circlabel='mcal')
print("测量校准矩阵:")
print(meas_fitter.cal_matrix)

# 运行带噪声的电路
noisy_result = execute(meas_bell, qasm_sim, 
                      shots=1024, 
                      noise_model=noise_model).result()
noisy_counts = noisy_result.get_counts()

# 应用误差缓解
mitigated_result = meas_fitter.filter.apply(noisy_result)
mitigated_counts = mitigated_result.get_counts()

print("\n无噪声结果:")
print(counts)
print("\n噪声结果:")
print(noisy_counts)
print("\n误差缓解后结果:")
print(mitigated_counts)

# 10. 量子态层析
print("\n10. 量子态层析")
print("量子态层析是一种重构量子态的技术，通过多种基测量来估计量子态")

# 创建一个量子态
state_circuit = QuantumCircuit(2)
state_circuit.h(0)
state_circuit.cx(0, 1)

# 生成层析电路
qst_circuits = state_tomography_circuits(state_circuit, [0, 1])
print(f"层析需要{len(qst_circuits)}个电路")

# 执行层析电路
job = execute(qst_circuits, qasm_sim, shots=1024)
qst_result = job.result()

# 重构密度矩阵
tomo_fitter = StateTomographyFitter(qst_result, qst_circuits)
rho_fit = tomo_fitter.fit(method='lstsq')

print("\n通过层析重构的密度矩阵:")
print(rho_fit)

# 理论密度矩阵
expected_rho = DensityMatrix.from_instruction(state_circuit)
print("\n理论密度矩阵:")
print(expected_rho)

# 计算保真度
fidelity = expected_rho.fidelity(rho_fit)
print(f"\n保真度: {fidelity}")

# 11. 总结
print("\n11. 总结")
print("1. Qiskit提供多种模拟器，包括状态向量、QASM和酉模拟器")
print("2. 可以通过测量获得量子电路的概率分布")
print("3. 可以使用噪声模型模拟真实量子计算机中的错误")
print("4. 提供测量误差缓解技术来减轻测量误差")
print("5. 支持量子态层析来重构量子态")
print("6. 可以使用密度矩阵表示纯态和混合态")
print("7. 提供丰富的可视化工具来分析模拟结果")

print("\n下一步学习:")
print("- 实现具体的量子算法")
print("- 在真实量子设备上运行电路")
print("- 学习更高级的量子错误校正技术")
print("- 探索量子机器学习应用") 