#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IBM Qiskit框架学习 3：Qiskit中的量子门
本文件详细介绍Qiskit中各种量子门的特性、矩阵表示和应用
"""

# 导入必要的库
from qiskit import QuantumCircuit, transpile, Aer
from qiskit.quantum_info import Operator
from qiskit.visualization import plot_bloch_multivector, array_to_latex
from qiskit.extensions import UnitaryGate
from qiskit.circuit.library import (
    HGate, XGate, YGate, ZGate, SGate, TGate, CXGate, CZGate, 
    SwapGate, CCXGate, PhaseGate, RXGate, RYGate, RZGate
)
import numpy as np
import matplotlib.pyplot as plt

print("===== Qiskit中的量子门 =====")

# 1. 单量子比特门
print("\n1. 单量子比特门 (Single-qubit gates)")

# 创建一个量子电路并应用各种单比特门
qc_single = QuantumCircuit(1)

# 应用几个常见的单比特门并展示
print("应用各种单比特门到|0⟩上，观察不同效果")

# 初始状态 |0⟩
print("\n初始状态 |0⟩:")
init_circ = QuantumCircuit(1)
backend = Aer.get_backend('statevector_simulator')
init_result = backend.run(transpile(init_circ, backend)).result()
init_state = init_result.get_statevector()
print(f"状态向量: {init_state}")

# X门 (NOT门) |0⟩ → |1⟩
print("\nX门 (NOT门) - 比特翻转:")
x_circ = QuantumCircuit(1)
x_circ.x(0)
x_result = backend.run(transpile(x_circ, backend)).result()
x_state = x_result.get_statevector()
print(f"状态向量: {x_state}")
print(f"矩阵表示:\n{Operator(XGate()).data}")

# H门 (Hadamard) |0⟩ → (|0⟩+|1⟩)/√2
print("\nH门 (Hadamard) - 创建叠加态:")
h_circ = QuantumCircuit(1)
h_circ.h(0)
h_result = backend.run(transpile(h_circ, backend)).result()
h_state = h_result.get_statevector()
print(f"状态向量: {h_state}")
print(f"矩阵表示:\n{Operator(HGate()).data}")

# Z门 - 相位翻转
print("\nZ门 - 相位翻转:")
# 先创建叠加态再应用Z门
z_circ = QuantumCircuit(1)
z_circ.h(0)  # 先创建叠加态
z_circ.z(0)  # 再应用Z门
z_result = backend.run(transpile(z_circ, backend)).result()
z_state = z_result.get_statevector()
print(f"状态向量: {z_state}")
print(f"矩阵表示:\n{Operator(ZGate()).data}")

# Y门
print("\nY门:")
y_circ = QuantumCircuit(1)
y_circ.y(0)
y_result = backend.run(transpile(y_circ, backend)).result()
y_state = y_result.get_statevector()
print(f"状态向量: {y_state}")
print(f"矩阵表示:\n{Operator(YGate()).data}")

# S门 (√Z)
print("\nS门 (√Z):")
s_circ = QuantumCircuit(1)
s_circ.h(0)  # 先创建叠加态
s_circ.s(0)  # 再应用S门
s_result = backend.run(transpile(s_circ, backend)).result()
s_state = s_result.get_statevector()
print(f"状态向量: {s_state}")
print(f"矩阵表示:\n{Operator(SGate()).data}")

# T门 (√S)
print("\nT门 (√S):")
t_circ = QuantumCircuit(1)
t_circ.h(0)  # 先创建叠加态
t_circ.t(0)  # 再应用T门
t_result = backend.run(transpile(t_circ, backend)).result()
t_state = t_result.get_statevector()
print(f"状态向量: {t_state}")
print(f"矩阵表示:\n{Operator(TGate()).data}")

# 2. 旋转门
print("\n2. 旋转门 (Rotation gates)")

# RX门 (绕X轴旋转)
print("\nRX门 (绕X轴旋转π/2):")
rx_circ = QuantumCircuit(1)
rx_circ.rx(np.pi/2, 0)
rx_result = backend.run(transpile(rx_circ, backend)).result()
rx_state = rx_result.get_statevector()
print(f"状态向量: {rx_state}")
print(f"矩阵表示:\n{Operator(RXGate(np.pi/2)).data}")

# RY门 (绕Y轴旋转)
print("\nRY门 (绕Y轴旋转π/2):")
ry_circ = QuantumCircuit(1)
ry_circ.ry(np.pi/2, 0)
ry_result = backend.run(transpile(ry_circ, backend)).result()
ry_state = ry_result.get_statevector()
print(f"状态向量: {ry_state}")
print(f"矩阵表示:\n{Operator(RYGate(np.pi/2)).data}")

# RZ门 (绕Z轴旋转)
print("\nRZ门 (绕Z轴旋转π/2):")
rz_circ = QuantumCircuit(1)
rz_circ.h(0)  # 先创建叠加态
rz_circ.rz(np.pi/2, 0)
rz_result = backend.run(transpile(rz_circ, backend)).result()
rz_state = rz_result.get_statevector()
print(f"状态向量: {rz_state}")
print(f"矩阵表示:\n{Operator(RZGate(np.pi/2)).data}")

# Phase门
print("\nPhase门 (旋转相位π/4):")
p_circ = QuantumCircuit(1)
p_circ.h(0)  # 先创建叠加态
p_circ.p(np.pi/4, 0)
p_result = backend.run(transpile(p_circ, backend)).result()
p_state = p_result.get_statevector()
print(f"状态向量: {p_state}")
print(f"矩阵表示:\n{Operator(PhaseGate(np.pi/4)).data}")

# 3. 多量子比特门
print("\n3. 多量子比特门 (Multi-qubit gates)")

# CNOT门 (CX)
print("\nCNOT门 (CX) - 受控X门:")
cx_circ = QuantumCircuit(2)
cx_circ.h(0)  # 控制位创建叠加态
cx_circ.cx(0, 1)  # 0控制1
cx_result = backend.run(transpile(cx_circ, backend)).result()
cx_state = cx_result.get_statevector()
print(f"状态向量: {cx_state}")
print(f"矩阵表示:\n{Operator(CXGate()).data}")

# CZ门 - 受控Z门
print("\nCZ门 - 受控Z门:")
cz_circ = QuantumCircuit(2)
cz_circ.h(0)  # 控制位创建叠加态
cz_circ.h(1)  # 目标位也创建叠加态
cz_circ.cz(0, 1)
cz_result = backend.run(transpile(cz_circ, backend)).result()
cz_state = cz_result.get_statevector()
print(f"状态向量: {cz_state}")
print(f"矩阵表示:\n{Operator(CZGate()).data}")

# SWAP门 - 交换两个量子比特的状态
print("\nSWAP门 - 交换两个量子比特的状态:")
swap_circ = QuantumCircuit(2)
swap_circ.x(0)  # 将第一个比特置为|1⟩
swap_circ.swap(0, 1)  # 交换0和1的状态
swap_result = backend.run(transpile(swap_circ, backend)).result()
swap_state = swap_result.get_statevector()
print(f"状态向量: {swap_state}")
print(f"矩阵表示:\n{Operator(SwapGate()).data}")

# Toffoli门 (CCX) - 两个控制位的X门
print("\nToffoli门 (CCX) - 两个控制位的X门:")
ccx_circ = QuantumCircuit(3)
ccx_circ.x(0)
ccx_circ.x(1)
ccx_circ.ccx(0, 1, 2)
ccx_result = backend.run(transpile(ccx_circ, backend)).result()
ccx_state = ccx_result.get_statevector()
print(f"状态向量: {ccx_state}")

# 4. 在Bloch球上可视化单量子比特门的效果
print("\n4. 在Bloch球上可视化单量子比特门的效果")
print("各种单比特门对|0⟩状态的效果（图像将保存到文件）")

# 准备几个单比特电路
bloch_circuits = {
    "初始态|0⟩": QuantumCircuit(1),
    "X门": QuantumCircuit(1).compose(XGate(), [0]),
    "H门": QuantumCircuit(1).compose(HGate(), [0]),
    "Y门": QuantumCircuit(1).compose(YGate(), [0]),
    "RX(π/4)": QuantumCircuit(1).compose(RXGate(np.pi/4), [0]),
    "RY(π/4)": QuantumCircuit(1).compose(RYGate(np.pi/4), [0])
}

# 运行并保存Bloch球可视化
for name, circ in bloch_circuits.items():
    result = backend.run(transpile(circ, backend)).result()
    state = result.get_statevector()
    
    # 保存Bloch球可视化
    fig = plot_bloch_multivector(state)
    fig.savefig(f'bloch_{name.replace("|", "").replace("⟩", "").replace("(", "").replace(")", "").replace("/", "_")}.png')
    plt.close(fig)
    print(f"保存了{name}的Bloch球可视化")

# 5. 定制量子门
print("\n5. 定制量子门")

# 自定义一个量子门 - 例如创建一个√X门
sqrtx_matrix = np.array([
    [0.5+0.5j, 0.5-0.5j],
    [0.5-0.5j, 0.5+0.5j]
])

# 创建一个自定义门
sqrtx_gate = UnitaryGate(sqrtx_matrix, label='√X')

# 使用自定义门
custom_circ = QuantumCircuit(1)
custom_circ.append(sqrtx_gate, [0])
custom_result = backend.run(transpile(custom_circ, backend)).result()
custom_state = custom_result.get_statevector()
print("自定义√X门:")
print(f"状态向量: {custom_state}")
print(f"矩阵表示:\n{Operator(sqrtx_gate).data}")
print("应用两次√X门等同于一次X门")

# 应用两次√X门
custom_circ2 = QuantumCircuit(1)
custom_circ2.append(sqrtx_gate, [0])
custom_circ2.append(sqrtx_gate, [0])
custom_result2 = backend.run(transpile(custom_circ2, backend)).result()
custom_state2 = custom_result2.get_statevector()
print(f"应用两次后的状态向量: {custom_state2}")
print(f"对比X门的状态向量: {x_state}")

# 6. 常用量子门组合和等价关系
print("\n6. 常用量子门组合和等价关系")

# X门 = H-Z-H
print("X门 = H-Z-H:")
equiv_circ = QuantumCircuit(1)
equiv_circ.h(0)
equiv_circ.z(0)
equiv_circ.h(0)
equiv_result = backend.run(transpile(equiv_circ, backend)).result()
equiv_state = equiv_result.get_statevector()
print(f"HZH的状态向量: {equiv_state}")
print(f"X门的状态向量: {x_state}")

# CNOT = H-CZ-H
print("\nCNOT(0,1) = H(1)-CZ(0,1)-H(1):")
cnot_equiv_circ = QuantumCircuit(2)
cnot_equiv_circ.h(0)
cnot_equiv_circ.h(1)
cnot_equiv_circ.cz(0, 1)
cnot_equiv_circ.h(1)
cnot_result = backend.run(transpile(cnot_equiv_circ, backend)).result()
cnot_state = cnot_result.get_statevector()
print(f"H-CZ-H的状态向量: {cnot_state}")

# 7. 量子门的代数性质
print("\n7. 量子门的代数性质")

# 1. 幂等性 - 应用两次相同的门，例如X^2 = I
print("幂等性 - 例如X^2 = I:")
xx_circ = QuantumCircuit(1)
xx_circ.x(0)
xx_circ.x(0)
xx_result = backend.run(transpile(xx_circ, backend)).result()
xx_state = xx_result.get_statevector()
print(f"X^2的状态向量: {xx_state}")
print(f"等同于单位操作的状态向量: {init_state}")

# 2. 可逆性 - 每个量子门都是可逆的
print("\n可逆性 - 每个量子门都是可逆的")
print("例如H^2 = I:")
hh_circ = QuantumCircuit(1)
hh_circ.h(0)
hh_circ.h(0)
hh_result = backend.run(transpile(hh_circ, backend)).result()
hh_state = hh_result.get_statevector()
print(f"H^2的状态向量: {hh_state}")
print(f"等同于单位操作的状态向量: {init_state}")

# 3. 交换关系 - 某些门之间的交换性质
print("\n交换关系 - 例如X和Z门不满足交换律:")
xz_circ = QuantumCircuit(1)
xz_circ.x(0)
xz_circ.z(0)
xz_result = backend.run(transpile(xz_circ, backend)).result()
xz_state = xz_result.get_statevector()

zx_circ = QuantumCircuit(1)
zx_circ.z(0)
zx_circ.x(0)
zx_result = backend.run(transpile(zx_circ, backend)).result()
zx_state = zx_result.get_statevector()

print(f"XZ的状态向量: {xz_state}")
print(f"ZX的状态向量: {zx_state}")
print("注意两个结果之间的相位差异")

# 8. 量子门的完备性
print("\n8. 量子门的完备性")
print("量子计算的通用门集:任何量子电路原则上都可以由以下门组成：")
print("1. H门")
print("2. T门")
print("3. CNOT门")
print("这是一个万能门集，可以近似实现任何酉变换")

print("\n另一个通用门集包括:")
print("1. 单量子比特旋转门（RX, RY, RZ）")
print("2. CNOT门")

# 9. 量子门和经典逻辑门的比较
print("\n9. 量子门和经典逻辑门的比较")
print("经典门与量子门对比:")
print("- NOT (经典) ⟷ X (量子): 比特翻转")
print("- AND (经典) ⟷ Toffoli/CCX (量子): 在量子计算中可逆版本的AND")
print("- XOR (经典) ⟷ CNOT (量子): 量子条件非门")
print("- 没有直接对应: H, Z, Phase, 旋转门")
print("关键区别: 量子门必须是可逆的（酉的），经典门不需要")

# 10. 总结
print("\n10. 总结")
print("1. 量子门是量子计算的基本构建块")
print("2. 单量子比特门操作单个量子比特的状态")
print("3. 多量子比特门允许量子比特之间的交互和纠缠")
print("4. 旋转门能够在Bloch球上实现任意的单量子比特操作")
print("5. 量子门必须是酉的（可逆的）")
print("6. 一小组量子门可以组成通用门集，能够实现任意量子电路")
print("7. 量子门的组合可以创建复杂的量子算法和协议")

print("\n下一步学习:")
print("- 学习如何使用这些量子门实现经典量子算法")
print("- 探索量子电路优化和编译技术")
print("- 实现特定问题的量子解决方案") 