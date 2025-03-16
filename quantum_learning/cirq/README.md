# Google Cirq 框架学习

这个目录包含了Google Cirq量子计算框架的学习资料和代码示例。Cirq是由Google开发的开源量子计算框架，专为NISQ（嘈杂的中等规模量子）设备设计。

## 学习路径

1. **Cirq基础和特性** - `01_cirq_basics.py`
   - Cirq的基本概念和数据结构
   - 量子比特表示
   - 命名和线路操作

2. **量子门和电路** - `02_gates_and_circuits.py`
   - Cirq中的单量子比特门和多量子比特门
   - 参数化门
   - 电路构建和可视化

3. **模拟和测量** - `03_simulation.py`
   - 状态向量模拟
   - 密度矩阵模拟
   - 测量和采样
   - 噪声模型

4. **量子算法实现** - `04_quantum_algorithms.py`
   - Deutsch-Jozsa算法
   - Grover搜索算法
   - 量子相位估计
   - QAOA (量子近似优化算法)

5. **与TensorFlow Quantum集成** - `05_tensorflow_quantum.py`
   - TFQ基础
   - 混合量子-经典模型
   - 量子机器学习

6. **量子计算资源优化** - `06_optimization.py`
   - 电路优化
   - 编译策略
   - 噪声缓解技术
   - 资源分析与估计

## 安装指南

安装Cirq和相关依赖：

```bash
pip install cirq cirq-core cirq-google matplotlib numpy

# 如果需要TensorFlow Quantum
pip install tensorflow tensorflow-quantum
```

## 参考资源

- [Cirq官方文档](https://quantumai.google/cirq)
- [Cirq GitHub仓库](https://github.com/quantumlib/Cirq)
- [TensorFlow Quantum文档](https://www.tensorflow.org/quantum)
- [Google量子AI](https://quantumai.google/)

## 学习建议

- 尝试在本地运行每个示例，并修改参数观察结果变化
- 比较Cirq和其他框架（如Qiskit）的差异
- 关注Cirq针对NISQ设备的特殊优化 