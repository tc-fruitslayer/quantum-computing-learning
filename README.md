# 量子计算学习框架

这个仓库包含量子计算学习的代码和教程，覆盖了多种量子计算框架和应用。

## 目录结构

```
量子计算/
├── README.md                 # 本文件
├── environment.yml           # Conda环境配置文件
└── quantum_learning/         # 量子学习资源目录
    ├── basics/               # 量子计算基础知识
    │   ├── 01_intro.py       # 量子计算简介
    │   ├── 02_qubits.py      # 量子比特和叠加
    │   ├── 03_entanglement.py # 量子纠缠
    │   └── 04_algorithms.py  # 量子算法基础
    ├── cirq/                 # Google Cirq框架学习
    │   ├── README.md         # Cirq学习指南
    │   ├── 01_cirq_basics.py # Cirq基础和特性
    │   ├── 02_gates_and_circuits.py # 量子门和电路
    │   ├── 03_simulation.py  # 模拟和测量
    │   ├── 04_quantum_algorithms.py # 量子算法实现
    │   ├── 05_tensorflow_quantum.py # TensorFlow Quantum集成
    │   └── 06_optimization.py # 优化技术
    ├── pennylane/            # PennyLane框架学习（待开发）
    ├── qiskit/               # IBM Qiskit框架学习
    │   ├── README.md         # Qiskit学习指南
    │   ├── 01_qiskit_basics.py # Qiskit基础和特性
    │   ├── 02_quantum_circuits.py # 量子电路创建和可视化
    │   ├── 03_quantum_gates.py # Qiskit中的量子门
    │   ├── 04_simulation.py  # 模拟和测量
    │   ├── 05_quantum_algorithms.py # 经典量子算法实现
    │   ├── 06_real_hardware.py # 在真实量子计算机上运行
    │   ├── 07_noise_and_error.py # 量子误差纠正和降噪
    │   └── exercises/        # 练习文件夹
    │       ├── ex01_basics.py # 基础练习
    │       ├── ex02_circuits.py # 电路练习
    │       ├── ex03_algorithms.py # 算法练习
    │       └── solutions/    # 习题解答
    └── projects/             # 量子计算项目（待开发）
```

## 学习路径

1. **量子计算基础** - 从 `quantum_learning/basics/` 开始，了解量子计算的基本概念
2. **框架学习** - 选择一个或多个框架深入学习：
   - **Google Cirq** - Google的量子编程框架
   - **IBM Qiskit** - IBM的量子计算生态系统
   - **PennyLane** - 用于量子机器学习的框架（待开发）
3. **项目实践** - 通过 `quantum_learning/projects/` 中的项目巩固所学知识（待开发）

## 框架比较

| 框架 | 开发者 | 特点 | 适用场景 |
|------|-------|------|---------|
| Cirq | Google | 为NISQ设备优化，与TensorFlow集成 | 量子机器学习，NISQ算法研究 |
| Qiskit | IBM | 完整生态系统，真实硬件访问 | 量子算法开发，教育，量子化学 |
| PennyLane | Xanadu | 自动微分，混合量子-经典计算 | 量子机器学习，变分算法 |

## 环境配置

使用提供的Conda环境文件创建环境：

```bash
conda env create -f environment.yml
conda activate quantum-learning
```

## 特色内容

1. **基础教程** - 量子计算核心概念的直观解释
2. **多框架支持** - 学习和比较不同的量子编程方法
3. **交互式示例** - 每个示例都可以运行和修改
4. **练习与解答** - 巩固学习的实践练习

### Qiskit学习特色

IBM Qiskit部分现包含完整的学习路径，从基础到高级内容：

- **教学文件** - 7个详细的教学文件，涵盖从基础到高级的所有主题
- **交互式练习** - 3套练习题，帮助巩固学习内容：
  - 基础练习：创建电路、制备量子态、测量等
  - 电路练习：电路组合、寄存器、参数化电路等高级特性
  - 算法练习：实现德意志-约扎、格罗弗搜索等经典量子算法
- **详细解答** - 每个练习都有完整的参考解答
- **实用指南** - 关于如何在真实量子计算机上运行代码的详细指导

### Cirq学习特色

Google Cirq部分提供了对Google量子生态系统的深入探索：

- **Cirq基础** - 了解Cirq独特的设计理念和数据结构
- **TensorFlow Quantum集成** - 探索量子-经典混合计算
- **优化技术** - 学习如何为NISQ设备优化量子电路

## 后续开发计划

1. 添加PennyLane框架学习资料
2. 开发实际项目示例
3. 添加量子机器学习专题
4. 扩展练习和挑战
5. 性能比较和基准测试

## 参考资源

- [Qiskit文档](https://qiskit.org/documentation/)
- [Cirq文档](https://quantumai.google/cirq)
- [PennyLane文档](https://pennylane.ai/docs/)
- [量子计算导论](https://quantum.country/)
- [IBM Quantum Experience](https://quantum-computing.ibm.com/)

## 贡献

欢迎通过Pull Request贡献代码、修复错误或添加新教程。请确保所有代码都有详细注释和文档。 