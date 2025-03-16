# 量子计算学习资源库

这个仓库包含了一系列用于学习量子计算的实践代码和教程，涵盖了主流量子计算框架：PennyLane、Qiskit和Cirq。

## 项目概述

本项目旨在提供一个全面的量子计算学习路径，从基础概念到高级应用。每个框架的学习路径都遵循相似的进阶结构，帮助读者逐步掌握量子计算技能。

## 框架介绍

### PennyLane
PennyLane是一个面向量子机器学习的开源软件框架，它提供了与多种量子硬件和经典机器学习库的无缝集成。

### Qiskit
Qiskit是IBM开发的开源量子计算框架，提供了完整的量子计算工具集，包括电路构建、模拟和在真实量子硬件上运行代码的能力。

### Cirq
Cirq是Google开发的专注于噪声中级量子(NISQ)计算的框架，它与TensorFlow Quantum紧密集成，支持量子机器学习应用。

## 项目结构

```
quantum_learning/
├── pennylane/             # PennyLane学习路径
│   ├── 01_pennylane_basics.py
│   ├── 02_hybrid_computing.py
│   ├── 03_variational_circuits.py
│   ├── 04_quantum_gradients.py
│   ├── 05_quantum_ml.py
│   ├── 06_advanced_applications.py
│   ├── exercises/        # 练习和解答
│   ├── notebooks/        # Jupyter笔记本
│   ├── examples/         # 具体应用示例
│   └── images/           # 生成的图像和可视化
├── qiskit/                # Qiskit学习路径
│   ├── 01_qiskit_basics.py
│   ├── ...
│   └── exercises/
├── cirq/                  # Cirq学习路径
│   ├── 01_cirq_basics.py
│   ├── ...
│   └── exercises/
├── environment.yml        # Conda环境配置
└── requirements.txt       # 依赖包列表
```

## 入门指南

### 环境设置

1. 克隆仓库：
```bash
git clone https://github.com/tc-fruitslayer/quantum-computing-learning.git
cd quantum-computing-learning
```

2. 设置环境（使用Conda）：
```bash
conda env create -f environment.yml
conda activate quantum-env
```

或者使用pip：
```bash
pip install -r requirements.txt
```

### 运行代码

按照以下顺序运行脚本以获得最佳学习体验：

1. 基础知识（01_*_basics.py）
2. 电路构建和操作（02_*）
3. 变分电路和优化（03_*）
4. 进阶主题（04_*及以后）

示例：
```bash
cd quantum_learning/pennylane
python 01_pennylane_basics.py
```

## 学习路径

### 初学者路径
1. PennyLane基础（01_pennylane_basics.py）
2. Qiskit基础（01_qiskit_basics.py）
3. 量子门和电路（02_*）
4. 基本模拟（03_*）

### 中级路径
1. 变分量子电路（PennyLane: 03_variational_circuits.py）
2. 量子梯度和优化（PennyLane: 04_quantum_gradients.py）
3. 量子算法实现（Qiskit: 05_quantum_algorithms.py）

### 高级路径
1. 量子机器学习（PennyLane: 05_quantum_ml.py）
2. 高级应用（PennyLane: 06_advanced_applications.py）
3. 噪声和错误缓解（Qiskit: 07_noise_and_error.py）
4. TensorFlow Quantum（Cirq: 05_tensorflow_quantum.py）

## 贡献指南

欢迎提交问题和改进建议！请遵循以下步骤：
1. Fork仓库
2. 创建您的特性分支（`git checkout -b feature/amazing-feature`）
3. 提交您的更改（`git commit -m 'Add some amazing feature'`）
4. 推送到分支（`git push origin feature/amazing-feature`）
5. 打开Pull Request

## 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 致谢

- 感谢[PennyLane](https://pennylane.ai/)、[Qiskit](https://qiskit.org/)和[Cirq](https://quantumai.google/cirq)团队开发这些优秀的框架
- 特别感谢量子计算社区的所有贡献者和教育者 