# UniKE — 知识图谱嵌入工具包

[![Documentation Status](https://readthedocs.org/projects/unike/badge/?version=latest)](https://unike.readthedocs.io/)
![GitHub last commit](https://img.shields.io/github/last-commit/CPU-DS/UniKE)


基于 [OpenKE-PyTorch](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch) 开发的知识图谱嵌入工具包，支持跨平台运行，具备自动超参数搜索、高效并行训练以及实验结果记录功能，为研究与应用提供强大助力。

教程和 API 参考文档可以访问 [unike.readthedocs.io](https://unike.readthedocs.io/)。

## 项目结构

- 📁 [`unike/`](https://github.com/CPU-DS/UniKE/tree/main/unike/)  
  UniKE 源代码保存在 `unike/`
- 📚 [`docs/`](https://github.com/CPU-DS/UniKE/tree/main/docs/)  
  所有的文档源文件保存在 `docs/`，所有的 `*.rst` 构成了文档中的各个部分
- 🌰 [`examples/`](https://github.com/CPU-DS/UniKE/tree/main/examples/)  
  UniKE 的例子保存在 `examples/`，修改自 `OpenKE-PyTorch`
- 📍 [`docs/_static/logs/`](https://github.com/CPU-DS/UniKE/tree/main/docs/_static/logs/)  
  UniKE 的例子运行日志保存在 `docs/_static/logs/`
- 💡 [`benchmarks/`](https://github.com/CPU-DS/UniKE/tree/main/benchmarks/)  
  常用的知识图谱保存在 `benchmarks/`
- 📜 [`README.rst`](https://github.com/CPU-DS/UniKE/tree/main/README.rst)  
  项目主页

### ⁉️ Questions / comments
如果你有任何问题，可以在 [Github issue](https://github.com/CPU-DS/UniKE/issues) 提问。

## 快速开始

使用 `pip`
```bash
pip install unike
```

## 新特性

### **易用的**

- **1.0.0 版本**: 利用 C++ 重写底层数据处理、C++11 的线程库实现并行、[pybind11](https://github.com/pybind/pybind11) 实现 Python 和 C++ 的交互，进而能够做到跨平台 (Windows, Linux)
- **2.0.0 版本**: 使用 Python 重写底层数据处理，进而能够做到跨平台 (Windows, Linux)
- 增加了文档

### **正确的**

- 增加了 `R-GCN`模型
- 增加了 `CompGCN`模型
- 修复了 [SimplE模型实现的问题](https://github.com/thunlp/OpenKE/issues/151)
- 修复了 [HolE](#details_hole) 深度学习框架（pytorch）的版本适配问题

### **高效的**

- 使用 `torch.nn.parallel.DistributedDataParallel` 完成数据并行（**2.0.0 版本** 使用 [accelerate](https://github.com/huggingface/accelerate) 实现），使得 `UniKE` 能够利用多个 `GPU` 同时训练
- 增加超参数扫描功能（随机搜索、网格搜索和贝叶斯搜索）

### **扩展的**

- 在模型训练过程中，能够在验证集上评估模型（模型能够一次评估多个三元组（batch），能够大大加速模型评估）
- 增加了学习率调度器
- 能够利用 [wandb](https://wandb.ai/) 输出日志
- 实现了早停止
- 能够自定义 Hits@N

## 支持的知识图谱嵌入模型

| 类型 | 模型 |
| --- | --- |
| 平移模型 | `TransE`, `TransH`, `TransR`, `TransD`, `RotatE` |
| 语义匹配模型 | `RESCAL`, `DistMult`, `HolE`, `ComplEx`, `Analogy`, `SimplE`, `ANALOGY` |
| 图神经网络模型 | `R-GCN`, `CompGCN` |

## 如何引用这个项目？

如果您发现 UniKE 对您的研究有用，请考虑使用以下 BibTeX 模板引用 UniKE：

```bibtex
@misc{UniKE,
   author = {Lu, Yanfeng and Hou, Fengzhen},
   year = {2024},
   note = {https://github.com/CPU-DS/UniKE},
   title = {UniKE: An Open Source Library for Knowledge Graph Embeddings}
}
```

该仓库主要由 [Yanfeng Lu](https://github.com/LuYF-Lemon-love)，[Fengzhen Hou](https://github.com/houfz-cpu)，[Tao Wang](https://github.com/wangtao2001)提供（按时间顺序排列）。
