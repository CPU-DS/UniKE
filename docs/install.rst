安装
==================================

使用
----------------------------------

1 使用 `pip` 安装：

.. prompt:: bash

    pip install unike

2. 验证：

::

    >>> import unike
    >>> unike.__version__
    '3.0.5'
    >>>

开发
----------------------------------

1. 克隆 main 分支：

.. prompt:: bash

    git clone https://github.com/CPU-DS/UniKE.git
    cd UniKE/
    uv sync

2. 快速开始：

.. prompt:: bash

    cd examples/TransE/
    python single_gpu_transe_FB15K.py