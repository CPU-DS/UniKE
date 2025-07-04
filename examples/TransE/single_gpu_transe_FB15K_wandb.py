"""
`TransE-FB15K-single-gpu <single_gpu_transe_FB15K.html>`_ ||
**TransE-FB15K-single-gpu-wandb** ||
`TransE-FB15K-single-gpu-hpo <single_gpu_transe_FB15K_hpo.html>`_ ||
`TransE-FB15K-accelerate <accelerate_transe_FB15K.html>`_ ||
`TransE-FB15K-accelerate-wandb <accelerate_transe_FB15K_wandb.html>`_ ||
`TransE-FB15K237-single-gpu-wandb <single_gpu_transe_FB15K237_wandb.html>`_ ||
`TransE-WN18RR-single-gpu-adv-wandb <single_gpu_transe_WN18_adv_sigmoidloss_wandb.html>`_

TransE-FB15K-single-gpu-wandb
====================================================================

.. Note:: created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023

.. Note:: updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 9, 2024

.. Note:: last run by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 9, 2024

这一部分介绍如何用一个 GPU 在 ``FB15k`` 知识图谱上训练 ``TransE`` :cite:`TransE`，使用 ``wandb`` 记录实验结果。

导入数据
-----------------
UniKE 有 1 个工具用于导入数据: :py:class:`unike.data.KGEDataLoader`。
"""

import os
from unike.utils import WandbLogger
from unike.data import KGEDataLoader, BernSampler, TradTestSampler
from unike.module.model import TransE
from unike.module.loss import MarginLoss
from unike.module.strategy import NegativeSampling
from unike.config import Trainer, Tester

######################################################################
# 首先初始化 :py:class:`unike.utils.WandbLogger` 日志记录器，它是对 wandb 初始化操作的一层简单封装。

wandb_logger = WandbLogger(
	project="pybind11-ke",
	name="TransE-FB15K",
	config=dict(
		in_path = os.path.join(os.path.dirname(__file__), '../../benchmarks/FB15K/'),
		batch_size = 8192,
		neg_ent = 25,
		test = True,
		test_batch_size = 256,
		num_workers = 16,
		dim = 50,
		p_norm = 1,
		norm_flag = True,
		margin = 1.0,
		use_gpu = True,
		device = 'cuda:0',
		epochs = 1000,
		lr = 0.01,
		valid_interval = 100,
		log_interval = 100,
		save_interval = 100,
		save_path = '../../checkpoint/transe.pth',
		delta = 0.01
	)
)

config = wandb_logger.config

######################################################################
# UniKE 提供了很多数据集，它们很多都是 KGE 原论文发表时附带的数据集。
# :py:class:`unike.data.KGEDataLoader` 包含 ``in_path`` 用于传递数据集目录。

# dataloader for training
dataloader = KGEDataLoader(
	in_path = config.in_path, 
	batch_size = config.batch_size,
	neg_ent = config.neg_ent,
	test = config.test,
	test_batch_size = config.test_batch_size,
	num_workers = config.num_workers,
	train_sampler = BernSampler,
	test_sampler = TradTestSampler
)

######################################################################
# --------------
#

################################
# 导入模型
# ------------------
# UniKE 提供了很多 KGE 模型，它们都是目前最常用的基线模型。我们下面将要导入
# :py:class:`unike.module.model.TransE`，它是最简单的平移模型。

# define the model
transe = TransE(
	ent_tol = dataloader.get_ent_tol(),
	rel_tol = dataloader.get_rel_tol(),
	dim = config.dim, 
	p_norm = config.p_norm, 
	norm_flag = config.norm_flag)

######################################################################
# --------------
#


#####################################################################
# 损失函数
# ----------------------------------------
# 我们这里使用了 ``TransE`` :cite:`TransE` 原论文使用的损失函数：:py:class:`unike.module.loss.MarginLoss`，
# :py:class:`unike.module.strategy.NegativeSampling` 对
# :py:class:`unike.module.loss.MarginLoss` 进行了封装，加入权重衰减等额外项。

# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = config.margin)
)

######################################################################
# --------------
#

######################################################################
# 训练模型
# -------------
# UniKE 将训练循环包装成了 :py:class:`unike.config.Trainer`，
# 可以运行它的 :py:meth:`unike.config.Trainer.run` 函数进行模型学习；
# 也可以通过传入 :py:class:`unike.config.Tester`，
# 使得训练器能够在训练过程中评估模型。
	
# test the model
tester = Tester(model = transe, data_loader = dataloader, use_gpu = config.use_gpu, device = config.device)

# train the model
trainer = Trainer(model = model, data_loader = dataloader.train_dataloader(),
	epochs = config.epochs, lr = config.lr, use_gpu = config.use_gpu, device = config.device,
	tester = tester, test = config.test, valid_interval = config.valid_interval,
	log_interval = config.log_interval, save_interval = config.save_interval,
	save_path = config.save_path, delta = config.delta, use_wandb = True)
trainer.run()

# close your wandb run
wandb_logger.finish()

######################################################################
# .. Note:: 上述代码的运行日志可以从 `此处 </zh-cn/latest/_static/logs/examples/TransE/single_gpu_transe_FB15K_wandb.txt>`_ 下载。
# .. Note:: 上述代码的运行报告可以从 `此处 </zh-cn/latest/_static/pdfs/examples/TransE/TransE单卡训练示例（一）.pdf>`_ 下载。

######################################################################
# --------------
#