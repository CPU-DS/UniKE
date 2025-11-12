# coding:utf-8
#
# unike/config/HPOTrainer.py
#
# created by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jan 2, 2024
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 6, 2024
#
# 该脚本定义了并行训练循环函数.

"""
hpo_train - 超参数优化训练循环函数。
"""

import wandb
import typing
from ..utils import import_class
from ..module.model import TransE
from ..module.loss import MarginLoss
from ..module.strategy import NegativeSampling
from ..config import Trainer, Tester
from ..data import KGEDataLoader
from ..utils import WandbLogger
from loguru import logger
from typing import Literal, Any, Optional


def set_hpo_config(
	method: str = 'bayes',
	sweep_name: str = 'unike_hpo',
	metric_name: str = 'val/hits@10',
	metric_goal: Literal['maximize', 'minimize'] = 'maximize',
	data_loader_config: dict[str, dict[str, typing.Any]] = {},
	kge_config: dict[str, dict[str, typing.Any]] = {},
	loss_config: dict[str, dict[str, typing.Any]] = {},
	strategy_config: dict[str, dict[str, typing.Any]] = {},
	tester_config: dict[str, dict[str, typing.Any]] = {},
	trainer_config: dict[str, dict[str, typing.Any]] = {}) -> dict[str, Any]:

	"""设置超参数优化范围。
	
	:param method: 超参数优化的方法，``grid`` 或 ``random`` 或 ``bayes``
	:type param: str
	:param sweep_name: 超参数优化 sweep 的名字
	:type sweep_name: str
	:param metric_name: 超参数优化的指标名字
	:type metric_name: str
	:param metric_goal: 超参数优化的指标目标，``maximize`` 或 ``minimize``
	:type metric_goal: str
	:param data_loader_config: :py:class:`unike.data.KGEDataLoader` 的超参数优化配置
	:type data_loader_config: dict
	:param kge_config: :py:class:`unike.module.model.Model` 的超参数优化配置
	:type kge_config: dict
	:param loss_config: :py:class:`unike.module.loss.Loss` 的超参数优化配置
	:type loss_config: dict
	:param strategy_config: :py:class:`unike.module.strategy.Strategy` 的超参数优化配置
	:type strategy_config: dict
	:param tester_config: :py:class:`unike.config.Tester` 的超参数优化配置
	:type tester_config: dict
	:param trainer_config: :py:class:`unike.config.Trainer` 的超参数优化配置
	:type trainer_config: dict
	:returns: 超参数优化范围
	:rtype: dict
	"""

	sweep_config: dict[str, Any] = {
		'method': method,
		'name': sweep_name
	}

	metric: dict[str, str] = {
		'name': metric_name,
		'goal': metric_goal
	}

	parameters_dict: dict[str, dict[str, typing.Any]] | None = {}
	for key, value in data_loader_config.items():
		parameters_dict[f"data_loader_{key}"] = value
	for key, value in kge_config.items():
		parameters_dict[f"kge_{key}"] = value
	for key, value in loss_config.items():
		parameters_dict[f"loss_{key}"] = value
	for key, value in strategy_config.items():
		parameters_dict[f"strategy_{key}"] = value
	for key, value in tester_config.items():
		parameters_dict[f"tester_{key}"] = value
	for key, value in trainer_config.items():
		parameters_dict[f"trainer_{key}"] = value

	sweep_config['metric'] = metric
	sweep_config['parameters'] = parameters_dict

	return sweep_config

def set_hpo_hits(
    new_hits: list[int] = [1, 3, 10]):
	
	"""设置 Hits 指标。
	
	:param new_hits: 准备报告的指标 Hit@N 的列表，默认为 [1, 3, 10], 表示报告 Hits@1, Hits@3, Hits@10
	:type new_hits: list[int]
    """
	
	tmp = Tester.hits
	Tester.hits = new_hits
	logger.info(f"Hits@N 指标由 {tmp} 变为 {Tester.hits}")

def start_hpo_train(
	config: dict[str, Any],
	project: str = "unike-sweeps",
	count: int = 2,
	prior_runs: Optional[list[str]] = None,
 	resume_sweep_id: str | None = None) -> None:

	"""开启超参数优化。
	
	:param config: wandb 的超参数优化配置。
	:type config: dict
	:param project: 项目名
	:type param: str
	:param count: 进行几次尝试。
	:type count: int
	:param resume_sweep_id: 恢复的超参数 sweep id
	:type resume_sweep_id: str | None
	"""
	wandb.login()
 
	if resume_sweep_id:
		api = wandb.Api()
		sweep_id = f'{api.default_entity}/{project}/{resume_sweep_id}'
	else:
		sweep_id = wandb.sweep(config, project=project, prior_runs=prior_runs)

	wandb.agent(sweep_id, hpo_train, count=count)

def hpo_train(config: dict[str, Any] | None = None):

	"""超参数优化训练循环函数。
	
	:param config: wandb 的项目配置如超参数。
	:type config: dict[str, Any]
	"""
	
	with wandb.init(config = config):
		
		config_ = wandb.config

		# dataloader for training
		dataloader_class: type[KGEDataLoader] = import_class(f"unike.data.{config_.data_loader_dataloader}")
		dataloader = dataloader_class(
			in_path = config_.data_loader_in_path,
			ent_file = config_.data_loader_ent_file,
			rel_file = config_.data_loader_rel_file,
			train_file = config_.data_loader_train_file,
			valid_file = config_.data_loader_valid_file,
			test_file = config_.data_loader_test_file,
			batch_size = config_.data_loader_batch_size,
			neg_ent = config_.data_loader_neg_ent,
			test = True,
			test_batch_size = config_.data_loader_test_batch_size,
			type_constrain = config_.data_loader_type_constrain,
			num_workers = config_.data_loader_num_workers,
			train_sampler = import_class(f"unike.data.{config_.data_loader_train_sampler}"),
			test_sampler = import_class(f"unike.data.{config_.data_loader_test_sampler}")
		)

		# define the model
		if config_.kge_model in ["TransE", "TransH"]:
			model_class = import_class(f"unike.module.model.{config_.kge_model}")
			kge_model = model_class(
			    ent_tol = dataloader.get_ent_tol(),
			    rel_tol = dataloader.get_rel_tol(),
			    dim = config_.kge_dim,
			    p_norm = config_.kge_p_norm,
			    norm_flag = config_.kge_norm_flag
			)
		elif config_.kge_model == "TransR":
			model_class = import_class(f"unike.module.model.{config_.kge_model}")
			transe = TransE(
				ent_tol = dataloader.get_ent_tol(),
				rel_tol = dataloader.get_rel_tol(),
				dim = config_.kge_dim,
				p_norm = config_.kge_p_norm,
				norm_flag = config_.kge_norm_flag
			)
			kge_model = model_class(
				ent_tol = dataloader.get_ent_tol(),
				rel_tol = dataloader.get_rel_tol(),
				dim_e = config_.kge_dim,
				dim_r = config_.kge_dim,
				p_norm = config_.kge_p_norm,
				norm_flag = config_.kge_norm_flag,
				rand_init = config_.kge_rand_init)
			model_e = NegativeSampling(
				model = transe,
				loss = MarginLoss(margin = config_.kge_margin_e)
			)
			trainer_e = Trainer(
				model = model_e,
				data_loader = dataloader.train_dataloader(),
				epochs = 1,
				lr = config_.kge_lr_e,
				opt_method = config_.kge_opt_method_e,
				use_gpu = config_.trainer_use_gpu,
				device = config_.trainer_device
			)
			trainer_e.run()
			parameters = transe.get_parameters()
			transe.save_parameters("./transr_transe.json")
			kge_model.set_parameters(parameters)
		elif config_.kge_model == "TransD":
			model_class = import_class(f"unike.module.model.{config_.kge_model}")
			kge_model = model_class(
				ent_tol = dataloader.get_ent_tol(),
				rel_tol = dataloader.get_rel_tol(),
				dim_e = config_.kge_dim_e,
				dim_r = config_.kge_dim_r,
				p_norm = config_.kge_p_norm,
				norm_flag = config_.kge_norm_flag)
		elif config_.kge_model == "RotatE":
			model_class = import_class(f"unike.module.model.{config_.kge_model}")
			kge_model = model_class(
				ent_tol = dataloader.get_ent_tol(),
				rel_tol = dataloader.get_rel_tol(),
				dim = config_.kge_dim,
				margin = config_.kge_margin,
				epsilon = config_.kge_epsilon)
		elif config_.kge_model in ["RESCAL", "DistMult", "HolE", "ComplEx", "Analogy", "SimplE"]:
			model_class = import_class(f"unike.module.model.{config_.kge_model}")
			kge_model = model_class(
			    ent_tol = dataloader.get_ent_tol(),
			    rel_tol = dataloader.get_rel_tol(),
			    dim = config_.kge_dim)
		elif config_.kge_model == "RGCN":
			model_class = import_class(f"unike.module.model.{config_.kge_model}")
			kge_model = model_class(
				ent_tol = dataloader.get_ent_tol(),
				rel_tol = dataloader.get_rel_tol(),
				dim = config_.kge_dim,
				num_layers = config_.kge_num_layers)
		elif config_.kge_model == "CompGCN":
			model_class = import_class(f"unike.module.model.{config_.kge_model}")
			kge_model = model_class(
				ent_tol = dataloader.get_ent_tol(),
				rel_tol = dataloader.get_rel_tol(),
				dim = config_.kge_dim,
				opn = config_.kge_opn,
				fet_drop = config_.kge_fet_drop,
				hid_drop = config_.kge_hid_drop,
				margin = config_.kge_margin,
				decoder_model = config_.kge_decoder_model)
		else:
			import sys
			sys.path.append(config_.model_class_path)
			model_class = import_class(f"{config_.model_class}")
			params_config = {}
			for p_name in model_class.__init__.__code__.co_varnames:
				if f"kge_{p_name}" in config_:
					params_config[p_name] = config_[f"kge_{p_name}"]
			kge_model = model_class(
       			ent_tol = dataloader.get_ent_tol(),
			    rel_tol = dataloader.get_rel_tol(),
       			**params_config
          	)

		# define the loss function
		loss_class = import_class(f"unike.module.loss.{config_.loss_loss}")
		if config_.loss_loss == 'MarginLoss':
			loss = loss_class(
				adv_temperature = config_.loss_adv_temperature,
				margin = config_.loss_margin
			)
		elif config_.loss_loss in ['SigmoidLoss', 'SoftplusLoss']:
			loss = loss_class(adv_temperature = config_.loss_adv_temperature)
		elif config_.loss_loss == 'RGCNLoss':
			loss = loss_class(
				model = kge_model,
				regularization = config_.loss_regularization
			)
		else: 
      		# config_.loss_loss == 'CompGCNLoss':
			loss = loss_class(model = kge_model)
		
		# define the strategy
		strategy_class = import_class(f"unike.module.strategy.{config_.strategy_strategy}")
		if config_.strategy_strategy == 'NegativeSampling':
			model = strategy_class(
				model = kge_model,
				loss = loss,
				regul_rate = config_.strategy_regul_rate,
				l3_regul_rate = config_.strategy_l3_regul_rate
			)
		elif config_.strategy_strategy == 'RGCNSampling':
			model = strategy_class(
				model = kge_model,
				loss = loss
			)
		else:
      		# config_.strategy_strategy == 'CompGCNSampling':
			model = strategy_class(
				model = kge_model,
				loss = loss,
				smoothing = config_.strategy_smoothing,
				ent_tol = dataloader.train_sampler.ent_tol
			)

		# test the model
		tester_class: type[Tester] = import_class(f"unike.config.{config_.tester_tester}")
		tester = tester_class(
			model = kge_model,
			data_loader = dataloader,
			prediction = config_.tester_prediction,
			use_tqdm = config_.tester_use_tqdm,
			use_gpu = config_.tester_use_gpu,
			device = config_.tester_device
		)
  
		wandb_logger = WandbLogger(endpoint='wandb')
		wandb_logger.logger = wandb

		# # train the model
		trainer_class: type[Trainer] = import_class(f"unike.config.{config_.trainer_trainer}")
		trainer = trainer_class(
			model = model,
			data_loader = dataloader.train_dataloader(),
			epochs = config_.trainer_epochs,
			lr = config_.trainer_lr,
			opt_method = config_.trainer_opt_method,
			use_gpu = config_.trainer_use_gpu,
			device = config_.trainer_device,
			tester = tester,
			test = True,
			valid_interval = config_.trainer_valid_interval,
			log_interval = config_.trainer_log_interval,
			save_path = config_.trainer_save_path,
			use_early_stopping = config_.trainer_use_early_stopping,
			metric = config_.trainer_metric,
			patience = config_.trainer_patience,
			delta = config_.trainer_delta,
			use_tqdm = config_.trainer_use_tqdm,
			wandb_logger = wandb_logger
		)
		trainer.run()
