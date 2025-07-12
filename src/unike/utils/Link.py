# coding:utf-8
#
# unike/utils/Link.py
#
# created by wangtao <wangtao.cpu@gmail.com> on Jul 10, 2025
#
# 该脚本定义了 Link 类.

"""
Link - 链接分数计算.
"""

import torch
import pandas as pd
import os
from ..module.model import Model

class Link:
    """
    使用 KGE 模型对感兴趣的三元组计算链接分数。
    """
    def __init__(self, 
                model: Model,
                in_path: str = './',
                ent_file: str = "entity2id.txt", 
                rel_file: str = "relation2id.txt",
                all_file: str = "all2id.txt"):
        
        """创建 Link 对象。
        
        :param model: 模型
        :type model: Model
        :param in_path: 数据集目录
        :type in_path: str
        :param ent_file: entity2id.txt
        :type ent_file: str
        :param rel_file: relation2id.txt
        :type rel_file: str
        :param all_file: all2id.txt
        :type all_file: str
        """
    
        self.model = model
        self.in_path = in_path
        self.ent_file = ent_file
        self.rel_file = rel_file
        self.all_file = all_file
        
        self.load_data()
        self.load_all()
        
        
    def load_data(self) -> None:
        
        """读取 :py:attr:`ent_file` 文件和 :py:attr:`rel_file` 文件。"""
        
        self.ent2id = {}
        self.id2ent = {}

        with open(os.path.join(self.in_path, self.ent_file)) as f:
            _ = (int)(f.readline())
            for line in f:
                entity, id_ = line.strip().split("\t")
                self.ent2id[entity] = int(id_)
                self.id2ent[int(id_)] = entity

        self.rel2id = {}
        self.id2rel = {}

        with open(os.path.join(self.in_path, self.rel_file)) as f:
            _ = (int)(f.readline())
            for line in f:
                relation, id_ = line.strip().split("\t")
                self.rel2id[relation] = int(id_)
                self.id2rel[int(id_)] = relation
    
    def load_all(self) -> None:
        
        """读取 :py:attr:`all_file` 文件。"""
        
        self.all = []
        with open(os.path.join(self.in_path, self.all_file)) as f:
            _ = f.readline()
            for line in f:
                head, tail, rel = line.strip().split()
                self.all.append((int(head), int(rel), int(tail)))
        
        
    def link(self, head_ids: list[int], rel_ids: list[int], tail_ids: list[int], topk: int = 50, device: str = 'cpu') -> pd.DataFrame:
        
        """对给定的头实体、关系和尾实体进行组合并计算链接分数。
        
        :param head_ids: 头实体列表
        :type head_ids: list[int]
        :param rel_ids: 关系列表
        :type rel_ids: list[int]
        :param tail_ids: 尾实体列表
        :type tail_ids: list[int]
        :returns: 结果数据框
        :rtype: pd.DataFrame
        :param topk: 返回的 topk 个结果
        :type topk: int
        :param device: 设备
        :type device: str
        """
        
        head_ids = torch.tensor(head_ids).long().to(device)
        rel_ids = torch.tensor(rel_ids).long().to(device)
        tail_ids = torch.tensor(tail_ids).long().to(device)
        
        triples = []
        scores = []
        
        self.model.eval()
        self.model.to(device)
        with torch.no_grad():
            for h_idx in range(len(head_ids)):
                for r_idx in range(len(rel_ids)):
                    h_id = head_ids[h_idx]
                    r_id = rel_ids[r_idx]
                    
                    h_id = h_id.tile((tail_ids.shape[0], ))
                    r_id = r_id.tile((tail_ids.shape[0], ))
                    
                    triple = torch.stack((h_id, r_id, tail_ids)).T
                    data = {
                        "positive_sample": triple
                    }
                    score = self.model.predict(data, "single")
                    
                    scores.append(score)
                    triples.append(triple)
        
        triples = torch.cat(triples)
        scores = torch.cat(scores)
        
        sorted_scores, indices = torch.sort(scores.squeeze(), descending=True)
        sorted_triples = triples[indices]

        triples = list(tuple(triple) for triple in sorted_triples.tolist()[:topk])
        scores = sorted_scores.tolist()[:topk]

        result = [[   
                    head, rel, tail, 
                    tuple([head, rel, tail]) in self.all, 
                    self.id2ent[head], self.id2rel[rel], self.id2ent[tail],
                    scores[idx]
            ]
            for idx, (head, rel, tail) in enumerate(triples)
        ]
        
        df = pd.DataFrame(result, columns=["head", "rel", "tail", "in", "head_ent", "rel_ent", "tail_ent", "score"])
        return df
