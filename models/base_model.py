import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List
import time


class BaseNERModel(ABC):
    def __init__(self, config):
        self.config = config
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def build_model(self, vocab_size: int, num_labels: int):
        """构建模型架构"""
        pass

    @abstractmethod
    def train_epoch(self, train_loader, optimizer, criterion):
        """训练一个epoch"""
        pass

    @abstractmethod
    def evaluate(self, test_loader, metrics):
        """评估模型"""
        pass

    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, path)

    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
