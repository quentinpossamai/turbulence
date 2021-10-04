import torch
from torch import nn
from abc import ABC, abstractmethod


class Run(ABC):
    @abstractmethod
    def forward(self, **kwargs):
        pass

    @abstractmethod
    def rollout(self, **kwargs):
        pass
