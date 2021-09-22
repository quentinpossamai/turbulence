import torch
from torch import nn
from abc import ABC, abstractmethod


class Run(ABC):

    @abstractmethod
    def configure_optimizers(self, *args):
        pass

    @abstractmethod
    def forward(self, *args):
        pass

    @abstractmethod
    def training_step(self, *args):
        pass

    @abstractmethod
    def validation_step(self, *args):
        pass

    @abstractmethod
    def rollout(self, *args):
        pass
