import torch
from torch import nn
from abc import ABC, abstractmethod


class Model(ABC):

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def configure_optimizers(self):
        pass

    @abstractmethod
    def training_step(self, train_batch, batch_idx):
        pass

    @abstractmethod
    def validation_step(self, val_batch, batch_idx):
        pass
