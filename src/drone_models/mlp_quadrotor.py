import torch
import torch.nn as nn
from torch.nn.functional import relu


class MLPQuadrotor(nn.Module):
    def __init__(self):
        super(MLPQuadrotor, self).__init__()
        self.linear1 = nn.Linear(in_features=16, out_features=256, bias=True).double()
        self.linear2 = nn.Linear(in_features=256, out_features=256, bias=True).double()
        self.linear3 = nn.Linear(in_features=256, out_features=256, bias=True).double()
        self.linear4 = nn.Linear(in_features=256, out_features=12, bias=True).double()

    def forward(self, state, motor):
        state_motor = torch.stack([state, motor], dim=1)
        out = relu(self.linear1(state_motor))
        out = relu(self.linear2(out))
        out = relu(self.linear3(out))
        out = self.linear4(out)
        return out
