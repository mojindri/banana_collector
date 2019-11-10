import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, h1_size=32, h2_size=32,h3_size=16, use_dueling=False):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input = nn.Linear(in_features=state_size, out_features= h1_size)
        self.h = nn.Linear(in_features=h1_size,out_features= h2_size)
        self.h = nn.Linear(in_features=h2_size, out_features=h3_size)
        self.output = nn.Linear(in_features=h3_size, out_features=action_size)
        self.advantage_value = nn.Linear(in_features=h3_size, out_features=1)
        self.use_dueling = use_dueling

    def forward(self, state):
        x = F.relu(self.input(state))
        x = F.relu(self.h(x))
        if self.use_dueling:
            return self.output(x) + self.advantage_value(x)
        else:
            return self.output(x)
