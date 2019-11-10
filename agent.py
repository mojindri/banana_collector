from torch import tensor

from QNetwork import QNetwork
import random
import torch
import torch.nn.functional as F
import numpy as np
from ReplayBuffer import ReplayBuffer
import os

BATCH_SIZE = 64
GAMMA = 0.99  #
TAU = 1e-3


class Agent:
    def __init__(self, env, action_size, state_size, use_dueling=False,
                 use_double=False, network_file=None):
        self.device = torch.device('cpu')
        self.action_size = action_size
        self.env = env
        self.state_size = state_size
        self.seed = 1234
        self.target_network = QNetwork(state_size=state_size, action_size=action_size, seed=self.seed,
                                       use_dueling=use_dueling).to(self.device)
        self.local_network = QNetwork(state_size=state_size, action_size=action_size, seed=self.seed,
                                      use_dueling=use_dueling).to(self.device)
        self.optimizer = torch.optim.Adam(self.local_network.parameters(), lr=5e-4)

        if network_file is not None:
            if os.path.exists(network_file):
                checkpoints = torch.load(network_file)
                self.local_network.load_state_dict(checkpoints['local'])
                self.target_network.load_state_dict(checkpoints['target'])
                self.optimizer.load_state_dict(checkpoints['optimizer'])

        self.memory = ReplayBuffer(self.seed, batch_size=BATCH_SIZE, device=self.device)
        self.use_double = use_double

    def act(self, state, epsilon):
        if random.random() > epsilon:
            self.local_network.eval()
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            ans = None
            with torch.no_grad():
                ans = self.local_network.forward(state).cpu().data.numpy()
                ans = np.argmax(ans).astype(int)
            self.local_network.train()
            return ans
        else:
            return random.choice(range(self.action_size))

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)

    def step(self, timestep, state, next_state, action, reward, done):
        self.memory.add(state, next_state, action, reward, done)

        if timestep % 4 == 0:
            if len(self.memory) >= BATCH_SIZE:
                states, next_states, actions, rewards, dones = self.memory.sample()
                q_target = None
                if not self.use_double:
                    q_target = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)  # need to be fixed
                else:
                    indice = torch.argmax(self.local_network(next_states).detach(), 1)
                    q_target = self.target_network(next_states).detach().gather(1, indice.unsqueeze(1))

                target = rewards + (GAMMA * q_target * (1 - dones))
                q_expected = self.local_network(states).gather(1, actions)
                loss = F.mse_loss(q_expected, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.soft_update(self.local_network, self.target_network)
