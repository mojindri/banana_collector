import random
from collections import deque, namedtuple

# an implementation for experience replay approach
import numpy as np
import torch

device = torch.device('cpu')

NOISE= 0.02
ALPHA = 0.7
BETA = 0.4
beta_increment_per_sampling = 0.001

class ReplayBufferWithPriority:
    def __init__(self, seed, batch_size, device):
        self.device = device
        self.batch_size = batch_size
        self.memory = deque(maxlen=batch_size)
        self.seed = seed
        self.experience = namedtuple(typename="Experience",
                                     field_names=['state', 'next_state', 'action', 'reward', 'done','probability'])

    def add(self, state, next_state, action, reward, done,probability):
        self.memory.append(self.experience(state, next_state, action, reward, done,np.absolute(probability) + NOISE))

    def getALlProbablitiesMean(self):
        return np.mean(([e.probability ** ALPHA for e in self.memory if e is not None]))

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(
            self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(
            self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)
        probablities = torch.from_numpy(np.vstack([e.probability ** ALPHA for e in experiences if e is not None] / self.getALlProbablitiesMean() )).float().to(
            self.device)

        return (states, next_states, actions, rewards, dones,probablities)

    def __len__(self):
        return len(self.memory)
