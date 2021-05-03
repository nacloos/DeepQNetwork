import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class DQNAgent():
    def __init__(self):
        pass

    def select_action(self, obs):
        return np.random.choice(3)

    def update(self, obs, action, reward, next_obs):
        pass 

    def training_step(self, obs, action, reward, next_obs):
        pass