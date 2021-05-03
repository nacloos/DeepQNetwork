import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from replay_buffer import ReplayMemory


class DQN(nn.Module):
    def __init__(self, input_shape, n_outputs, config):
        super(DQN, self).__init__()
        n_hidden = config['n_hidden']
        self.fc1 = nn.Linear(np.prod(input_shape), n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_outputs)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = F.relu(x)
        # x = self.fc3(x)
        return x


class DQNAgent():
    def __init__(self, obs_shape, n_actions, log_writer, config):
        self.n_actions = n_actions
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
    
        self.epsilon_start = config['epsilon_start']
        self.epsilon = self.epsilon_start
        self.epsilon_final = config['epsilon_final']
        self.decay_steps = config['epsilon_decay_steps']

        self.replay_buffer = ReplayMemory(config['replay_capacity']) 
        self.replay_min = config['replay_min']

        self.net = DQN(obs_shape, self.n_actions, config)
        self.target_net = DQN(obs_shape, self.n_actions, config)
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval() # evaluation mode (don't compute gradients, etc)

        self.log_writer = log_writer

    def select_action(self, obs, iter):
        self.epsilon = max(self.epsilon_final, self.epsilon_start - iter / self.decay_steps)
        self.log_writer.add_scalar('Agent/epsilon', self.epsilon, iter)

        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            Q = self.net(torch.tensor(obs).type(torch.float)).detach().numpy()
            action = np.argmax(Q)
            return action


    def training_step(self, obs, action, reward, next_obs, iter):
        # populate the replay buffer
        self.replay_buffer.store((obs, action, reward, next_obs))
        self.log_writer.add_scalar('Agent/replay memory size', len(self.replay_buffer), iter)

        if len(self.replay_buffer) > self.replay_min:
            obs, action, reward, next_obs = self.replay_buffer.sample(self.batch_size)
            # add a wrapper to the env
            obs, action, reward, next_obs = torch.tensor(obs).type(torch.float), torch.tensor(action), torch.tensor(reward), torch.tensor(next_obs).type(torch.float) 


            Q = self.net(obs).gather(1, action.reshape(-1, 1)).reshape(-1)

            next_Q = self.target_net(next_obs).max(dim=1)[0].detach()
            Q_target = reward + self.gamma*next_Q

            # if the residual variance is close to 1, the network doesn't learn to predict the reward (the error varies as much as the target -> not learning)
            self.log_writer.add_scalar('Agent/residual variance', torch.var(Q - Q_target)/torch.var(Q_target), iter)
            p = torch.bincount(action)/self.batch_size
            p[p == 0] = 1e-5
            policy_entropy = torch.sum(-p*torch.log(p))
            self.log_writer.add_scalar('Agent/policy entropy', policy_entropy, iter)

            loss = nn.MSELoss()(Q, Q_target)
            return loss
        else:
            return None