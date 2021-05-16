import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from replay_buffer import ReplayMemory

from time import perf_counter



class CNN_DQN(nn.Module):
    def __init__(self, input_shape, n_outputs, config):
        super(CNN_DQN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out((input_shape[2], *input_shape[:2]))

        self.dueling = config['dueling']        
        if not self.dueling:
            self.fc_layers = nn.Sequential(
                nn.Linear(conv_out_size, 64),
                nn.ReLU(),
                nn.Linear(64, n_outputs)
            )
        else:
            self.fc_A = nn.Sequential(
                nn.Linear(conv_out_size, 64),
                nn.ReLU(),
                nn.Linear(64, n_outputs)
            )
            self.fc_V = nn.Sequential(
                nn.Linear(conv_out_size, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )


    def _get_conv_out(self, shape):
        o = self.conv_layers(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.transpose(1, 3).transpose(2, 3)
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        if not self.dueling:
            x = self.fc_layers(x)
        else:
            V = self.fc_V(x)
            A = self.fc_A(x)
            x = V + A - A.mean()
        return x


class DQN(nn.Module):
    def __init__(self, input_shape, n_outputs, config):
        super(DQN, self).__init__()
        n_hidden = config['n_hidden']
        self.fc1 = nn.Linear(np.prod(input_shape), n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_outputs)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class DQNAgent():
    def __init__(self, obs_shape, n_actions, log_writer, config, device='cpu'):
        self.n_actions = n_actions
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
    
        self.epsilon_start = config['epsilon_start']
        self.epsilon = self.epsilon_start
        self.epsilon_final = config['epsilon_final']
        self.decay_steps = config['epsilon_decay_steps']

        self.replay_buffer = ReplayMemory(config['replay_capacity']) 
        self.replay_min = config['replay_min']

        NetClass = CNN_DQN if config['cnn'] else DQN
        self.net = NetClass(obs_shape, self.n_actions, config).to(device)
        self.target_net = NetClass(obs_shape, self.n_actions, config).to(device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval() # evaluation mode (don't compute gradients, etc)

        self.double = config['double']

        self.log_writer = log_writer
        self.device = device
        self.total_time = 0

    def select_action(self, obs, iter):
        self.epsilon = max(self.epsilon_final, self.epsilon_start - iter / self.decay_steps)
        self.log_writer.add_scalar('Agent/epsilon', self.epsilon, iter)

        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            Q = self.net(torch.tensor([obs], device=self.device).type(torch.float)).detach()
            action = torch.argmax(Q)
            return action


    def training_step(self, obs, action, reward, next_obs, done, iter):
        # populate the replay buffer
        self.replay_buffer.store((obs, action, reward, next_obs, done))
        self.log_writer.add_scalar('Agent/replay memory size', len(self.replay_buffer), iter)

        if len(self.replay_buffer) > self.replay_min:
            # 0.001
            obs, action, reward, next_obs, done = self.replay_buffer.sample(self.batch_size)
            
            # TODO this is what takes most of the time
            tic = perf_counter()
            obs, action, reward, next_obs = torch.tensor(obs, device=self.device, dtype=torch.float), \
                torch.tensor(action, device=self.device), torch.tensor(reward, device=self.device), torch.tensor(next_obs, device=self.device, dtype=torch.float) 
            done_mask = torch.tensor(done, device=self.device, dtype=torch.bool)

            self.total_time += perf_counter() - tic
            if iter > 0 and iter % 500 == 0:
                print("Tensor convertion: {:.3f}s".format(self.total_time/iter))
                self.total_time = 0


            Q = self.net(obs).gather(1, action.reshape(-1, 1)).reshape(-1)

            next_Q_values = self.target_net(next_obs).detach()
            if not self.double:
                next_Q = next_Q_values.max(dim=1)[0]
            else:
                a = self.net(next_obs).argmax(dim=1).detach()
                next_Q = next_Q_values.gather(1, a.reshape(-1, 1)).reshape(-1)

            next_Q[done_mask] = 0.0
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