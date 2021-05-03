from datetime import datetime
from pathlib import Path
import argparse
import json
from array2gif import write_gif
import numpy as np

import gym_minigrid
from gym_minigrid.wrappers import *
import gym

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import DQNAgent


env_name = "MiniGrid-Empty-5x5-v0"
# env_name = "MiniGrid-Empty-Random-5x5-v0"

save_dir = Path("runs")
model_name = "DQN"

config = {
    'n_episodes': 100,
    'episode_max_steps': 500,
    'video_iter': 1000,

    'batch_size': 1000,
    'lr': 0.001,
    'gamma': 0.99,
    'epsilon_start': 1,
    'epsilon_final': 0.01,
    'epsilon_decay_steps': 3000,
    'replay_capacity': 10000,
    'replay_min': 1000, # min replay size before training
    'target_update': 5,

    'n_hidden': 64 # number of hidden units in DQN
}

parser = argparse.ArgumentParser()
parser.add_argument("--load", type=str, default="", help="load a run. Let empty to start a new run")
args = parser.parse_args()

run_name = "run-" + datetime.utcnow().strftime("%Y%m%d%H%M%S") if args.load == "" else args.load
save_path = save_dir / env_name / model_name / run_name


def preprocess_obs(obs):
    obs = obs / 10
    obs = obs.reshape(-1)
    return obs


def save_model(net, config, save_path):
    torch.save(net.state_dict(), save_path / "model.pt")
    with open(save_path / "config.json", 'w') as f:
        json.dump(config, f, indent=4)


def train(env):
    save_path.mkdir(parents=True, exist_ok=True)
    log_writer = SummaryWriter(save_path)
    agent = DQNAgent(env.observation_space.shape, env.action_space.n, log_writer, config)

    optimizer = optim.Adam(agent.net.parameters(), lr=config['lr'])
    
    training_iter = 0
    play(agent, env, save_video=save_path / "iter-{}-init.gif".format(training_iter))
    
    for episode in range(config['n_episodes']):
        total_loss = 0 
        total_reward = 0 # sum of rewards received in the episode
        obs = env.reset() 
        obs = preprocess_obs(obs)

        for step in range(config['episode_max_steps']):
            action = agent.select_action(obs, training_iter)
            next_obs, reward, done, _ = env.step(action)
            next_obs = preprocess_obs(next_obs)

            loss = agent.training_step(obs, action, reward, next_obs, training_iter)
            obs = next_obs

            # loss is None when the replay buffer is not filled enough
            if loss is not None:
                optimizer.zero_grad()
                loss.backward()
            
                optimizer.step()

                log_writer.add_scalar('Net/loss', loss.item(), training_iter)
                total_loss += loss.item()
                training_iter += 1

            total_reward += reward

            if training_iter >  0 and training_iter % config['video_iter']  == 0:
                play(agent, env, save_video=save_path / "iter-{}.gif".format(training_iter))

            if done:
                break

        log_writer.add_scalar('Episode/steps', step, episode)
        log_writer.add_scalar('Episode/total reward', total_reward, episode)
        log_writer.add_scalar('Net/episode averaged loss', total_loss/(step+1), episode)
        # print("Episode {}: total_reward= {}, loss={}".format(episode, total_reward, total_loss/(step+1)))
        print("Episode {}: steps = {}".format(episode, step))


        if episode % config['target_update'] == 0:
            agent.target_net.load_state_dict(agent.net.state_dict())

    save_model(agent.net, config, save_path)
    play(agent, env, save_video=save_path / "iter-{}-final.gif".format(training_iter))


def play(agent, env, save_video=None, max_steps=50):
    obs = env.reset() 

    if save_video is None:
        env.render()
    else:
        frames = []

    for i in range(max_steps):
        obs = preprocess_obs(obs)
        obs = torch.tensor(obs).type(torch.float)

        if agent is not None:
            Q = agent.net(obs).detach().numpy()
            action = np.argmax(Q)
        else:
            action = np.random.randint(env.action_space.n)
        next_obs, reward, done, _ = env.step(action) 
        obs = next_obs

        if save_video is None:
            env.render()
        else:
            frames.append(env.render('rgb_array'))

        if done:
            break
    
    if save_video is not None:
        write_gif(np.array(frames), save_video)


env = gym.make(env_name)
env = FullyObsWrapper(env)
env = ImgObsWrapper(env)

train(env)
# play(None, env)
# for i in range(3):
#     play(None, env)
