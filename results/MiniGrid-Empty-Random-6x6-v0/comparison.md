## Comparison

Env: MiniGrid-Empty-Random-6x6-v0

It is a bit difficult to compare algorithms using epsilon greedy for the exploration as the number of iterations needed to solve the game must be specified in advance throught the espilon decay scheduling.



### Effect of update_target:

Remove dueling and it seems to not work anymore... Have to increase a bit update_target.

Double, dueling =  False, n_iter = 2500

*run-20210515091638* (red): update_target = 200

*run-20210515090927* (darkblue): update_target = 500

<img src="D:\UCL\Master\Q2\Data mining and decision making\projects\project-RL\DeepQNetwork\results\MiniGrid-Empty-Random-6x6-v0\figures\target_update_reward.png" alt="target_update_reward" style="zoom:70%;" />

<img src="D:\UCL\Master\Q2\Data mining and decision making\projects\project-RL\DeepQNetwork\results\MiniGrid-Empty-Random-6x6-v0\figures\target_update_loss.png" alt="target_update_reward" style="zoom:70%;" />

Increase the number of training iteration ? *run-20210515082502* (light blue): 5000 iter but still fails.

Add double ? *run-20210515075832*: 2500 iter, double = True but still fails.



### Effect of Dueling

target_update = 500

*run-20210515103018* (red): double=dueling=false

*run-20210515102300* (blue): double=true, dueling=false

*run-20210515101746* (orange): double=dueling=true

<img src="D:\UCL\Master\Q2\Data mining and decision making\projects\project-RL\DeepQNetwork\results\MiniGrid-Empty-Random-6x6-v0\figures\dueling_reward.PNG" alt="dueling_reward" style="zoom:67%;" />

<img src="D:\UCL\Master\Q2\Data mining and decision making\projects\project-RL\DeepQNetwork\results\MiniGrid-Empty-Random-6x6-v0\figures\dueling_loss.PNG" alt="dueling_loss" style="zoom:67%;" />

With dueling can take update_target = 200 with dueling and not without, but has no effect on the performance.

### Effect of replay memory

Train with batch_size = 1 so that can compare with replay_memory = 1. Have to increase the number of training iterations and decrease the learning rate compared to batch_size = 256.

*run-20210515145646* (blue): replay_memory = 10000

*run-20210515150106* (pink): replay_memory = 1

<img src="D:\UCL\Master\Q2\Data mining and decision making\projects\project-RL\DeepQNetwork\results\MiniGrid-Empty-Random-6x6-v0\figures\replay_memory_reward.PNG" alt="replay_memory_reward" style="zoom:67%;" />

### Conclusion

It seems that with this small and easy environment, improvements such as double and dueling don't have any effect on the performance. But target network and replay memory are necessary.





Problem with 16x16 env: the reward is sparse and there is a very small probability to reach the goal by taking random actions.