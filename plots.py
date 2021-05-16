from pathlib import Path
import numpy as np
from scipy.ndimage.filters import uniform_filter1d, gaussian_filter1d
import matplotlib.pyplot as plt


data_path = Path("results\MiniGrid-Empty-Random-6x6-v0\data")

plotname = "update_target"
filenames = [
    "run-MiniGrid-Empty-Random-6x6-v0_DQN_run-20210515091638-tag-Episode_total reward.csv",
    "run-MiniGrid-Empty-Random-6x6-v0_DQN_run-20210515090927-tag-Episode_total reward.csv",
    "run-MiniGrid-Empty-Random-6x6-v0_DQN_run-20210515181546-tag-Episode_total reward.csv",
]
labels = ["Update target = 200", "Update target = 500", "Update target = 1000"]
colors = ["cornflowerblue", "coral", "darkgray"]

# plotname = "replay_memory"
# filenames = [
#     "run-MiniGrid-Empty-Random-6x6-v0_DQN_run-20210515145646-tag-Episode_total reward.csv",
#     "run-MiniGrid-Empty-Random-6x6-v0_DQN_run-20210515150106-tag-Episode_total reward.csv",
# ]
# labels = ["Replay memory", "No replay memory"]
# colors = ["darkseagreen", "gray"]


plt.figure(figsize=(5
, 3.5), dpi=130)
ax = plt.gca()

for i, filename in enumerate(filenames):
    data = np.genfromtxt(data_path / filename, delimiter=',', skip_header=True)[:,1:]
    averaged_rewards = uniform_filter1d(data[:,1], 10)

    plt.plot(data[:,0], averaged_rewards, label=labels[i], color=colors[i], lw=1.8, zorder=10 if i == 1 else 0)

plt.xlabel("Number of episodes")
plt.ylabel("Total reward")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend()
plt.tight_layout()
plt.savefig("D:\\UCL\\Master\\Q2\\Data mining and decision making\\projects\\project-RL\\report\\figures\\" + plotname + ".pdf", transparent=True)
plt.show()

