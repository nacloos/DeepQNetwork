import random
from collections import deque


class ReplayMemory():
    # See https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    def __init__(self, capacity):
        self.buffer = deque([], maxlen=capacity)

    def store(self, *args):
        self.buffer.append(*args)

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        return list(zip(*transitions)) # transpose the batch (see https://stackoverflow.com/a/19343/3343043)

    def __len__(self):
        return len(self.buffer)



if __name__ == '__main__':
    replay_memory = ReplayMemory(3)
    replay_memory.store((1, 2, 4, 5))
    replay_memory.store((1, 0, 4, 5))
    replay_memory.store((10, 255, -4, 0))
    print(replay_memory.buffer)
    x = replay_memory.sample(2)
    print(x)

