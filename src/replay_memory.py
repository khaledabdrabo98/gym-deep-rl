import collections
import random

Experience = collections.namedtuple(
    'Experience', ['state', 'action', 'next_state', 'reward', 'done'])


class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = collections.deque([], maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, *args):
        self.buffer.append(Experience(*args))

    # Sample a random batch of experiences from the agent's memory
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
