import torch
import gym
import numpy as np

import minerl
import environment

from src.my_utils import resize
from src.wrappers import make_env


# ENV_NAME = 'Mineline-v0'
HIDDEN_DIM = 64
N_EPISODES = 200 

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("Mineline-v0")
env = make_env(env)

NB_MAX_STEP = env.unwrapped.spec.max_episode_steps
NOOP = None if not hasattr(env.action_space, 'noop') else env.action_space.noop()

def main():
    state = env.unwrapped.reset()
    state = state['pov']

    print("state[pov]", state)
    print("pov shape", state.shape)
    state = np.ascontiguousarray(state, dtype=np.float32) / 255
    print("np.ascontiguousarray", state)
    state = torch.from_numpy(state.copy())
    print("state tensor", state)
    state_after = resize(state).unsqueeze(1)
    print("state unsqueeze", state_after)

    print(env.action_space['attack'])
    print(env.unwrapped.observation_space['pov'])
    
    for episode in range(N_EPISODES):
        next_state = env.unwrapped.reset()
        
        for step in range(NB_MAX_STEP):
            state = next_state
            next_state, reward, done, _ = env.step(NOOP)

            print('Episode {:3}  |  Step {:4}  |  Reward {:2}'.format(episode, step, reward))

            if done: break

if __name__ == "__main__":
    main()
